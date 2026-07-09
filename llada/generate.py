# This code is adapted from: https://github.com/ML-GSAI/LLaDA and https://github.com/NVlabs/Fast-dLLM
import torch
import numpy as np
import torch.nn.functional as F
from model.modeling_llada import LLaDAModelLM
from model.small_model import LogisticRegression
from model.cora_diff import (
    CoraDiffConfig,
    CoraDiffState,
    calibrate_cora_channel_ordering,
    prepare_cora_channel_ordering,
)
from vendor_methods.dapd.generation import generate_dapd_with_blocks
from vendor_methods.klass.llada_klass import stable_confident_decode as klass_stable_confident_decode
from vendor_methods.prophet.generate_earlyexit import generate as prophet_generate

from transformers import AutoTokenizer


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def _masked_zscore(values, mask, eps=1e-6):
    out = torch.zeros_like(values, dtype=torch.float32)
    selected = values[mask].to(dtype=torch.float32)
    if selected.numel() <= 1:
        return out
    out[mask] = (selected - selected.mean()) / (selected.std(unbiased=False) + eps)
    return out


def _topk_distribution_drift(current_ids, current_probs, prev_ids, prev_probs):
    """Top-k approximation to L1 drift over the union of current/previous top-k ids."""
    if prev_ids is None or prev_probs is None:
        return torch.zeros(current_ids.shape[:-1], device=current_probs.device, dtype=torch.float32)

    current_probs = current_probs.to(dtype=torch.float32)
    prev_probs = prev_probs.to(device=current_probs.device, dtype=torch.float32)
    prev_ids = prev_ids.to(device=current_ids.device)

    # match[b, l, k_current, k_prev] says whether a current top-k id was also in previous top-k.
    match = current_ids.unsqueeze(-1) == prev_ids.unsqueeze(-2)
    prev_prob_for_current = (match.to(dtype=torch.float32) * prev_probs.unsqueeze(-2)).sum(dim=-1)
    current_union_drift = (current_probs - prev_prob_for_current).abs().sum(dim=-1)

    prev_seen_in_current = match.any(dim=-2)
    missing_prev_drift = prev_probs.masked_fill(prev_seen_in_current, 0.0).sum(dim=-1)
    return current_union_drift + missing_prev_drift


@ torch.no_grad()
def generate_predict_eot(model: LLaDAModelLM, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if not mask_index.any():
                break
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            p = None
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            # before x0 and x0_p get masked
            end_token_conf = torch.where((x0 == 126081) & mask_index, x0_p, -np.inf)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            if (end_token_conf > -np.inf).any():
                # If the end token is present, we find its position and truncate the sequence.
                # This is to ensure that the generation stops at the end token.
                position_of_end_token = end_token_conf.argmax(dim=-1)
                # print(f"Position of end token: {position_of_end_token}/{total_length}")
                x = x[:, :position_of_end_token + 1]

    return x


@ torch.no_grad()
def generate_learn2parallel(model: LLaDAModelLM, small_model: LogisticRegression, accept_thres: float, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # results = []
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        # prev_data = None
        while (x[:, block_start:block_end] == mask_id).any():
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            p = None
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            # small model predict
            block_confidence = x0_p[:, block_start:block_end]
            block_logists = small_model(block_confidence.to(dtype=torch.float32))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            transfer_index[:, block_start:block_end] = (torch.sigmoid(block_logists) > accept_thres)
            transfer_index = torch.where(mask_index, transfer_index, False)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                # unmask the highest
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
            i += 1

    return x


@ torch.no_grad()
def generate_l2p_eot(model: LLaDAModelLM, small_model: LogisticRegression, accept_thres: float, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while (x[:, block_start:block_end] == mask_id).any():
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            # small model predict
            block_confidence = x0_p[:, block_start:block_end]
            block_logists = small_model(block_confidence.to(dtype=torch.float32))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            transfer_index[:, block_start:block_end] = (torch.sigmoid(block_logists) > accept_thres)
            transfer_index = torch.where(mask_index, transfer_index, False)

            # Get small_model's 'endoftext' prediction
            small_model_eot = ((x0 == 126081) & transfer_index)
            # before x0 and x0_p get masked
            end_token_conf = torch.where((x0 == 126081) & mask_index, x0_p, -np.inf)
            
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            for j in range(confidence.shape[0]):
                # unmask the highest
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]

            # ignore any token after 'endoftext' token
            if small_model_eot.any():
                position_of_end_token = small_model_eot.to(torch.int8).argmax(dim=-1)[0]
                x = x[:, :position_of_end_token + 1]
            elif (end_token_conf > -np.inf).any():
                # If the end token is present, we find its position and truncate the sequence.
                # This is to ensure that the generation stops at the end token.
                position_of_end_token = end_token_conf.argmax(dim=-1)
                x = x[:, :position_of_end_token + 1]
            i += 1

    return x


@ torch.no_grad()
def generate(model: LLaDAModelLM, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


@ torch.no_grad()
def generate_prophet(
    model: LLaDAModelLM,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    constraints=None,
    answer_start_pos=None,
    answer_start_offset=None,
    answer_length=5,
    early_exit_thresholds=None,
    return_stats=False,
):
    """LLaDA generation with Prophet logits-gap early exit."""
    if answer_start_pos is None and answer_start_offset is not None:
        answer_start_pos = prompt.shape[1] + int(answer_start_offset)
    out, gap_data = prophet_generate(
        model,
        prompt,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        constraints=constraints,
        analyze_gap=True,
        answer_start_pos=answer_start_pos,
        answer_length=answer_length,
        early_exit_thresholds=early_exit_thresholds,
    )
    if return_stats:
        exit_info = gap_data.get("exit_info", {})
        actual_steps = float(exit_info.get("actual_steps", steps))
        return out, {
            "planned_denoising_steps": float(steps),
            "actual_denoising_steps": actual_steps,
            "step_ratio": float(actual_steps / max(1, steps)),
            "early_exit_triggered": float(bool(exit_info.get("early_exit_triggered", False))),
            "exit_decision_step": -1.0
            if exit_info.get("exit_decision_step") is None
            else float(exit_info.get("exit_decision_step")),
        }
    return out


@ torch.no_grad()
def generate_klass(
    model: LLaDAModelLM,
    prompt,
    tokenizer=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    cfg_scale=0.,
    mask_id=126336,
    conf_threshold=0.9,
    kl_threshold=0.01,
    history_length=2,
    unmask_strategy="all",
    confidence_metric="prob",
    alg="klass",
    step_save_dir=None,
    example_idx=0,
    return_stats=False,
):
    """KLASS: high-confidence and low-KL-stability token unmasking.

    Adapted from https://github.com/shkim0116/KLASS.
    """
    if cfg_scale > 0.:
        raise NotImplementedError("The vendored KLASS LLaDA implementation supports cfg_scale=0 only.")
    out, used_steps = klass_stable_confident_decode(
        model,
        tokenizer,
        prompt,
        gen_length,
        steps,
        block_length,
        temperature=temperature,
        mask_id=mask_id,
        conf_threshold=conf_threshold,
        kl_threshold=kl_threshold,
        kl_history_length=history_length,
        step_save_dir=step_save_dir,
        example_idx=example_idx,
        alg=alg,
        unmask_strategy=unmask_strategy,
    )
    if return_stats:
        return out, {
            "planned_denoising_steps": float(steps),
            "actual_denoising_steps": float(used_steps),
            "step_ratio": float(used_steps / max(1, steps)),
            "ready_tokens": 0.0,
            "fallback_tokens": 0.0,
            "avg_kl": 0.0,
        }
    return out


@ torch.no_grad()
def generate_dapd(
    model: LLaDAModelLM,
    prompt,
    gen_length=128,
    block_length=128,
    mask_id=126336,
    dapd_alg="dapd_staged",
    dapd_layer_ratio=0.3,
    dapd_tau_min=0.01,
    dapd_tau_max=0.15,
    dapd_verbose=False,
    dapd_collect_step_history=False,
    return_stats=False,
):
    out, stats = generate_dapd_with_blocks(
        model=model,
        prompt=prompt,
        gen_length=gen_length,
        block_length=block_length,
        mask_id=mask_id,
        alg=dapd_alg,
        layer_ratio=dapd_layer_ratio,
        tau_min=dapd_tau_min,
        tau_max=dapd_tau_max,
        verbose=dapd_verbose,
        collect_step_history=dapd_collect_step_history,
    )
    if return_stats:
        return out, stats
    return out


@ torch.no_grad()
def generate_cora(model: LLaDAModelLM, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, cora_num_groups=4,
             cora_core_ratio=0.5, cora_active_ratio=0.6, cora_alpha=0.3, cora_beta=1.0,
             cora_gamma=1.0, cora_dependency_topk=16, cora_order_channels=True,
             cora_channel_ordering='activation', cora_extra_commit=True,
             cora_commit_residual_threshold=0.8, cora_commit_confidence_threshold=0.9,
             cora_budget_schedule='constant', cora_active_ratio_final=None,
             cora_routing_mode='adaptive',
             cora_fast_accept=False, cora_accept_confidence_threshold=0.9,
             cora_accept_stability_steps=1, cora_eot=False,
             cora_eot_id=126081, cora_eot_confidence_threshold=0.0,
             cora_residual_accept=False, cora_residual_threshold=0.0,
             cora_drift_topk=8, cora_residual_beta=1.0,
             cora_residual_gamma=1.0, cora_persistence_temperature=2.0,
             cora_score_eps=1e-6,
             return_cora_stats=False):
    '''
    CORA-Diff style generation for LLaDA.

    This keeps the normal diffusion token transfer rule, but routes unresolved
    tokens through token-wise MLP refinement depths inside the model forward.
    The first denoising pass runs dense MLP to initialize prediction evidence.
    '''
    if cfg_scale > 0.:
        raise NotImplementedError("generate_cora currently supports cfg_scale=0 only.")

    cora_state = CoraDiffState(
        CoraDiffConfig(
            routing_mode=cora_routing_mode,
            num_refinement_groups=cora_num_groups,
            core_ratio=cora_core_ratio,
            active_ratio=cora_active_ratio,
            active_ratio_final=cora_active_ratio_final,
            budget_schedule=cora_budget_schedule,
            alpha=cora_alpha,
            beta=cora_beta,
            gamma=cora_gamma,
            dependency_topk=cora_dependency_topk,
        )
    )

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if cora_order_channels and str(cora_routing_mode).lower() not in ('disabled', 'off', 'none'):
        if str(cora_channel_ordering).lower() == 'activation':
            calibrate_cora_channel_ordering(model, x)
        elif str(cora_channel_ordering).lower() == 'weight':
            prepare_cora_channel_ordering(model)
        else:
            raise ValueError("cora_channel_ordering must be 'activation' or 'weight'.")

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    total_denoising_steps = num_blocks * steps
    actual_denoising_steps = 0
    fast_accept_tokens = 0
    residual_accept_tokens = 0
    residual_score_sum = 0.0
    residual_score_count = 0
    eot_truncated = 0

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        if block_start >= x.shape[1]:
            break
        block_end = min(prompt.shape[1] + (num_block + 1) * block_length, x.shape[1])
        block_slice = slice(block_start, block_end)
        block_mask_index = (x[:, block_slice] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        stability_count = torch.zeros_like(x, dtype=torch.int16, device=x.device)
        prev_topk_ids = None
        prev_topk_probs = None
        for i in range(steps):
            if block_start >= x.shape[1] or not (x[:, block_slice] == mask_id).any():
                break

            cora_state.set_budget_position(num_block * steps + i, total_denoising_steps)
            mask_index = (x == mask_id)
            current_block_mask = torch.zeros_like(mask_index, dtype=torch.bool, device=x.device)
            current_block_mask[:, block_slice] = True
            logits = model(x, cora_state=cora_state, cora_unresolved_mask=mask_index).logits
            actual_denoising_steps += 1

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            p = None
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            state_pred = x0.clone()
            state_confidence = x0_p.clone()
            raw_confidence = x0_p.clone()
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            if cora_fast_accept or cora_residual_accept:
                if cora_state.prev_pred is not None and cora_state.prev_pred.shape == x0.shape:
                    stable_prediction = x0 == cora_state.prev_pred.to(device=x0.device)
                else:
                    stable_prediction = torch.zeros_like(mask_index, dtype=torch.bool, device=x0.device)
                score_mask = mask_index & current_block_mask
                stability_count = torch.where(
                    score_mask & stable_prediction,
                    stability_count + 1,
                    torch.zeros_like(stability_count),
                )

                if cora_residual_accept:
                    if p is None:
                        p = F.softmax(logits, dim=-1)
                    drift_topk = max(1, min(int(cora_drift_topk), p.shape[-1]))
                    current_topk_probs, current_topk_ids = torch.topk(p, k=drift_topk, dim=-1)
                    drift = _topk_distribution_drift(
                        current_topk_ids,
                        current_topk_probs,
                        prev_topk_ids,
                        prev_topk_probs,
                    )
                    uncertainty = 1.0 - raw_confidence.to(dtype=torch.float32)
                    uncertainty_z = _masked_zscore(uncertainty, score_mask, eps=float(cora_score_eps))
                    drift_z = _masked_zscore(drift, score_mask, eps=float(cora_score_eps))
                    persistence_penalty = torch.exp(
                        -stability_count.to(dtype=torch.float32)
                        / max(float(cora_persistence_temperature), float(cora_score_eps))
                    )
                    residual_score = (
                        uncertainty_z
                        + float(cora_residual_beta) * drift_z
                        + float(cora_residual_gamma) * persistence_penalty
                    )
                    selected_scores = residual_score[score_mask]
                    if selected_scores.numel() > 0:
                        residual_score_sum += float(selected_scores.sum().item())
                        residual_score_count += int(selected_scores.numel())
                    residual_allowed = residual_score < float(cora_residual_threshold)
                    prev_topk_ids = current_topk_ids.detach()
                    prev_topk_probs = current_topk_probs.detach()
                else:
                    residual_allowed = torch.ones_like(mask_index, dtype=torch.bool, device=x0.device)

                fast_accept_index = (
                    score_mask
                    & residual_allowed
                    & (raw_confidence >= float(cora_accept_confidence_threshold))
                    & (stability_count >= int(cora_accept_stability_steps))
                )
                fast_accept_tokens += int(fast_accept_index.sum().item())
                if cora_residual_accept:
                    residual_accept_tokens += int(fast_accept_index.sum().item())
                transfer_index |= fast_accept_index

            if cora_extra_commit:
                transfer_index |= cora_state.stable_commit_mask(
                    x0,
                    confidence,
                    mask_index,
                    residual_threshold=cora_commit_residual_threshold,
                    confidence_threshold=cora_commit_confidence_threshold,
                )

            for j in range(confidence.shape[0]):
                remaining = int(num_transfer_tokens[j, i].item()) - int(transfer_index[j].sum().item())
                if remaining <= 0:
                    continue
                topk_confidence = confidence[j].clone()
                topk_confidence[transfer_index[j]] = -np.inf
                _, select_index = torch.topk(topk_confidence, k=remaining)
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            cora_state.update_predictions(state_pred, state_confidence)

            if cora_eot:
                eot_candidate = (
                    transfer_index
                    & current_block_mask
                    & (x0 == int(cora_eot_id))
                    & (raw_confidence >= float(cora_eot_confidence_threshold))
                )
                if eot_candidate.any():
                    position_of_end_token = eot_candidate.to(torch.int8).argmax(dim=-1)[0]
                    x = x[:, :position_of_end_token + 1]
                    eot_truncated = 1
                    break

    if return_cora_stats:
        stats = cora_state.stats_dict()
        stats.update(
            {
                "planned_denoising_steps": float(total_denoising_steps),
                "actual_denoising_steps": float(actual_denoising_steps),
                "step_ratio": float(actual_denoising_steps / max(1, total_denoising_steps)),
                "fast_accept_tokens": float(fast_accept_tokens),
                "residual_accept_tokens": float(residual_accept_tokens),
                "avg_residual_score": float(residual_score_sum / max(1, residual_score_count)),
                "eot_truncated": float(eot_truncated),
            }
        )
        return x, stats
    return x


#------Beginning of code adopted from Fast-dLLM-----#
@ torch.no_grad()
def generate_with_prefix_cache(model, small_model: LogisticRegression, accept_thres: float, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        if current_block_start >= x.shape[1]:
            break
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index, position_of_end_token = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0], small_model, accept_thres, current_block_start, current_block_end)
        x[transfer_index] = x0[transfer_index]
        if position_of_end_token is not None:
            x = x[:, :position_of_end_token + 1]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            x0, transfer_index, position_of_end_token = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i], small_model, accept_thres, block_start=0, block_end=current_block_end-current_block_start)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            if position_of_end_token is not None:
                x = x[:, :current_block_start + position_of_end_token + 1]
            
            i += 1

    return x


@ torch.no_grad()
def generate_with_dual_cache(model, small_model: LogisticRegression, accept_thres: float, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        if current_block_start >= x.shape[1]:
            break
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index, position_of_end_token = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0], small_model, accept_thres, current_block_start, current_block_end)
        x[transfer_index] = x0[transfer_index]
        if position_of_end_token is not None:
            x = x[:, :position_of_end_token + 1]

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            x0, transfer_index, position_of_end_token = get_transfer_index(logits, temperature, remasking, mask_index, 
                                            x[:, current_block_start:current_block_end], num_transfer_tokens[:, i], small_model, accept_thres)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            if position_of_end_token is not None:
                x = x[:, :current_block_start + position_of_end_token + 1]
                replace_position[:, current_block_start + position_of_end_token + 1:] = 0
            i += 1

    return x


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, small_model: LogisticRegression, accept_thres: float, block_start = None, block_end = None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    # small model predict
    if block_start is not None and block_end is not None:
        block_confidence = x0_p[:, block_start:block_end]
        block_logists = small_model(block_confidence.to(dtype=torch.float32))
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        transfer_index[:, block_start:block_end] = (torch.sigmoid(block_logists) > accept_thres)
    else:
        block_confidence = x0_p
        block_logists = small_model(block_confidence.to(dtype=torch.float32))
        transfer_index = (torch.sigmoid(block_logists) > accept_thres)
    transfer_index = torch.where(mask_index, transfer_index, False)

    # Get small_model's 'endoftext' prediction
    small_model_eot = ((x0 == 126081) & transfer_index)
    # before x0 and x0_p get masked
    end_token_conf = torch.where((x0 == 126081) & mask_index, x0_p, -np.inf)
    # ignore any token after 'endoftext' token
    position_of_end_token = None
    if small_model_eot.any():
        position_of_end_token = small_model_eot.to(torch.int8).argmax(dim=-1)[0]
    elif (end_token_conf > -np.inf).any():
        # If the end token is present, we find its position and truncate the sequence.
        # This is to ensure that the generation stops at the end token.
        position_of_end_token = end_token_conf.argmax(dim=-1)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    for j in range(confidence.shape[0]):
        # unmask the highest
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
    return x0, transfer_index, position_of_end_token
#----Beginning of code adopted from Fast-dLLM----#


def main():
    device = 'cuda:0'

    small_model = LogisticRegression(32)
    small_model.load_state_dict(torch.load('layer_2_flan.pth'))
    small_model.to(device).eval()

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours? Give me the number only"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    import time
    start = time.time()
    out = generate(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_predict_eot(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_learn2parallel(model, small_model, 0.96, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_l2p_eot(model, small_model, 0.96, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_with_prefix_cache(model, small_model, 0.96, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., remasking='low_confidence')
    # out = generate_with_dual_cache(model, small_model, 0.96, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., remasking='low_confidence')
    print(f'{time.time() - start}')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
