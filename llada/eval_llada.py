# This code is adapted from: https://github.com/ML-GSAI/LLaDA
'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from model.modeling_llada import LLaDAModelLM

from transformers import AutoTokenizer, AutoModel
from generate import (
    generate,
    generate_predict_eot,
    generate_learn2parallel,
    generate_l2p_eot,
    generate_cora,
    generate_prophet,
    generate_klass,
    generate_dapd,
    generate_with_prefix_cache,
    generate_with_dual_cache,
)
from model.small_model import LogisticRegression
import time
import os

# prevent crash due to waiting for other processes for more than 10 minutes
import datetime
if all(name in os.environ for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")):
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().strip("'\"").lower() in ("1", "true", "yes", "y", "t")


def _parse_constraints(text, tokenizer):
    """Parse Prophet constraints like '200:The|201:answer|202:is'."""
    constraints = {}
    if text is None or str(text).strip() == "":
        return constraints
    for part in str(text).split("|"):
        if ":" not in part:
            continue
        pos_str, token_text = part.split(":", 1)
        try:
            pos = int(pos_str.strip())
        except ValueError:
            continue
        token_ids = tokenizer.encode(" " + token_text.strip(), add_special_tokens=False)
        for offset, token_id in enumerate(token_ids):
            constraints[pos + offset] = token_id
    return constraints


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        method="original",
        accept_thres=None,
        cora_num_groups=4,
        cora_core_ratio=0.5,
        cora_active_ratio=0.6,
        cora_alpha=0.3,
        cora_beta=1.0,
        cora_gamma=1.0,
        cora_dependency_topk=16,
        cora_order_channels=True,
        cora_channel_ordering='activation',
        cora_extra_commit=True,
        cora_commit_residual_threshold=0.8,
        cora_commit_confidence_threshold=0.9,
        cora_budget_schedule='constant',
        cora_active_ratio_final=None,
        cora_routing_mode='adaptive',
        cora_fast_accept=False,
        cora_accept_confidence_threshold=0.9,
        cora_accept_stability_steps=1,
        cora_eot=False,
        cora_eot_id=126081,
        cora_eot_confidence_threshold=0.0,
        cora_residual_accept=False,
        cora_residual_threshold=0.0,
        cora_drift_topk=8,
        cora_residual_beta=1.0,
        cora_residual_gamma=1.0,
        cora_persistence_temperature=2.0,
        prophet_constraints_text='',
        prophet_answer_start=None,
        prophet_answer_length=5,
        prophet_early_threshold=7.5,
        prophet_mid_threshold=5.0,
        prophet_late_threshold=2.5,
        klass_conf_threshold=0.9,
        klass_kl_threshold=0.01,
        klass_history_length=2,
        klass_unmask_strategy='all',
        klass_confidence_metric='prob',
        klass_alg='klass',
        klass_step_save_dir=None,
        dapd_alg='dapd_staged',
        dapd_layer_ratio=0.3,
        dapd_tau_min=0.01,
        dapd_tau_max=0.15,
        dapd_verbose=False,
        dapd_collect_step_history=False,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        small_model_path='layer_2_flan.pth'
        self.small_model = LogisticRegression(32)
        self.small_model.load_state_dict(torch.load(small_model_path))
        self.small_model.eval()

        self.model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.small_model = self.accelerator.prepare(self.small_model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)
            self.small_model = self.small_model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.is_instruct = True if 'instruct' in model_path.lower() else False
        
        self.method = method
        self.accept_thres = accept_thres
        self.cora_num_groups = int(cora_num_groups)
        self.cora_core_ratio = float(cora_core_ratio)
        self.cora_active_ratio = float(cora_active_ratio)
        self.cora_alpha = float(cora_alpha)
        self.cora_beta = float(cora_beta)
        self.cora_gamma = float(cora_gamma)
        self.cora_dependency_topk = int(cora_dependency_topk)
        self.cora_order_channels = str(cora_order_channels).lower() not in ("false", "0", "no")
        self.cora_channel_ordering = str(cora_channel_ordering)
        self.cora_extra_commit = str(cora_extra_commit).lower() not in ("false", "0", "no")
        self.cora_commit_residual_threshold = float(cora_commit_residual_threshold)
        self.cora_commit_confidence_threshold = float(cora_commit_confidence_threshold)
        self.cora_budget_schedule = str(cora_budget_schedule)
        self.cora_active_ratio_final = None if cora_active_ratio_final is None else float(cora_active_ratio_final)
        self.cora_routing_mode = str(cora_routing_mode)
        self.cora_fast_accept = str(cora_fast_accept).lower() not in ("false", "0", "no")
        self.cora_accept_confidence_threshold = float(cora_accept_confidence_threshold)
        self.cora_accept_stability_steps = max(1, int(cora_accept_stability_steps))
        self.cora_eot = str(cora_eot).lower() not in ("false", "0", "no")
        self.cora_eot_id = int(cora_eot_id)
        self.cora_eot_confidence_threshold = float(cora_eot_confidence_threshold)
        self.cora_residual_accept = str(cora_residual_accept).lower() not in ("false", "0", "no")
        self.cora_residual_threshold = float(cora_residual_threshold)
        self.cora_drift_topk = int(cora_drift_topk)
        self.cora_residual_beta = float(cora_residual_beta)
        self.cora_residual_gamma = float(cora_residual_gamma)
        self.cora_persistence_temperature = float(cora_persistence_temperature)
        self.prophet_constraints_text = str(prophet_constraints_text)
        self.prophet_answer_start = None if prophet_answer_start is None else int(prophet_answer_start)
        self.prophet_answer_length = int(prophet_answer_length)
        self.prophet_early_threshold = float(prophet_early_threshold)
        self.prophet_mid_threshold = float(prophet_mid_threshold)
        self.prophet_late_threshold = float(prophet_late_threshold)
        self.klass_conf_threshold = float(klass_conf_threshold)
        self.klass_kl_threshold = float(klass_kl_threshold)
        self.klass_history_length = max(1, int(klass_history_length))
        self.klass_unmask_strategy = str(klass_unmask_strategy)
        self.klass_confidence_metric = str(klass_confidence_metric)
        self.klass_alg = str(klass_alg)
        self.klass_step_save_dir = None if klass_step_save_dir is None else str(klass_step_save_dir)
        self.dapd_alg = str(dapd_alg)
        if method == "DAPD-Direct":
            self.dapd_alg = "dapd_direct"
        elif method == "DAPD-Staged":
            self.dapd_alg = "dapd_staged"
        self.dapd_layer_ratio = float(dapd_layer_ratio)
        self.dapd_tau_min = float(dapd_tau_min)
        self.dapd_tau_max = float(dapd_tau_max)
        self.dapd_verbose = _as_bool(dapd_verbose)
        self.dapd_collect_step_history = _as_bool(dapd_collect_step_history)
        # verify input
        acceptable_methods = [
            "original",
            "L2P",
            "Learn2PD",
            "EoT",
            "L2P+EoT",
            "L2P+EoT+dual_cache",
            "L2P+EoT+prefix_cache",
            "CORA",
            "Prophet",
            "KLASS",
            "DAPD",
            "DAPD-Direct",
            "DAPD-Staged",
        ]
        assert method in acceptable_methods, f"method must be one of {acceptable_methods}"
        if "L2P" in method or method == "Learn2PD":
            assert accept_thres != None, "accept_thres must be set!"

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        out = []
        num_tokens = 0
        cora_stats_count = 0
        cora_refinement_ratio_sum = 0.0
        cora_ffn_ratio_sum = 0.0
        cora_step_ratio_sum = 0.0
        cora_fast_accept_tokens_sum = 0.0
        cora_eot_truncated_sum = 0.0
        cora_residual_accept_tokens_sum = 0.0
        cora_residual_score_sum = 0.0
        prophet_stats_count = 0
        prophet_step_ratio_sum = 0.0
        prophet_actual_steps_sum = 0.0
        prophet_early_exit_sum = 0.0
        klass_stats_count = 0
        klass_step_ratio_sum = 0.0
        klass_actual_steps_sum = 0.0
        klass_ready_tokens_sum = 0.0
        klass_fallback_tokens_sum = 0.0
        klass_kl_sum = 0.0
        dapd_stats_count = 0
        dapd_step_ratio_sum = 0.0
        dapd_total_steps_sum = 0.0
        dapd_avg_tokens_per_step_sum = 0.0
        start_time = time.perf_counter()
        for example_idx, req in enumerate(tqdm(requests, desc="Generating...")):
            question = req.args[0]
            if (not self.is_instruct) or ('task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval')):
                user_input = question
                input_ids = self.tokenizer(user_input)['input_ids']
            else:
                m = [{"role": "user", "content": question}]
                user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                input_ids = self.tokenizer(user_input)['input_ids']

            stop_tokens = req.args[1]['until']
            input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
            
            if self.method == "original":
                generated_answer = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, temperature=0, cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_id)
            elif self.method == "CORA":
                generated_answer, cora_stats = generate_cora(
                    self.model,
                    input_ids,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    cora_num_groups=self.cora_num_groups,
                    cora_core_ratio=self.cora_core_ratio,
                    cora_active_ratio=self.cora_active_ratio,
                    cora_alpha=self.cora_alpha,
                    cora_beta=self.cora_beta,
                    cora_gamma=self.cora_gamma,
                    cora_dependency_topk=self.cora_dependency_topk,
                    cora_order_channels=self.cora_order_channels,
                    cora_channel_ordering=self.cora_channel_ordering,
                    cora_extra_commit=self.cora_extra_commit,
                    cora_commit_residual_threshold=self.cora_commit_residual_threshold,
                    cora_commit_confidence_threshold=self.cora_commit_confidence_threshold,
                    cora_budget_schedule=self.cora_budget_schedule,
                    cora_active_ratio_final=self.cora_active_ratio_final,
                    cora_routing_mode=self.cora_routing_mode,
                    cora_fast_accept=self.cora_fast_accept,
                    cora_accept_confidence_threshold=self.cora_accept_confidence_threshold,
                    cora_accept_stability_steps=self.cora_accept_stability_steps,
                    cora_eot=self.cora_eot,
                    cora_eot_id=self.cora_eot_id,
                    cora_eot_confidence_threshold=self.cora_eot_confidence_threshold,
                    cora_residual_accept=self.cora_residual_accept,
                    cora_residual_threshold=self.cora_residual_threshold,
                    cora_drift_topk=self.cora_drift_topk,
                    cora_residual_beta=self.cora_residual_beta,
                    cora_residual_gamma=self.cora_residual_gamma,
                    cora_persistence_temperature=self.cora_persistence_temperature,
                    return_cora_stats=True,
                )
                cora_stats_count += 1
                cora_refinement_ratio_sum += cora_stats["avg_refinement_ratio"]
                cora_ffn_ratio_sum += cora_stats["estimated_activated_ffn_ratio"]
                cora_step_ratio_sum += cora_stats["step_ratio"]
                cora_fast_accept_tokens_sum += cora_stats["fast_accept_tokens"]
                cora_eot_truncated_sum += cora_stats["eot_truncated"]
                cora_residual_accept_tokens_sum += cora_stats["residual_accept_tokens"]
                cora_residual_score_sum += cora_stats["avg_residual_score"]
            elif self.method == "Prophet":
                prophet_constraints = _parse_constraints(self.prophet_constraints_text, self.tokenizer)
                prophet_answer_start = self.prophet_answer_start
                if prophet_answer_start is None and prophet_constraints:
                    prophet_answer_start = max(prophet_constraints.keys()) + 2
                generated_answer, prophet_stats = generate_prophet(
                    self.model,
                    input_ids,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    constraints=prophet_constraints,
                    answer_start_offset=prophet_answer_start,
                    answer_length=self.prophet_answer_length,
                    early_exit_thresholds={
                        "early": self.prophet_early_threshold,
                        "mid": self.prophet_mid_threshold,
                        "late": self.prophet_late_threshold,
                    },
                    return_stats=True,
                )
                prophet_stats_count += 1
                prophet_step_ratio_sum += prophet_stats["step_ratio"]
                prophet_actual_steps_sum += prophet_stats["actual_denoising_steps"]
                prophet_early_exit_sum += prophet_stats["early_exit_triggered"]
            elif self.method == "KLASS":
                generated_answer, klass_stats = generate_klass(
                    self.model,
                    input_ids,
                    tokenizer=self.tokenizer,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    mask_id=self.mask_id,
                    conf_threshold=self.klass_conf_threshold,
                    kl_threshold=self.klass_kl_threshold,
                    history_length=self.klass_history_length,
                    unmask_strategy=self.klass_unmask_strategy,
                    confidence_metric=self.klass_confidence_metric,
                    alg=self.klass_alg,
                    step_save_dir=self.klass_step_save_dir,
                    example_idx=example_idx,
                    return_stats=True,
                )
                klass_stats_count += 1
                klass_step_ratio_sum += klass_stats["step_ratio"]
                klass_actual_steps_sum += klass_stats["actual_denoising_steps"]
                klass_ready_tokens_sum += klass_stats["ready_tokens"]
                klass_fallback_tokens_sum += klass_stats["fallback_tokens"]
                klass_kl_sum += klass_stats["avg_kl"]
            elif self.method in ("DAPD", "DAPD-Direct", "DAPD-Staged"):
                generated_answer, dapd_stats = generate_dapd(
                    self.model,
                    input_ids,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    mask_id=self.mask_id,
                    dapd_alg=self.dapd_alg,
                    dapd_layer_ratio=self.dapd_layer_ratio,
                    dapd_tau_min=self.dapd_tau_min,
                    dapd_tau_max=self.dapd_tau_max,
                    dapd_verbose=self.dapd_verbose,
                    dapd_collect_step_history=self.dapd_collect_step_history,
                    return_stats=True,
                )
                dapd_stats_count += 1
                dapd_step_ratio_sum += dapd_stats["total_steps"] / max(1, self.steps)
                dapd_total_steps_sum += dapd_stats["total_steps"]
                dapd_avg_tokens_per_step_sum += dapd_stats["avg_tokens_per_step"]
            elif self.method in ("L2P", "Learn2PD"):
                generated_answer = generate_learn2parallel(self.model, self.small_model, self.accept_thres, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, temperature=0, cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_id)
            elif self.method == "EoT":
                generated_answer = generate_predict_eot(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, temperature=0, cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_id)
            elif self.method == "L2P+EoT":
                generated_answer = generate_l2p_eot(self.model, self.small_model, self.accept_thres, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, temperature=0, cfg_scale=self.cfg, remasking=self.remasking)
            elif self.method == "L2P+EoT+dual_cache":
                generated_answer = generate_with_dual_cache(self.model, self.small_model, self.accept_thres, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, temperature=0, remasking=self.remasking)
            elif self.method == "L2P+EoT+prefix_cache":
                generated_answer = generate_with_prefix_cache(self.model, self.small_model, self.accept_thres, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, temperature=0, remasking=self.remasking)

            generated_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = torch.tensor(self.tokenizer(generated_answer)["input_ids"])
            num_tokens += (generated_answer_ids != 126081).sum()
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            # self.accelerator.wait_for_everyone()
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        process_index = self.accelerator.process_index if self.accelerator is not None else 0
        print(f"[GPU{process_index}] Number of tokens: {num_tokens}")
        print(f"[GPU{process_index}] Generation time: {elapsed} seconds")
        print(f"[GPU{process_index}] Tokens per second: {num_tokens / elapsed}")
        if cora_stats_count > 0:
            print(f"[GPU{process_index}] CORA avg refinement ratio: {cora_refinement_ratio_sum / cora_stats_count}")
            print(f"[GPU{process_index}] CORA estimated activated FFN ratio: {cora_ffn_ratio_sum / cora_stats_count}")
            print(f"[GPU{process_index}] CORA actual denoising step ratio: {cora_step_ratio_sum / cora_stats_count}")
            print(f"[GPU{process_index}] CORA fast accept tokens: {cora_fast_accept_tokens_sum}")
            print(f"[GPU{process_index}] CORA residual accept tokens: {cora_residual_accept_tokens_sum}")
            print(f"[GPU{process_index}] CORA avg residual score: {cora_residual_score_sum / cora_stats_count}")
            print(f"[GPU{process_index}] CORA EoT truncated samples: {cora_eot_truncated_sum}")
        if prophet_stats_count > 0:
            print(f"[GPU{process_index}] Prophet avg actual denoising steps: {prophet_actual_steps_sum / prophet_stats_count}")
            print(f"[GPU{process_index}] Prophet actual denoising step ratio: {prophet_step_ratio_sum / prophet_stats_count}")
            print(f"[GPU{process_index}] Prophet early-exit samples: {prophet_early_exit_sum}")
        if klass_stats_count > 0:
            print(f"[GPU{process_index}] KLASS avg actual denoising steps: {klass_actual_steps_sum / klass_stats_count}")
            print(f"[GPU{process_index}] KLASS actual denoising step ratio: {klass_step_ratio_sum / klass_stats_count}")
            print(f"[GPU{process_index}] KLASS ready tokens: {klass_ready_tokens_sum}")
            print(f"[GPU{process_index}] KLASS fallback tokens: {klass_fallback_tokens_sum}")
            print(f"[GPU{process_index}] KLASS avg KL: {klass_kl_sum / klass_stats_count}")
        if dapd_stats_count > 0:
            print(f"[GPU{process_index}] DAPD avg total steps: {dapd_total_steps_sum / dapd_stats_count}")
            print(f"[GPU{process_index}] DAPD step ratio vs configured steps: {dapd_step_ratio_sum / dapd_stats_count}")
            print(f"[GPU{process_index}] DAPD avg tokens per step: {dapd_avg_tokens_per_step_sum / dapd_stats_count}")

        if self.accelerator is None:
            print(f"Number of tokens: {num_tokens}")
            print(f"Generation time: {elapsed} seconds")
            print(f"Tokens per second: {num_tokens / elapsed}")
            if cora_stats_count > 0:
                print(f"CORA avg refinement ratio: {cora_refinement_ratio_sum / cora_stats_count}")
                print(f"CORA estimated activated FFN ratio: {cora_ffn_ratio_sum / cora_stats_count}")
                print(f"CORA actual denoising step ratio: {cora_step_ratio_sum / cora_stats_count}")
                print(f"CORA fast accept tokens: {cora_fast_accept_tokens_sum}")
                print(f"CORA residual accept tokens: {cora_residual_accept_tokens_sum}")
                print(f"CORA avg residual score: {cora_residual_score_sum / cora_stats_count}")
                print(f"CORA EoT truncated samples: {cora_eot_truncated_sum}")
            if prophet_stats_count > 0:
                print(f"Prophet avg actual denoising steps: {prophet_actual_steps_sum / prophet_stats_count}")
                print(f"Prophet actual denoising step ratio: {prophet_step_ratio_sum / prophet_stats_count}")
                print(f"Prophet early-exit samples: {prophet_early_exit_sum}")
            if klass_stats_count > 0:
                print(f"KLASS avg actual denoising steps: {klass_actual_steps_sum / klass_stats_count}")
                print(f"KLASS actual denoising step ratio: {klass_step_ratio_sum / klass_stats_count}")
                print(f"KLASS ready tokens: {klass_ready_tokens_sum}")
                print(f"KLASS fallback tokens: {klass_fallback_tokens_sum}")
                print(f"KLASS avg KL: {klass_kl_sum / klass_stats_count}")
            if dapd_stats_count > 0:
                print(f"DAPD avg total steps: {dapd_total_steps_sum / dapd_stats_count}")
                print(f"DAPD step ratio vs configured steps: {dapd_step_ratio_sum / dapd_stats_count}")
                print(f"DAPD avg tokens per step: {dapd_avg_tokens_per_step_sum / dapd_stats_count}")
            return out
        
        self.accelerator.wait_for_everyone()
        total_elapsed = self.accelerator.reduce(torch.tensor(elapsed, device=self.accelerator.device), reduction="sum").item()
        total_num_tokens = self.accelerator.reduce(torch.tensor(num_tokens, device=self.accelerator.device), reduction="sum").item()
        total_cora_stats_count = self.accelerator.reduce(torch.tensor(cora_stats_count, device=self.accelerator.device), reduction="sum").item()
        total_cora_refinement_ratio_sum = self.accelerator.reduce(torch.tensor(cora_refinement_ratio_sum, device=self.accelerator.device), reduction="sum").item()
        total_cora_ffn_ratio_sum = self.accelerator.reduce(torch.tensor(cora_ffn_ratio_sum, device=self.accelerator.device), reduction="sum").item()
        total_cora_step_ratio_sum = self.accelerator.reduce(torch.tensor(cora_step_ratio_sum, device=self.accelerator.device), reduction="sum").item()
        total_cora_fast_accept_tokens_sum = self.accelerator.reduce(torch.tensor(cora_fast_accept_tokens_sum, device=self.accelerator.device), reduction="sum").item()
        total_cora_eot_truncated_sum = self.accelerator.reduce(torch.tensor(cora_eot_truncated_sum, device=self.accelerator.device), reduction="sum").item()
        total_cora_residual_accept_tokens_sum = self.accelerator.reduce(torch.tensor(cora_residual_accept_tokens_sum, device=self.accelerator.device), reduction="sum").item()
        total_cora_residual_score_sum = self.accelerator.reduce(torch.tensor(cora_residual_score_sum, device=self.accelerator.device), reduction="sum").item()
        total_prophet_stats_count = self.accelerator.reduce(torch.tensor(prophet_stats_count, device=self.accelerator.device), reduction="sum").item()
        total_prophet_step_ratio_sum = self.accelerator.reduce(torch.tensor(prophet_step_ratio_sum, device=self.accelerator.device), reduction="sum").item()
        total_prophet_actual_steps_sum = self.accelerator.reduce(torch.tensor(prophet_actual_steps_sum, device=self.accelerator.device), reduction="sum").item()
        total_prophet_early_exit_sum = self.accelerator.reduce(torch.tensor(prophet_early_exit_sum, device=self.accelerator.device), reduction="sum").item()
        total_klass_stats_count = self.accelerator.reduce(torch.tensor(klass_stats_count, device=self.accelerator.device), reduction="sum").item()
        total_klass_step_ratio_sum = self.accelerator.reduce(torch.tensor(klass_step_ratio_sum, device=self.accelerator.device), reduction="sum").item()
        total_klass_actual_steps_sum = self.accelerator.reduce(torch.tensor(klass_actual_steps_sum, device=self.accelerator.device), reduction="sum").item()
        total_klass_ready_tokens_sum = self.accelerator.reduce(torch.tensor(klass_ready_tokens_sum, device=self.accelerator.device), reduction="sum").item()
        total_klass_fallback_tokens_sum = self.accelerator.reduce(torch.tensor(klass_fallback_tokens_sum, device=self.accelerator.device), reduction="sum").item()
        total_klass_kl_sum = self.accelerator.reduce(torch.tensor(klass_kl_sum, device=self.accelerator.device), reduction="sum").item()
        total_dapd_stats_count = self.accelerator.reduce(torch.tensor(dapd_stats_count, device=self.accelerator.device), reduction="sum").item()
        total_dapd_step_ratio_sum = self.accelerator.reduce(torch.tensor(dapd_step_ratio_sum, device=self.accelerator.device), reduction="sum").item()
        total_dapd_total_steps_sum = self.accelerator.reduce(torch.tensor(dapd_total_steps_sum, device=self.accelerator.device), reduction="sum").item()
        total_dapd_avg_tokens_per_step_sum = self.accelerator.reduce(torch.tensor(dapd_avg_tokens_per_step_sum, device=self.accelerator.device), reduction="sum").item()
        if self.accelerator.is_main_process:
            print(f"Number of tokens: {total_num_tokens}")
            print(f"Generation time: {total_elapsed} seconds")
            print(f"Tokens per second: {total_num_tokens / total_elapsed}")
            if total_cora_stats_count > 0:
                print(f"CORA avg refinement ratio: {total_cora_refinement_ratio_sum / total_cora_stats_count}")
                print(f"CORA estimated activated FFN ratio: {total_cora_ffn_ratio_sum / total_cora_stats_count}")
                print(f"CORA actual denoising step ratio: {total_cora_step_ratio_sum / total_cora_stats_count}")
                print(f"CORA fast accept tokens: {total_cora_fast_accept_tokens_sum}")
                print(f"CORA residual accept tokens: {total_cora_residual_accept_tokens_sum}")
                print(f"CORA avg residual score: {total_cora_residual_score_sum / total_cora_stats_count}")
                print(f"CORA EoT truncated samples: {total_cora_eot_truncated_sum}")
            if total_prophet_stats_count > 0:
                print(f"Prophet avg actual denoising steps: {total_prophet_actual_steps_sum / total_prophet_stats_count}")
                print(f"Prophet actual denoising step ratio: {total_prophet_step_ratio_sum / total_prophet_stats_count}")
                print(f"Prophet early-exit samples: {total_prophet_early_exit_sum}")
            if total_klass_stats_count > 0:
                print(f"KLASS avg actual denoising steps: {total_klass_actual_steps_sum / total_klass_stats_count}")
                print(f"KLASS actual denoising step ratio: {total_klass_step_ratio_sum / total_klass_stats_count}")
                print(f"KLASS ready tokens: {total_klass_ready_tokens_sum}")
                print(f"KLASS fallback tokens: {total_klass_fallback_tokens_sum}")
                print(f"KLASS avg KL: {total_klass_kl_sum / total_klass_stats_count}")
            if total_dapd_stats_count > 0:
                print(f"DAPD avg total steps: {total_dapd_total_steps_sum / total_dapd_stats_count}")
                print(f"DAPD step ratio vs configured steps: {total_dapd_step_ratio_sum / total_dapd_stats_count}")
                print(f"DAPD avg tokens per step: {total_dapd_avg_tokens_per_step_sum / total_dapd_stats_count}")

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
    
