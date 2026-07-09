"""Paper-facing DAPD generation.

This module intentionally contains only the public DAPD decoding path used in
the paper. Historical research variants are not part of this deploy branch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import (
    AttentionCaptureHook,
    build_dependency_graph,
    compute_dependency_scores as compute_dapd_dependency_scores,
    select_independent_set,
    validate_dapd_algorithm,
)


def _resolve_model_num_layers(model) -> int:
    config = getattr(model, "config", None)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return int(len(model.model.layers))
    if hasattr(model, "layers"):
        return int(len(model.layers))
    raise ValueError("Could not resolve num_hidden_layers from model.")


def _resolve_layer_count(num_layers: int, layer_ratio: float) -> int:
    if not (0 < float(layer_ratio) <= 1.0):
        raise ValueError(f"layer_ratio must be in (0, 1], got {layer_ratio}")
    return max(1, num_layers - int(num_layers * (1 - float(layer_ratio))))


def _resolve_capture_layers(
    model,
    *,
    layer_ratio: float,
) -> list[int]:
    num_layers = _resolve_model_num_layers(model)
    count = _resolve_layer_count(num_layers, layer_ratio)
    return list(range(num_layers - count, num_layers))


def _compute_confidence_and_predictions(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)
    return confidence, predictions, probs


def _compute_attention(hook: AttentionCaptureHook, layer_ratio: float) -> torch.Tensor:
    return hook.compute_attention_weights(layer_ratio=layer_ratio)


def _build_paper_dependency(
    attention: torch.Tensor,
    mask_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return build_dependency_graph(
        attention,
        mask_index,
    )


def _compute_tau(
    *,
    remaining_masks: int,
    gen_length: int,
    tau_min: float,
    tau_max: float,
) -> float:
    remaining_ratio = remaining_masks / gen_length
    progress = 1.0 - remaining_ratio
    return float(tau_min + (tau_max - tau_min) * progress)


def _init_dapd_algorithm_stats(stats: dict, alg: str) -> None:
    stats["dapd_alg"] = alg
    if alg == "dapd_staged":
        stats["staged_conf_unmasked_per_step"] = []
    elif alg == "dapd_direct":
        stats["direct_conf_unmasked_per_step"] = []


def _compute_dapd_direct_selection(
    *,
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    selected = (confidence >= 1.0) & mask_index
    return selected, {"direct_added": int(selected.sum().item())}


def _apply_dapd_staged_selection(
    *,
    selected: torch.Tensor,
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    progress: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    if progress < 0.5:
        return selected, {"staged_added": 0}
    staged = (confidence >= 0.9) & mask_index
    added = staged & ~selected
    return selected | staged, {"staged_added": int(added.sum().item())}


@torch.no_grad()
def generate_dapd(
    model,
    prompt,
    attention_mask=None,
    gen_length=128,
    mask_id=126336,
    layer_ratio=0.3,
    tau_min=0.01,
    tau_max=0.15,
    verbose=False,
    alg="dapd_staged",
    collect_step_history=True,
    **removed_options,
):
    """Generate with the paper DAPD algorithm."""
    if removed_options:
        raise ValueError(f"Unsupported DAPD options in deploy path: {sorted(removed_options)}")

    alg = validate_dapd_algorithm(alg)
    capture_layers = _resolve_capture_layers(
        model,
        layer_ratio=layer_ratio,
    )
    device = prompt.device
    batch_size = prompt.shape[0]
    prompt_len = prompt.shape[1]

    x = torch.cat(
        [prompt, torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=device)],
        dim=1,
    )
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
    full_attention_mask = torch.cat(
        [attention_mask, torch.ones(batch_size, gen_length, dtype=torch.long, device=device)],
        dim=1,
    )

    stats = {
        "total_steps": 0,
        "tokens_per_step": [],
        "total_forward_passes": 0,
        "avg_dependency": [],
        "tau_per_step": [],
    }
    if collect_step_history:
        stats["generation_order"] = torch.zeros(batch_size, gen_length, dtype=torch.long, device=device)
        stats["step_positions"] = []
        stats["step_confidences"] = []
        stats["step_dependency_scores"] = []
    _init_dapd_algorithm_stats(stats, alg)

    hook = AttentionCaptureHook(capture_layers=capture_layers)
    step = 0
    hook.register(model)
    try:
        while True:
            mask_index = x == mask_id
            remaining_masks = int(mask_index.sum().item())
            if remaining_masks == 0:
                break

            current_tau = _compute_tau(
                remaining_masks=remaining_masks,
                gen_length=gen_length,
                tau_min=float(tau_min),
                tau_max=float(tau_max),
            )
            stats["tau_per_step"].append(current_tau)

            if verbose:
                print(f"Step {step}: {remaining_masks} masks remaining, tau={current_tau:.6f}")

            hook.clear()
            logits = model(x, attention_mask=full_attention_mask).logits
            stats["total_forward_passes"] += 1
            attention = _compute_attention(hook=hook, layer_ratio=layer_ratio)

            raw_dependency, dependency = _build_paper_dependency(attention, mask_index)
            dependency_scores = compute_dapd_dependency_scores(raw_dependency, mask_index)
            confidence, predictions, _ = _compute_confidence_and_predictions(logits)

            greedy_mask_index = mask_index
            direct_selected = torch.zeros_like(mask_index)
            if alg == "dapd_direct":
                direct_selected, direct_counts = _compute_dapd_direct_selection(
                    confidence=confidence,
                    mask_index=mask_index,
                )
                greedy_mask_index = mask_index & ~direct_selected
                stats["direct_conf_unmasked_per_step"].append(direct_counts["direct_added"])

            selected = select_independent_set(
                confidence=confidence,
                dependency_score=dependency_scores,
                dependency=dependency,
                mask_index=greedy_mask_index,
                tau=current_tau,
            )
            selected = selected | direct_selected

            if alg == "dapd_staged":
                progress = 1.0 - (remaining_masks / gen_length)
                selected, staged_counts = _apply_dapd_staged_selection(
                    selected=selected,
                    confidence=confidence,
                    mask_index=mask_index,
                    progress=progress,
                )
                stats["staged_conf_unmasked_per_step"].append(staged_counts["staged_added"])

            num_selected = int(selected.sum().item())
            if num_selected < 1 and remaining_masks > 0:
                mask_positions = mask_index.nonzero(as_tuple=True)
                mask_confidences = confidence[mask_positions]
                top_k = min(1, len(mask_confidences))
                _, top_indices = mask_confidences.topk(top_k)
                for idx in top_indices:
                    b, t = mask_positions[0][idx].item(), mask_positions[1][idx].item()
                    selected[b, t] = True
                num_selected = int(selected.sum().item())

            if collect_step_history:
                selected_positions = selected.nonzero(as_tuple=False).tolist()
                stats["step_positions"].append(selected_positions)
                for b, pos in selected_positions:
                    if pos >= prompt_len:
                        stats["generation_order"][b, pos - prompt_len] = step + 1
                if selected.any():
                    stats["step_confidences"].append(confidence[selected].cpu().tolist())
                    stats["step_dependency_scores"].append(dependency_scores[selected].cpu().tolist())

            if mask_index.any():
                dep_values = dependency[mask_index.unsqueeze(-1) & mask_index.unsqueeze(-2)]
                if dep_values.numel() > 0:
                    stats["avg_dependency"].append(float(dep_values.mean().item()))
            stats["tokens_per_step"].append(num_selected)

            x = torch.where(selected, predictions, x)

            step += 1
            stats["total_steps"] = step
            if step > gen_length:
                if verbose:
                    print("Warning: Max iterations reached")
                break
    finally:
        hook.restore()

    stats["avg_tokens_per_step"] = sum(stats["tokens_per_step"]) / max(len(stats["tokens_per_step"]), 1)
    return x, stats


@torch.no_grad()
def generate_dapd_with_blocks(
    model,
    prompt,
    attention_mask=None,
    gen_length=128,
    block_length=128,
    attention_scope="global",
    **kwargs,
):
    """Generate with the same block semantics as LLaDA's semi-autoregressive path.

    The full ``prompt + gen_length`` sequence is allocated once. Future blocks
    remain visible as mask tokens to the model, but DAPD selection is restricted
    to the active block. This mirrors the original LLaDA rule:
    ``mask_index[:, block_end:] = 0``.
    """
    if attention_scope != "global":
        raise ValueError("DAPD block generation keeps attention_scope fixed to 'global'.")

    block_length = int(block_length) if block_length else int(gen_length)
    gen_length = int(gen_length)
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if block_length >= gen_length:
        return generate_dapd(
            model=model,
            prompt=prompt,
            attention_mask=attention_mask,
            gen_length=gen_length,
            **kwargs,
        )
    if gen_length % block_length != 0:
        raise ValueError(f"block_length ({block_length}) must divide gen_length ({gen_length}).")

    layer_ratio = float(kwargs.pop("layer_ratio", 0.3))
    tau_min = float(kwargs.pop("tau_min", 0.01))
    tau_max = float(kwargs.pop("tau_max", 0.15))
    mask_id = int(kwargs.pop("mask_id", 126336))
    verbose = bool(kwargs.pop("verbose", False))
    alg = validate_dapd_algorithm(kwargs.pop("alg", "dapd_staged"))
    collect_step_history = bool(kwargs.pop("collect_step_history", True))
    if kwargs:
        raise ValueError(f"Unsupported DAPD options in deploy path: {sorted(kwargs)}")

    capture_layers = _resolve_capture_layers(
        model,
        layer_ratio=layer_ratio,
    )
    device = prompt.device
    batch_size = prompt.shape[0]
    prompt_len = prompt.shape[1]
    num_blocks = gen_length // block_length

    x = torch.cat(
        [prompt, torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=device)],
        dim=1,
    )
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
    full_attention_mask = torch.cat(
        [attention_mask, torch.ones(batch_size, gen_length, dtype=torch.long, device=device)],
        dim=1,
    )

    stats = {
        "total_steps": 0,
        "tokens_per_step": [],
        "total_forward_passes": 0,
        "avg_dependency": [],
        "tau_per_step": [],
        "num_blocks": num_blocks,
        "block_length": block_length,
        "per_block_stats": [],
    }
    _init_dapd_algorithm_stats(stats, alg)
    if collect_step_history:
        stats["generation_order"] = torch.zeros(
            batch_size,
            gen_length,
            dtype=torch.long,
            device=device,
        )
        stats["step_positions"] = []
        stats["step_confidences"] = []
        stats["step_dependency_scores"] = []

    hook = AttentionCaptureHook(capture_layers=capture_layers)
    step = 0
    hook.register(model)
    try:
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = block_start + block_length
            block_step_start = step
            block_forward_start = stats["total_forward_passes"]
            block_token_start = len(stats["tokens_per_step"])

            if verbose:
                print(f"Block {block_idx + 1}/{num_blocks}: positions [{block_start}, {block_end})")

            while True:
                full_mask_index = x == mask_id
                block_mask_index = full_mask_index.clone()
                block_mask_index[:, :block_start] = False
                block_mask_index[:, block_end:] = False

                remaining_in_block = int(block_mask_index.sum().item())
                if remaining_in_block == 0:
                    break

                remaining_masks = int(full_mask_index[:, prompt_len:].sum().item())
                current_tau = _compute_tau(
                    remaining_masks=remaining_masks,
                    gen_length=gen_length,
                    tau_min=tau_min,
                    tau_max=tau_max,
                )
                stats["tau_per_step"].append(current_tau)

                if verbose:
                    print(
                        f"Step {step}: block_remaining={remaining_in_block}, "
                        f"total_remaining={remaining_masks}, tau={current_tau:.6f}"
                    )

                hook.clear()
                logits = model(x, attention_mask=full_attention_mask).logits
                stats["total_forward_passes"] += 1
                attention = _compute_attention(hook=hook, layer_ratio=layer_ratio)

                raw_dependency, dependency = _build_paper_dependency(attention, block_mask_index)
                dependency_scores = compute_dapd_dependency_scores(raw_dependency, block_mask_index)
                confidence, predictions, _ = _compute_confidence_and_predictions(logits)

                greedy_mask_index = block_mask_index
                direct_selected = torch.zeros_like(block_mask_index)
                if alg == "dapd_direct":
                    direct_selected, direct_counts = _compute_dapd_direct_selection(
                        confidence=confidence,
                        mask_index=block_mask_index,
                    )
                    greedy_mask_index = block_mask_index & ~direct_selected
                    stats["direct_conf_unmasked_per_step"].append(direct_counts["direct_added"])

                selected = select_independent_set(
                    confidence=confidence,
                    dependency_score=dependency_scores,
                    dependency=dependency,
                    mask_index=greedy_mask_index,
                    tau=current_tau,
                )
                selected = selected | direct_selected

                if alg == "dapd_staged":
                    progress = 1.0 - (remaining_masks / gen_length)
                    selected, staged_counts = _apply_dapd_staged_selection(
                        selected=selected,
                        confidence=confidence,
                        mask_index=block_mask_index,
                        progress=progress,
                    )
                    stats["staged_conf_unmasked_per_step"].append(staged_counts["staged_added"])

                num_selected = int(selected.sum().item())
                if num_selected < 1:
                    mask_positions = block_mask_index.nonzero(as_tuple=True)
                    mask_confidences = confidence[mask_positions]
                    top_k = min(1, len(mask_confidences))
                    _, top_indices = mask_confidences.topk(top_k)
                    for idx in top_indices:
                        b, t = mask_positions[0][idx].item(), mask_positions[1][idx].item()
                        selected[b, t] = True
                    num_selected = int(selected.sum().item())

                if collect_step_history:
                    selected_positions = selected.nonzero(as_tuple=False).tolist()
                    stats["step_positions"].append(selected_positions)
                    for b, pos in selected_positions:
                        if pos >= prompt_len:
                            stats["generation_order"][b, pos - prompt_len] = step + 1
                    if selected.any():
                        stats["step_confidences"].append(confidence[selected].cpu().tolist())
                        stats["step_dependency_scores"].append(dependency_scores[selected].cpu().tolist())

                dep_values = dependency[
                    block_mask_index.unsqueeze(-1) & block_mask_index.unsqueeze(-2)
                ]
                if dep_values.numel() > 0:
                    stats["avg_dependency"].append(float(dep_values.mean().item()))
                stats["tokens_per_step"].append(num_selected)

                x = torch.where(selected, predictions, x)

                step += 1
                stats["total_steps"] = step
                if (step - block_step_start) > block_length:
                    if verbose:
                        print(f"Warning: Block {block_idx + 1} exceeded block_length iterations")
                    break

            block_steps = step - block_step_start
            block_tokens = stats["tokens_per_step"][block_token_start:]
            stats["per_block_stats"].append(
                {
                    "block_idx": block_idx,
                    "block_range": f"[{block_start}:{block_end}]",
                    "steps": block_steps,
                    "forward_passes": stats["total_forward_passes"] - block_forward_start,
                    "avg_tokens_per_step": (
                        sum(block_tokens) / len(block_tokens) if block_tokens else 0.0
                    ),
                }
            )
    finally:
        hook.restore()

    stats["avg_tokens_per_step"] = sum(stats["tokens_per_step"]) / max(len(stats["tokens_per_step"]), 1)
    stats["avg_steps_per_block"] = stats["total_steps"] / num_blocks if num_blocks > 0 else 0.0
    return x, stats


__all__ = [
    "generate_dapd",
    "generate_dapd_with_blocks",
]
