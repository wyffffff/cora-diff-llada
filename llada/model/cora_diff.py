from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoraDiffConfig:
    enabled: bool = True
    routing_mode: str = "adaptive"
    num_refinement_groups: int = 4
    core_ratio: float = 0.5
    active_ratio: float = 0.6
    active_ratio_final: Optional[float] = None
    budget_schedule: str = "constant"
    alpha: float = 0.0
    beta: float = 1.0
    gamma: float = 1.0
    dependency_topk: int = 0
    min_active_tokens: int = 1
    eps: float = 1e-6


@dataclass
class CoraDiffStats:
    routed_tokens: int = 0
    routed_refinement_levels: int = 0
    full_refinement_levels: int = 0

    @property
    def avg_refinement_ratio(self) -> float:
        if self.full_refinement_levels == 0:
            return 1.0
        return self.routed_refinement_levels / self.full_refinement_levels

    def as_dict(self, core_ratio: float) -> Dict[str, float]:
        refinement_ratio = self.avg_refinement_ratio
        return {
            "routed_tokens": float(self.routed_tokens),
            "avg_refinement_ratio": float(refinement_ratio),
            "estimated_activated_ffn_ratio": float(core_ratio + (1.0 - core_ratio) * refinement_ratio),
        }


class CoraDiffState:
    """Per-generation state for CORA-Diff style routed MLP refinement."""

    def __init__(self, config: Optional[CoraDiffConfig] = None):
        self.config = config or CoraDiffConfig()
        self.prev_pred: Optional[torch.Tensor] = None
        self.prev_conf: Optional[torch.Tensor] = None
        self.older_pred: Optional[torch.Tensor] = None
        self.older_conf: Optional[torch.Tensor] = None
        self.prev_hidden: Dict[int, torch.Tensor] = {}
        self.attention_features: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.latest_scores: Dict[int, torch.Tensor] = {}
        self.stats = CoraDiffStats()
        self.current_step = 0
        self.total_steps = 1

    def set_budget_position(self, step: int, total_steps: int) -> None:
        self.current_step = max(0, int(step))
        self.total_steps = max(1, int(total_steps))

    def update_predictions(self, pred: torch.Tensor, confidence: torch.Tensor) -> None:
        self.older_pred = self.prev_pred
        self.older_conf = self.prev_conf
        self.prev_pred = pred.detach()
        self.prev_conf = confidence.detach().to(dtype=torch.float32)

    def set_attention_features(self, layer_id: int, q: torch.Tensor, k: torch.Tensor) -> None:
        cfg = self.config
        if cfg.alpha <= 0.0 or cfg.dependency_topk <= 0:
            return
        if q.ndim != 4 or k.ndim != 4 or q.shape[-2] != k.shape[-2]:
            return

        q_mean = q.detach().to(dtype=torch.float32).mean(dim=1)
        k_mean = k.detach().to(dtype=torch.float32).mean(dim=1)
        self.attention_features[layer_id] = (
            F.normalize(q_mean, dim=-1),
            F.normalize(k_mean, dim=-1),
        )

    def refinement_levels(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        unresolved_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        cfg = self.config
        routing_mode = str(cfg.routing_mode).lower()
        if routing_mode in ("disabled", "off", "none"):
            return None
        if routing_mode not in ("adaptive", "dense_routed"):
            raise ValueError("routing_mode must be 'adaptive', 'dense_routed', or 'disabled'.")
        if not cfg.enabled or unresolved_mask is None or cfg.num_refinement_groups <= 0:
            return None

        if unresolved_mask.shape != hidden.shape[:2]:
            return None

        unresolved = unresolved_mask.to(device=hidden.device, dtype=torch.bool)
        levels = torch.full(
            unresolved.shape,
            cfg.num_refinement_groups,
            dtype=torch.long,
            device=hidden.device,
        )
        if not unresolved.any():
            self.prev_hidden[layer_id] = hidden.detach()
            return levels

        if routing_mode == "dense_routed":
            self.prev_hidden[layer_id] = hidden.detach()
            routed = unresolved.sum().item()
            self.stats.routed_tokens += int(routed)
            self.stats.routed_refinement_levels += int(routed * cfg.num_refinement_groups)
            self.stats.full_refinement_levels += int(routed * cfg.num_refinement_groups)
            return levels

        # The first denoising pass initializes prediction evidence and runs dense MLP.
        if self.prev_conf is None or self.prev_conf.shape != unresolved.shape:
            self.prev_hidden[layer_id] = hidden.detach()
            return levels

        conf = self.prev_conf.to(device=hidden.device, dtype=torch.float32).clamp(0.0, 1.0)
        uncertainty = 1.0 - conf

        drift = torch.zeros_like(uncertainty)
        if self.older_conf is not None and self.older_conf.shape == unresolved.shape:
            older_conf = self.older_conf.to(device=hidden.device, dtype=torch.float32).clamp(0.0, 1.0)
            drift = (conf - older_conf).abs()
            if (
                self.prev_pred is not None
                and self.older_pred is not None
                and self.prev_pred.shape == unresolved.shape
                and self.older_pred.shape == unresolved.shape
            ):
                drift = drift + (
                    self.prev_pred.to(device=hidden.device) != self.older_pred.to(device=hidden.device)
                ).to(dtype=torch.float32)

        hidden_drift = torch.zeros_like(uncertainty)
        prev_hidden = self.prev_hidden.get(layer_id)
        if prev_hidden is not None and prev_hidden.shape == hidden.shape:
            hidden_drift = (
                hidden.detach().to(dtype=torch.float32) - prev_hidden.to(device=hidden.device, dtype=torch.float32)
            ).pow(2).mean(dim=-1).sqrt()

        unary = (
            self._masked_zscore(uncertainty, unresolved)
            + cfg.beta * self._masked_zscore(drift, unresolved)
            + cfg.gamma * self._masked_zscore(hidden_drift, unresolved)
        )
        risk = F.softplus(unary)

        score = risk
        if cfg.alpha > 0.0 and cfg.dependency_topk > 0:
            propagated = self._propagate_with_attention_affinity(layer_id, risk, hidden, unresolved)
            score = (1.0 - cfg.alpha) * risk + cfg.alpha * propagated

        self.latest_scores[layer_id] = score.detach()
        self._assign_rank_levels(levels, score, unresolved)
        self.prev_hidden[layer_id] = hidden.detach()

        routed = unresolved.sum().item()
        self.stats.routed_tokens += int(routed)
        self.stats.routed_refinement_levels += int(levels[unresolved].sum().item())
        self.stats.full_refinement_levels += int(routed * cfg.num_refinement_groups)
        return levels

    def stable_commit_mask(
        self,
        pred: torch.Tensor,
        confidence: torch.Tensor,
        unresolved_mask: torch.Tensor,
        residual_threshold: float,
        confidence_threshold: float,
    ) -> torch.Tensor:
        if self.prev_pred is None or not self.latest_scores:
            return torch.zeros_like(unresolved_mask, dtype=torch.bool)

        residual = self.aggregate_residual(pred.shape, pred.device)
        if residual is None:
            return torch.zeros_like(unresolved_mask, dtype=torch.bool)

        stable = pred == self.prev_pred.to(device=pred.device)
        return (
            unresolved_mask.to(device=pred.device, dtype=torch.bool)
            & stable
            & (confidence.to(device=pred.device) > confidence_threshold)
            & (residual < residual_threshold)
        )

    def aggregate_residual(self, shape: torch.Size, device: torch.device) -> Optional[torch.Tensor]:
        scores = [
            score.to(device=device, dtype=torch.float32)
            for score in self.latest_scores.values()
            if score.shape == shape
        ]
        if not scores:
            return None
        return torch.stack(scores, dim=0).amax(dim=0)

    def _assign_rank_levels(self, levels: torch.Tensor, score: torch.Tensor, unresolved: torch.Tensor) -> None:
        cfg = self.config
        active_ratio = self.current_active_ratio()
        levels[unresolved] = 0
        for batch_idx in range(levels.shape[0]):
            positions = torch.nonzero(unresolved[batch_idx], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue

            active_count = int(torch.ceil(torch.tensor(positions.numel() * active_ratio)).item())
            active_count = max(cfg.min_active_tokens, active_count)
            active_count = min(active_count, positions.numel())

            order = torch.argsort(score[batch_idx, positions], descending=True)
            active_positions = positions[order[:active_count]]
            ranks = torch.arange(active_count, device=levels.device)
            rank_bands = cfg.num_refinement_groups - torch.div(
                ranks * cfg.num_refinement_groups,
                active_count,
                rounding_mode="floor",
            )
            rank_bands = rank_bands.clamp(1, cfg.num_refinement_groups)
            levels[batch_idx, active_positions] = rank_bands.to(dtype=torch.long)

    def current_active_ratio(self) -> float:
        cfg = self.config
        start = float(cfg.active_ratio)
        end = start if cfg.active_ratio_final is None else float(cfg.active_ratio_final)
        schedule = str(cfg.budget_schedule).lower()
        if schedule == "constant" or self.total_steps <= 1:
            value = start
        else:
            progress = min(1.0, max(0.0, self.current_step / max(1, self.total_steps - 1)))
            if schedule in ("linear", "linear_decay"):
                value = start + (end - start) * progress
            elif schedule in ("cosine", "cosine_decay"):
                weight = 0.5 - 0.5 * torch.cos(torch.tensor(progress * torch.pi)).item()
                value = start + (end - start) * weight
            else:
                raise ValueError("budget_schedule must be 'constant', 'linear', or 'cosine'.")
        return min(1.0, max(0.0, value))

    def stats_dict(self) -> Dict[str, float]:
        return self.stats.as_dict(core_ratio=self.config.core_ratio)

    def _masked_zscore(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(values, dtype=torch.float32)
        selected = values[mask].to(dtype=torch.float32)
        if selected.numel() <= 1:
            return out
        mean = selected.mean()
        std = selected.std(unbiased=False)
        out[mask] = (selected - mean) / (std + self.config.eps)
        return out

    def _propagate_with_attention_affinity(
        self,
        layer_id: int,
        risk: torch.Tensor,
        hidden: torch.Tensor,
        unresolved: torch.Tensor,
    ) -> torch.Tensor:
        features = self.attention_features.get(layer_id)
        if features is None:
            return self._propagate_with_hidden_affinity(risk, hidden, unresolved)

        q_mean, k_mean = features
        if q_mean.shape[:2] != risk.shape or k_mean.shape[:2] != risk.shape:
            return self._propagate_with_hidden_affinity(risk, hidden, unresolved)

        cfg = self.config
        propagated = risk.clone()
        for batch_idx in range(risk.shape[0]):
            positions = torch.nonzero(unresolved[batch_idx], as_tuple=False).flatten()
            if positions.numel() <= 1:
                continue
            topk = min(cfg.dependency_topk, positions.numel() - 1)
            if topk <= 0:
                continue

            scores = q_mean[batch_idx, positions] @ k_mean[batch_idx, positions].transpose(0, 1)
            scores.fill_diagonal_(-float("inf"))
            values, neighbor_idx = torch.topk(scores, k=topk, dim=-1)
            finite = torch.isfinite(values)
            safe_values = torch.where(finite, values, torch.zeros_like(values))
            weights = torch.softmax(safe_values, dim=-1) * finite.to(dtype=torch.float32)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
            weights = weights / denom
            neighbor_risk = risk[batch_idx, positions[neighbor_idx]]
            propagated[batch_idx, positions] = (weights * neighbor_risk).sum(dim=-1)
        return propagated

    def _propagate_with_hidden_affinity(
        self,
        risk: torch.Tensor,
        hidden: torch.Tensor,
        unresolved: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        propagated = risk.clone()
        for batch_idx in range(hidden.shape[0]):
            positions = torch.nonzero(unresolved[batch_idx], as_tuple=False).flatten()
            if positions.numel() <= 1:
                continue
            topk = min(cfg.dependency_topk, positions.numel() - 1)
            if topk <= 0:
                continue

            h = F.normalize(hidden[batch_idx, positions].to(dtype=torch.float32), dim=-1)
            affinity = h @ h.transpose(0, 1)
            affinity.fill_diagonal_(-float("inf"))
            values, neighbor_idx = torch.topk(affinity, k=topk, dim=-1)
            weights = torch.softmax(values, dim=-1)
            neighbor_risk = risk[batch_idx, positions[neighbor_idx]]
            propagated[batch_idx, positions] = (weights * neighbor_risk).sum(dim=-1)
        return propagated


def cora_channel_ranges(hidden_size: int, num_refinement_groups: int, core_ratio: float) -> Tuple[Tuple[int, int], ...]:
    num_refinement_groups = max(0, int(num_refinement_groups))
    if num_refinement_groups == 0:
        return ((0, hidden_size),)

    core_size = int(round(hidden_size * core_ratio))
    core_size = max(1, min(core_size, hidden_size))
    remaining = hidden_size - core_size
    if remaining == 0:
        return ((0, hidden_size),)

    ranges = [(0, core_size)]
    base = remaining // num_refinement_groups
    extra = remaining % num_refinement_groups
    start = core_size
    for group_idx in range(num_refinement_groups):
        width = base + (1 if group_idx < extra else 0)
        end = start + width
        if end > start:
            ranges.append((start, end))
        start = end
    return tuple(ranges)


@torch.no_grad()
def prepare_cora_channel_ordering(
    model: nn.Module,
    activation_energy: Optional[Mapping[nn.Module, torch.Tensor]] = None,
) -> None:
    """Order SwiGLU intermediate channels so prefix groups keep larger contributions first.

    The paper uses an activation-energy estimate from a small calibration set. This
    helper accepts such estimates when available and falls back to a weight-energy
    proxy otherwise. The dense path remains exactly equivalent after permutation.
    """
    for module in _iter_cora_mlp_modules(model):
        if getattr(module, "_cora_channels_ordered", False):
            continue

        ff_proj = module.ff_proj
        up_proj = module.up_proj
        ff_out = module.ff_out
        energy = None if activation_energy is None else activation_energy.get(module)
        if energy is None:
            in_energy = 0.5 * (
                ff_proj.weight.detach().to(dtype=torch.float32).norm(dim=1)
                + up_proj.weight.detach().to(dtype=torch.float32).norm(dim=1)
            )
            out_energy = ff_out.weight.detach().to(dtype=torch.float32).norm(dim=0)
            energy = in_energy * out_energy
        else:
            energy = energy.to(device=ff_proj.weight.device, dtype=torch.float32)

        permutation = torch.argsort(energy, descending=True).to(device=ff_proj.weight.device)
        _permute_linear_rows(ff_proj, permutation)
        _permute_linear_rows(up_proj, permutation)
        _permute_linear_columns(ff_out, permutation)
        module._cora_channels_ordered = True


@torch.no_grad()
def calibrate_cora_channel_ordering(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> None:
    """Estimate activation energy with a dense calibration pass, then order channels."""
    if not any(not getattr(module, "_cora_channels_ordered", False) for module in _iter_cora_mlp_modules(model)):
        return

    activation_sum: Dict[nn.Module, torch.Tensor] = {}
    activation_count: Dict[nn.Module, int] = {}
    handles = []

    def make_hook(module: nn.Module):
        def hook(_layer: nn.Module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return
            if hidden.ndim != 3:
                return

            ff_proj = module.ff_proj
            up_proj = module.up_proj
            ff_out = module.ff_out
            gate = module.act(ff_proj(hidden))
            up = up_proj(hidden)
            if gate.shape != up.shape or ff_out.in_features != gate.shape[-1]:
                return

            intermediate = gate * up
            out_energy = ff_out.weight.detach().to(device=intermediate.device, dtype=torch.float32).norm(dim=0)
            energy = intermediate.detach().to(dtype=torch.float32).abs().mean(dim=(0, 1)) * out_energy
            energy = energy.cpu()
            if module in activation_sum:
                activation_sum[module] += energy
                activation_count[module] += 1
            else:
                activation_sum[module] = energy
                activation_count[module] = 1

        return hook

    for module in _iter_cora_mlp_modules(model):
        if hasattr(module, "ff_norm"):
            handles.append(module.ff_norm.register_forward_hook(make_hook(module)))

    was_training = model.training
    model.eval()
    try:
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    activation_energy = {
        module: energy / activation_count[module]
        for module, energy in activation_sum.items()
        if activation_count.get(module, 0) > 0
    }
    prepare_cora_channel_ordering(model, activation_energy=activation_energy)


def _iter_cora_mlp_modules(model: nn.Module):
    for module in model.modules():
        if not all(hasattr(module, name) for name in ("ff_proj", "up_proj", "ff_out")):
            continue
        ff_proj = module.ff_proj
        up_proj = module.up_proj
        ff_out = module.ff_out
        if not isinstance(ff_proj, nn.Linear) or not isinstance(up_proj, nn.Linear) or not isinstance(ff_out, nn.Linear):
            continue
        if ff_proj.out_features != up_proj.out_features or ff_out.in_features != ff_proj.out_features:
            continue
        yield module


def _permute_linear_rows(layer: nn.Linear, permutation: torch.Tensor) -> None:
    layer.weight.data = layer.weight.data.index_select(0, permutation)
    if layer.bias is not None:
        layer.bias.data = layer.bias.data.index_select(0, permutation)


def _permute_linear_columns(layer: nn.Linear, permutation: torch.Tensor) -> None:
    layer.weight.data = layer.weight.data.index_select(1, permutation)


def routed_llama_mlp(
    hidden: torch.Tensor,
    levels: Optional[torch.Tensor],
    ff_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    ff_out: torch.nn.Linear,
    act: torch.nn.Module,
    config: CoraDiffConfig,
) -> torch.Tensor:
    if levels is None or ff_out.in_features != ff_proj.out_features:
        gate = act(ff_proj(hidden))
        return ff_out(gate * up_proj(hidden))

    batch_size, seq_len, d_model = hidden.shape
    hidden_size = ff_proj.out_features
    ranges = cora_channel_ranges(hidden_size, config.num_refinement_groups, config.core_ratio)
    if len(ranges) <= 1:
        gate = act(ff_proj(hidden))
        return ff_out(gate * up_proj(hidden))

    flat_hidden = hidden.reshape(batch_size * seq_len, d_model)
    flat_levels = levels.reshape(batch_size * seq_len)
    flat_out = hidden.new_zeros((batch_size * seq_len, ff_out.out_features))

    for group_idx, (start, end) in enumerate(ranges):
        if group_idx == 0:
            selected = torch.ones_like(flat_levels, dtype=torch.bool)
        else:
            selected = flat_levels >= group_idx
        if not selected.any():
            continue

        selected_hidden = flat_hidden[selected]
        ff_bias = None if ff_proj.bias is None else ff_proj.bias[start:end]
        up_bias = None if up_proj.bias is None else up_proj.bias[start:end]
        gate = F.linear(selected_hidden, ff_proj.weight[start:end], ff_bias)
        up = F.linear(selected_hidden, up_proj.weight[start:end], up_bias)
        intermediate = act(gate) * up
        flat_out[selected] += F.linear(intermediate, ff_out.weight[:, start:end], None)

    if ff_out.bias is not None:
        flat_out += ff_out.bias
    return flat_out.view(batch_size, seq_len, ff_out.out_features)
