"""Core utilities for DAPD decoding."""
import math
import os
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union

VALID_DAPD_ALGORITHMS = ("dapd_staged", "dapd_direct")


def validate_dapd_algorithm(alg: Optional[str]) -> str:
    """Validate the two public DAPD algorithms without accepting aliases."""
    if alg is None:
        raise ValueError(
            f"DAPD alg is required. Expected one of {VALID_DAPD_ALGORITHMS} "
        )
    text = str(alg).strip()
    if text not in VALID_DAPD_ALGORITHMS:
        raise ValueError(
            f"Invalid DAPD alg='{alg}'. Expected exactly one of {VALID_DAPD_ALGORITHMS}."
        )
    return text


class AttentionCaptureHook:
    """
    Capture Q and K tensors from LLaDA's _scaled_dot_product_attention.
    
    Uses method wrapping since _scaled_dot_product_attention is a method, not a Module.
    
    Usage:
        hook = AttentionCaptureHook()
        hook.register(model)
        output = model(input_ids)
        attention_weights = hook.compute_attention_weights()
        hook.restore(model)
    """
    
    def __init__(self, capture_layers: Optional[List[int]] = None):
        """
        Args:
            capture_layers: Internal list of transformer layer ids to capture.
                If None, capture from all layers.
        """
        self.capture_layers = capture_layers
        self.captured_qk: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.layer_configs: List[Dict[str, int]] = []
        self._original_methods: Dict[int, Any] = {}  # Store original methods for restoration
        self._blocks: List = []  # Store block references
        self._restore_callbacks: List[Any] = []
    
    def register(self, model):
        """
        Replace _scaled_dot_product_attention with wrapper that captures Q, K.
        """
        self.captured_qk = []
        self.layer_configs = []
        self._original_methods = {}
        self._restore_callbacks = []

        if self._register_llada1(model):
            return
        raise ValueError("Cannot find supported LLaDA attention structure in model")

    def _register_llada1(self, model) -> bool:
        # Handle LLaDAModelLM wrapper: model.model.transformer
        if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            transformer = model.model.transformer
        elif hasattr(model, 'transformer'):
            transformer = model.transformer
        else:
            return False
        
        # Get blocks from transformer
        if hasattr(transformer, 'blocks'):
            blocks = list(transformer.blocks)
        elif hasattr(transformer, 'block_groups'):
            blocks = []
            for group in transformer.block_groups:
                blocks.extend(list(group))
        else:
            return False
        
        self._blocks = blocks
        
        for idx, block in enumerate(blocks):
            if self.capture_layers is not None and idx not in self.capture_layers:
                continue
            
            # Store original method
            self._original_methods[idx] = block._scaled_dot_product_attention
            
            # Create wrapper that captures Q, K
            def make_wrapper(original_method, layer_idx, blk):
                def wrapper(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
                    # Capture Q, K (post-RoPE, as passed to SDPA)
                    self.captured_qk.append((q.detach().clone(), k.detach().clone()))
                    self.layer_configs.append({
                        'n_heads': blk.config.n_heads,
                        'n_kv_heads': blk.config.effective_n_kv_heads,
                        'layer_idx': layer_idx,
                    })
                    # Call original method
                    return original_method(q, k, v, attn_mask, dropout_p, is_causal)
                return wrapper
            
            # Replace method with wrapper
            import types
            block._scaled_dot_product_attention = types.MethodType(
                lambda self_block, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                       _orig=self._original_methods[idx], _idx=idx, _blk=block, **extra_kwargs:
                    self._capture_and_call(
                        _orig,
                        _idx,
                        _blk,
                        q,
                        k,
                        v,
                        attn_mask,
                        dropout_p,
                        is_causal,
                        extra_kwargs,
                    ),
                block
            )
        return True

    def _capture_and_call(self, original_method, layer_idx, block, q, k, v, attn_mask, dropout_p, is_causal, extra_kwargs=None):
        """Helper to capture Q,K and call original method."""
        self.captured_qk.append((q.detach().clone(), k.detach().clone()))
        self.layer_configs.append({
            'n_heads': block.config.n_heads,
            'n_kv_heads': block.config.effective_n_kv_heads,
            'layer_idx': layer_idx,
        })
        extra_kwargs = extra_kwargs or {}
        try:
            return original_method(q, k, v, attn_mask, dropout_p, is_causal, **extra_kwargs)
        except TypeError as exc:
            if extra_kwargs and "unexpected keyword argument" in str(exc):
                return original_method(q, k, v, attn_mask, dropout_p, is_causal)
            raise
    
    def restore(self, model=None):
        """Restore original _scaled_dot_product_attention methods."""
        while self._restore_callbacks:
            callback = self._restore_callbacks.pop()
            callback()
        for idx, original in self._original_methods.items():
            if idx < len(self._blocks):
                self._blocks[idx]._scaled_dot_product_attention = original
        self._original_methods = {}
    
    def clear(self):
        """Clear captured data."""
        self.captured_qk = []
        self.layer_configs = []

    
    def compute_attention_weights(
        self,
        layer_ratio: float = 0.3,
    ) -> torch.Tensor:
        """
        Compute attention weights from captured Q, K tensors.
        
        Args:
            layer_ratio: Use top layer_ratio% of layers (e.g., 0.3 = top 30%)
                when capturing all layers. If this hook was initialized with
                internal ``capture_layers``, all captured layers are used and
                ``layer_ratio`` is ignored.
        
        Returns:
            Aggregated attention weights: (batch, seq, seq)
        """
        if not self.captured_qk:
            raise ValueError("No Q, K captured. Did you run forward pass?")

        num_layers = len(self.captured_qk)

        if self.capture_layers is not None:
            selected_qk = self.captured_qk
            selected_configs = self.layer_configs
        else:
            # Preserve the historical "top layer_ratio" behavior, which is
            # equivalent to selecting ceil(num_layers * layer_ratio) layers.
            start_layer = int(num_layers * (1 - layer_ratio))
            selected_qk = self.captured_qk[start_layer:]
            selected_configs = self.layer_configs[start_layer:]
        
        running_attn = None
        used_layers = 0
        head_chunk = int(os.environ.get("DAPD_ATTENTION_HEAD_CHUNK", "0") or "0")
        
        for (q, k), config in zip(selected_qk, selected_configs):
            # Handle GQA: expand k to match q's head count
            n_heads = config['n_heads']
            n_kv_heads = config['n_kv_heads']
            
            if n_heads != n_kv_heads:
                # Repeat k for GQA
                k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)
            
            # Compute attention: softmax(QK^T / sqrt(d_k)) in float32 for precision
            d_k = q.size(-1)
            if head_chunk > 0 and head_chunk < n_heads:
                attn_sum = torch.zeros(
                    q.size(0),
                    q.size(-2),
                    k.size(-2),
                    device=q.device,
                    dtype=torch.float32,
                )
                for head_start in range(0, n_heads, head_chunk):
                    head_end = min(n_heads, head_start + head_chunk)
                    q_chunk = q[:, head_start:head_end].float()
                    k_chunk = k[:, head_start:head_end].float()
                    scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(d_k)
                    attn_sum += F.softmax(scores, dim=-1).sum(dim=1)
                attn_avg = attn_sum / n_heads  # (B, T, T)
            else:
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(d_k)
                attn_weights = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)
                attn_avg = attn_weights.mean(dim=1)  # (B, T, T)

            if running_attn is None:
                running_attn = attn_avg
            else:
                running_attn = running_attn + attn_avg.to(running_attn.device)
            used_layers += 1

        if running_attn is None:
            raise ValueError("No Q, K selected for attention computation.")
        return running_attn / used_layers  # (B, T, T)


def build_dependency_graph(
    attention: torch.Tensor,
    mask_index: torch.Tensor,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Build dependency scores used by DAPD."""
    dependency = (attention + attention.transpose(-1, -2)) / 2

    _, T, _ = dependency.shape
    diag_mask = torch.eye(T, dtype=torch.bool, device=dependency.device)
    dependency = dependency.masked_fill(diag_mask.unsqueeze(0), 0.0)

    mask_2d = mask_index.unsqueeze(-1) & mask_index.unsqueeze(-2)
    dependency = dependency * mask_2d.float()
    raw_dependency = dependency.clone()

    masked_dependency = dependency.masked_fill(~mask_2d, float("-inf"))
    normalizer = masked_dependency.max(dim=-1, keepdim=True)[0]
    normalizer = torch.where(normalizer == float("-inf"), torch.ones_like(normalizer), normalizer)
    normalizer = normalizer.clamp(min=1e-8)
    normalized_dependency = dependency / normalizer
    normalized_dependency = normalized_dependency * mask_2d.float()

    stats = getattr(build_dependency_graph, "_norm_stats", None)
    if stats is not None:
        valid = mask_2d
        if valid.any():
            before_values = raw_dependency[valid]
            after_values = normalized_dependency[valid]
            stats["count"] += 1
            stats["before_max_sum"] += float(before_values.max().item())
            stats["before_mean_sum"] += float(before_values.mean().item())
            stats["after_max_sum"] += float(after_values.max().item())
            stats["after_mean_sum"] += float(after_values.mean().item())

    return raw_dependency, normalized_dependency


def compute_dependency_scores(
    dependency: torch.Tensor,
    mask_index: torch.Tensor,
) -> torch.Tensor:
    """Compute DAPD dependency scores."""
    dependency_scores = dependency.sum(dim=-1)
    dependency_scores = dependency_scores * mask_index.float()
    return dependency_scores


def select_independent_set(
    confidence: torch.Tensor,
    dependency_score: torch.Tensor,
    dependency: torch.Tensor,
    mask_index: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Greedy independent-set selection fixed to the paper DAPD rule.

    Tokens are sorted by normalized_dependency_score * normalized_confidence.
    A candidate is accepted only when its dependency to all already selected
    tokens is <= tau.
    """
    B, T = confidence.shape
    device = confidence.device

    dependency_norm = dependency_score / (dependency_score.max(dim=-1, keepdim=True)[0] + 1e-8)
    conf_norm = confidence / (confidence.max(dim=-1, keepdim=True)[0] + 1e-8)
    combined_score = dependency_norm * conf_norm
    combined_score = combined_score * mask_index.float()  # Zero non-mask

    selected = torch.zeros(B, T, dtype=torch.bool, device=device)

    for b in range(B):
        dep_b = dependency[b]

        mask_positions = mask_index[b].nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            continue

        scores_at_mask = combined_score[b, mask_positions]
        sorted_indices = torch.argsort(scores_at_mask, descending=True)
        sorted_positions = mask_positions[sorted_indices]

        sorted_positions = sorted_positions.tolist()
        selected_positions = []

        for pos_item in sorted_positions:
            can_select = True
            for sel_pos in selected_positions:
                if dep_b[pos_item, sel_pos] > tau:
                    can_select = False
                    break

            if can_select:
                selected_positions.append(pos_item)
                selected[b, pos_item] = True
    
    return selected
