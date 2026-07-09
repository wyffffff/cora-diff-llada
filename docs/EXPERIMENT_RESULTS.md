# CORA-Diff Current Snapshot And Results

Snapshot date: 2026-05-16

This package contains the current CORA-Diff implementation for LLaDA-style diffusion decoding.

## Current Best Configuration

The current best validated path is the step-reduction path:

```text
method=CORA
cora_routing_mode=disabled
cora_fast_accept=true
cora_accept_confidence_threshold=0.97
cora_accept_stability_steps=2
cora_eot=false
```

This configuration keeps the native dense Transformer path and accelerates inference by reducing actual denoising forward passes.
It does not rely on routed FFN, sparse FFN execution, or EoT truncation.

## Main Results

### GSM8K, gen_length=256, steps=256, limit=50

| Method | Time | TPS | Flexible EM | Strict EM | Step Ratio | Fast Accept Tokens | EoT Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 1477.68s | 8.07 | 0.80 | 0.36 | 1.000 | - | - |
| CORA-step best | 803.29s | 14.85 | 0.80 | 0.36 | 0.541 | 7693 | 0 |

Speedup:

```text
14.846 / 8.071 = 1.84x
```

Theoretical step-ratio prediction:

```text
1 / 0.541 = 1.85x
```

### GSM8K, gen_length=1024, steps=256, limit=50

| Method | Time | TPS | Flexible EM | Strict EM | Step Ratio | Fast Accept Tokens | EoT Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 2398.32s | 5.93 | 0.56 | 0.28 | 1.000 | - | - |
| CORA-step best | 999.55s | 14.08 | 0.54 | 0.26 | 0.416 | 36715 | 0 |

Speedup:

```text
14.080 / 5.928 = 2.38x
```

Theoretical step-ratio prediction:

```text
1 / 0.416 = 2.40x
```

The 1024-length result shows that speedup becomes stronger for longer generation, while accuracy changes remain small under the sampled setting.

## Interpretation

The effective speedup comes from reducing actual denoising steps:

```text
CORA actual denoising step ratio < 1
```

The current best results are not caused by:

```text
routed FFN
sparse FFN FLOP reduction
EoT truncation
shorter generated outputs
```

Instead, the speedup comes from accepting high-confidence predictions that remain stable across consecutive denoising steps, then terminating completed blocks early.

## Important Logs

The server logs were produced under paths like:

```text
output/best_compare/gsm8k_g256_s256_limit50/
output/best_compare/gsm8k_g1024_s256_limit50/
```

Run summary extraction with:

```bash
grep -E "Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|CORA EoT truncated samples|exact_match" output/best_compare/*/*.log
```
