# AutoDL Quick Start

This guide is for running the packaged `cora_diff_llada` folder on an AutoDL GPU server.

## 1. Upload And Unzip

Upload `cora_diff_llada_autodl.zip` to AutoDL, then run:

```bash
unzip cora_diff_llada_autodl.zip
cd cora_diff_llada
```

If you uploaded the folder directly, just:

```bash
cd cora_diff_llada
```

## 2. Install

AutoDL images usually include CUDA-matched PyTorch. The default setup script keeps the existing torch and installs the rest:

```bash
bash autodl_setup.sh
conda activate cora
```

If shell scripts are not executable after upload, run:

```bash
chmod +x autodl_setup.sh llada/run_cora_gsm8k.sh llada/run_cora_calibrate.sh
```

If your image does not have torch installed, run:

```bash
INSTALL_TORCH=1 bash autodl_setup.sh
conda activate cora
```

Optional Hugging Face login:

```bash
huggingface-cli login
```

Model and dataset caches default to:

```text
/root/autodl-tmp/huggingface
```

## 3. First Smoke Run

Start with a shorter generation length:

```bash
cd llada
bash run_cora_gsm8k.sh
```

By default this runs the GPU-friendlier step-reduction path:

```text
TASK=gsm8k
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
GEN_LENGTH=128
STEPS=128
BLOCK_LENGTH=32
CORA_ROUTING_MODE=disabled
CORA_FAST_ACCEPT=true
CORA_EOT=true
```

The log is written to:

```text
llada/output/eval_results/gsm8k/eval_CORA_128.log
```

## 4. Full 256 Run

```bash
cd llada
GEN_LENGTH=256 STEPS=256 bash run_cora_gsm8k.sh
```

Useful overrides:

```bash
TASK=math GEN_LENGTH=256 STEPS=256 bash run_cora_gsm8k.sh
TASK=humaneval GEN_LENGTH=256 STEPS=256 bash run_cora_gsm8k.sh
TASK=mbpp GEN_LENGTH=256 STEPS=256 bash run_cora_gsm8k.sh
```

## 4.5 Tune Step Reduction

After diagnostics show that `cora_step_native` is faster, sweep a few conservative thresholds:

```bash
cd llada
LIMIT=20 GEN_LENGTH=128 STEPS=128 bash run_cora_step_sweep.sh
```

Read the summary:

```bash
grep -E "Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|CORA EoT truncated samples|exact_match" output/step_sweep/gsm8k_g128_s128_limit20/*.log
```

Choose the fastest setting whose strict exact match is close to `original`, then rerun it with a larger `LIMIT`.

To compare the current best setting against `original`:

```bash
cd llada
LIMIT=50 GEN_LENGTH=128 STEPS=128 bash run_cora_best_compare.sh
```

For a longer generation setting:

```bash
LIMIT=50 GEN_LENGTH=256 STEPS=256 bash run_cora_best_compare.sh
```

## 4.6 Aligned Method Compare

To compare the main decoding methods in one run:

```bash
cd llada
LIMIT=50 GEN_LENGTH=256 STEPS=256 bash run_three_method_compare.sh
```

This runs:

```text
original   dense LLaDA decoding
learn2pd   Learn2PD/L2P filtering with layer_2_flan.pth
cora_best  CORA-Diff best validated step-reduction path
cora_plus  CORA-Diff with the added residual/top-k drift scoring module
```

The shared decoding hyperparameters are aligned across all three methods:

```text
TASK=gsm8k
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
GEN_LENGTH=256
STEPS=256
BLOCK_LENGTH=32
REMASKING=low_confidence
CFG=0.0
temperature=0
```

Learn2PD additionally uses `LEARN2PD_ACCEPT_THRES=0.96`, matching the original LLaDA script.
The two CORA variants share the same step-reduction settings; `cora_plus` only turns on the added residual/top-k drift scoring module.

For the 1024-length setting used in the long-generation comparison:

```bash
LIMIT=50 GEN_LENGTH=1024 STEPS=256 bash run_three_method_compare.sh
```

Useful overrides:

```bash
LEARN2PD_METHOD=Learn2PD LEARN2PD_ACCEPT_THRES=0.96 bash run_three_method_compare.sh
LEARN2PD_METHOD=L2P+EoT bash run_three_method_compare.sh
CORA_PLUS_RESIDUAL_THRESHOLD=0.5 CORA_PLUS_DRIFT_TOPK=8 bash run_three_method_compare.sh
```

Logs are written to:

```text
llada/output/three_method_compare/
```

## 4.6.5 EoTP Increment Compare

In this repo, EoTP is implemented as `method=EoT` for standalone end-token truncation and `method=L2P+EoT` for Learn2PD plus EoTP. To compare EoTP against Learn2PD and CORA-Diff under aligned decoding settings:

```bash
cd llada
LIMIT=50 GEN_LENGTH=256 STEPS=256 bash run_eotp_compare.sh
```

For the long-generation setting:

```bash
LIMIT=50 GEN_LENGTH=1024 STEPS=256 bash run_eotp_compare.sh
```

The script runs:

```text
original         dense LLaDA decoding
eot              standalone EoTP/EoT truncation
learn2pd         Learn2PD/L2P filtering
learn2pd_eot     Learn2PD plus EoTP/EoT
cora_best_noeot  CORA-Diff without EoT truncation
cora_best_eot    CORA-Diff with EoT truncation
cora_plus_noeot  CORA-Diff residual/top-k variant without EoT
cora_plus_eot    CORA-Diff residual/top-k variant with EoT
```

Use these paired comparisons:

```text
learn2pd_eot - learn2pd       EoTP increment on Learn2PD
cora_best_eot - cora_best_noeot   EoTP increment on CORA-Diff
cora_plus_eot - cora_plus_noeot   EoTP increment on CORA-Diff plus residual/top-k scoring
```

Logs are written to:

```text
llada/output/eotp_compare/
```

## 4.7 Hidden Drift Ablation

Hidden-state drift only affects the adaptive routed-FFN path. To test whether it helps, keep all routing parameters fixed and compare `cora_gamma=0/1/2`:

```bash
cd llada
LIMIT=20 GEN_LENGTH=128 STEPS=128 bash run_cora_hidden_drift_ablation.sh
```

Useful reading:

```bash
grep -E "Tokens per second|CORA avg refinement ratio|CORA estimated activated FFN ratio|exact_match" output/hidden_drift_ablation/gsm8k_g128_s128_limit20/*.log
```

The module is useful if `gamma=1` or `gamma=2` gives higher exact match at a similar activated FFN ratio, or a lower activated FFN ratio at similar exact match. It is not expected to improve the current `cora_step_best` path because that path disables routed FFN.

## 4.8 Residual Stability Sweep

To test the paper-style residual score with top-k distribution drift:

```bash
cd llada
LIMIT=20 GEN_LENGTH=256 STEPS=256 bash run_cora_residual_sweep.sh
```

The sweep compares the current persistence-only path with residual-score variants:

```text
persistence_only
residual_t0_top8
residual_t05_top8
residual_t10_top8
drift_only_t05_top8
```

Useful summary:

```bash
grep -E "Tokens per second|CORA actual denoising step ratio|CORA residual accept tokens|CORA avg residual score|exact_match" output/residual_sweep/gsm8k_g256_s256_limit20/*.log
```

Enable one residual configuration directly with:

```bash
LIMIT=50 GEN_LENGTH=256 STEPS=256 \
CORA_RESIDUAL_ACCEPT=true \
CORA_RESIDUAL_THRESHOLD=0.5 \
CORA_DRIFT_TOPK=8 \
bash run_cora_gsm8k.sh
```

## 5. Calibrate CORA Budget

To search for an active-token ratio near a target activated FFN ratio:

```bash
cd llada
TARGET_FFN_RATIO=0.70 GEN_LENGTH=128 STEPS=128 bash run_cora_calibrate.sh
```

Copy the suggested `cora_active_ratio` into:

```bash
CORA_ACTIVE_RATIO=0.xxxx bash run_cora_gsm8k.sh
```

## 5.5 Diagnose Why CORA Is Slower

If CORA lowers the estimated FFN ratio but wall-clock TPS is slower, run:

```bash
cd llada
LIMIT=5 GEN_LENGTH=128 STEPS=128 bash run_cora_diagnostics.sh
```

This runs five comparable settings:

```text
original
cora_control_native_mlp
cora_step_native
cora_dense_routed
cora_aggressive
```

How to read it:

- If `cora_control_native_mlp` is much slower than `original`, the overhead is mostly outside routed FFN.
- If `cora_control_native_mlp` is close to `original` but `cora_dense_routed` is much slower, the routed FFN implementation is the bottleneck.
- If `cora_step_native` is faster than `original`, the real speedup is coming from fewer denoising forwards, which is closer to Learn2PD/EoTP.
- If `cora_aggressive` is still slower, fine-grained dynamic routing is not GPU-friendly enough and you should switch to coarser routing or step-reduction/EoT-style acceleration.

To reproduce the old fine-grained routed FFN path explicitly:

```bash
CORA_ROUTING_MODE=adaptive CORA_ORDER_CHANNELS=true CORA_FAST_ACCEPT=false CORA_EOT=false bash run_cora_gsm8k.sh
```

## 6. If CUDA OOM Happens

Try these in order:

```bash
GEN_LENGTH=128 STEPS=128 bash run_cora_gsm8k.sh
```

```bash
CORA_DEPENDENCY_TOPK=8 bash run_cora_gsm8k.sh
```

```bash
CORA_CHANNEL_ORDERING=weight bash run_cora_gsm8k.sh
```

If `bfloat16` is not supported by the GPU, edit `llada/eval_llada.py` and replace:

```python
torch_dtype=torch.bfloat16
```

with:

```python
torch_dtype=torch.float16
```

## 7. What To Check In Logs

Look for:

```text
Tokens per second
CORA avg refinement ratio
CORA estimated activated FFN ratio
CORA actual denoising step ratio
CORA fast accept tokens
CORA EoT truncated samples
```

The CORA FFN ratio is the approximate activated FFN/MLP compute ratio used for budget comparisons. The step ratio is usually more important for wall-clock speed: values below `1.0` mean fewer model forwards were actually run.
