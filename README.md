# CORA-Diff LLaDA

This is the open-source release folder for CORA-Diff experiments on LLaDA. It
contains the local CORA-Diff implementation plus the comparison methods
Prophet, KLASS, and DAPD behind one shared evaluation interface.

The layout is intentionally conservative: the runnable LLaDA code stays under
`llada/`, because the evaluator and scripts use local imports from that
directory. The root-level `scripts/`, `configs/`, and `docs/` folders provide a
clean public entry point without breaking the tested experiment code.

## Directory Layout

```text
cora_diff_llada_open_source/
  README.md                         # Start here
  LICENSE
  requirements.txt                  # Minimal Python dependencies
  requirements-autodl.txt           # Server/AutoDL-oriented dependencies
  autodl_setup.sh                   # Optional AutoDL environment setup

  configs/
    default.env.example             # Common environment variables

  docs/
    ARCHITECTURE.md                 # Detailed directory and code map
    AUTODL.md                       # AutoDL/server quick start
    EXPERIMENT_RESULTS.md           # Current experiment notes/results

  figures/
    cora_diff_overview.svg

  scripts/
    smoke.sh                        # Recommended quick sanity check
    reproduce_main_table.sh          # Full fair comparison entry
    dapd_remaining_no_math1024.sh    # DAPD resume helper
    three_method_compare.sh          # Original/Learn2PD/CORA helper

  llada/
    eval_llada.py                   # lm-evaluation-harness model wrapper
    generate.py                     # Generation functions for all methods
    extract_log_table.py            # Log summarization helper
    layer_2_flan.pth                # Learn2PD small filter checkpoint

    model/
      cora_diff.py                  # CORA-Diff routing/state implementation
      modeling_llada.py             # LLaDA model wrapper with CORA hooks
      configuration_llada.py
      small_model.py

    vendor_methods/
      prophet/                      # Prophet vendored implementation
      klass/                        # KLASS vendored implementation
      dapd/                         # DAPD vendored implementation
      UPSTREAM_SOURCES.md

    run_*.sh                        # Detailed experiment/ablation scripts
```

## What Goes Where

- Use `scripts/` for public entry points that a new user should run first.
- Use `configs/` for environment-variable examples and reproducible settings.
- Use `docs/` for explanations, server notes, result notes, and design details.
- Use `figures/` for images referenced by docs.
- Use `llada/eval_llada.py` for the lm-evaluation-harness adapter.
- Use `llada/generate.py` for decoding method dispatch and generation logic.
- Use `llada/model/` for LLaDA/CORA-Diff model code.
- Use `llada/vendor_methods/` for third-party baseline method implementations.
- Keep generated logs and benchmark outputs under `llada/output/`; this path is
  ignored and should not be committed.

## Methods Included

The evaluator exposes these method names through `--model_args`:

```text
method=original
method=Learn2PD
method=L2P
method=L2P+EoT
method=CORA
method=Prophet
method=KLASS
method=DAPD
method=DAPD-Direct
method=DAPD-Staged
```

Comparison methods are vendored from:

- Prophet: https://github.com/pixeli99/Prophet
- KLASS: https://github.com/shkim0116/KLASS
- DAPD: https://github.com/quasar529/DAPD

See `llada/vendor_methods/UPSTREAM_SOURCES.md` for local compatibility notes.

## Installation

Recommended Python version: 3.10.

```bash
cd cora_diff_llada_open_source
conda create -n cora python=3.10 -y
conda activate cora
pip install -r requirements-autodl.txt
```

If you are not using AutoDL, `requirements.txt` may be enough:

```bash
pip install -r requirements.txt
```

The scripts default to:

```text
GSAI-ML/LLaDA-8B-Instruct
```

For mainland China mirrors:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

You can also copy and edit the environment example:

```bash
cp configs/default.env.example .env
source .env
```

## One-Command Smoke Test

Run this first after cloning. It runs one short GSM8K example across Original,
CORA-Diff, Prophet, KLASS, and DAPD.

```bash
cd cora_diff_llada_open_source
bash scripts/smoke.sh
```

Common overrides:

```bash
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct \
TASK=gsm8k \
LIMIT=2 \
GEN_LENGTH=64 \
STEPS=64 \
bash scripts/smoke.sh
```

Run only selected methods:

```bash
METHODS="original CORA KLASS" bash scripts/smoke.sh
```

Smoke logs are written to:

```text
llada/output/smoke_open_source/
```

## Full Reproduction

The main reproduction script runs the fair Prophet/KLASS/DAPD comparison table
with shared model, task, generation length, steps, remasking, cfg, and block
length.

```bash
cd cora_diff_llada_open_source
bash scripts/reproduce_main_table.sh
```

On a server, use tmux:

```bash
tmux new -d -s cora_table "bash -lc 'cd /path/to/cora_diff_llada_open_source && bash scripts/reproduce_main_table.sh 2>&1 | tee llada/output/main_table_tmux.log'"
tmux attach -t cora_table
```

Detach with `Ctrl-b` then `d`.

## Fair Comparison Protocol

The recommended fair table uses:

```text
model:        GSAI-ML/LLaDA-8B-Instruct
remasking:    low_confidence
cfg:          0.0
block_length: 32
settings:     256/256 and 1024/1024
tasks:        gsm8k, score_non_greedy_robustness_math, humaneval, mbpp
```

The full script keeps the base decoding budget aligned and applies the
method-specific best or official hyperparameters for Prophet, KLASS, and DAPD.

## Important Note About `layer_2_flan.pth`

`llada/eval_llada.py` currently initializes the small Learn2PD filter checkpoint
`llada/layer_2_flan.pth` during model setup. It is only used by Learn2PD/L2P
methods, but the file must stay in the repository so the evaluator can start.

This small checkpoint does not affect `original`, `CORA`, `Prophet`, `KLASS`, or
`DAPD` generation results, because those method branches do not call the
Learn2PD small model.

## Output Metrics

The evaluator prints standard task metrics plus throughput information:

```text
Number of tokens
Generation time
Tokens per second
exact_match / pass_at_1 / non_greedy_accuracy
```

Method-specific summaries include:

```text
CORA actual denoising step ratio
CORA fast accept tokens
Prophet actual denoising step ratio
KLASS actual denoising step ratio
DAPD step ratio vs configured steps
```

Extract a compact table from logs:

```bash
cd llada
python extract_log_table.py output/ablations/best_hparams_fair_block32_full_prophet_answerstart200
```

## More Documentation

- `docs/ARCHITECTURE.md`: detailed file-by-file map.
- `docs/AUTODL.md`: AutoDL/server setup and older experiment commands.
- `docs/EXPERIMENT_RESULTS.md`: current result notes.
- `llada/vendor_methods/UPSTREAM_SOURCES.md`: upstream baseline attribution.

## Citation And Attribution

This code builds on LLaDA and lm-evaluation-harness, and vendors code adapted
from Prophet, KLASS, and DAPD. Please cite the corresponding upstream projects
and papers when using the comparison implementations.
