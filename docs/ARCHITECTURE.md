# Architecture

This document explains how the open-source folder is organized and where new
code should go.

## Top-Level Rule

The repository has two layers:

1. Public project layer: `README.md`, `docs/`, `configs/`, and `scripts/`.
2. Runnable experiment layer: `llada/`.

The public project layer makes the release readable. The runnable experiment
layer keeps the tested LLaDA code in one place so existing imports such as
`from model...` and `from generate...` continue to work.

## Directory Map

```text
cora_diff_llada_open_source/
  README.md
  LICENSE
  requirements.txt
  requirements-autodl.txt
  autodl_setup.sh
  configs/
  docs/
  figures/
  scripts/
  llada/
```

## `configs/`

Environment examples and reproducible settings live here.

Current file:

```text
configs/default.env.example
```

Use this for variables such as:

- `MODEL_PATH`
- `HF_HOME`
- `HF_ENDPOINT`
- `KLASS_PROB_DTYPE`
- `DAPD_ATTENTION_HEAD_CHUNK`

Do not put private tokens or server passwords in config files.

## `docs/`

Long-form documentation lives here.

```text
docs/ARCHITECTURE.md        # this file
```

Keep the root README focused on quick start and major concepts. Put longer
design and maintenance notes here.

## `scripts/`

Root-level scripts are user-facing launchers. They should be short wrappers
that `cd` into `llada/` and call the tested experiment scripts there.

```text
scripts/smoke.sh
scripts/reproduce_main_table.sh
scripts/three_method_compare.sh
```

This keeps the public entry points clean while preserving the original runnable
scripts under `llada/`.

## `figures/`

Figures used in README or documentation live here.

Do not put generated result logs or model checkpoints in `figures/`.

## `llada/`

This is the executable experiment code. Most commands eventually run from this
directory.

```text
llada/
  eval_llada.py
  generate.py
  extract_log_table.py
  layer_2_flan.pth
  model/
  vendor_methods/
  run_smoke_open_source.sh
  run_reproduce_main_table.sh
  run_best_hparams_fair_block32_full.sh
  run_three_method_compare.sh
```

### `llada/eval_llada.py`

This is the lm-evaluation-harness model wrapper. It parses `--model_args`,
loads the LLaDA model, dispatches to the selected decoding method, and prints
throughput/method statistics.

Add new command-line method arguments here when a method needs user-visible
hyperparameters.

### `llada/generate.py`

This file contains method dispatch helpers and generation implementations:

```text
generate(...)
generate_cora(...)
generate_prophet(...)
generate_klass(...)
generate_dapd(...)
generate_learn2parallel(...)
generate_l2p_eot(...)
```

Add a new decoding method here only if it shares the LLaDA generation loop.
Large third-party method code should live under `llada/vendor_methods/`.

### `llada/model/`

Model-side implementation code lives here.

```text
model/cora_diff.py          # CORA-Diff routing/state/scoring
model/modeling_llada.py     # LLaDA model wrapper and CORA hooks
model/configuration_llada.py
model/small_model.py        # Learn2PD small filter architecture
```

Put model architecture changes here, not in scripts.

### `llada/vendor_methods/`

Vendored baseline implementations live here.

```text
vendor_methods/prophet/
vendor_methods/klass/
vendor_methods/dapd/
vendor_methods/UPSTREAM_SOURCES.md
```

Keep third-party code isolated. If local compatibility patches are needed,
document them in `UPSTREAM_SOURCES.md`.

### Maintained `llada/run_*.sh` Launchers

The maintained low-level experiment scripts live next to `eval_llada.py`
because they invoke it directly:

```text
run_smoke_open_source.sh
run_reproduce_main_table.sh
run_best_hparams_fair_block32_full.sh
run_three_method_compare.sh
```

Recommended public entry points are still in root `scripts/`.

## Adding A New Baseline Method

1. Put third-party method code under `llada/vendor_methods/<method_name>/`.
2. Add a thin adapter in `llada/generate.py`.
3. Add method arguments and dispatch in `llada/eval_llada.py`.
4. Add a short usage example to `README.md`.
5. Add upstream attribution to `llada/vendor_methods/UPSTREAM_SOURCES.md`.
6. Add a smoke-test command or wrapper under `scripts/` if needed.

## Adding A New Experiment Script

1. Put low-level maintained experiment scripts under `llada/run_<name>.sh`.
2. If it is a recommended public command, add a root wrapper under `scripts/`.
3. Write outputs under `llada/output/<experiment_name>/`.
4. Do not commit generated logs or benchmark outputs.

## Runtime Outputs

Generated files should go under:

```text
llada/output/
```

Local caches may go under:

```text
llada/.hf_cache/
.hf_cache/
```

These paths are ignored by git.

## Learn2PD Checkpoint Note

`llada/layer_2_flan.pth` is a small Learn2PD filter checkpoint. The current
`eval_llada.py` initializes it during setup, so the file must be present even if
the selected method is CORA, Prophet, KLASS, or DAPD.

Only Learn2PD/L2P method branches call the small model, so it does not change
the behavior of the other methods.
