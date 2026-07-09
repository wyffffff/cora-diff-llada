# Vendored Decoding Methods

This directory vendors the LLaDA-facing decoding implementations used by
`llada/generate.py` and `llada/eval_llada.py`.

- Prophet: https://github.com/pixeli99/Prophet
  - Files: `prophet/generate_earlyexit.py`, `prophet/generate.py`
  - Local compatibility patch: `generate_earlyexit.py` honors the documented
    `answer_length` argument instead of hard-coding `5`.
- KLASS: https://github.com/shkim0116/KLASS
  - File: `klass/llada_klass.py`
  - Local memory patch: optional `KLASS_PROB_DTYPE=float32` and
    `KLASS_KL_VOCAB_CHUNK=<int>` reduce peak memory on long prompts while
    preserving KLASS's public hyperparameters and decoding logic.
- DAPD: https://github.com/quasar529/DAPD
  - Files: `dapd/core.py`, `dapd/generation.py`
  - Local memory patch: optional `DAPD_ATTENTION_HEAD_CHUNK=<int>` computes
    attention dependency heads in chunks instead of allocating one large QK
    tensor for all heads at once.

The public evaluator methods are thin adapters: `method=Prophet`,
`method=KLASS`, and `method=DAPD`.
