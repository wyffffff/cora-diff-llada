import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import generate_cora
from model.modeling_llada import LLaDAModelLM


DEFAULT_PROMPTS = [
    "Solve: If Lily runs 12 kilometers per hour for 4 hours and then 6 kilometers per hour for 4 hours, how far does she run?",
    "Write a Python function that returns the Fibonacci number at index n.",
    "A store sells pencils in packs of 8. How many packs are needed for 67 pencils?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate CORA-Diff active-token budget for LLaDA.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--target_ffn_ratio", type=float, default=0.7)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--core_ratio", type=float, default=0.5)
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--active_ratio_final", type=float, default=None)
    parser.add_argument("--budget_schedule", type=str, default="cosine", choices=["constant", "linear", "cosine"])
    parser.add_argument("--dependency_topk", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--channel_ordering", type=str, default="activation", choices=["activation", "weight"])
    return parser.parse_args()


def load_prompts(prompt_file):
    if prompt_file is None:
        return DEFAULT_PROMPTS
    path = Path(prompt_file)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dtype_from_name(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


@torch.no_grad()
def evaluate_budget(model, tokenizer, prompts, args, active_ratio):
    ffn_ratios = []
    refinement_ratios = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            text = prompt
        input_ids = torch.tensor(tokenizer(text)["input_ids"], device=model.device).unsqueeze(0)

        _, stats = generate_cora(
            model,
            input_ids,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=0,
            cfg_scale=0,
            cora_num_groups=args.num_groups,
            cora_core_ratio=args.core_ratio,
            cora_active_ratio=active_ratio,
            cora_active_ratio_final=args.active_ratio_final,
            cora_budget_schedule=args.budget_schedule,
            cora_alpha=args.alpha,
            cora_dependency_topk=args.dependency_topk,
            cora_order_channels=True,
            cora_channel_ordering=args.channel_ordering,
            cora_extra_commit=True,
            return_cora_stats=True,
        )
        ffn_ratios.append(stats["estimated_activated_ffn_ratio"])
        refinement_ratios.append(stats["avg_refinement_ratio"])

    return sum(ffn_ratios) / len(ffn_ratios), sum(refinement_ratios) / len(refinement_ratios)


def main():
    args = parse_args()
    prompts = load_prompts(args.prompt_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(args.model_path, dtype=dtype_from_name(args.dtype)).to(args.device).eval()

    low, high = 0.05, 1.0
    best = None
    for _ in range(args.iterations):
        mid = (low + high) / 2
        ffn_ratio, refinement_ratio = evaluate_budget(model, tokenizer, prompts, args, mid)
        best = (mid, ffn_ratio, refinement_ratio)
        if ffn_ratio > args.target_ffn_ratio:
            high = mid
        else:
            low = mid

    active_ratio, ffn_ratio, refinement_ratio = best
    print(f"Suggested cora_active_ratio={active_ratio:.4f}")
    print(f"Observed estimated_activated_ffn_ratio={ffn_ratio:.4f}")
    print(f"Observed avg_refinement_ratio={refinement_ratio:.4f}")
    final_ratio_arg = (
        ""
        if args.active_ratio_final is None
        else f"cora_active_ratio_final={args.active_ratio_final},"
    )
    print(
        "model_args snippet: "
        f"method=CORA,cora_core_ratio={args.core_ratio},cora_active_ratio={active_ratio:.4f},"
        f"{final_ratio_arg}cora_budget_schedule={args.budget_schedule},"
        f"cora_alpha={args.alpha},cora_dependency_topk={args.dependency_topk},"
        f"cora_channel_ordering={args.channel_ordering}"
    )


if __name__ == "__main__":
    main()
