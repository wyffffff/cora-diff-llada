import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate import add_gumbel_noise, get_num_transfer_tokens
from model.modeling_llada import LLaDAModelLM


DEFAULT_PROMPTS = [
    "Solve: If Lily runs 12 kilometers per hour for 4 hours and then 6 kilometers per hour for 4 hours, how far does she run?",
    "Write a Python function that returns the Fibonacci number at index n.",
    "A store sells pencils in packs of 8. How many packs are needed for 67 pencils?",
]


TRACE_FIELDS = [
    "prompt_id",
    "global_step",
    "block_id",
    "block_step",
    "abs_pos",
    "gen_pos",
    "pred_token",
    "confidence",
    "persistence_m",
    "dense_final_token",
    "dense_disagree",
    "valid_final",
]


AGG_FIELDS = [
    "confidence_bucket",
    "bucket_low",
    "bucket_high",
    "persistence_m",
    "n",
    "disagreements",
    "disagreement_probability",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect token-level temporal-persistence residual-risk data by "
            "comparing intermediate dense predictions with the final dense decode."
        )
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", action="append", default=None, help="Prompt text. Can be passed multiple times.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Text, JSONL, or JSON prompt file.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of prompts to run.")
    parser.add_argument("--output_dir", type=str, default="output/persistence_risk")
    parser.add_argument("--trace_name", type=str, default="persistence_trace.csv")
    parser.add_argument("--aggregate_name", type=str, default="persistence_risk.csv")
    parser.add_argument("--plot_name", type=str, default="persistence_risk.png")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--eot_id", type=int, default=126081)
    parser.add_argument(
        "--disable_eot_truncation",
        action="store_true",
        help="Do not truncate the dense trajectory when an EoT token is predicted.",
    )
    parser.add_argument(
        "--confidence_bins",
        type=str,
        default="0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated bucket edges, for example 0.5,0.6,0.7,0.8,0.9.",
    )
    parser.add_argument("--max_m", type=int, default=6, help="Clip persistence length to this value for aggregation.")
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Use raw prompt text instead of tokenizer.apply_chat_template.",
    )
    return parser.parse_args()


def dtype_from_name(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_confidence_bins(raw):
    edges = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(edges) < 2:
        raise ValueError("--confidence_bins must contain at least two edges.")
    if any(high <= low for low, high in zip(edges, edges[1:])):
        raise ValueError("--confidence_bins must be strictly increasing.")
    return edges


def load_prompts(prompt_file, prompt_args, limit):
    prompts = []
    if prompt_args:
        prompts.extend(prompt_args)

    if prompt_file:
        path = Path(prompt_file)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                prompts.append(prompt_from_json_item(item))
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                prompts.extend(prompt_from_json_item(item) for item in data)
            else:
                prompts.append(prompt_from_json_item(data))
        else:
            prompts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    if not prompts:
        prompts = list(DEFAULT_PROMPTS)

    if limit is not None:
        prompts = prompts[: max(0, int(limit))]
    return prompts


def prompt_from_json_item(item):
    if isinstance(item, str):
        return item
    for key in ("prompt", "question", "text", "input"):
        if key in item and item[key] is not None:
            return str(item[key])
    raise ValueError(f"Could not find a prompt field in JSON item keys={list(item.keys())}")


def encode_prompt(tokenizer, prompt, model_path, no_chat_template):
    use_chat_template = (
        not no_chat_template
        and "instruct" in model_path.lower()
        and hasattr(tokenizer, "apply_chat_template")
    )
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(prompt)["input_ids"]


def confidence_bucket(confidence, bins):
    for index, (low, high) in enumerate(zip(bins, bins[1:])):
        if low <= confidence < high or (index == len(bins) - 2 and confidence == high):
            return index, low, high
    return None


def model_forward_logits(model, x, prompt_length, cfg_scale, mask_id):
    if cfg_scale <= 0.0:
        return model(x).logits

    prompt_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    prompt_index[:, : min(prompt_length, x.shape[1])] = True
    un_x = x.clone()
    un_x[prompt_index] = mask_id
    x_ = torch.cat([x, un_x], dim=0)
    logits = model(x_).logits
    logits, un_logits = torch.chunk(logits, 2, dim=0)
    return un_logits + (cfg_scale + 1.0) * (logits - un_logits)


@torch.no_grad()
def dense_decode_with_persistence_trace(model, prompt_ids, prompt_id, args):
    if args.gen_length % args.block_length != 0:
        raise ValueError("--gen_length must be divisible by --block_length.")
    num_blocks = args.gen_length // args.block_length
    if args.steps % num_blocks != 0:
        raise ValueError("--steps must be divisible by gen_length / block_length.")

    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    prompt_length = prompt.shape[1]
    steps_per_block = args.steps // num_blocks

    x = torch.full(
        (1, prompt_length + args.gen_length),
        args.mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_length] = prompt
    records = []
    global_step = 0

    for block_id in range(num_blocks):
        block_start = prompt_length + block_id * args.block_length
        if block_start >= x.shape[1]:
            break
        block_end = min(prompt_length + (block_id + 1) * args.block_length, x.shape[1])
        block_slice = slice(block_start, block_end)
        block_mask_index = x[:, block_slice] == args.mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        stability_count = torch.zeros_like(x, dtype=torch.int16, device=x.device)
        prev_pred = None

        for block_step in range(steps_per_block):
            if block_start >= x.shape[1] or not (x[:, block_slice] == args.mask_id).any():
                break

            mask_index = x == args.mask_id
            current_block_mask = torch.zeros_like(mask_index, dtype=torch.bool, device=x.device)
            current_block_mask[:, block_slice] = True

            logits = model_forward_logits(model, x, prompt_length, args.cfg_scale, args.mask_id)
            logits_with_noise = add_gumbel_noise(logits, temperature=args.temperature)
            pred = torch.argmax(logits_with_noise, dim=-1)

            probs = F.softmax(logits, dim=-1)
            pred_confidence = torch.squeeze(
                torch.gather(probs, dim=-1, index=torch.unsqueeze(pred, -1)),
                -1,
            )

            score_mask = mask_index & current_block_mask
            if prev_pred is not None and prev_pred.shape == pred.shape:
                stable_prediction = pred == prev_pred.to(device=pred.device)
            else:
                stable_prediction = torch.zeros_like(mask_index, dtype=torch.bool, device=x.device)
            stability_count = torch.where(
                score_mask & stable_prediction,
                stability_count + 1,
                torch.zeros_like(stability_count),
            )

            positions = torch.nonzero(score_mask[0], as_tuple=False).flatten()
            for pos in positions.tolist():
                records.append(
                    {
                        "prompt_id": prompt_id,
                        "global_step": global_step,
                        "block_id": block_id,
                        "block_step": block_step,
                        "abs_pos": pos,
                        "gen_pos": pos - prompt_length,
                        "pred_token": int(pred[0, pos].item()),
                        "confidence": float(pred_confidence[0, pos].item()),
                        "persistence_m": int(stability_count[0, pos].item()),
                    }
                )

            dense_pred = torch.where(mask_index, pred, x)
            confidence = pred_confidence.clone()
            confidence[:, block_end:] = -float("inf")
            confidence = torch.where(mask_index, confidence, torch.full_like(confidence, -float("inf")))

            transfer_index = torch.zeros_like(dense_pred, dtype=torch.bool, device=x.device)
            for batch_id in range(confidence.shape[0]):
                transfer_count = int(num_transfer_tokens[batch_id, block_step].item())
                if transfer_count <= 0:
                    continue
                _, select_index = torch.topk(confidence[batch_id], k=transfer_count)
                transfer_index[batch_id, select_index] = True
            x[transfer_index] = dense_pred[transfer_index]

            prev_pred = pred.detach()
            global_step += 1

            if not args.disable_eot_truncation:
                eot_confidence = torch.where(
                    (pred == args.eot_id) & mask_index,
                    pred_confidence,
                    torch.full_like(pred_confidence, -float("inf")),
                )
                if (eot_confidence > -float("inf")).any():
                    eot_pos = int(eot_confidence.argmax(dim=-1)[0].item())
                    x = x[:, : eot_pos + 1]
                    if prev_pred.shape[1] > x.shape[1]:
                        prev_pred = prev_pred[:, : x.shape[1]]
                    break

    final_tokens = x[0].detach().cpu()
    for record in records:
        pos = record["abs_pos"]
        valid = pos < final_tokens.numel() and int(final_tokens[pos].item()) != args.mask_id
        record["valid_final"] = int(valid)
        if valid:
            dense_final_token = int(final_tokens[pos].item())
            record["dense_final_token"] = dense_final_token
            record["dense_disagree"] = int(record["pred_token"] != dense_final_token)
        else:
            record["dense_final_token"] = ""
            record["dense_disagree"] = ""
    return records, final_tokens


def aggregate_records(records, bins, max_m):
    counts = defaultdict(lambda: [0, 0])
    bucket_meta = {}
    for record in records:
        if int(record["valid_final"]) != 1:
            continue
        bucket = confidence_bucket(float(record["confidence"]), bins)
        if bucket is None:
            continue
        bucket_id, low, high = bucket
        m = min(int(record["persistence_m"]), max_m)
        key = (bucket_id, m)
        counts[key][0] += 1
        counts[key][1] += int(record["dense_disagree"])
        bucket_meta[bucket_id] = (low, high)

    rows = []
    for bucket_id, m in sorted(counts):
        low, high = bucket_meta[bucket_id]
        n, disagreements = counts[(bucket_id, m)]
        rows.append(
            {
                "confidence_bucket": f"{low:.2f}-{high:.2f}",
                "bucket_low": f"{low:.6g}",
                "bucket_high": f"{high:.6g}",
                "persistence_m": m,
                "n": n,
                "disagreements": disagreements,
                "disagreement_probability": f"{disagreements / max(1, n):.8f}",
            }
        )
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregate(path, rows, max_m):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row["confidence_bucket"]].append(row)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for bucket in sorted(by_bucket):
        points = sorted(by_bucket[bucket], key=lambda item: int(item["persistence_m"]))
        xs = [int(item["persistence_m"]) for item in points]
        ys = [float(item["disagreement_probability"]) for item in points]
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"Confidence {bucket}")

    ax.set_title("Temporal Persistence Reduces Residual Risk")
    ax.set_xlabel("Persistence length m")
    ax.set_ylabel("Probability of disagreement with dense decoder")
    ax.set_xlim(0, max_m)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    bins = parse_confidence_bins(args.confidence_bins)
    prompts = load_prompts(args.prompt_file, args.prompt, args.limit)
    output_dir = Path(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_name(args.dtype),
    ).to(args.device)
    model.eval()

    all_records = []
    for prompt_id, prompt in enumerate(tqdm(prompts, desc="Collecting persistence traces")):
        prompt_ids = encode_prompt(tokenizer, prompt, args.model_path, args.no_chat_template)
        records, _ = dense_decode_with_persistence_trace(model, prompt_ids, prompt_id, args)
        all_records.extend(records)

    aggregate_rows = aggregate_records(all_records, bins, args.max_m)
    trace_path = output_dir / args.trace_name
    aggregate_path = output_dir / args.aggregate_name
    write_csv(trace_path, all_records, TRACE_FIELDS)
    write_csv(aggregate_path, aggregate_rows, AGG_FIELDS)

    print(f"Wrote token trace: {trace_path}")
    print(f"Wrote aggregate risk table: {aggregate_path}")
    if not args.no_plot:
        plot_path = output_dir / args.plot_name
        plot_aggregate(plot_path, aggregate_rows, args.max_m)
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
