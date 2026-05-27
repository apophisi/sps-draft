"""Run dynamic draft-depth experiments (Proposal Part 2).

Compares dynamic K (p_max early stop) against fixed K=4 and K=8, writes
results/dynamic_k_results.jsonl. Optional top-1 / top-2 margin strategy.

Proposal settings: greedy decoding (temperature=0), K_max=8,
threshold = 0.6 / 0.7 / 0.8.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from dynamic_policy import (
    DynamicGenerationResult,
    FixedDepthPolicy,
    PMaxEarlyStopPolicy,
    Top1Top2MarginEarlyStopPolicy,
    speculative_generate_dynamic,
)
from pipeline import load_draft_and_target, prefill_both
from runtime import encode_prompt
from runtime.model import ModelRunner, PrefillState
from speculative.sampling import logits_to_probs, sample_token


DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts" / "default_prompts.txt"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "dynamic_k_results.jsonl"

# Proposal symbols: K_max = 8; threshold = 0.6, 0.7, 0.8; fixed K = 4, 8
K_max = 8
THRESHOLDS = (0.6, 0.7, 0.8)
FIXED_K = (4, 8)


@dataclass(frozen=True)
class RunMetrics:
    """One experiment row; field names follow Proposal section 3 / 5."""

    method: str
    strategy: str
    K: int | None
    K_max: int | None
    threshold: float | None
    margin: float | None
    prompt_id: str
    prompt: str
    max_new_tokens: int
    generated_tokens: int
    ar_tokens: int
    ar_elapsed_sec: float
    ar_tokens_s: float
    rounds: int
    acceptance_rate: float
    average_acceptance_length: float
    average_draft_length: float
    tokens_s: float
    elapsed_sec: float
    speedup: float | None


def load_prompts(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def ar_baseline_generate(
    target: ModelRunner,
    state: PrefillState,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    rng: np.random.Generator,
    temperature: float = 0.0,
) -> tuple[list[int], float]:
    """Target-only autoregressive baseline with KV cache.

    This baseline uses the same target model and the same prefilling state as
    speculative decoding. It excludes prefill time, matching the speculative
    timing below, and measures only decode throughput.
    """

    token_ids: list[int] = []
    current = state
    synchronize_runner(target)
    start = time.perf_counter()

    while len(token_ids) < max_new_tokens:
        if temperature <= 0.0:
            logits = current.next_token_logits
            token_id = int(target.torch.argmax(logits, dim=-1).item())
        else:
            probs = logits_to_probs(current.next_token_logits, temperature=temperature)
            token_id = sample_token(probs, rng)
        token_ids.append(token_id)
        current = target.decode_one(token_id, current)
        if eos_token_id is not None and token_id == eos_token_id:
            break

    synchronize_runner(target)
    elapsed = time.perf_counter() - start
    return token_ids, elapsed


def average_draft_length_from_stats(stats) -> float:
    if stats.rounds == 0:
        return 0.0
    return stats.proposed_tokens / stats.rounds


def measure_run(
    *,
    method: str,
    strategy: str,
    prompt_id: str,
    prompt: str,
    max_new_tokens: int,
    elapsed_sec: float,
    generated_tokens: int,
    ar_tokens: int,
    ar_elapsed_sec: float,
    acceptance_rate: float,
    average_acceptance_length: float,
    average_draft_length: float,
    rounds: int,
    K: int | None = None,
    K_max: int | None = None,
    threshold: float | None = None,
    margin: float | None = None,
    speedup: float | None = None,
) -> RunMetrics:
    tokens_s = generated_tokens / elapsed_sec if elapsed_sec > 0 else 0.0
    ar_tokens_s = ar_tokens / ar_elapsed_sec if ar_elapsed_sec > 0 else 0.0
    return RunMetrics(
        method=method,
        strategy=strategy,
        K=K,
        K_max=K_max,
        threshold=threshold,
        margin=margin,
        prompt_id=prompt_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        generated_tokens=generated_tokens,
        ar_tokens=ar_tokens,
        ar_elapsed_sec=ar_elapsed_sec,
        ar_tokens_s=ar_tokens_s,
        rounds=rounds,
        acceptance_rate=acceptance_rate,
        average_acceptance_length=average_acceptance_length,
        average_draft_length=average_draft_length,
        tokens_s=tokens_s,
        elapsed_sec=elapsed_sec,
        speedup=speedup,
    )


def run_with_policy(
    draft: ModelRunner,
    target: ModelRunner,
    draft_state: PrefillState,
    target_state: PrefillState,
    policy,
    *,
    max_new_tokens: int,
    rng: np.random.Generator,
    eos_token_id: int | None,
    temperature: float,
) -> tuple[DynamicGenerationResult, float]:
    synchronize_runner(target)
    start = time.perf_counter()
    result = speculative_generate_dynamic(
        draft,
        target,
        draft_state,
        target_state,
        policy,
        max_new_tokens=max_new_tokens,
        rng=rng,
        eos_token_id=eos_token_id,
        temperature=temperature,
    )
    synchronize_runner(target)
    elapsed = time.perf_counter() - start
    return result, elapsed


def synchronize_runner(runner: ModelRunner) -> None:
    torch = runner.torch
    if torch.cuda.is_available() and runner.model_device.type == "cuda":
        torch.cuda.synchronize(runner.model_device)


def append_jsonl(path: Path, record: RunMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def compute_speedup(
    generated_tokens: int,
    elapsed_sec: float,
    ar_tokens: int,
    ar_elapsed_sec: float,
) -> float | None:
    """speedup = speculative tokens/s / AR baseline tokens/s (Proposal 3.1)."""

    if elapsed_sec <= 0 or ar_elapsed_sec <= 0 or ar_tokens <= 0:
        return None
    tokens_s = generated_tokens / elapsed_sec
    ar_tokens_s = ar_tokens / ar_elapsed_sec
    return tokens_s / ar_tokens_s


def print_summary(records: list[RunMetrics]) -> None:
    """Print aggregate comparison table (Proposal section 5)."""

    grouped: dict[str, list[RunMetrics]] = defaultdict(list)
    for record in records:
        grouped[record.strategy].append(record)

    def avg_strategy(key: str, field: str) -> float:
        rows = grouped[key]
        return mean(getattr(r, field) for r in rows)

    print()
    print("=== Dynamic K vs Fixed K (mean over prompts) ===")
    print(
        f"{'Method':<12} {'Strategy':<16} {'Tokens/s':>10} {'AR Tok/s':>10} "
        f"{'Speedup':>8} {'Acceptance Rate':>16} {'Avg Accept Length':>18} "
        f"{'Avg Draft Length':>17}"
    )
    print("-" * 112)

    strategy_order = [f"K={K}" for K in FIXED_K] + [f"p_max > {t}" for t in THRESHOLDS]
    for strategy in strategy_order:
        if strategy not in grouped:
            continue
        method = "Fixed K" if strategy.startswith("K=") else "Dynamic K"
        print(
            f"{method:<12} {strategy:<16} "
            f"{avg_strategy(strategy, 'tokens_s'):>10.2f} "
            f"{avg_strategy(strategy, 'ar_tokens_s'):>10.2f} "
            f"{avg_strategy(strategy, 'speedup'):>8.3f} "
            f"{avg_strategy(strategy, 'acceptance_rate'):>16.3f} "
            f"{avg_strategy(strategy, 'average_acceptance_length'):>18.3f} "
            f"{avg_strategy(strategy, 'average_draft_length'):>17.2f}"
        )

    print()
    print("Notes:")
    print("- speedup = speculative tokens/s divided by AR baseline tokens/s.")
    print("- p_max strategy stops drafting when p_max falls below threshold.")


def load_records(path: Path) -> list[RunMetrics]:
    if not path.exists():
        return []
    records: list[RunMetrics] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "ar_tokens_s" not in row:
            speedup = row.get("speedup")
            tokens_s = row.get("tokens_s", 0.0)
            row["ar_tokens_s"] = tokens_s / speedup if speedup else 0.0
        if "ar_elapsed_sec" not in row:
            ar_tokens_s = row.get("ar_tokens_s", 0.0)
            row["ar_elapsed_sec"] = (
                row.get("ar_tokens", 0) / ar_tokens_s if ar_tokens_s else 0.0
            )
        records.append(RunMetrics(**row))
    return records


def run_experiments(
    args: argparse.Namespace,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> list[RunMetrics]:
    prompts = load_prompts(Path(args.prompts))
    output = Path(args.output)
    if args.overwrite and output.exists():
        output.unlink()

    rng = np.random.default_rng(args.seed)
    tokenizer, draft, target = load_draft_and_target(
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        hf_endpoint=args.hf_endpoint,
        local_files_only=args.local_files_only,
    )
    eos_token_id = tokenizer.eos_token_id
    temperature = 0.0
    target_temperature = 1.0 if temperature <= 0.0 else temperature
    all_records: list[RunMetrics] = []

    for prompt_idx, prompt in enumerate(prompts):
        prompt_id = f"p{prompt_idx:03d}"
        batch = encode_prompt(
            tokenizer,
            prompt,
            mode=args.prompt_mode,
            enable_thinking=False,
        )

        _, target_state = prefill_both(draft, target, batch)
        ar_rng = np.random.default_rng(args.seed + prompt_idx)
        ar_token_ids, ar_elapsed_sec = ar_baseline_generate(
            target,
            target_state,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            rng=ar_rng,
            temperature=target_temperature,
        )
        ar_tokens = len(ar_token_ids)

        for K in FIXED_K:
            draft_state, target_state = prefill_both(draft, target, batch)
            policy = FixedDepthPolicy(K=K)
            result, elapsed_sec = run_with_policy(
                draft,
                target,
                draft_state,
                target_state,
                policy,
                max_new_tokens=args.max_new_tokens,
                rng=rng,
                eos_token_id=eos_token_id,
                temperature=temperature,
            )
            record = measure_run(
                method="Fixed K",
                strategy=policy.strategy_name,
                prompt_id=prompt_id,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                elapsed_sec=elapsed_sec,
                generated_tokens=len(result.token_ids),
                ar_tokens=ar_tokens,
                ar_elapsed_sec=ar_elapsed_sec,
                acceptance_rate=result.stats.accept_rate,
                average_acceptance_length=result.stats.avg_accept,
                average_draft_length=average_draft_length_from_stats(result.stats),
                rounds=result.stats.rounds,
                K=K,
                speedup=compute_speedup(
                    len(result.token_ids), elapsed_sec, ar_tokens, ar_elapsed_sec
                ),
            )
            append_jsonl(output, record)
            all_records.append(record)
            print(f"[Fixed K={K}] {prompt_id} tokens/s={record.tokens_s:.2f}")

        for threshold in thresholds:
            policy = PMaxEarlyStopPolicy(threshold=threshold, K_max=K_max)
            draft_state, target_state = prefill_both(draft, target, batch)
            result, elapsed_sec = run_with_policy(
                draft,
                target,
                draft_state,
                target_state,
                policy,
                max_new_tokens=args.max_new_tokens,
                rng=rng,
                eos_token_id=eos_token_id,
                temperature=temperature,
            )
            record = measure_run(
                method="Dynamic K",
                strategy=policy.strategy_name,
                prompt_id=prompt_id,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                elapsed_sec=elapsed_sec,
                generated_tokens=len(result.token_ids),
                ar_tokens=ar_tokens,
                ar_elapsed_sec=ar_elapsed_sec,
                acceptance_rate=result.stats.accept_rate,
                average_acceptance_length=result.stats.avg_accept,
                average_draft_length=result.stats.average_draft_length,
                rounds=result.stats.rounds,
                K_max=K_max,
                threshold=threshold,
                speedup=compute_speedup(
                    len(result.token_ids), elapsed_sec, ar_tokens, ar_elapsed_sec
                ),
            )
            append_jsonl(output, record)
            all_records.append(record)
            print(
                f"[Dynamic K {policy.strategy_name}] {prompt_id} "
                f"average_draft_length={record.average_draft_length:.2f} "
                f"tokens/s={record.tokens_s:.2f}"
            )

        if args.run_margin:
            for margin in args.margin_values:
                policy = Top1Top2MarginEarlyStopPolicy(margin=margin, K_max=K_max)
                draft_state, target_state = prefill_both(draft, target, batch)
                result, elapsed_sec = run_with_policy(
                    draft,
                    target,
                    draft_state,
                    target_state,
                    policy,
                    max_new_tokens=args.max_new_tokens,
                    rng=rng,
                    eos_token_id=eos_token_id,
                    temperature=temperature,
                )
                record = measure_run(
                    method="Dynamic K",
                    strategy=policy.strategy_name,
                    prompt_id=prompt_id,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    elapsed_sec=elapsed_sec,
                    generated_tokens=len(result.token_ids),
                    ar_tokens=ar_tokens,
                    ar_elapsed_sec=ar_elapsed_sec,
                    acceptance_rate=result.stats.accept_rate,
                    average_acceptance_length=result.stats.avg_accept,
                    average_draft_length=result.stats.average_draft_length,
                    rounds=result.stats.rounds,
                    K_max=K_max,
                    margin=margin,
                    speedup=compute_speedup(
                        len(result.token_ids), elapsed_sec, ar_tokens, ar_elapsed_sec
                    ),
                )
                append_jsonl(output, record)
                all_records.append(record)
                print(f"[Dynamic K {policy.strategy_name}] {prompt_id} done")

    return all_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic draft-depth experiments (Part 2)")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-mode", choices=["plain", "chat"], default="chat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace output JSONL")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary from existing JSONL without re-running experiments",
    )
    parser.add_argument(
        "--run-margin",
        action="store_true",
        help="Also run top-1 / top-2 margin strategy (optional per proposal)",
    )
    parser.add_argument(
        "--margin-values",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2],
        help="Margin thresholds for top-1 / top-2 strategy",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Single prompt, threshold=0.7 only (smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)

    if args.summary_only:
        records = load_records(output)
        if not records:
            raise SystemExit(f"No records found at {output}")
        print_summary(records)
        return

    thresholds = THRESHOLDS
    if args.quick:
        quick_prompts = Path(__file__).resolve().parent / "prompts" / "_quick.txt"
        quick_prompts.write_text(
            "请用一句话解释 speculative decoding。\n",
            encoding="utf-8",
        )
        args.prompts = str(quick_prompts)
        thresholds = (0.7,)

    records = run_experiments(args, thresholds=thresholds)
    print(f"Wrote results to {args.output}")
    print_summary(records)


if __name__ == "__main__":
    main()
