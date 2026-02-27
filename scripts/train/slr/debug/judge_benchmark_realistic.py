"""
Realistic judge-model benchmark using ACTUAL OLMo rollout data.

Loads a sample of rollouts from a previous GRPO run, decodes them with the
original tokenizer, wraps them in the same quality_ref judge template used
during training, and measures throughput / parse success / latency for both
reasoning-on and reasoning-off modes.

All parameters read from environment variables (set by test_judge_model.sh).
"""
import asyncio
import json
import os
import random
import re
import statistics
import time

from openai import AsyncOpenAI

# ── Config from env ──────────────────────────────────────────────────────────
MODEL = os.environ["LLM_JUDGE_MODEL"]
PORT = int(os.environ.get("LLM_JUDGE_PORT", "8000"))
MAX_TOKENS = int(os.environ.get("MAX_COMPLETION_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
NUM_REQUESTS = int(os.environ.get("NUM_REQUESTS", "50"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
# Rollout data path
ROLLOUT_FILE = os.environ.get(
    "ROLLOUT_FILE",
    "/stage/output/RLVR-soofi-Olmo-IsomorphicRL/rollouts/"
    "RLVR-soofi-Olmo-IsomorphicRL__1__1771357951_rollouts_000000.jsonl",
)
ROLLOUT_TOKENIZER = os.environ.get("ROLLOUT_TOKENIZER", "allenai/Olmo-3-7B-Think-DPO")
# How many rollout lines to sample from (we read this many, then sample NUM_REQUESTS)
ROLLOUT_SAMPLE_POOL = int(os.environ.get("ROLLOUT_SAMPLE_POOL", "500"))
# Whether to run both reasoning modes or just one
# "both" | "reasoning" | "no_reasoning"
BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "both")

client = AsyncOpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy")

# ── Judge template (matches production quality_ref) ──────────────────────────
JUDGE_TEMPLATE = """\
### Task Description
Please act as an impartial judge and evaluate the quality of the answer provided by an
AI assistant to the conversation history leading up to the answer displayed below.
Judge whether the provided answer is good by comparing it to the reference answer.

Notes:
- Besides comparing to the reference answer, your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.
- Note that sometimes the reference answer is not the only answer. So any valid variation of the reference answer is also acceptable and can get a full score.
- If there is a system prompt, ensure the AI answer prioritizes following it.
- Begin your evaluation by providing a short explanation.
- Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10.
- Please adhere to the following format.

[Conversation History]
{input}

[AI Answer]
{output}

[Reference Gold Answer]
{label}

[Your judgement]
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}"""


def extract_final_answer(prediction: str) -> str:
    """Extract answer from model output (mirrors open_instruct.utils.extract_final_answer)."""
    answer_match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    think_match = re.search(r"</think>(.*)", prediction, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    cleaned = re.sub(r"<\|assistant\|>", "", prediction)
    if cleaned != prediction:
        return cleaned.strip()
    return prediction


# ── Load and decode rollouts ─────────────────────────────────────────────────
def load_rollout_prompts(num_prompts: int) -> list[dict]:
    """
    Load rollout records, decode tokens → text, format as judge prompts.
    Returns list of dicts with 'prompt_text' and metadata.
    """
    print(f"[data] Loading tokenizer: {ROLLOUT_TOKENIZER} ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROLLOUT_TOKENIZER, trust_remote_code=True)

    print(f"[data] Reading up to {ROLLOUT_SAMPLE_POOL} rollouts from {ROLLOUT_FILE} ...")
    records = []
    with open(ROLLOUT_FILE) as f:
        for i, line in enumerate(f):
            if i >= ROLLOUT_SAMPLE_POOL:
                break
            d = json.loads(line)
            records.append(d)

    print(f"[data] Loaded {len(records)} rollout records")

    # Decode and format
    formatted = []
    for rec in records:
        prompt_text = tokenizer.decode(rec["prompt_tokens"], skip_special_tokens=True)
        response_text = tokenizer.decode(rec["response_tokens"], skip_special_tokens=True)
        ground_truth = rec.get("ground_truth", [""])
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0] if ground_truth else ""

        # Extract final answer (like production code does)
        final_answer = extract_final_answer(response_text)

        # Build judge prompt using the quality_ref template
        judge_prompt = JUDGE_TEMPLATE.format(
            input=prompt_text,
            output=final_answer,
            label=ground_truth,
        )

        formatted.append(
            {
                "judge_prompt": judge_prompt,
                "prompt_tokens_len": len(rec["prompt_tokens"]),
                "response_tokens_len": len(rec["response_tokens"]),
                "reward": rec.get("reward", None),
                "judge_prompt_chars": len(judge_prompt),
            }
        )

    # Sample a diverse set (mix of short and long responses)
    if len(formatted) > num_prompts:
        # Sort by response length and take evenly spaced samples to get length diversity
        formatted.sort(key=lambda x: x["response_tokens_len"])
        step = len(formatted) / num_prompts
        indices = [int(i * step) for i in range(num_prompts)]
        formatted = [formatted[i] for i in indices]

    # Print length stats
    prompt_chars = [f["judge_prompt_chars"] for f in formatted]
    resp_lens = [f["response_tokens_len"] for f in formatted]
    print(f"[data] Selected {len(formatted)} judge prompts")
    print(
        f"[data] Judge prompt chars: "
        f"min={min(prompt_chars):,}, max={max(prompt_chars):,}, "
        f"mean={statistics.mean(prompt_chars):,.0f}, median={statistics.median(prompt_chars):,.0f}"
    )
    print(
        f"[data] Original response tokens: "
        f"min={min(resp_lens)}, max={max(resp_lens)}, "
        f"mean={statistics.mean(resp_lens):.0f}, median={statistics.median(resp_lens):.0f}"
    )
    return formatted


# ── Parse judge output ───────────────────────────────────────────────────────
def try_parse_score(text: str) -> tuple:
    """Returns (success, score, reason)."""
    if not text or not text.strip():
        return False, 0.0, "empty response"

    # Strip <think>...</think> blocks
    cleaned = re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<answer>", "").replace("</answer>", "")

    # Unclosed <think> means model hit token limit during reasoning
    if "<think>" in cleaned and "</think>" not in text:
        return False, 0.0, "unclosed <think> (hit token limit during reasoning)"

    cleaned = cleaned.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        score = float(data.get("SCORE", 0.0))
        return True, score, "json_ok"
    except (json.JSONDecodeError, TypeError, ValueError):
        score_match = re.search(r'"SCORE"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)"?', cleaned)
        if score_match:
            return True, float(score_match.group(1)), "regex_fallback"
        return False, 0.0, "parse_failed"


# ── Send requests ────────────────────────────────────────────────────────────
async def send_request(sem, prompt: str, idx: int, enable_thinking: bool):
    async with sem:
        t0 = time.monotonic()
        try:
            extra_body = {}
            kwargs = {}
            if not enable_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                # Force JSON output to match production LMJudgeVerifier behavior
                kwargs["response_format"] = {"type": "json_object"}

            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
                seed=42,
                timeout=300,
                extra_body=extra_body if extra_body else None,
                **kwargs,
            )
            elapsed = time.monotonic() - t0
            content = resp.choices[0].message.content or ""
            finish = resp.choices[0].finish_reason
            usage = resp.usage
            ok, score, reason = try_parse_score(content)
            return {
                "idx": idx,
                "ok": ok,
                "score": score,
                "reason": reason,
                "finish_reason": finish,
                "latency": elapsed,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "content_full": content,
            }
        except Exception as e:
            elapsed = time.monotonic() - t0
            return {
                "idx": idx,
                "ok": False,
                "score": 0.0,
                "reason": f"error: {e}",
                "finish_reason": "error",
                "latency": elapsed,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "content_full": "",
            }


def print_results(results: list[dict], mode_label: str, t_total: float):
    """Print benchmark results for one mode."""
    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    latencies = [r["latency"] for r in results]
    scores = [r["score"] for r in successes]
    total_prompt_tok = sum(r["prompt_tokens"] for r in results)
    total_compl_tok = sum(r["completion_tokens"] for r in results)

    finish_reasons: dict = {}
    for r in results:
        fr = r["finish_reason"]
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

    failure_reasons: dict = {}
    for r in failures:
        reason = r["reason"]
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    print("\n" + "=" * 70)
    print(f"  MODE: {mode_label}")
    print(f"  MODEL: {MODEL}")
    print(f"  MAX_COMPLETION_TOKENS: {MAX_TOKENS}  |  TEMPERATURE: {TEMPERATURE}")
    print("=" * 70)

    print(f"\n{'── Throughput ──':-<60}")
    print(f"  Wall time:        {t_total:.1f}s")
    print(f"  Requests/sec:     {len(results) / t_total:.2f}")
    print(f"  Prompt tok:       {total_prompt_tok:,}")
    print(f"  Completion tok:   {total_compl_tok:,}")
    print(f"  Total tok/sec:    {(total_prompt_tok + total_compl_tok) / t_total:,.0f}")
    print(f"  Completion tok/s: {total_compl_tok / t_total:,.0f}")

    print(f"\n{'── Parse Success ──':-<60}")
    print(f"  Success:  {len(successes)}/{len(results)} ({100 * len(successes) / len(results):.1f}%)")
    print(f"  Failed:   {len(failures)}/{len(results)} ({100 * len(failures) / len(results):.1f}%)")
    if failure_reasons:
        print("  Failure breakdown:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    print(f"\n{'── Finish Reasons ──':-<60}")
    for reason, count in sorted(finish_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    if latencies:
        print(f"\n{'── Latency (seconds) ──':-<60}")
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        print(f"  min:  {latencies_sorted[0]:.2f}")
        print(f"  p50:  {latencies_sorted[n // 2]:.2f}")
        print(f"  p90:  {latencies_sorted[int(n * 0.9)]:.2f}")
        print(f"  p99:  {latencies_sorted[int(n * 0.99)]:.2f}")
        print(f"  max:  {latencies_sorted[-1]:.2f}")
        print(f"  mean: {statistics.mean(latencies):.2f}")

    if scores:
        print(f"\n{'── Score Distribution ──':-<60}")
        print(f"  mean:   {statistics.mean(scores):.2f}")
        if len(scores) > 1:
            print(f"  stdev:  {statistics.stdev(scores):.2f}")
        print(f"  min:    {min(scores):.1f}")
        print(f"  max:    {max(scores):.1f}")

    print(f"\n{'── Sample Outputs (first 3) ──':-<60}")
    for r in results[:3]:
        status = "OK" if r["ok"] else "FAIL"
        print(
            f"\n  [{status}] Request {r['idx']} | score={r['score']} | "
            f"finish={r['finish_reason']} | {r['latency']:.1f}s | "
            f"tokens={r['completion_tokens']} | reason={r['reason']}"
        )
        print(f"  >>> {r['content_full']}")

    if failures:
        print(f"\n{'── Sample Failures (up to 3) ──':-<60}")
        for r in failures[:3]:
            print(
                f"\n  Request {r['idx']} | finish={r['finish_reason']} | "
                f"{r['latency']:.1f}s | tokens={r['completion_tokens']} | "
                f"reason={r['reason']}"
            )
            print(f"  >>> {r['content_full']}")

    return {
        "mode": mode_label,
        "wall_time": t_total,
        "req_per_sec": len(results) / t_total,
        "completion_tok_per_sec": total_compl_tok / t_total,
        "parse_success_rate": len(successes) / len(results),
        "mean_latency": statistics.mean(latencies) if latencies else 0,
        "p50_latency": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p90_latency": sorted(latencies)[int(len(latencies) * 0.9)] if latencies else 0,
        "mean_score": statistics.mean(scores) if scores else 0,
        "total_prompt_tok": total_prompt_tok,
        "total_compl_tok": total_compl_tok,
    }


async def run_benchmark(prompts: list[str], enable_thinking: bool, mode_label: str) -> dict:
    """Run all requests and print results. Returns summary dict."""
    sem = asyncio.Semaphore(CONCURRENCY)
    print(f"\n{'#' * 70}")
    print(f"# Benchmark: {mode_label}  (thinking={'ON' if enable_thinking else 'OFF'})")
    print(f"# {len(prompts)} requests, concurrency={CONCURRENCY}")
    print(f"{'#' * 70}")

    t_start = time.monotonic()
    tasks = [send_request(sem, p, i, enable_thinking) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    t_total = time.monotonic() - t_start

    summary = print_results(results, mode_label, t_total)
    return summary


async def main():
    # Load real rollout data
    formatted = load_rollout_prompts(NUM_REQUESTS)
    prompts = [f["judge_prompt"] for f in formatted]

    summaries = []

    modes_to_run = []
    if BENCHMARK_MODE in ("both", "no_reasoning"):
        modes_to_run.append(("NO REASONING (thinking=OFF)", False))
    if BENCHMARK_MODE in ("both", "reasoning"):
        modes_to_run.append(("WITH REASONING (thinking=ON)", True))

    for mode_label, enable_thinking in modes_to_run:
        summary = await run_benchmark(prompts, enable_thinking, mode_label)
        summaries.append(summary)

    # ── Comparison table ─────────────────────────────────────────────────
    if len(summaries) > 1:
        print("\n\n" + "=" * 70)
        print("  COMPARISON SUMMARY")
        print("=" * 70)
        header = f"{'Metric':<30} {'No Reasoning':>18} {'With Reasoning':>18}"
        print(header)
        print("-" * 70)
        s_no, s_yes = summaries[0], summaries[1]
        rows = [
            ("Wall time (s)", f"{s_no['wall_time']:.1f}", f"{s_yes['wall_time']:.1f}"),
            ("Requests/sec", f"{s_no['req_per_sec']:.2f}", f"{s_yes['req_per_sec']:.2f}"),
            ("Completion tok/sec", f"{s_no['completion_tok_per_sec']:,.0f}", f"{s_yes['completion_tok_per_sec']:,.0f}"),
            ("Total completion tok", f"{s_no['total_compl_tok']:,}", f"{s_yes['total_compl_tok']:,}"),
            ("Parse success %", f"{100*s_no['parse_success_rate']:.1f}%", f"{100*s_yes['parse_success_rate']:.1f}%"),
            ("Mean latency (s)", f"{s_no['mean_latency']:.2f}", f"{s_yes['mean_latency']:.2f}"),
            ("p50 latency (s)", f"{s_no['p50_latency']:.2f}", f"{s_yes['p50_latency']:.2f}"),
            ("p90 latency (s)", f"{s_no['p90_latency']:.2f}", f"{s_yes['p90_latency']:.2f}"),
            ("Mean score (1-10)", f"{s_no['mean_score']:.2f}", f"{s_yes['mean_score']:.2f}"),
        ]
        for label, v1, v2 in rows:
            print(f"  {label:<28} {v1:>18} {v2:>18}")

        # Speedup
        if s_yes["wall_time"] > 0:
            speedup = s_yes["wall_time"] / s_no["wall_time"]
            print(f"\n  No-reasoning is {speedup:.1f}x faster (wall time)")
        print("=" * 70)
    elif summaries:
        s = summaries[0]
        print(f"\n  SINGLE MODE: {s['mode']}")
        print(f"  Parse success: {100*s['parse_success_rate']:.1f}%")
        print(f"  Completion tok/sec: {s['completion_tok_per_sec']:,.0f}")
        print(f"  Mean latency: {s['mean_latency']:.2f}s")

    # Final verdict
    all_pass = all(s["parse_success_rate"] >= 0.90 for s in summaries)
    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n  VERDICT: {verdict}  (target >= 90% parse success per mode)")
    print("=" * 70)


asyncio.run(main())
