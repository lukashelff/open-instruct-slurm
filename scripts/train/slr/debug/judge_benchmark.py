"""
Judge model benchmark: sends realistic judge prompts to a running vLLM server
and reports throughput, parse-success rate, and latency.

All parameters are read from environment variables (set by test_judge_model.sh).
"""
import asyncio
import json
import os
import re
import statistics
import time

from openai import AsyncOpenAI

# ── Config from env ──
MODEL = os.environ["LLM_JUDGE_MODEL"]
PORT = int(os.environ.get("LLM_JUDGE_PORT", "8000"))
MAX_TOKENS = int(os.environ.get("MAX_COMPLETION_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
NUM_REQUESTS = int(os.environ.get("NUM_REQUESTS", "50"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))

client = AsyncOpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy")

# ── Realistic judge prompts (quality_ref style) ──
# These mirror what LMJudgeVerifier sends during training.
JUDGE_PROMPTS = [
    {
        "input": "Solve the equation: 2x + 5 = 13",
        "output": (
            "<think>\nI need to solve 2x + 5 = 13.\n"
            "Subtract 5 from both sides: 2x = 8\nDivide by 2: x = 4\n</think>\n\n"
            "To solve 2x + 5 = 13:\n\n"
            "1. Subtract 5 from both sides: 2x = 8\n"
            "2. Divide by 2: x = 4\n\n"
            "The answer is **x = 4**."
        ),
        "label": "x = 4",
    },
    {
        "input": "Write a Python function to check if a string is a palindrome.",
        "output": (
            "<think>\nA palindrome reads the same forwards and backwards. "
            "I can compare the string to its reverse.\n</think>\n\n"
            "```python\ndef is_palindrome(s):\n    s = s.lower().strip()\n    return s == s[::-1]\n```\n\n"
            "This function converts to lowercase and checks if the string equals its reverse."
        ),
        "label": "def is_palindrome(s):\n    return s == s[::-1]",
    },
    {
        "input": "Explain the difference between TCP and UDP protocols.",
        "output": (
            "<think>\nTCP and UDP are both transport layer protocols. "
            "TCP is connection-oriented and reliable, UDP is connectionless and faster but unreliable.\n</think>\n\n"
            "**TCP (Transmission Control Protocol):**\n"
            "- Connection-oriented (3-way handshake)\n"
            "- Reliable delivery with acknowledgments\n"
            "- Order guaranteed\n"
            "- Slower due to overhead\n"
            "- Used for: web browsing, email, file transfer\n\n"
            "**UDP (User Datagram Protocol):**\n"
            "- Connectionless\n- No delivery guarantee\n"
            "- No order guarantee\n- Faster, lower latency\n"
            "- Used for: streaming, gaming, DNS"
        ),
        "label": (
            "TCP is connection-oriented, reliable, ordered, and slower. "
            "UDP is connectionless, unreliable, unordered, and faster. "
            "TCP is used for web/email/file transfer; UDP for streaming/gaming/DNS."
        ),
    },
    {
        "input": "A farmer has 100 meters of fencing. What dimensions of a rectangular field maximize the enclosed area?",
        "output": (
            "<think>\nPerimeter = 2l + 2w = 100, so l + w = 50.\n"
            "Area = l * w = l * (50 - l) = 50l - l^2\n"
            "Maximize: dA/dl = 50 - 2l = 0 => l = 25\n"
            "So w = 25 as well. It is a square.\n</think>\n\n"
            "The field should be a **25m x 25m square**, giving a maximum area of **625 m²**.\n\n"
            "Proof: With perimeter P = 100m, we have l + w = 50. "
            "Area A = l(50-l) is maximized when l = 25 (by completing the square or calculus)."
        ),
        "label": "25m x 25m square, area = 625 m²",
    },
    {
        "input": "Translate to French: The cat sat on the mat and watched the birds fly by.",
        "output": "Le chat s assis sur le tapis et a regarde les oiseaux passer en volant.",
        "label": "Le chat s'est assis sur le tapis et a regardé les oiseaux passer en volant.",
    },
    {
        "input": "What are the main causes of World War I?",
        "output": (
            "<think>\nWWI causes: MANIA - Militarism, Alliances, Nationalism, Imperialism, Assassination.\n</think>\n\n"
            "The main causes of World War I include:\n\n"
            "1. **Militarism** - European powers engaged in an arms race\n"
            "2. **Alliance System** - Complex web of treaties (Triple Alliance vs Triple Entente)\n"
            "3. **Nationalism** - Ethnic tensions, especially in the Balkans\n"
            "4. **Imperialism** - Competition for colonies and resources\n"
            "5. **Assassination** - The assassination of Archduke Franz Ferdinand in 1914 was the immediate trigger"
        ),
        "label": "Militarism, alliances, nationalism, imperialism, and the assassination of Archduke Franz Ferdinand.",
    },
    {
        "input": "Find the longest increasing subsequence. Input: [10, 9, 2, 5, 3, 7, 101, 18]",
        "output": (
            "<think>\nUsing dynamic programming. LIS for each element:\n"
            "10: [10] len=1\n9: [9] len=1\n2: [2] len=1\n5: [2,5] len=2\n"
            "3: [2,3] len=2\n7: [2,3,7] or [2,5,7] len=3\n"
            "101: [2,3,7,101] len=4\n18: [2,3,7,18] len=4\nLIS length = 4\n</think>\n\n"
            "The length of the longest increasing subsequence is **4**.\n\n"
            "One such subsequence is: [2, 3, 7, 101] or [2, 3, 7, 18] or [2, 5, 7, 101]."
        ),
        "label": "4. Example subsequence: [2, 3, 7, 101]",
    },
    {
        "input": "Explain quantum entanglement in simple terms.",
        "output": (
            "<think>\nQuantum entanglement is when two particles become linked and measuring one "
            "instantly affects the other, regardless of distance. Einstein called it spooky action at a distance.\n</think>\n\n"
            "Quantum entanglement is like having two magic coins. When you flip them, they always land on "
            "matching sides, no matter how far apart they are. In physics, when two particles are entangled, "
            "measuring the state of one instantly tells you the state of the other, even across the universe. "
            "Einstein famously called this spooky action at a distance."
        ),
        "label": (
            "Quantum entanglement is a phenomenon where two particles become correlated such that "
            "measuring one instantly determines the state of the other, regardless of distance."
        ),
    },
    {
        "input": "SQL: top 5 customers by total order amount. Tables: customers(id, name), orders(id, customer_id, amount).",
        "output": (
            "SELECT c.name, SUM(o.amount) as total FROM customers c "
            "JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total DESC LIMIT 5;"
        ),
        "label": (
            "SELECT c.name, SUM(o.amount) AS total_amount FROM customers c "
            "JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total_amount DESC LIMIT 5;"
        ),
    },
    {
        "input": "A ball is thrown upward at 20 m/s. Max height and time to return? (g=10 m/s²)",
        "output": (
            "<think>\nUsing v = v0 - gt at max height v=0:\n"
            "0 = 20 - 10t => t = 2s\n"
            "Max height: h = v0*t - 0.5*g*t^2 = 20*2 - 0.5*10*4 = 40 - 20 = 20m\n"
            "Total time = 2 * 2 = 4s\n</think>\n\n"
            "The ball reaches a maximum height of **20 meters** at t = 2 seconds, "
            "and returns to the ground at **t = 4 seconds**."
        ),
        "label": "Maximum height: 20 meters at t=2s. Returns to ground at t=4s.",
    },
]

JUDGE_TEMPLATE = (
    "### Task Description\n"
    "Please act as an impartial judge and evaluate the quality of the answer provided by an\n"
    "AI assistant to the conversation history leading up to the answer displayed below.\n"
    "Judge whether the provided answer is good by comparing it to the reference answer.\n\n"
    "Notes:\n"
    "- Besides comparing to the reference answer, your evaluation should consider factors such as "
    "the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well "
    "the response satisfies the user's explicit constraints or accurately follows their instructions.\n"
    "- Note that sometimes the reference answer is not the only answer. So any valid variation of "
    "the reference answer is also acceptable and can get a full score.\n"
    "- If there is a system prompt, ensure the AI answer prioritizes following it.\n"
    "- Begin your evaluation by providing a short explanation.\n"
    "- Be as objective as possible. After providing your short explanation, please output a score "
    "on a scale of 1 to 10.\n"
    "- Please adhere to the following format.\n\n"
    "[Conversation History]\n{input}\n\n"
    "[AI Answer]\n{output}\n\n"
    "[Reference Gold Answer]\n{label}\n\n"
    "[Your judgement]\n"
    'Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}'
)


def make_prompts():
    prompts = []
    for i in range(NUM_REQUESTS):
        ex = JUDGE_PROMPTS[i % len(JUDGE_PROMPTS)]
        prompts.append(JUDGE_TEMPLATE.format(**ex))
    return prompts


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

    cleaned = cleaned.replace("\r\n", "\n").replace("\n", "\\n")
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned)
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


async def send_request(sem, prompt, idx):
    async with sem:
        t0 = time.monotonic()
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
                seed=42,
                timeout=300,
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
                "content_preview": content[:500],
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
                "content_preview": "",
            }


async def main():
    prompts = make_prompts()
    sem = asyncio.Semaphore(CONCURRENCY)

    print(f"\nSending {len(prompts)} requests (concurrency={CONCURRENCY}) ...")
    t_start = time.monotonic()
    tasks = [send_request(sem, p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    t_total = time.monotonic() - t_start

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
    print(f"  Success:  {len(successes)}/{len(results)} ({100*len(successes)/len(results):.1f}%)")
    print(f"  Failed:   {len(failures)}/{len(results)} ({100*len(failures)/len(results):.1f}%)")
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

    print(f"\n{'── Sample Outputs (first 5) ──':-<60}")
    for r in results[:5]:
        status = "OK" if r["ok"] else "FAIL"
        print(
            f"\n  [{status}] Request {r['idx']} | score={r['score']} | "
            f"finish={r['finish_reason']} | {r['latency']:.1f}s | "
            f"tokens={r['completion_tokens']} | reason={r['reason']}"
        )
        preview = r["content_preview"].replace("\n", "\\n")[:300]
        print(f"  >>> {preview}")

    if failures:
        print(f"\n{'── Sample Failures (up to 5) ──':-<60}")
        for r in failures[:5]:
            print(
                f"\n  Request {r['idx']} | finish={r['finish_reason']} | "
                f"{r['latency']:.1f}s | tokens={r['completion_tokens']} | "
                f"reason={r['reason']}"
            )
            preview = r["content_preview"].replace("\n", "\\n")[:300]
            print(f"  >>> {preview}")

    print("\n" + "=" * 70)
    verdict = "PASS" if len(successes) / len(results) >= 0.95 else "FAIL"
    print(f"  VERDICT: {verdict}  (target >= 95% parse success)")
    print("=" * 70)


asyncio.run(main())
