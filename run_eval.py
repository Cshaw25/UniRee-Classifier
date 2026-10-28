"""
Step 4: evaluate the classifier on the labeled dataset.

Runs every report in a split through the model, compares the model's answer
to your labels, prints a scorecard, and saves every result to a file so you
can re-score or compare runs later without paying for the calls again.

Usage:
    python run_eval.py                          # gemini, tune split
    python run_eval.py --limit 10               # quick check on 10 reports
    python run_eval.py --split test             # ONLY at the very end
    python run_eval.py --rescore results/gemini_tune_20261022-1900.jsonl
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from classifier import ReportInput, classify

FAULTS = ["OWNER", "CUSTOMER", "NEITHER", "UNCLEAR"]
PRIORITIES = ["LOW", "MEDIUM", "HIGH", "URGENT"]


class ExpectedLabel(BaseModel):
    """The three fields you labeled. Confidence and rationale aren't labeled."""
    suggested_fault: Literal["OWNER", "CUSTOMER", "NEITHER", "UNCLEAR"]
    hidden_safety_concern: bool
    priority: Literal["LOW", "MEDIUM", "HIGH", "URGENT"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line"] = line_number
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------
def run_one(row: dict, provider: str, retries: int = 3) -> dict:
    """Classify one report. Never raises: failures are recorded, not crashed on."""
    report = ReportInput.model_validate(row["input"])
    expected = ExpectedLabel.model_validate(row["expected"])
    result = {
        "report_id": report.report_id,
        "line": row["_line"],
        "tags": row.get("tags", []),
        "expected": expected.model_dump(),
        "predicted": None,
        "error": None,
        "latency_ms": None,
    }
    for attempt in range(1, retries + 1):
        start = time.perf_counter()
        try:
            output = classify(report, provider=provider)
            result["latency_ms"] = round((time.perf_counter() - start) * 1000)
            result["predicted"] = output.model_dump()
            result["error"] = None
            return result
        except Exception as e:  # API error, rate limit, or invalid model output
            result["error"] = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(2 ** attempt)  # back off: 2s, 4s, ...
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def confusion_matrix(results: list[dict], field: str, labels: list[str]) -> None:
    counts = Counter((r["expected"][field], r["predicted"][field]) for r in results)
    width = max(len(x) for x in labels) + 2
    print(f"\n  {field} confusion (rows = expected, columns = predicted)")
    print("  " + " " * width + "".join(x[:8].rjust(10) for x in labels))
    for exp in labels:
        row = "".join(str(counts[(exp, pred)]).rjust(10) for pred in labels)
        print("  " + exp.ljust(width) + row)


def score(results: list[dict]) -> None:
    ok = [r for r in results if r["predicted"]]
    failed = [r for r in results if not r["predicted"]]

    print("\n" + "=" * 70)
    print(f"Reports: {len(results)}   Succeeded: {len(ok)}   Failed: {len(failed)}")
    for r in failed:
        print(f"  FAILED report {r['report_id']} (line {r['line']}): {r['error']}")
    if not ok:
        return

    # --- Per-field accuracy ---
    print("\nAccuracy")
    for field in ("suggested_fault", "hidden_safety_concern", "priority"):
        hits = sum(r["expected"][field] == r["predicted"][field] for r in ok)
        print(f"  {field:<24}{pct(hits, len(ok))}")
    all3 = sum(all(r["expected"][f] == r["predicted"][f]
                   for f in ("suggested_fault", "hidden_safety_concern", "priority")) for r in ok)
    print(f"  {'all three correct':<24}{pct(all3, len(ok))}")

    # --- Safety: the metric that matters most ---
    tp = sum(r["expected"]["hidden_safety_concern"] and r["predicted"]["hidden_safety_concern"] for r in ok)
    fn = sum(r["expected"]["hidden_safety_concern"] and not r["predicted"]["hidden_safety_concern"] for r in ok)
    fp = sum(not r["expected"]["hidden_safety_concern"] and r["predicted"]["hidden_safety_concern"] for r in ok)
    print("\nSafety")
    print(f"  recall (caught / real safety cases)   {pct(tp, tp + fn)}   <- most important")
    print(f"  precision (real / flagged)            {pct(tp, tp + fp)}")
    print(f"  missed safety cases: {fn}   false alarms: {fp}")

    # Priority: being too LOW on a serious case is worse than too high.
    rank = {p: i for i, p in enumerate(PRIORITIES)}
    under = [r for r in ok if rank[r["predicted"]["priority"]] < rank[r["expected"]["priority"]]]
    over = [r for r in ok if rank[r["predicted"]["priority"]] > rank[r["expected"]["priority"]]]
    print(f"\nPriority: under-prioritized {len(under)}, over-prioritized {len(over)}")

    confusion_matrix(ok, "suggested_fault", FAULTS)
    confusion_matrix(ok, "priority", PRIORITIES)

    # --- Is confidence meaningful? ---
    print("\nFault accuracy by model confidence")
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    for lo, hi in buckets:
        b = [r for r in ok if lo <= r["predicted"]["confidence"] < hi]
        hits = sum(r["expected"]["suggested_fault"] == r["predicted"]["suggested_fault"] for r in b)
        print(f"  {lo:.2f}-{min(hi, 1.0):.2f}   {pct(hits, len(b))}")

    # --- Weak spots by tag ---
    by_tag = defaultdict(list)
    for r in ok:
        for t in r["tags"]:
            by_tag[t].append(r)
    print("\nWeakest tags (any field wrong, tags with 3+ reports)")
    tag_rows = []
    for tag, rs in by_tag.items():
        if len(rs) < 3:
            continue
        wrong = sum(any(r["expected"][f] != r["predicted"][f]
                        for f in ("suggested_fault", "hidden_safety_concern", "priority")) for r in rs)
        tag_rows.append((wrong / len(rs), tag, wrong, len(rs)))
    for frac, tag, wrong, n in sorted(tag_rows, reverse=True)[:10]:
        print(f"  {tag:<28}{wrong}/{n} wrong")

    # --- Latency ---
    lat = sorted(r["latency_ms"] for r in ok if r["latency_ms"] is not None)
    if lat:
        print(f"\nLatency: median {lat[len(lat) // 2]} ms, max {lat[-1]} ms")

    # --- Every miss, so you can read the rationale ---
    print("\nMisses")
    for r in ok:
        wrong = [f for f in ("suggested_fault", "hidden_safety_concern", "priority")
                 if r["expected"][f] != r["predicted"][f]]
        if not wrong:
            continue
        diffs = ", ".join(f"{f}: expected {r['expected'][f]}, got {r['predicted'][f]}" for f in wrong)
        print(f"\n  Report {r['report_id']} (line {r['line']}) [{', '.join(r['tags'])}]")
        print(f"    {diffs}")
        print(f"    confidence {r['predicted']['confidence']}: {r['predicted']['rationale']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="gemini", choices=["gemini", "claude"])
    parser.add_argument("--split", default="tune", choices=["tune", "test"])
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--limit", type=int, help="only run the first N reports")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between calls")
    parser.add_argument("--rescore", help="score a saved results file instead of calling the model")
    args = parser.parse_args()

    if args.rescore:
        with open(args.rescore, encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        score(results)
        return

    rows = [r for r in load_dataset(args.dataset) if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]

    if args.split == "test":
        answer = input("You're about to run the TEST split. Is your prompt final? (yes/no) ")
        if answer.strip().lower() != "yes":
            return

    os.makedirs("results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = f"results/{args.provider}_{args.split}_{stamp}.jsonl"
    print(f"Running {len(rows)} reports with {args.provider}. Saving to {out_path}")

    results = []
    with open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows, start=1):
            result = run_one(row, args.provider)
            results.append(result)
            out.write(json.dumps(result) + "\n")  # saved as we go, so a crash loses nothing
            out.flush()
            mark = "ok" if result["predicted"] else "FAILED"
            print(f"  [{i}/{len(rows)}] report {result['report_id']}: {mark}")
            time.sleep(args.delay)

    score(results)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()