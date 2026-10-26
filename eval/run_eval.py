import os
import csv
import sys
import time

# Ensure project root and src are on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rag_udemy import ask, verify, config


def check_source_correct(expected_source: str, sources: list[str]) -> bool:
    exp = (expected_source or "").strip().lower()
    if exp in ("", "none"):
        return len(sources) == 0
    return any(exp in s.lower() for s in sources)


def check_expected_in_answer(expected: str, answer: str, category: str) -> bool:
    if category.strip().lower() != "answerable":
        return False
    exp = (expected or "").strip()
    if not exp or exp.upper() == "VERIFY":
        return False
    return exp.lower() in answer.lower()


def safe_ask(question: str, max_retries: int = 3, delay: float = 2.0) -> dict:
    for attempt in range(max_retries):
        try:
            return ask(question)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
    return {}


def safe_verify(answer: str, chunks: list[str], max_retries: int = 3, delay: float = 2.0) -> list[dict]:
    for attempt in range(max_retries):
        try:
            return verify(answer, chunks)
        except Exception as e:
            if attempt == max_retries - 1:
                return []
            time.sleep(delay)
    return []



def run_evaluation():
    questions_path = os.path.join(BASE_DIR, "questions.csv")
    results_path = os.path.join(BASE_DIR, "results.csv")
    comparison_path = os.path.join(BASE_DIR, "comparison.csv")

    if not os.path.exists(questions_path):
        print(f"Error: {questions_path} not found.")
        sys.exit(1)

    with open(questions_path, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    print(f"Loaded {len(rows)} questions from {questions_path}.\nRunning evaluation with USE_VERIFIER off and on...\n")

    results_fieldnames = [
        "id",
        "question",
        "category",
        "answer",
        "sources",
        "top_score",
        "refused",
        "expected",
        "refusal_correct",
        "refusal_with_sources",
        "expected_in_answer",
        "source_correct",
    ]

    comparison_fieldnames = [
        "id",
        "question",
        "category",
        "should_refuse",
        "latency_off_sec",
        "latency_on_sec",
        "extra_api_calls",
        "total_sentences",
        "supported_sentences",
        "unsupported_sentences_caught",
        "wrongly_flagged",
        "unsupported_details",
    ]

    results_rows = []
    comparison_rows = []

    total_latency_off = 0.0
    total_latency_on = 0.0
    total_extra_calls = 0
    total_unsupported_caught = 0
    total_wrongly_flagged = 0

    cat_stats = {}

    for row in rows:
        q_id = row.get("id", "").strip()
        question = row.get("question", "").strip()
        category = row.get("category", "").strip()
        expected = row.get("expected", "").strip()
        expected_source = row.get("source", "").strip()
        should_refuse_str = row.get("should_refuse", "").strip().lower()
        should_refuse = (should_refuse_str == "yes")

        print(f"--- Question [{q_id}]: {question} ---")

        # ---------------- Run 1: USE_VERIFIER OFF ----------------
        config.USE_VERIFIER = False
        t0 = time.time()
        res_off = safe_ask(question)
        latency_off = time.time() - t0
        extra_calls_off = 0

        # Save standard evaluation row from baseline
        ans_off = res_off.get("answer", "")
        chunks_off = res_off.get("chunks", [])
        scores_off = res_off.get("scores", [])
        sources_off = res_off.get("sources", [])
        top_score_off = round(scores_off[0], 4) if scores_off else ""
        refused_off = "I couldn't find" in ans_off
        refusal_correct = (refused_off == should_refuse)
        refusal_with_sources = bool(refused_off and len(sources_off) > 0)
        expected_in_ans = check_expected_in_answer(expected, ans_off, category)
        source_correct = check_source_correct(expected_source, sources_off)

        results_rows.append({
            "id": q_id,
            "question": question,
            "category": category,
            "answer": ans_off,
            "sources": ", ".join(sources_off),
            "top_score": top_score_off,
            "refused": refused_off,
            "expected": expected,
            "refusal_correct": refusal_correct,
            "refusal_with_sources": refusal_with_sources,
            "expected_in_answer": expected_in_ans,
            "source_correct": source_correct,
        })

        # ---------------- Run 2: USE_VERIFIER ON ----------------
        config.USE_VERIFIER = True
        t1 = time.time()
        res_on = safe_ask(question)

        ans_on = res_on.get("answer", "")
        chunks_on = res_on.get("chunks", [])
        sources_on = res_on.get("sources", [])
        refused_on = "I couldn't find" in ans_on

        # Run verifier if there is context and answer wasn't an early refusal
        extra_calls_on = 0
        verifications = []
        if chunks_on and not refused_on:
            verifications = safe_verify(ans_on, chunks_on)
            extra_calls_on = 1
        latency_on = time.time() - t1

        unsupported_items = [
            v for v in verifications if str(v.get("label", "")).strip().upper() == "UNSUPPORTED"
        ]
        supported_items = [
            v for v in verifications if str(v.get("label", "")).strip().upper() == "SUPPORTED"
        ]

        unsupported_count = len(unsupported_items)
        supported_count = len(supported_items)
        total_sentences = len(verifications)

        # A "correct answer wrongly flagged" happens when an answerable, correctly retrieved
        # answer was provided, but the verifier incorrectly marked one or more sentences as unsupported.
        is_correct_answer = (category.lower() == "answerable" and not should_refuse and not refused_on and source_correct)
        wrongly_flagged = bool(is_correct_answer and unsupported_count > 0)

        unsupported_details = "; ".join(
            f"\"{v.get('sentence', '')}\" [Reason: {v.get('reason', '')}]" for v in unsupported_items
        )

        comparison_rows.append({
            "id": q_id,
            "question": question,
            "category": category,
            "should_refuse": row.get("should_refuse", ""),
            "latency_off_sec": round(latency_off, 3),
            "latency_on_sec": round(latency_on, 3),
            "extra_api_calls": extra_calls_on,
            "total_sentences": total_sentences,
            "supported_sentences": supported_count,
            "unsupported_sentences_caught": unsupported_count,
            "wrongly_flagged": wrongly_flagged,
            "unsupported_details": unsupported_details,
        })

        # Aggregation
        total_latency_off += latency_off
        total_latency_on += latency_on
        total_extra_calls += extra_calls_on
        total_unsupported_caught += unsupported_count
        if wrongly_flagged:
            total_wrongly_flagged += 1

        if category not in cat_stats:
            cat_stats[category] = {
                "total": 0,
                "latency_off": 0.0,
                "latency_on": 0.0,
                "extra_calls": 0,
                "unsupported": 0,
                "wrongly_flagged": 0,
            }
        cat_stats[category]["total"] += 1
        cat_stats[category]["latency_off"] += latency_off
        cat_stats[category]["latency_on"] += latency_on
        cat_stats[category]["extra_calls"] += extra_calls_on
        cat_stats[category]["unsupported"] += unsupported_count
        if wrongly_flagged:
            cat_stats[category]["wrongly_flagged"] += 1

        print(f"  OFF: {latency_off:.2f}s (0 extra calls)")
        print(f"  ON:  {latency_on:.2f}s ({extra_calls_on} extra calls) | sentences: {total_sentences}, unsupported: {unsupported_count}, wrongly_flagged: {wrongly_flagged}")

    # Write results.csv
    with open(results_path, mode="w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results_fieldnames)
        w.writeheader()
        w.writerows(results_rows)

    # Write comparison.csv
    with open(comparison_path, mode="w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=comparison_fieldnames)
        w.writeheader()
        w.writerows(comparison_rows)

    print(f"\nFiles successfully written:")
    print(f"  - {results_path}")
    print(f"  - {comparison_path}\n")

    # Summary Report
    num_q = len(rows)
    avg_lat_off = total_latency_off / num_q if num_q else 0
    avg_lat_on = total_latency_on / num_q if num_q else 0
    lat_diff = avg_lat_on - avg_lat_off

    header = f"{'Metric':<38} | {'Verifier OFF':<16} | {'Verifier ON':<16} | {'Delta':<12}"
    sep = "=" * len(header)
    sub_sep = "-" * len(header)

    print(sep)
    print("                    VERIFIER COMPARISON EVALUATION REPORT")
    print(sep)
    print(header)
    print(sub_sep)
    print(f"{'Total Questions Tested':<38} | {num_q:<16} | {num_q:<16} | {'-':<12}")
    print(f"{'Total Extra API Calls':<38} | {0:<16} | {total_extra_calls:<16} | {f'+{total_extra_calls}':<12}")
    print(f"{'Average Latency per Question':<38} | {f'{avg_lat_off:.3f}s':<16} | {f'{avg_lat_on:.3f}s':<16} | {f'+{lat_diff:.3f}s':<12}")
    print(f"{'Unsupported Sentences Caught':<38} | {0:<16} | {total_unsupported_caught:<16} | {f'+{total_unsupported_caught}':<12}")
    print(f"{'Correct Answers Wrongly Flagged (FP)':<38} | {0:<16} | {total_wrongly_flagged:<16} | {f'+{total_wrongly_flagged}':<12}")
    print(sep)

    # Category breakdown
    print("\nCategory Breakdown (Verifier ON):")
    cat_header = f"{'Category':<30} | {'Total':<6} | {'Avg Latency':<12} | {'Extra Calls':<12} | {'Unsupported':<12} | {'Wrongly Flagged':<15}"
    cat_sep = "-" * len(cat_header)
    print(cat_sep)
    print(cat_header)
    print(cat_sep)
    for cat, s in cat_stats.items():
        n = s["total"]
        avg_l = s["latency_on"] / n if n else 0
        print(f"{cat:<30} | {n:<6} | {f'{avg_l:.2f}s':<12} | {s['extra_calls']:<12} | {s['unsupported']:<12} | {s['wrongly_flagged']:<15}")
    print(cat_sep)


if __name__ == "__main__":
    run_evaluation()
