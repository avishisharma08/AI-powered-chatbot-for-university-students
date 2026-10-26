import os
import csv
import sys

# Ensure project root and src are on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rag_udemy import ask


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


def run_evaluation():
    questions_path = os.path.join(BASE_DIR, "questions.csv")
    results_path = os.path.join(BASE_DIR, "results.csv")

    if not os.path.exists(questions_path):
        print(f"Error: {questions_path} not found.")
        sys.exit(1)

    with open(questions_path, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    print(f"Loaded {len(rows)} questions from {questions_path}...")

    fieldnames = [
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

    results = []
    category_stats = {}

    for row in rows:
        q_id = row.get("id", "").strip()
        question = row.get("question", "").strip()
        category = row.get("category", "").strip()
        expected = row.get("expected", "").strip()
        expected_source = row.get("source", "").strip()
        should_refuse_str = row.get("should_refuse", "").strip().lower()
        should_refuse = (should_refuse_str == "yes")

        print(f"[{q_id}] Asking: {question}")
        res = ask(question)

        answer = res.get("answer", "")
        chunks = res.get("chunks", [])
        scores = res.get("scores", [])
        sources = res.get("sources", [])

        top_score = round(scores[0], 4) if scores else ""
        refused = "I couldn't find" in answer

        refusal_correct = (refused == should_refuse)
        refusal_with_sources = bool(refused and len(sources) > 0)
        expected_in_ans = check_expected_in_answer(expected, answer, category)
        source_correct = check_source_correct(expected_source, sources)

        # Overall row pass criteria:
        # 1. Refusal behavior must match expected
        # 2. Source retrieval must be correct
        # 3. If answerable with explicit expected text, text must appear in answer
        if category.lower() == "answerable":
            if expected and expected.upper() != "VERIFY":
                overall_pass = refusal_correct and source_correct and expected_in_ans
            else:
                overall_pass = refusal_correct and source_correct
        else:
            overall_pass = refusal_correct and source_correct

        result_row = {
            "id": q_id,
            "question": question,
            "category": category,
            "answer": answer,
            "sources": ", ".join(sources),
            "top_score": top_score,
            "refused": refused,
            "expected": expected,
            "refusal_correct": refusal_correct,
            "refusal_with_sources": refusal_with_sources,
            "expected_in_answer": expected_in_ans,
            "source_correct": source_correct,
        }
        results.append(result_row)

        if category not in category_stats:
            category_stats[category] = {
                "total": 0,
                "refusal_correct": 0,
                "refusal_with_sources": 0,
                "expected_in_answer": 0,
                "source_correct": 0,
                "overall_passed": 0,
            }
        category_stats[category]["total"] += 1
        if refusal_correct:
            category_stats[category]["refusal_correct"] += 1
        if refusal_with_sources:
            category_stats[category]["refusal_with_sources"] += 1
        if expected_in_ans:
            category_stats[category]["expected_in_answer"] += 1
        if source_correct:
            category_stats[category]["source_correct"] += 1
        if overall_pass:
            category_stats[category]["overall_passed"] += 1

    with open(results_path, mode="w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults successfully written to {results_path}\n")

    # Print summary table
    header = f"{'Category':<30} | {'Total':<6} | {'Refusal Pass':<13} | {'Source Pass':<12} | {'Answer Pass':<12} | {'Overall Pass Rate':<17}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    total_all = 0
    refusal_all = 0
    source_all = 0
    answer_all = 0
    overall_all = 0

    for cat, stats in category_stats.items():
        total = stats["total"]
        refusal_p = f"{stats['refusal_correct']}/{total} ({stats['refusal_correct']/total*100:.1f}%)"
        source_p = f"{stats['source_correct']}/{total} ({stats['source_correct']/total*100:.1f}%)"
        answer_p = f"{stats['expected_in_answer']}/{total} ({stats['expected_in_answer']/total*100:.1f}%)"
        overall_p = f"{stats['overall_passed']}/{total} ({stats['overall_passed']/total*100:.1f}%)"

        print(f"{cat:<30} | {total:<6} | {refusal_p:<13} | {source_p:<12} | {answer_p:<12} | {overall_p:<17}")

        total_all += total
        refusal_all += stats["refusal_correct"]
        source_all += stats["source_correct"]
        answer_all += stats["expected_in_answer"]
        overall_all += stats["overall_passed"]

    print(sep)
    tot_refusal_p = f"{refusal_all}/{total_all} ({refusal_all/total_all*100:.1f}%)" if total_all else "N/A"
    tot_source_p = f"{source_all}/{total_all} ({source_all/total_all*100:.1f}%)" if total_all else "N/A"
    tot_answer_p = f"{answer_all}/{total_all} ({answer_all/total_all*100:.1f}%)" if total_all else "N/A"
    tot_overall_p = f"{overall_all}/{total_all} ({overall_all/total_all*100:.1f}%)" if total_all else "N/A"
    print(f"{'Total / Overall':<30} | {total_all:<6} | {tot_refusal_p:<13} | {tot_source_p:<12} | {tot_answer_p:<12} | {tot_overall_p:<17}")
    print(sep)


if __name__ == "__main__":
    run_evaluation()
