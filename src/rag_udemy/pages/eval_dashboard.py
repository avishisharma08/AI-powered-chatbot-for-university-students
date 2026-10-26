import os
import streamlit as st
import pandas as pd


def find_eval_dir() -> str:
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval"),
        os.path.join(os.getcwd(), "eval"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "results.csv")):
            return os.path.abspath(c)
    return os.path.abspath("eval")


def compute_pass_status(row: pd.Series) -> bool:
    refusal_ok = str(row.get("refusal_correct", "")).strip().lower() == "true"
    source_ok = str(row.get("source_correct", "")).strip().lower() == "true"
    cat = str(row.get("category", "")).strip().lower()
    exp = str(row.get("expected", "")).strip()

    if cat == "answerable" and exp and exp.upper() != "VERIFY":
        ans_ok = str(row.get("expected_in_answer", "")).strip().lower() == "true"
        return refusal_ok and source_ok and ans_ok
    return refusal_ok and source_ok


def render_dashboard():
    st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")

    st.title("📊 RAG Evaluation Dashboard")
    st.caption("Live evaluation metrics, category pass rates, failure breakdown, and verifier comparison")

    eval_dir = find_eval_dir()
    results_path = os.path.join(eval_dir, "results.csv")
    comparison_path = os.path.join(eval_dir, "comparison.csv")

    if not os.path.exists(results_path) or not os.path.exists(comparison_path):
        st.error(f"Evaluation files not found in `{eval_dir}`. Please run `uv run python eval/run_eval.py` first.")
        return

    df_results = pd.read_csv(results_path)
    df_comp = pd.read_csv(comparison_path)

    df_results["passed"] = df_results.apply(compute_pass_status, axis=1)

    # ---------------- Top Level Metric Cards ----------------
    total_q = len(df_results)
    total_passed = int(df_results["passed"].sum())
    pass_rate = (total_passed / total_q * 100) if total_q else 0

    avg_lat_off = df_comp["latency_off_sec"].mean() if "latency_off_sec" in df_comp else 0
    avg_lat_on = df_comp["latency_on_sec"].mean() if "latency_on_sec" in df_comp else 0
    lat_delta = avg_lat_on - avg_lat_off
    extra_calls = int(df_comp["extra_api_calls"].sum()) if "extra_api_calls" in df_comp else 0
    unsupported_caught = int(df_comp["unsupported_sentences_caught"].sum()) if "unsupported_sentences_caught" in df_comp else 0
    wrongly_flagged_count = int(df_comp["wrongly_flagged"].astype(str).str.lower().eq("true").sum()) if "wrongly_flagged" in df_comp else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Pass Rate", f"{pass_rate:.1f}%", f"{total_passed}/{total_q} Passed")
    c2.metric("Avg Latency (Off / On)", f"{avg_lat_off:.2f}s → {avg_lat_on:.2f}s", f"+{lat_delta:.2f}s overhead", delta_color="inverse")
    c3.metric("Extra Verifier API Calls", f"{extra_calls}", f"{total_q} questions")
    c4.metric("Unsupported Sentences Caught", f"{unsupported_caught}", f"False Positives: {wrongly_flagged_count}")

    st.divider()

    # ---------------- Section 1: Pass Rate Per Category ----------------
    st.subheader("1. 📈 Pass Rate Per Category")
    cat_summary = []
    for cat, group in df_results.groupby("category"):
        total = len(group)
        passed = int(group["passed"].sum())
        failed = total - passed
        rate = (passed / total * 100) if total else 0
        cat_summary.append({
            "Category": cat,
            "Total Questions": total,
            "Passed": passed,
            "Failed": failed,
            "Pass Rate": f"{rate:.1f}%",
        })

    cat_df = pd.DataFrame(cat_summary)
    st.dataframe(cat_df, use_container_width=True, hide_index=True)

    # ---------------- Section 2: Table of Failed Questions ----------------
    st.subheader("2. ⚠️ Failed Questions")
    failed_df = df_results[~df_results["passed"]].copy()

    if failed_df.empty:
        st.success("🎉 All test questions passed their criteria!")
    else:
        st.caption(f"Found {len(failed_df)} question(s) that did not meet all pass criteria:")
        display_cols = ["id", "question", "category", "expected", "refusal_correct", "source_correct", "expected_in_answer"]
        st.dataframe(
            failed_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("🔍 View Detailed Failure Diagnostics"):
            for _, r in failed_df.iterrows():
                st.markdown(f"**[{r['id']}] {r['question']}** (`{r['category']}`)")
                st.markdown(f"- **Expected:** `{r.get('expected', '')}`")
                st.markdown(f"- **Answer:** {r.get('answer', '')}")
                reasons = []
                if not str(r.get("refusal_correct", "")).lower() == "true":
                    reasons.append("Refusal behavior did not match expected.")
                if not str(r.get("source_correct", "")).lower() == "true":
                    reasons.append("Retrieved source does not match expected source.")
                if not str(r.get("expected_in_answer", "")).lower() == "true":
                    reasons.append("Expected phrase did not appear verbatim in answer.")
                st.warning("Failure Reason: " + " | ".join(reasons))
                st.divider()

    # ---------------- Section 3: With / Without Verifier Comparison ----------------
    st.subheader("3. ⚖️ With / Without Verifier Comparison")
    st.caption("Compares latency, extra API calls, and validation results across both configurations.")

    comp_cols = [
        "id",
        "question",
        "category",
        "latency_off_sec",
        "latency_on_sec",
        "extra_api_calls",
        "total_sentences",
        "supported_sentences",
        "unsupported_sentences_caught",
        "wrongly_flagged",
    ]
    available_cols = [c for c in comp_cols if c in df_comp.columns]
    st.dataframe(df_comp[available_cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_dashboard()
