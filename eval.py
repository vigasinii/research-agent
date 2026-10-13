"""Evaluation harness for the research agent."""
import json
import sys
from datetime import datetime
from pathlib import Path
from agent import run_agent

CASES_FILE = Path(__file__).parent / "eval_cases.json"
RESULTS_FILE = Path(__file__).parent / "eval_results.json"


def _log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] [EVAL] {msg}")


def run_eval() -> dict:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    results = []

    TP = FP = FN = TN = 0
    grounding_correct = 0

    for case in cases:
        cid = case["id"]
        question = case["question"]
        should_answer = case["should_answer"]
        expected_grounded = case["expected_grounded"]
        expected_contains = case["expected_answer_contains"]

        _log(f"Running {cid} ({case['scenario']}): {question!r}")

        try:
            output = run_agent(question)
        except Exception as exc:
            _log(f"  ERROR: {exc}")
            output = {"answer": "", "sources": [], "grounded": False}

        answer = output["answer"]
        sources = output["sources"]
        grounded = output["grounded"]

        # Grounding accuracy
        grounding_ok = grounded == expected_grounded
        if grounding_ok:
            grounding_correct += 1

        # Precision / recall classification
        # Agent "answered" = grounded is True
        agent_answered = grounded

        if should_answer and agent_answered:
            TP += 1
        elif not should_answer and agent_answered:
            FP += 1
        elif should_answer and not agent_answered:
            FN += 1
        else:
            TN += 1

        # Answer content check
        answer_ok = True
        if should_answer and expected_contains:
            answer_lower = answer.lower()
            answer_ok = all(kw.lower() in answer_lower for kw in expected_contains)

        passed = grounding_ok and answer_ok
        _log(
            f"  → grounded={grounded} expected_grounded={expected_grounded} | "
            f"grounding_ok={grounding_ok} answer_ok={answer_ok} | passed={passed}"
        )

        results.append(
            {
                "id": cid,
                "scenario": case["scenario"],
                "question": question,
                "passed": passed,
                "grounded_correct": grounding_ok,
                "answer_correct": answer_ok,
                "answer": answer[:200] + ("..." if len(answer) > 200 else ""),
                "sources": sources,
            }
        )

    total = len(cases)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    grounding_accuracy = grounding_correct / total
    pass_rate = sum(r["passed"] for r in results) / total

    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)
    print(f"  Total cases  : {total}")
    print(f"  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"  Precision    : {precision:.3f}  ({TP}/{TP+FP})")
    print(f"  Recall       : {recall:.3f}  ({TP}/{TP+FN})")
    print(f"  Grounding acc: {grounding_accuracy:.3f}  ({grounding_correct}/{total})")
    print(f"  Pass rate    : {pass_rate:.3f}  ({sum(r['passed'] for r in results)}/{total})")
    print("=" * 60)

    summary = {
        "total": total,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "grounding_accuracy": round(grounding_accuracy, 3),
        "pass_rate": round(pass_rate, 3),
        "results": results,
    }

    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"Results saved to {RESULTS_FILE}")
    return summary


if __name__ == "__main__":
    run_eval()
