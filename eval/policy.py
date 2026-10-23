#!/usr/bin/env python3
"""The one scorer every arm goes through.

An arm produces, per item, a score in [0,1] = "how safe is it to leave this until
tomorrow?". Every arm is judged by this same policy, or the comparison is noise.

The product's promise is a MISS BUDGET, not accuracy:
  * items are deferred from highest defer-score first,
  * we stop before wrongly deferring more than BUDGET of the items that genuinely
    needed a maintainer today,
  * within that budget, report how much of the stream never reached the human.

Accuracy is deliberately not reported. An arm that says "needs you" to everything
is 45% "accurate" here and saves the user precisely nothing.
"""
import json, pathlib
from typing import Dict, List

HERE = pathlib.Path(__file__).parent
LABEL = "needs_today"
BUDGET = 0.05


def load_test() -> List[dict]:
    return [json.loads(l) for l in (HERE.parent / "data" / "test.jsonl").open()]


def roc_auc(scores: List[float], labels: List[int]) -> float:
    """Rank quality, threshold-free. 0.5 = coin flip, 1.0 = perfect ordering."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(p[1] for p in pairs[i:j + 1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def budget_curve(scores: List[float], labels: List[int], budget: float = BUDGET):
    total_pos = sum(labels)
    # A constant score is not a ranking: it can only defer 0 items legitimately.
    if len(set(scores)) <= 1:
        return {"deferred": 0, "deferred_frac": 0.0, "wrongly_deferred": 0,
                "miss_rate": 0.0, "auc": 0.5}
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    deferred = missed = 0
    for idx in order:
        if labels[idx] == 1:
            if (missed + 1) / total_pos > budget:
                break
            missed += 1
        deferred += 1
    # AUC against "could wait", so higher always means a better arm to read.
    return {"deferred": deferred, "deferred_frac": deferred / len(scores),
            "wrongly_deferred": missed, "miss_rate": missed / total_pos,
            "auc": roc_auc(scores, [1 - l for l in labels])}


def report(arms: Dict[str, Dict[str, float]], rows: List[dict], budgets=(0.03, 0.05, 0.10)):
    labels = [1 if r[LABEL] else 0 for r in rows]
    ids = [r["id"] for r in rows]
    print(f"n={len(rows)}  needed a human today={sum(labels)}  could wait={len(labels)-sum(labels)}")
    print(f"  ceiling: a perfect ranker defers {len(labels)-sum(labels)}/"
          f"{len(labels)} = {(len(labels)-sum(labels))/len(labels):.1%}\n")
    out = {}
    for budget in budgets:
        print(f"  --- miss budget {budget:.0%} of the items that genuinely needed someone ---")
        print(f"  {'arm':<20} {'deferred':>9} {'of pool':>8} {'wrong':>6} {'AUC':>6}")
        for name, sc in arms.items():
            s = [sc.get(i, 0.5) for i in ids]
            m = budget_curve(s, labels, budget)
            out[(name, budget)] = m
            print(f"  {name:<20} {m['deferred']:>9} {m['deferred_frac']:>7.1%} "
                  f"{m['wrongly_deferred']:>6} {m['auc']:>6.3f}")
        print()
    return out


if __name__ == "__main__":
    import sys
    rows = load_test()
    arms = {}
    for f in sys.argv[1:]:
        p = pathlib.Path(f)
        arms[p.stem.replace("scores_", "")] = json.loads(p.read_text())
    arms["oracle (upper bound)"] = {r["id"]: (0.0 if r[LABEL] else 1.0) for r in rows}
    arms["defer everything"] = {r["id"]: 1.0 for r in rows}
    arms["defer nothing"] = {r["id"]: 0.0 for r in rows}
    report(arms, rows)
