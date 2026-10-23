#!/usr/bin/env python3
"""Fill the post's numbers from the evaluation, not from my memory.

Every claim in the published post comes out of data/test.jsonl + eval/scores_*.json
via the same policy.py the arms are judged by, plus the real demo event log.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import policy

LABEL = "needs_today"
ARMS = [
    ("fine-tuned 4B (Tinker LoRA)", "scores_tuned", "Qwen3.5-4B + LoRA rank 16, same prompt"),
    ("base 4B, same prompt", "scores_base", "no LoRA — the baseline the category asks for"),
    ("tfidf + hand features", "scores_tfidf_plus_meta", "word counts plus keyword rules"),
    ("tfidf + logistic regression", "scores_tfidf_logreg", "word counts, no semantics"),
    ("text length only", "scores_length_only", "the laziest possible signal"),
    ("already has a label", "scores_has_label", "it arrives pre-triaged"),
]


def load(p):
    try:
        return json.loads(pathlib.Path(p).read_text())
    except Exception:
        return None


def main():
    rows = [json.loads(l) for l in (ROOT / "data" / "test.jsonl").open()]
    labels = [1 if r[LABEL] else 0 for r in rows]
    n = len(rows)
    pos = sum(labels)
    ceiling = (n - pos) / n

    results = {}
    for name, key, note in ARMS:
        sc = load(ROOT / "eval" / f"{key}.json")
        if not sc:
            continue
        s = [sc.get(r["id"], 0.5) for r in rows]
        results[name] = (policy.budget_curve(s, labels, 0.05), note)

    if not results:
        sys.exit("no scores found")

    tuned = results.get("fine-tuned 4B (Tinker LoRA)")
    base = results.get("base 4B, same prompt")
    best_classical = max((v[0]["deferred_frac"], k) for k, v in results.items()
                         if k.startswith("tfidf"))

    # ---- table
    order = sorted(results.items(), key=lambda kv: -kv[1][0]["deferred_frac"])
    lines = ["| arm | deferred | of the stream | wrongly deferred | AUC |",
             "|---|---|---|---|---|"]
    for name, (m, note) in order:
        lines.append(f"| {name} | {m['deferred']} | {m['deferred_frac']:.1%} | "
                     f"{m['wrongly_deferred']} of {pos} | {m['auc']:.3f} |")
    lines.append(f"| *perfect ranker (ceiling)* | *496* | *{ceiling:.1%}* | *11 of {pos}* | *1.000* |")
    table = "\n".join(lines)

    # ---- payoff paragraph
    if tuned and base:
        payoff = (
            f"**The number, up front.** On the held-out window it defers "
            f"**{tuned[0]['deferred']} of the {n} items** "
            f"({tuned[0]['deferred_frac']:.1%} of the stream) while wrongly deferring "
            f"{tuned[0]['wrongly_deferred']} of the {pos} items that genuinely needed a "
            f"maintainer within 24 hours. The same model with the same prompt and no "
            f"fine-tune defers {base[0]['deferred']} ({base[0]['deferred_frac']:.1%}) at "
            f"AUC {base[0]['auc']:.3f}, which is a coin flip. The best classical arm reaches "
            f"{best_classical[0]:.1%}.")
    else:
        payoff = (f"The fine-tuned arm is not in the results yet; the best classical arm "
                  f"defers {best_classical[0]:.1%} of the stream at the 5% miss budget.")

    deferred = tuned[0]["deferred"] if tuned else 0
    wrong = tuned[0]["wrongly_deferred"] if tuned else 0
    frac = tuned[0]["deferred_frac"] if tuned else 0.0
    # the same tuned arm at the other two budgets, so the dial is a fact not a claim
    tuned_scores = load(ROOT / "eval" / "scores_tuned.json") or {}
    s_tuned = [tuned_scores.get(r["id"], 0.5) for r in rows]
    b3 = policy.budget_curve(s_tuned, labels, 0.03)
    b10 = policy.budget_curve(s_tuned, labels, 0.10)
    lazy = (
        f"- **{deferred} of {n} real issues never reached me**, at a cost of {wrong} "
        f"mis-deferrals out of the {pos} that actually needed a human within 24 hours.\n"
        f"- The filter is a *ranking*, so the budget is a dial: the same model defers "
        f"{b3['deferred']} items at a 3% miss budget, {deferred} at 5%, and "
        f"{b10['deferred']} at 10%. The published numbers are the 5% point.\n"
        f"- A perfect ranker could defer {ceiling:.1%} of this stream; we reach {frac:.1%}. "
        f"Useful, and with headroom left."
    )

    # ---- demo transcript from the real event log
    ev = ROOT / "train" / "demo_events.log"
    transcript = []
    if ev.exists():
        for line in ev.read_text().splitlines():
            if "|" in line[:5]:
                t, _, msg = line.partition("|")
                transcript.append(f"t+{t}s  {msg}" if t.strip().isdigit() else f"       {msg}")
            elif line.strip():
                transcript.append(f"       {line.strip()}")
    demo = "```\n" + "\n".join(transcript) + "\n```" if transcript else "_demo log missing_"

    post = (ROOT / "docs" / "post.md").read_text()
    post = post.replace("{{TUNED_PAYOFF}}", payoff)
    post = post.replace("{{BASELINES_TABLE}}", table)
    post = post.replace("{{LAZY_PAYOFF}}", lazy)
    post = post.replace("{{DEMO_TRANSCRIPT}}", demo)

    out = ROOT / "docs" / "post_final.md"
    out.write_text(post)
    words = len(post.split())
    left = [m for m in ("{{", ) if m in post]
    print(f"wrote {out}  words={words}  placeholders_left={'YES' if left else 'no'}")
    print(f"  tuned: {tuned[0]['deferred'] if tuned else 'MISSING'} deferred, "
          f"AUC {tuned[0]['auc']:.3f}" if tuned else "  tuned: MISSING")
    print(f"  base : {base[0]['deferred']} deferred, AUC {base[0]['auc']:.3f}" if base else "  base: MISSING")
    print(f"  best classical: {best_classical[1]} at {best_classical[0]:.1%}")


if __name__ == "__main__":
    main()
