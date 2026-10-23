#!/usr/bin/env python3
"""Pick the demo items out of the real held-out set, using a real arm's scores.

The demo replays numbers the post already reports instead of inventing nicer ones.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).parent.parent
arm = sys.argv[1] if len(sys.argv) > 1 else "tuned"
scores = json.loads((ROOT / "eval" / f"scores_{arm}.json").read_text())
rows = [json.loads(l) for l in (ROOT / "data" / "test.jsonl").open()]

# highest p(NOW) -> must be seen today; lowest -> safe to defer
rows_sorted = sorted(rows, key=lambda r: scores.get(r["id"], 0.5))
defer = rows_sorted[:2]                       # safest to defer -> seal
surface = rows_sorted[-1]                     # clearest needs-you -> shown
rescue = rows_sorted[1]                       # gets the attention signal mid-demo

def item(r, key):
    return {"key": key, "item_id": r["id"], "p_handle": round(scores.get(r["id"], 0.5), 4),
            "needs_today": r["needs_today"],
            "title": r["text"].splitlines()[0][:110]}

items = [item(defer[0], "A"), item(rescue, "B"), item(surface, "C")]
(ROOT / "app" / "demo_items.json").write_text(json.dumps(items, indent=2))
for i in items:
    print(f"  {i['key']}: {i['item_id']:<24} p_handle={i['p_handle']:.4f} "
          f"needs_today={i['needs_today']}  {i['title'][:60]}")
print(f"\nwrote app/demo_items.json from arm '{arm}'")
