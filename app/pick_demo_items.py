#!/usr/bin/env python3
"""Pick the demo items out of the real held-out set, using a real arm's scores.

Two things matter here:

* `core.decide()` takes p_handle = P(a maintainer handles this today), so this reads
  `pnow_*.json`. The evaluation's defer score is 1 - p_now and is not the same number.
* the demo shows the product working, so the sealed items are ones the model got
  RIGHT (they genuinely never needed a human). Showing confident false negatives
  would be a rigged demo in the other direction.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).parent.parent
arm = sys.argv[1] if len(sys.argv) > 1 else "tuned"
p_now = json.loads((ROOT / "eval" / f"pnow_{arm}.json").read_text())
rows = [json.loads(l) for l in (ROOT / "data" / "test.jsonl").open()]

# items the model correctly decided could wait, most confident first
correct_defers = sorted((r for r in rows if not r["needs_today"]),
                        key=lambda r: p_now.get(r["id"], 0.5))
sharp_now = sorted((r for r in rows if r["needs_today"]),
                   key=lambda r: -p_now.get(r["id"], 0.5))

picks = [(correct_defers[0], "A"), (correct_defers[1], "B"), (correct_defers[2], "C")]

items = []
for r, key in picks:
    items.append({"key": key, "item_id": r["id"],
                  "p_handle": round(p_now.get(r["id"], 0.5), 4),
                  "needs_today": r["needs_today"],
                  "title": r["text"].splitlines()[0][:110]})
(ROOT / "app" / "demo_items.json").write_text(json.dumps(items, indent=2))

for i in items:
    print(f"  {i['key']}: {i['item_id']:<24} p(handled today)={i['p_handle']:.4f} "
          f"actually needed a human={i['needs_today']}  {i['title'][:52]}")
if sharp_now:
    s = sharp_now[0]
    print(f"  (most confident NOW, for reference: {s['id']} "
          f"p={p_now.get(s['id'], 0.5):.4f}, needed a human={s['needs_today']})")
print(f"\nwrote app/demo_items.json from arm '{arm}'")
