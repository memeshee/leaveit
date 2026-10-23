#!/usr/bin/env python3
"""Build the labelled dataset.

The product decides ONE thing each morning: does this need a maintainer today?

  needs_today = a maintainer (MEMBER/OWNER/COLLABORATOR) commented, or the item
                was closed, within 24 hours of arrival.

'Did anyone reply' is the wrong label: in large repos passing users pile on with
"+1 me too", which is engagement without attention.

Split is TEMPORAL — train on older items, test on the newest observable window —
because the tool is always used on items that did not exist when it was trained.
"""
import json, pathlib
from datetime import datetime, timezone
from collections import Counter

HERE = pathlib.Path(__file__).parent
raw = [json.loads(l) for l in (HERE / "issues.jsonl").open()]

def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None

NOW = datetime.now(timezone.utc)
MARGIN_DAYS = 2            # the 24h outcome has to be fully observable
MAINTAINERS = {"MEMBER", "OWNER", "COLLABORATOR"}


def handled_within(r, hours):
    dl = ts(r["created_at"]).timestamp() + hours * 3600
    for c in r["comments"]:
        if (not c["by_author"]) and c["assoc"] in MAINTAINERS and ts(c["at"]).timestamp() <= dl:
            return True
    cl = ts(r["closed_at"])
    return bool(cl) and cl.timestamp() <= dl


rows, dropped = [], 0
for r in raw:
    if (NOW - ts(r["created_at"])).days < MARGIN_DAYS:
        dropped += 1
        continue
    rows.append({
        "id": f'{r["repo"]}#{r["number"]}',
        "created_at": r["created_at"],
        "text": (r["title"] + "\n\n" + r["body"]).strip(),
        "labels": r["labels"],
        "reactions": r["reactions"],
        "comment_count": r["comment_count"],
        "needs_today": handled_within(r, 24),
        "triaged_30d": handled_within(r, 720),
    })

rows.sort(key=lambda x: x["created_at"])
cut = int(len(rows) * 0.8)
train, test = rows[:cut], rows[cut:]

def dist(xs, name):
    c = Counter(x["needs_today"] for x in xs)
    t = len(xs)
    print(f"  {name:<6} n={t:<5} needs_today={c[True]:<5} ({c[True]/t:.1%})  "
          f"can_wait={c[False]:<5} ({c[False]/t:.1%})")

print(f"rows usable: {len(rows)}   too fresh to label: {dropped}")
dist(train, "train"); dist(test, "test")
print(f"  train window: {train[0]['created_at'][:10]} .. {train[-1]['created_at'][:10]}")
print(f"  test  window: {test[0]['created_at'][:10]} .. {test[-1]['created_at'][:10]}")
print("  repos:", dict(Counter(x['id'].split('#')[0] for x in rows)))

(HERE / "train.jsonl").write_text("\n".join(json.dumps(x) for x in train) + "\n")
(HERE / "test.jsonl").write_text("\n".join(json.dumps(x) for x in test) + "\n")
print(f"\n  a perfect filter would defer {sum(1 for x in test if not x['needs_today'])}/{len(test)} "
      f"of tomorrow's stream")
