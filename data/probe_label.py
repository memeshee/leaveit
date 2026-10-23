#!/usr/bin/env python3
"""Which label should the product actually decide?

Two candidate tasks:
  (a) "will a maintainer handle this within 30 days"  -> seal forever
  (b) "does this need a maintainer in the next 24h"   -> the morning filter

(b) is the product we actually want: it decides what makes today's card, not what
gets buried forever. Measure both before committing to one.
"""
import json
from collections import Counter
from datetime import datetime, timezone

raw = [json.loads(l) for l in open("data/issues.jsonl")]
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
NOW = datetime.now(timezone.utc)
M = {"MEMBER", "OWNER", "COLLABORATOR"}


def handled_within(r, hours):
    dl = ts(r["created_at"]).timestamp() + hours * 3600
    for c in r["comments"]:
        if (not c["by_author"]) and c["assoc"] in M and ts(c["at"]).timestamp() <= dl:
            return True
    cl = ts(r["closed_at"])
    return bool(cl) and cl.timestamp() <= dl


rows = []
for r in raw:
    if (NOW - ts(r["created_at"])).days < 2:      # 24h outcome must be observable
        continue
    rows.append({"d": r["created_at"][:10], "a24": handled_within(r, 24),
                 "a30d": handled_within(r, 720)})
rows.sort(key=lambda x: x["d"])

print("usable:", len(rows))
c = Counter(x["a24"] for x in rows)
print(f"  handled within 24h : {c[True]:<5} ({c[True]/len(rows):.1%})")
print(f"  NOT within 24h     : {c[False]:<5} ({c[False]/len(rows):.1%})")
c30 = Counter(x["a30d"] for x in rows)
print(f"  handled within 30d : {c30[True]:<5} ({c30[True]/len(rows):.1%})")
cut = int(len(rows) * 0.8)
print(f"  train: {rows[0]['d']} .. {rows[cut-1]['d']}   test: {rows[cut]['d']} .. {rows[-1]['d']}")
print(f"  ceiling for the morning filter: a perfect ranker defers "
      f"{c[False]}/{len(rows)} = {c[False]/len(rows):.1%} of the stream today")
