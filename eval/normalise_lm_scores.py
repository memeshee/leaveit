#!/usr/bin/env python3
"""Our LM arms score p(NOW) = 'this needs a human today'.

The policy expects a DEFER score = 'how safe is it to leave this until tomorrow'.
Those are opposites. The baselines were written as defer scores; the LM arms were
not, which made the fine-tune look anti-correlated (AUC 0.335) when it was simply
signed the other way. Keep p(NOW) and derive the defer score explicitly.
"""
import json, pathlib

EVAL = pathlib.Path(__file__).parent


def convert(arm: str) -> None:
    src = EVAL / f"scores_{arm}.json"
    if not src.exists():
        print(f"  {arm}: no scores, skipped")
        return
    p_now = json.loads(src.read_text())
    (EVAL / f"pnow_{arm}.json").write_text(json.dumps(p_now))
    defer = {k: 1.0 - v for k, v in p_now.items()}
    src.write_text(json.dumps(defer))
    print(f"  {arm}: {len(defer)} items -> defer score = 1 - p(NOW); p(NOW) kept in pnow_{arm}.json")


if __name__ == "__main__":
    for arm in ("tuned", "base"):
        convert(arm)
