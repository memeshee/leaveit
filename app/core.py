"""Leave It — the deterministic layer.

The model is allowed to have an opinion. It is not allowed to act.
Everything that can be undone by a bad prediction is decided here, in plain code:

  1. nothing is ever deleted, hidden forever, or moved,
  2. every NEVER verdict gets a fuse, and the fuse is a durable timer, not a sleep,
  3. a single outside signal (someone engages) unseals an item immediately,
  4. the ledger is append-only, so any verdict can be re-read and argued with later.

The model supplies one number. This file decides what that number is allowed to mean.
"""
from __future__ import annotations

import json, os, time, uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal

Verdict = Literal["NOW", "NEVER"]

# Threshold on p(maintainer will handle it). Below this, the item is sealed.
# Deliberately asymmetric: sealing something that mattered is the expensive mistake,
# so the default buys patience rather than quiet.
DEFAULT_THRESHOLD = 0.35
DEFAULT_FUSE_SECONDS = 14 * 86400        # two weeks to be wrong in

LEDGER = Path(os.environ.get("LEAVEIT_LEDGER", Path(__file__).parent.parent / "data" / "ledger.jsonl"))


@dataclass
class Seal:
    item_id: str
    p_handle: float
    verdict: Verdict
    reason: str
    sealed_at: float
    fuse_seconds: int
    state: str = "SEALED"                # SEALED -> LAPSED | RESCUED
    resolved_at: float | None = None
    resolved_why: str | None = None
    workflow_id: str = ""
    run_id: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def fuse_at(self) -> float:
        return self.sealed_at + self.fuse_seconds

    def to_json(self) -> str:
        return json.dumps(asdict(self) | {"fuse_at": self.fuse_at})


def decide(p_handle: float, threshold: float = DEFAULT_THRESHOLD) -> Verdict:
    """The whole model-facing interface. One number in, one verdict out."""
    if not (0.0 <= p_handle <= 1.0):
        raise ValueError(f"p_handle out of range: {p_handle}")
    return "NEVER" if p_handle < threshold else "NOW"


def reason_line(p_handle: float, verdict: Verdict, top_signal: str = "") -> str:
    """A twelve-word excuse the human can actually argue with. Never a black box."""
    if verdict == "NEVER":
        base = f"P(handled)={p_handle:.2f} — below the line; sealed with a fuse"
    else:
        base = f"P(handled)={p_handle:.2f} — this one has hands on it"
    return f"{base}{'; ' + top_signal if top_signal else ''}"


def append(seal: Seal) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(seal.to_json() + "\n")


def update_state(item_id: str, state: str, why: str) -> dict:
    """Ledger events are appended, never rewritten. State is derived by replay."""
    ev = {"event": "resolve", "item_id": item_id, "state": state,
          "why": why, "at": time.time(), "nonce": uuid.uuid4().hex[:8]}
    with LEDGER.open("a") as f:
        f.write(json.dumps(ev) + "\n")
    return ev


def replay() -> dict[str, dict]:
    """Fold the append-only ledger into current state. This is the audit path."""
    state: dict[str, dict] = {}
    if not LEDGER.exists():
        return state
    for line in LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "resolve":
            if rec["item_id"] in state:
                state[rec["item_id"]].update({"state": rec["state"], "why": rec["why"],
                                              "resolved_at": rec["at"]})
        else:
            state[rec["item_id"]] = {"state": rec.get("state", "SEALED"),
                                     "verdict": rec.get("verdict"), "p_handle": rec.get("p_handle"),
                                     "reason": rec.get("reason"), "fuse_at": rec.get("fuse_at"),
                                     "why": None, "resolved_at": None}
    return state


def morning_card(state: dict[str, dict]) -> list[dict]:
    """The product promise: you get ONE thing, not a list."""
    lapsed = [v | {"item_id": k} for k, v in state.items() if v["state"] == "LAPSED"]
    lapsed.sort(key=lambda r: r["resolved_at"] or 0)
    if not lapsed:
        return [{"kind": "quiet", "text": "Nothing came back. The fuses did their job."}]
    top = lapsed[-1]
    return [{"kind": "one", "item_id": top["item_id"],
             "text": f"One thing survived the fuse: {top['item_id']} "
                     f"(sealed at P={top['p_handle']:.2f}, nobody touched it for "
                     f"{int((top['resolved_at'] or 0) - (top['fuse_at'] or 0)) }s past the fuse)"}]
