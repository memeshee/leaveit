"""Start the seals. The worker may or may not be alive for any of it.

  python seal.py start    # open three seals with real fuses
  python seal.py poke B   # deliver an 'attention' signal to a sealed item
  python seal.py card     # print the morning card + ledger state

Deliberately dumb: no model calls here. The p_handle values are already recorded
from the held-out evaluation (eval/scores_*.json), so the demo replays the same
numbers the post reports instead of inventing new ones.
"""
import asyncio, json, sys, pathlib, time

from temporalio.client import Client

import workflows
from workflows import SealRequest

QUEUE = "leaveit"
STATE = pathlib.Path(__file__).parent.parent / "train" / "state.json"
DEMO_ITEMS = json.loads((pathlib.Path(__file__).parent / "demo_items.json").read_text())


async def start(fuses: dict[str, int]):
    client = await Client.connect("localhost:7233")
    out = []
    for item in DEMO_ITEMS:
        req = SealRequest(item_id=item["item_id"], p_handle=item["p_handle"],
                          fuse_seconds=fuses.get(item["key"], 30))
        wid = f"seal-{item['key']}-{int(time.time())}"
        h = await client.start_workflow(workflows.SealWorkflow.run, req, id=wid,
                                        task_queue=QUEUE)
        out.append({"key": item["key"], "item_id": item["item_id"], "workflow_id": wid,
                    "fuse_seconds": req.fuse_seconds, "p_handle": req.p_handle})
        print(f"  sealed {item['key']:<2} {item['item_id']:<26} p={req.p_handle:.2f} "
              f"fuse={req.fuse_seconds}s  wf={wid}", flush=True)
    pathlib.Path(pathlib.Path(__file__).parent / "demo_run.json").write_text(json.dumps(out, indent=2))


async def poke(key: str, why: str):
    client = await Client.connect("localhost:7233")
    runs = json.loads((pathlib.Path(__file__).parent / "demo_run.json").read_text())
    wid = next(r["workflow_id"] for r in runs if r["key"] == key)
    h = client.get_workflow_handle(wid)
    await h.signal(workflows.SealWorkflow.on_attention, why)
    print(f"  signal delivered to {wid} (worker alive: {'no' if '--dead' in sys.argv else 'yes'})", flush=True)


def card():
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import core
    st = core.replay()
    print("\n  LEDGER (append-only, folded):")
    for k, v in sorted(st.items(), key=lambda kv: kv[1].get("fuse_at") or 0):
        print(f"    {k:<28} {v['state']:<9} p={v.get('p_handle') or 0:.2f}  "
              f"{v.get('why') or v.get('reason') or ''}")
    print("\n  MORNING CARD:")
    for c in core.morning_card(st):
        print("   ", c["text"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "start":
        fuses = dict(a.split("=") for a in sys.argv[2:]) if len(sys.argv) > 2 else {}
        asyncio.run(start({k: int(v) for k, v in fuses.items()}))
    elif cmd == "poke":
        asyncio.run(poke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "a maintainer replied"))
    elif cmd == "card":
        card()
