#!/usr/bin/env python3
"""Leave It — the LM arm: fine-tune on Tinker, then score with calibrated probabilities.

  python train/lm.py probe                 # plumbing check, 5 items
  python train/lm.py train                 # LoRA SFT on the train split
  python train/lm.py score base            # score the held-out test split, base model
  python train/lm.py score tuned           # ... and the fine-tuned model
  python train/lm.py score tuned2          # optional second seed / config

Scoring reads p("SEAL") from the logprob of the SEAL token appended after the
assistant prefix, softmaxed against p("SHOW"). One forward pass per item, no
sampling noise, and the number is a real probability rather than a vibe.
"""
import json, math, os, pathlib, sys

import tinker
from tinker import types

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
EVAL = ROOT / "eval"
BASE_MODEL = "Qwen/Qwen3.5-4B"
STATE = ROOT / "train" / "state.json"

SYSTEM = (
    "You are Leave It, the morning filter for an open-source maintainer who is "
    "deliberately trying to read less. Every item you show them costs attention; "
    "every item you hide and get wrong costs trust.\n"
    "Decide whether a maintainer will handle this item within the next 24 hours.\n"
    "Answer NOW if they will (the maintainer must see it today), or WAIT if it "
    "will still be fine tomorrow (it can wait).\n"
    "Reply with exactly one word: NOW or WAIT."
)

def item_prompt(row):
    labels = ", ".join(row["labels"]) if row["labels"] else "(none)"
    return (f"Issue: {row['id']}\nLabels: {labels}\n\n{row['text'][:1600]}\n\n"
            f"Will a maintainer handle this within the next 24 hours?")

def messages(row):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": item_prompt(row)}]

def load(split):
    return [json.loads(l) for l in (DATA / f"{split}.jsonl").open()]

def client_for(arm):
    sc = tinker.ServiceClient()
    if arm == "base":
        return sc.create_sampling_client(base_model=BASE_MODEL)
    st = json.loads(STATE.read_text())
    path = st.get(arm)
    if not path:
        sys.exit(f"no saved weights for arm {arm}; run train first")
    return sc.create_sampling_client(base_model=BASE_MODEL, model_path=path)

def prefix_ids(tok, row):
    text = tok.apply_chat_template(messages(row), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text)["input_ids"]
    assert len(ids) > 16, f"suspiciously short prompt: {len(ids)}"
    return ids

def token_ids(tok):
    out = {}
    for w in ("NOW", "WAIT"):
        for cand in (w, " " + w):
            ids = tok(cand)["input_ids"]
            if len(ids) == 1:
                out[w] = ids[0]
                break
        else:
            sys.exit(f"token {w!r} is not a single token: {tok(w)['input_ids']}")
    return out

def p_now(cli, tok, row, tids):
    ids = prefix_ids(tok, row)
    best = {}
    for w, tid in tids.items():
        seq = types.ModelInput.from_ints(ids + [tid])
        lp = cli.compute_logprobs(prompt=seq, prompt_logprobs_last_n=1).result()
        best[w] = lp[-1]
        if best[w] is None:
            sys.exit("null logprob; prompt may be truncated")
    m = max(best.values())
    e = {w: math.exp(v - m) for w, v in best.items()}
    return e["NOW"] / (e["NOW"] + e["WAIT"]), best


async def p_now_async(cli, tok, row, tids, sem):
    """Both logprobs in flight at once, behind a concurrency gate."""
    import asyncio
    ids = prefix_ids(tok, row)
    async with sem:
        lps = await asyncio.gather(*[
            cli.compute_logprobs_async(prompt=types.ModelInput.from_ints(ids + [tid]),
                                       prompt_logprobs_last_n=1)
            for tid in tids.values()])
    best = dict(zip(tids.keys(), (lp[-1] for lp in lps)))
    if any(v is None for v in best.values()):
        raise RuntimeError(f"null logprob for {row['id']}")
    m = max(best.values())
    e = {w: math.exp(v - m) for w, v in best.items()}
    return e["NOW"] / (e["NOW"] + e["WAIT"])

# ---------------------------------------------------------------- training
def cmd_train(arm="tuned", steps=400, lr=1e-4, rank=16, bs=8, limit=None):
    rows = load("train")
    if limit:
        rows = rows[:limit]
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=rank,
                                        user_metadata={"project": "leaveit", "task": "now-or-later"})
    tok = tc.get_tokenizer()
    tids = token_ids(tok)

    data = []
    for r in rows:
        ids = prefix_ids(tok, r)
        target = "NOW" if r["needs_today"] else "WAIT"
        full = ids + [tids[target]]
        w = [0.0] * len(ids) + [1.0]
        data.append((full[:-1], full[1:], w[1:]))   # weights align with y, not x
    n_now = sum(1 for r in rows if r["needs_today"])
    print(f"training rows={len(data)} target NOW={n_now} WAIT={len(rows)-n_now} bs={bs}", flush=True)

    import math, random
    rng = random.Random(0)
    seq = list(range(len(data)))
    steps_per_epoch = math.ceil(len(data) / bs)
    losses = []
    for step in range(steps):
        if step % steps_per_epoch == 0:
            rng.shuffle(seq)
        idxs = [seq[(step * bs + k) % len(seq)] for k in range(bs)]
        batch, n_sup = [], 0
        for i in idxs:
            x, y, w = data[i]
            batch.append(types.Datum(model_input=types.ModelInput.from_ints(x),
                                     loss_fn_inputs={"target_tokens": types.TensorData(data=y, dtype="int64"),
                                                     "weights": types.TensorData(data=w, dtype="float32")}))
            n_sup += sum(w)
        fb = tc.forward_backward(batch, loss_fn="cross_entropy").result()
        tc.optim_step(adam_params=types.AdamParams(learning_rate=lr)).result()
        raw = fb.metrics.get("loss:sum", float("nan"))
        losses.append(raw / n_sup if n_sup else raw)      # loss:sum is already over weighted tokens
        if step % 20 == 0 or step == steps - 1:
            tail = losses[-20:]
            print(f"  step {step:>3}  loss/supervised-token={sum(tail)/len(tail):.4f}", flush=True)

    cli = tc.save_weights_and_get_sampling_client()
    handle = cli.get_sampler_handle()
    print("sampler handle type:", type(handle).__name__, flush=True)
    if not isinstance(handle, str):
        print("  handle repr:", repr(handle)[:300], flush=True)
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    st[arm] = handle if isinstance(handle, str) else getattr(handle, "path", None) or repr(handle)
    st[f"{arm}_meta"] = {"steps": steps, "lr": lr, "rank": rank, "rows": len(rows),
                         "final_loss": sum(losses[-20:]) / max(len(losses[-20:]), 1)}
    STATE.write_text(json.dumps(st, indent=2))
    print("saved handle for arm", arm, "->", STATE)
    return cli

# ---------------------------------------------------------------- scoring
def cmd_score(arm, limit=None):
    import asyncio
    rows = load("test")
    if limit:
        rows = rows[:limit]
    cli = client_for(arm)
    tok = cli.get_tokenizer()
    tids = token_ids(tok)
    concurrency = int(os.environ.get("LEAVEIT_CONCURRENCY", "12"))
    done = 0

    async def main():
        nonlocal done
        sem = asyncio.Semaphore(concurrency)
        out = {}

        async def one(r):
            nonlocal done
            try:
                out[r["id"]] = await p_now_async(cli, tok, r, tids, sem)
            except Exception as e:                      # one bad row must not sink the arm
                out[r["id"]] = 0.5
                print(f"  !! {r['id']}: {e}", flush=True)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(rows)}", flush=True)

        await asyncio.gather(*[one(r) for r in rows])
        return out

    scores = asyncio.run(main())
    path = EVAL / f"scores_{arm}.json"
    path.write_text(json.dumps(scores))
    print(f"wrote {path} ({len(scores)} items)")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "probe":
        cli = client_for("base"); tok = cli.get_tokenizer()
        print("token ids:", token_ids(tok))
        for r in load("test")[:3]:
            p, lp = p_now(cli, tok, r, token_ids(tok))
            print(f"  {r['id']:<28} p_now={p:.3f}  triaged={r['triaged_30d']}  lp={ {k: round(v,2) for k,v in lp.items()} }")
    elif cmd == "train":
        arm = sys.argv[2] if len(sys.argv) > 2 else "tuned"
        kw = {}
        if len(sys.argv) > 3:
            kw = dict(steps=int(sys.argv[3]))
        cmd_train(arm, **kw)
    elif cmd == "score":
        cmd_score(sys.argv[2])
    else:
        sys.exit(__doc__)
