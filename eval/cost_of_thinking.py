#!/usr/bin/env python3
"""What does thinking cost?

The category brief for Tinker is "a clear improvement in performance, latency, OR
cost over a baseline". Accuracy is one axis. This measures the other two honestly:

  base arm, thinking enabled   -> generates a reasoning block before the verdict
  tuned arm, thinking disabled -> emits the verdict as the first token

We count the tokens generated before the verdict appears and the wall-clock time
per decision, on the same items for both arms.
"""
import argparse, json, pathlib, time, importlib.util
import tinker
from tinker import types

ROOT = pathlib.Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("lm", ROOT / "train" / "lm.py")
lm = importlib.util.module_from_spec(spec); spec.loader.exec_module(lm)


def first_verdict_token(tok, toks, tids):
    inv = {v: k for k, v in tids.items()}
    for i, t in enumerate(toks):
        if t in inv:
            return inv[t], i
    return None, None


def run(arm, n, thinking):
    cli = lm.client_for(arm)
    tok = cli.get_tokenizer()
    tids = lm.token_ids(tok)
    rows = lm.load("test")[:n]
    total_tokens, decided, latencies, correct = 0, 0, [], 0
    for r in rows:
        if thinking:
            text = tok.apply_chat_template(lm.messages(r), tokenize=False, add_generation_prompt=True)
        else:
            text = tok.apply_chat_template(lm.messages(r), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        ids = tok(text)["input_ids"]
        t0 = time.time()
        out = cli.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1,
                         sampling_params=types.SamplingParams(max_tokens=700, temperature=0.0)).result()
        lat = time.time() - t0
        toks = list(out.sequences[0].tokens)
        verdict, idx = first_verdict_token(tok, toks, tids)
        total_tokens += len(toks)
        latencies.append(lat)
        if verdict:
            decided += 1
            gold = "NOW" if r["needs_today"] else "WAIT"
            correct += int(verdict == gold)
    latencies.sort()
    return {
        "arm": arm, "thinking": thinking, "items": len(rows),
        "verdict_found": decided,
        "accuracy_when_decided": round(correct / decided, 4) if decided else None,
        "mean_tokens_generated": round(total_tokens / len(rows), 1),
        "median_latency_s": round(latencies[len(latencies) // 2], 3),
        "total_tokens": total_tokens,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="base")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()
    res = run(a.arm, a.n, a.thinking)
    print(json.dumps(res, indent=2))
    tag = f"{a.arm}_{'thinking' if a.thinking else 'direct'}"
    (ROOT / "eval" / f"cost_{tag}.json").write_text(json.dumps(res, indent=2))
