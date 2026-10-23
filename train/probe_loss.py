#!/usr/bin/env python3
"""What does the loss actually look like? Print the raw metrics, do not assume keys."""
import json, pathlib
import tinker
from tinker import types

ROOT = pathlib.Path(__file__).parent.parent
rows = [json.loads(l) for l in (ROOT / "data" / "train.jsonl").open()][:4]

sc = tinker.ServiceClient()
tc = sc.create_lora_training_client(base_model="Qwen/Qwen3.5-4B", rank=16)
tok = tc.get_tokenizer()

import importlib.util
spec = importlib.util.spec_from_file_location("lm", ROOT / "train" / "lm.py")
lm = importlib.util.module_from_spec(spec); spec.loader.exec_module(lm)
tids = lm.token_ids(tok)
print("token ids:", tids)

batch, ws = [], []
for r in rows:
    ids = lm.prefix_ids(tok, r)
    target = "NOW" if r["needs_today"] else "WAIT"
    full = ids + [tids[target]]
    w = [0.0] * len(ids) + [1.0]
    x, y, ww = full[:-1], full[1:], w[1:]   # weights align with y, not x
    print(f"  {r['id']:<26} target={target:<5} len={len(full)} sum_w={sum(ww)} last_target_token={y[-1]}")
    batch.append(types.Datum(model_input=types.ModelInput.from_ints(x),
                             loss_fn_inputs={"target_tokens": types.TensorData(data=y, dtype="int64"),
                                             "weights": types.TensorData(data=ww, dtype="float32")}))
    ws.append(sum(ww))

fb = tc.forward_backward(batch, loss_fn="cross_entropy").result()
print("\nmetrics keys:", list(fb.metrics.keys()))
print("metrics     :", fb.metrics)
print("sum of supervised weights in batch:", sum(ws))
if "loss:sum" in fb.metrics:
    print("loss:sum / supervised tokens =", fb.metrics["loss:sum"] / sum(ws))
