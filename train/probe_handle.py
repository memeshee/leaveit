#!/usr/bin/env python3
"""The saved sampler handle is an opaque session blob, not a tinker:// path.

Find out whether it can be reused before paying for another training run.
"""
import inspect, json, pathlib
import tinker
from tinker import types

st = json.loads((pathlib.Path(__file__).parent / "state.json").read_text())
handle = st["tuned"]
print("handle looks like:", handle[:40], "...")
print("is tinker:// path:", handle.startswith("tinker://"))

print("\nSamplingClient.from_sampler_handle:", inspect.signature(tinker.SamplingClient.from_sampler_handle))
sc = tinker.ServiceClient()
print("ServiceClient methods:", [m for m in dir(sc) if "sampl" in m.lower() or "hold" in m.lower() or "save" in m.lower()])

try:
    cli = tinker.SamplingClient.from_sampler_handle(sc.holder, handle)
    tok = cli.get_tokenizer()
    print("REUSED OK; tokenizer:", type(tok).__name__)
    print("token ids:", {w: tok(w)["input_ids"] for w in ("NOW", "WAIT")})
except Exception as e:
    print("from_sampler_handle failed:", type(e).__name__, str(e)[:300])
