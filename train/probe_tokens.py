#!/usr/bin/env python3
"""The verdict must be one token per class, or the logprob comparison is invalid."""
import sys
import tinker

sc = tinker.ServiceClient()
tc = sc.create_lora_training_client(base_model="Qwen/Qwen3.5-4B", rank=8)
tok = tc.get_tokenizer()
for w in ["NOW", " WAIT", "WAIT", " LATER", "LATER", " HOLD", "HOLD", " SOON", "SOON",
          " DEFER", "HEAD", " READ", " OPEN", " YES", " NO", " TODAY", " NOW"]:
    ids = tok(w)["input_ids"]
    print(f"  {w!r:<10} -> {ids}  {'SINGLE' if len(ids) == 1 else ''}")
