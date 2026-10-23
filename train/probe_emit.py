#!/usr/bin/env python3
"""A reasoning model will not put the verdict on token 1 unless we make it.

Try the three ways of forcing the decision position and keep whichever gives a
single-token, well-formed answer.
"""
import json, pathlib, importlib.util
import tinker
from tinker import types

ROOT = pathlib.Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("lm", ROOT / "train" / "lm.py")
lm = importlib.util.module_from_spec(spec); spec.loader.exec_module(lm)
rows = [json.loads(l) for l in (ROOT / "data" / "train.jsonl").open()][:2]

sc = tinker.ServiceClient()
cli = sc.create_sampling_client(base_model=lm.BASE_MODEL)
tok = cli.get_tokenizer()

def show(tag, ids, n=6):
    out = cli.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1,
                     sampling_params=types.SamplingParams(max_tokens=n, temperature=0.0)).result()
    toks = list(out.sequences[0].tokens)
    print(f"  {tag:<34} next={tok.decode(toks)!r}  ids={toks[:3]}")

for r in rows:
    print(f"\n{r['id']}")
    # (a) plain generation prompt
    t = tok.apply_chat_template(lm.messages(r), tokenize=False, add_generation_prompt=True)
    show("plain add_generation_prompt", tok(t)["input_ids"])

    # (b) manual decision prefix
    t2 = t + "Decision:"
    show('assistant prefix "Decision:"', tok(t2)["input_ids"])

    # (c) template kwarg to disable thinking, if this template supports it
    try:
        t3 = tok.apply_chat_template(lm.messages(r), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
        show("enable_thinking=False", tok(t3)["input_ids"])
    except Exception as e:
        print(f"  enable_thinking=False              unsupported: {type(e).__name__}: {str(e)[:80]}")
