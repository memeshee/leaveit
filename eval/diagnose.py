#!/usr/bin/env python3
"""Is the TF-IDF inversion a real finding or a broken label? Diagnose before believing it."""
import json, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

D = Path(__file__).parent.parent / "data"
tr = [json.loads(l) for l in (D / "train.jsonl").open()]
te = [json.loads(l) for l in (D / "test.jsonl").open()]

print("=== sample IGNORED (label False) ===")
for r in [x for x in te if not x["triaged_30d"]][:3]:
    print(f"  {r['id']} comments={r['comment_count']} labels={r['labels']} :: {r['text'][:100]!r}")
print("=== sample TRIAGED (label True) ===")
for r in [x for x in te if x["triaged_30d"]][:3]:
    print(f"  {r['id']} comments={r['comment_count']} labels={r['labels']} :: {r['text'][:100]!r}")

def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    o = np.argsort(s, kind="mergesort")
    ys = y[o]
    pos, neg = ys.sum(), len(ys) - ys.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = np.empty(len(s)); ranks[o] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)

ytr = np.array([1 if r["triaged_30d"] else 0 for r in tr])
yte = np.array([1 if r["triaged_30d"] else 0 for r in te])
mk = lambda: make_pipeline(
    TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"),
    LogisticRegression(max_iter=2000, class_weight="balanced"))

rng = np.random.default_rng(0); allrows = tr + te
idx = rng.permutation(len(allrows)); cut = int(len(allrows) * 0.8)
A = [allrows[i] for i in idx[:cut]]; B = [allrows[i] for i in idx[cut:]]
ya = np.array([1 if r["triaged_30d"] else 0 for r in A]); yb = np.array([1 if r["triaged_30d"] else 0 for r in B])
p = mk().fit([r["text"] for r in A], ya).predict_proba([r["text"] for r in B])[:, 1]
print(f"\nRANDOM split (in-distribution)   AUC(p_triaged) = {auc(p, yb):.3f}   n={len(B)}")

p2 = mk().fit([r["text"] for r in tr], ytr).predict_proba([r["text"] for r in te])[:, 1]
print(f"TEMPORAL split (out-of-sample)   AUC(p_triaged) = {auc(p2, yte):.3f}   n={len(te)}")
print(f"  mean p_triaged | actually triaged : {p2[yte == 1].mean():.3f}")
print(f"  mean p_triaged | actually ignored : {p2[yte == 0].mean():.3f}")
print(f"  label rate train {ytr.mean():.3f}  test {yte.mean():.3f}")
print("  train repos:", dict(Counter(r['id'].split('#')[0] for r in tr)))
print("  test  repos:", dict(Counter(r['id'].split('#')[0] for r in te)))
