#!/usr/bin/env python3
"""Classical baselines. These are the arms the fine-tune has to beat.

If a logistic regression on TF-IDF matches the LoRA, the honest headline is
"the fine-tune bought nothing" — and that gets published too.
"""
import json, pathlib, sys
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"
LABEL = "needs_today"

def load(p):
    return [json.loads(l) for l in p.open()]

train, test = load(DATA / "train.jsonl"), load(DATA / "test.jsonl")
ytr = np.array([1 if r[LABEL] else 0 for r in train])
print(f"train n={len(train)} needs_today={ytr.sum()} ({ytr.mean():.1%})")
print(f"test  n={len(test)} needs_today={sum(1 for r in test if r[LABEL])} "
      f"({sum(1 for r in test if r[LABEL])/len(test):.1%})")

arms = {}

# 1. the laziest possible signal: does it have a maintainer label already?
arms["has_label"] = {r["id"]: float(bool(r["labels"])) for r in test}

# 2. text length — longer reports read as more substantive
lens = np.array([len(r["text"]) for r in test], dtype=float)
arms["length_only"] = {r["id"]: float(1 - l / max(lens.max(), 1)) for r, l in zip(test, lens)}

# 3. TF-IDF + logistic regression
vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2,
                      sublinear_tf=True, stop_words="english")
clf = make_pipeline(TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2,
                                    sublinear_tf=True, stop_words="english"),
                    LogisticRegression(max_iter=3000, class_weight="balanced"))
clf.fit([r["text"] for r in train], ytr)
p = clf.predict_proba([r["text"] for r in test])[:, 1]
arms["tfidf_logreg"] = {r["id"]: float(1 - v) for r, v in zip(test, p)}

# 4. TF-IDF + the cheap structural features a human eyeballs in half a second
def feats(r):
    t = r["text"].lower()
    return [len(r["text"]), len(r["labels"]), float(r["reactions"]),
            float(r["comment_count"]),
            float("stack trace" in t or "traceback" in t or "panic" in t),
            float(any(k in t for k in ("how do i", "how to", "question", "help")),
            ),
            float(c := len([1 for k in ("crash", "regression", "segfault", "security",
                                        "data loss", "use after free", "overflow") if k in t])),
    ]

Xtr_text = clf[:-1].transform([r["text"] for r in train])
Xte_text = clf[:-1].transform([r["text"] for r in test])
sc = StandardScaler().fit(np.array([feats(r) for r in train]))
Xtr = hstack([Xtr_text, csr_matrix(sc.transform(np.array([feats(r) for r in train])))])
Xte = hstack([Xte_text, csr_matrix(sc.transform(np.array([feats(r) for r in test])))])
lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
p2 = lr.predict_proba(Xte)[:, 1]
arms["tfidf_plus_meta"] = {r["id"]: float(1 - v) for r, v in zip(test, p2)}

for name, sc_ in arms.items():
    (HERE / f"scores_{name}.json").write_text(json.dumps(sc_))
    print("wrote", f"scores_{name}.json", len(sc_))

sys.path.insert(0, str(HERE))
from policy import report
report(arms, test)
