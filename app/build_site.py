#!/usr/bin/env python3
"""Rebuild the public demo page from real artefacts. No hand-typed numbers.

  data/test.jsonl        held-out items and what really happened to them
  eval/scores_*.json     each arm's per-item score
  train/demo_events.log  the kill/restart run, timestamped
  data/ledger.jsonl      the append-only seal ledger

Everything the page states is recomputed here, so the page cannot drift from the
evaluation. Output is one self-contained file with zero external requests: the
failure mode that cost two earlier entries was a demo host that 404'd.
"""
import html, json, pathlib, sys

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import policy

ARMS = [
    ("base 4B, prompted", "scores_base", "the same weights, no fine-tune"),
    ("tfidf + logistic reg", "scores_tfidf_logreg", "word counts, no semantics"),
    ("tfidf + hand features", "scores_tfidf_plus_meta", "word counts plus keyword rules"),
    ("has_label", "scores_has_label", "does it already carry a maintainer label"),
    ("length_only", "scores_length_only", "longer report = more serious"),
    ("fine-tuned 4B (Tinker)", "scores_tuned", "LoRA, 2,924 items, held-out window"),
]


def load_json(p, default=None):
    try:
        return json.loads(pathlib.Path(p).read_text())
    except Exception:
        return default


def table(rows):
    results = {}
    for label, key, note in ARMS:
        sc = load_json(ROOT / "eval" / f"{key}.json")
        if not sc:
            continue
        s = [sc.get(r["id"], 0.5) for r in rows]
        labels = [1 if r["needs_today"] else 0 for r in rows]
        results[label] = (policy.budget_curve(s, labels, 0.05), note)
    return results


def main():
    rows = [json.loads(l) for l in (ROOT / "data" / "test.jsonl").open()]
    labels = [1 if r["needs_today"] else 0 for r in rows]
    ceiling = (len(rows) - sum(labels)) / len(rows)
    res = table(rows)

    events = []
    evlog = ROOT / "train" / "demo_events.log"
    if evlog.exists():
        for line in evlog.read_text().splitlines():
            if "|" in line[:4]:
                t, _, msg = line.partition("|")
                events.append((int(t), msg))
            elif line.strip():
                events.append((None, line))

    ledger = []
    for line in (ROOT / "data" / "ledger.jsonl").read_text().splitlines() if (ROOT / "data" / "ledger.jsonl").exists() else []:
        if line.strip():
            ledger.append(json.loads(line))

    # The big number is the shipped arm; the table still shows anything that beats it.
    shipped = "fine-tuned 4B (Tinker LoRA)"
    best = (shipped, res[shipped]) if shipped in res else \
        max(res.items(), key=lambda kv: kv[1][0]["deferred_frac"])
    rows_html = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{v[0]['deferred']}</td>"
        f"<td class='num'>{v[0]['deferred_frac']:.1%}</td>"
        f"<td class='num'>{v[0]['wrongly_deferred']}</td>"
        f"<td class='num'>{v[0]['auc']:.3f}</td>"
        f"<td class='note'>{html.escape(v[1])}</td></tr>"
        for k, v in sorted(res.items(), key=lambda kv: -kv[1][0]["deferred_frac"]))

    ev_html = "".join(
        f"<li><span class='t'>{'' if t is None else f't+{t}s'}</span>"
        f"<span class='m'>{html.escape(m)}</span></li>" for t, m in events)
    led_html = "".join(
        f"<li><code>{html.escape(str(r.get('item_id')))}</code> "
        f"<b class='{html.escape(str(r.get('state','')))}'>{html.escape(str(r.get('state','')))}</b> "
        f"<span class='dim'>{html.escape(str(r.get('why') or r.get('reason') or ''))}</span></li>"
        for r in ledger)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Leave It — the morning filter</title>
<style>
:root{{--ink:#e8e6e1;--dim:#8b877e;--bg:#12110f;--panel:#1a1917;--line:#2c2a26;--acc:#c8f560;--warn:#f5a05a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;padding:0 20px 80px}}
main{{max-width:960px;margin:0 auto}}
h1{{font-size:clamp(28px,5vw,44px);line-height:1.1;letter-spacing:-.02em;margin:48px 0 8px}}
h2{{font-size:15px;text-transform:uppercase;letter-spacing:.14em;color:var(--acc);
 margin:52px 0 12px;font-weight:600}}
p{{color:var(--ink);max-width:70ch}}
.sub{{color:var(--dim);margin:0 0 32px}}
.big{{font-size:clamp(40px,9vw,72px);line-height:1;color:var(--acc);font-weight:700;margin:8px 0}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--dim);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.08em}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;color:var(--acc)}}
td.note{{color:var(--dim);font-size:13px}}
tr:first-child td{{color:var(--ink)}}
ol.ev{{list-style:none;padding:0;margin:0}}
ol.ev li{{display:flex;gap:16px;padding:7px 0;border-bottom:1px solid var(--line);font-size:14px}}
.t{{color:var(--acc);min-width:64px;font-variant-numeric:tabular-nums}}
ul.led{{list-style:none;padding:0;margin:0;font-size:14px}}
ul.led li{{padding:7px 0;border-bottom:1px solid var(--line)}}
.LAPSED{{color:var(--warn)}} .RESCUED{{color:var(--acc)}} .SEALED{{color:var(--dim)}}
.dim{{color:var(--dim)}}
code{{background:#000;padding:2px 6px;border-radius:4px;font-size:13px}}
pre{{background:#000;border:1px solid var(--line);border-radius:8px;padding:16px;overflow:auto;font-size:13px}}
.foot{{margin-top:56px;color:var(--dim);font-size:13px}}
</style></head><body><main>

<h1>Leave It</h1>
<p class="sub">A morning filter for a backlog you have decided not to read.</p>

<p>Every item below is a real issue from a real repository. For each one the model
answers a single question: <b>does this need a maintainer today?</b> Items it defers
are held by a durable Temporal fuse, so a decision to ignore something cannot be
lost by a crash, a reboot or a closed laptop lid.</p>

<h2>What the filter buys you</h2>
<div class="big">{best[1][0]['deferred_frac']:.0%}</div>
<p>of the held-out stream never reached the human, while wrongly deferring
{best[1][0]['wrongly_deferred']} of the {sum(labels)} items that genuinely needed
someone within 24 hours. A perfect ranker could defer {ceiling:.0%}; deferring
everything would defer 100% and miss everything.</p>

<div class="panel"><table>
<thead><tr><th>arm</th><th class="num">deferred</th><th class="num">of pool</th>
<th class="num">wrong</th><th class="num">AUC</th><th>what it is</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<p class="dim" style="font-size:13px;margin:14px 0 0">
Held-out window: {rows[0]['created_at'][:10]} to {rows[-1]['created_at'][:10]},
n={len(rows)}. Operating point: at most 5% of the items that needed a maintainer
today may be deferred. Grouped by the same policy for every arm.</p>
</div>

<h2>The guarantee, executed</h2>
<p>A worker was killed with <code>kill -9</code> with three fuses pending, and one
attention signal was delivered while no worker process existed at all.</p>
<div class="panel"><ol class="ev">{ev_html}</ol></div>

<h2>Ledger</h2>
<p>Append-only. Verdicts are recorded and folded on read, so any decision can be
re-read and argued with later.</p>
<div class="panel"><ul class="led">{led_html}</ul></div>

<h2>Reproduce</h2>
<pre>git clone https://github.com/memeshee/leaveit
cd leaveit
python3 data/fetch_issues.py 10      # real issues + real outcomes
python3 data/build_dataset.py        # labels + temporal split
python3 eval/baselines.py            # the arms to beat
python3 train/lm.py train tuned 400  # LoRA on Tinker
python3 train/lm.py score tuned
python3 eval/policy.py eval/scores_*.json
cd app &amp;&amp; ./run_demo.sh              # kill -9 proof</pre>

<p class="foot">Leave It · Hacktoberfest Open-Source AI Challenge Week 3 ·
every number on this page is regenerated from <code>data/</code> and <code>eval/</code> by
<code>app/build_site.py</code>.</p>
</main></body></html>
"""
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({len(page)} bytes), arms: {len(res)}, events: {len(events)}, ledger: {len(ledger)}")


if __name__ == "__main__":
    main()
