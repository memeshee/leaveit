#!/usr/bin/env python3
"""Publish the finished post to the existing draft.

  python app/publish_post.py            # dry run: show what would change
  python app/publish_post.py --go       # update draft 4906035 and publish it

Refuses to publish while any {{placeholder}} is still in the file, or if the body
is under the length a competitive entry needs.
"""
import json, pathlib, re, sys, urllib.error, urllib.request

ROOT = pathlib.Path(__file__).parent.parent
KEY = pathlib.Path.home().joinpath(".config", "hf26", "devto_api_key")
DRAFT_ID = 4906035
MIN_WORDS = 1800


def api(path, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request("https://dev.to/api" + path, data=data, method=method,
                                 headers={"api-key": KEY.read_text().strip(),
                                          "Accept": "application/json",
                                          "Content-Type": "application/json",
                                          "User-Agent": "leaveit-publisher"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]


def main():
    src = ROOT / "post" / "post_final.md"
    body = src.read_text()
    words = len(body.split())

    problems = []
    leftovers = re.findall(r"\{\{[A-Z_]+\}\}", body)
    if leftovers:
        problems.append(f"unfilled placeholders: {sorted(set(leftovers))}")
    if words < MIN_WORDS:
        problems.append(f"too short: {words} words < {MIN_WORDS}")
    if "hf26challenge" not in body and "devchallenge" not in body:
        problems.append("challenge tag not mentioned")
    for required in ["## What I Built", "## Demo", "## Code", "## How I Built It",
                     "## Why Does Open Innovation Matter?", "## Prize Categories"]:
        if required not in body:
            problems.append(f"missing template heading: {required}")
    if "https://github.com/memeshee/leaveit" not in body:
        problems.append("repo link missing")

    print(f"body: {words} words, {len(body)} chars")
    if problems:
        print("REFUSING TO PUBLISH:")
        for p in problems:
            print("  -", p)
        return 1

    title = body.splitlines()[0].lstrip("# ").strip()
    # the markdown starts with the italic submission line; take the real title from our arg
    title = sys.argv[2] if len(sys.argv) > 2 else \
        "I Fine-Tuned a 4B Model to Argue Me Out of Reading 731 GitHub Issues"
    payload = {"article": {"title": title, "body_markdown": body, "published": True,
                           "tags": ["devchallenge", "hf26challenge", "ai", "opensource"]}}
    if "--go" not in sys.argv:
        print("\nDRY RUN — would PUT /articles/%d with title:\n  %s" % (DRAFT_ID, title))
        st, cur = api(f"/articles/{DRAFT_ID}")
        print("current draft status:", st)
        return 0

    st, res = api(f"/articles/{DRAFT_ID}", "PUT", payload)
    print("PUT ->", st)
    if st in (200, 201):
        a = res
        print("  published:", a.get("published"), "url:", a.get("url"))
        print("  tags:", a.get("tag_list"), "reading time:", a.get("reading_time_minutes"))
    else:
        print("  ", res)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
