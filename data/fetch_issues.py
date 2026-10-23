#!/usr/bin/env python3
"""Pull real issue streams from public repos via GitHub GraphQL.

Each row = one real backlog item plus what actually happened to it.
We keep per-comment (timestamp, author association) so the label can be
"did a MAINTAINER ever look at this?" rather than "did anyone reply at all".
"""
import json, subprocess, sys, pathlib

OUT = pathlib.Path(__file__).parent.parent / "data" / "issues.jsonl"
REPOS = [("vercel", "next.js"), ("rust-lang", "rust"), ("microsoft", "vscode"), ("python", "cpython")]
PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 10

Q = """
query($owner:String!,$name:String!,$cursor:String){
  repository(owner:$owner,name:$name){
    issues(first:100,after:$cursor,orderBy:{field:CREATED_AT,direction:DESC}){
      pageInfo{hasNextPage endCursor}
      nodes{
        number title body createdAt state stateReason closedAt
        author{login}
        comments(first:10){ totalCount nodes{ createdAt authorAssociation author{login} } }
        reactions{ totalCount }
        labels(first:6){ nodes{ name } }
      }
    }
  }
}
"""

def gql(owner, name, cursor=None):
    args = ["gh", "api", "graphql", "-f", f"query={Q}", "-F", f"owner={owner}", "-F", f"name={name}"]
    if cursor:
        args += ["-F", f"cursor={cursor}"]
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[:400])
    return json.loads(p.stdout)

rows = 0
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    for owner, name in REPOS:
        cursor, page = None, 0
        while page < PAGES:
            d = gql(owner, name, cursor)
            iss = d["data"]["repository"]["issues"]
            for n in iss["nodes"]:
                author = (n.get("author") or {}).get("login")
                f.write(json.dumps({
                    "repo": f"{owner}/{name}", "number": n["number"],
                    "title": n["title"], "body": (n.get("body") or "")[:1500],
                    "created_at": n["createdAt"], "closed_at": n.get("closedAt"),
                    "state": n["state"], "state_reason": n.get("stateReason"),
                    "author": author,
                    "comment_count": n["comments"]["totalCount"],
                    "comments": [{"at": c["createdAt"], "assoc": c.get("authorAssociation"),
                                  "by_author": (c.get("author") or {}).get("login") == author}
                                 for c in n["comments"]["nodes"]],
                    "reactions": n["reactions"]["totalCount"],
                    "labels": [l["name"] for l in n["labels"]["nodes"]],
                }) + "\n")
                rows += 1
            cursor = iss["pageInfo"]["endCursor"]
            page += 1
            if not iss["pageInfo"]["hasNextPage"]:
                break
        print(f"{owner}/{name}: {page} pages", flush=True)

print("rows written:", rows, "->", OUT)
