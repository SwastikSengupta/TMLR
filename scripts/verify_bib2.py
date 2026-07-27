"""Second pass: re-check unresolved entries against Semantic Scholar, which
indexes CS conference proceedings that CrossRef largely misses."""
import json, re, time, urllib.parse, urllib.request
from verify_bib import entries, norm, similar
rep = json.load(open("results/bib_verification.json"))
todo = [r for r in rep if r["status"] != "VERIFIED"]
es = {e["key"]: e for e in entries("docs/paper.tex")}
UA = {"User-Agent": "bib-verify/1.0"}
def s2(title):
    q = urllib.parse.urlencode({"query": title, "limit": 3,
                                "fields": "title,year,venue,authors,externalIds"})
    try:
        with urllib.request.urlopen(urllib.request.Request(
                f"https://api.semanticscholar.org/graph/v1/paper/search?{q}",
                headers=UA), timeout=25) as r:
            return json.load(r).get("data", [])
    except Exception as e:
        return [{"error": str(e)[:40]}]
print(f"re-checking {len(todo)} entries against Semantic Scholar\n")
fixed = 0
for r in todo:
    e = es.get(r["key"])
    title = e["title"] if e else None
    if not title:
        print(f"  --  {r['key']:18s} (book/URL, verify by hand)"); continue
    hits = s2(title)
    best, bs = None, 0.0
    for h in hits:
        if "error" in h: break
        s = similar(title, h.get("title",""))
        if s > bs: best, bs = h, s
    if best and bs >= 0.6:
        yr = best.get("year")
        ok = e["year"] is None or yr is None or abs(e["year"]-yr) <= 1
        auth = [a.get("name","").split()[-1] for a in (best.get("authors") or [])][:3]
        r["s2"] = {"title": best.get("title"), "year": yr, "venue": best.get("venue"),
                   "authors": auth, "sim": round(bs,2)}
        r["status"] = "VERIFIED_S2" if ok else "YEAR_CHECK"
        fixed += ok
        print(f"  {'ok ' if ok else 'YR '} {r['key']:18s} {yr} {str(best.get('venue'))[:34]}")
    else:
        print(f"  ??  {r['key']:18s} no confident match - VERIFY BY HAND")
    time.sleep(0.9)
json.dump(rep, open("results/bib_verification.json","w"), indent=1)
c = {}
for r in rep: c[r["status"]] = c.get(r["status"],0)+1
print("\n=== final ==="); [print(f"  {k:16s} {v}") for k,v in sorted(c.items())]
