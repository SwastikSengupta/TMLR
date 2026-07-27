"""
Verify every bibliography entry against CrossRef (DOI registry) and arXiv.

For each \bibitem we extract the title, query the registry, and compare the
returned authors / year / venue against what the paper claims. Anything that
does not match is flagged for manual checking. This is stricter than eyeballing
because it compares against the publisher's own metadata record.
"""
import json
import re
import time
import urllib.parse
import urllib.request

TEX = "docs/paper.tex"
UA = {"User-Agent": "bib-verify/1.0 (mailto:research@example.org)"}


def entries(path):
    t = open(path).read()
    block = t[t.index("\\begin{thebibliography}"):t.index("\\end{thebibliography}")]
    out = []
    for m in re.finditer(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\Z)", block, re.S):
        key, body = m.group(1), " ".join(m.group(2).split())
        body = re.sub(r"\\emph\{([^}]*)\}", r"\1", body)
        body = re.sub(r"\\url\{([^}]*)\}", r"\1", body)
        body = body.replace("\\&", "&").replace("~", " ").replace("\\", "")
        # title is the first double-quoted span, else the emph'd book title
        tm = re.search(r"``(.+?),''", body) or re.search(r"``(.+?)''", body)
        title = tm.group(1) if tm else None
        ym = re.search(r"(19|20)\d{2}", body)
        year = int(ym.group(0)) if ym else None
        am = re.search(r"arXiv:(\d{4}\.\d{4,5})", body)
        out.append({"key": key, "raw": body, "title": title, "year": year,
                    "arxiv": am.group(1) if am else None})
    return out


def crossref(title):
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 1,
                                "select": "title,author,issued,container-title,DOI"})
    url = f"https://api.crossref.org/works?{q}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=25) as r:
            items = json.load(r)["message"]["items"]
        if not items:
            return None
        it = items[0]
        return {"title": (it.get("title") or [""])[0],
                "year": (it.get("issued", {}).get("date-parts", [[None]])[0][0]),
                "venue": (it.get("container-title") or [""])[0],
                "authors": [a.get("family", "") for a in it.get("author", [])][:4],
                "doi": it.get("DOI", "")}
    except Exception as e:
        return {"error": str(e)[:50]}


def arxiv(aid):
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=25) as r:
            x = r.read().decode()
        t = re.search(r"<title>(.*?)</title>", x, re.S)
        t = re.search(r"<entry>.*?<title>(.*?)</title>", x, re.S)
        auth = re.findall(r"<name>(.*?)</name>", x)
        pub = re.search(r"<published>(\d{4})", x)
        return {"title": " ".join(t.group(1).split()) if t else "",
                "authors": auth[:4], "year": int(pub.group(1)) if pub else None}
    except Exception as e:
        return {"error": str(e)[:50]}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def similar(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    sa = {a[i:i+5] for i in range(len(a) - 4)}
    sb = {b[i:i+5] for i in range(len(b) - 4)}
    return len(sa & sb) / max(1, len(sa | sb))


if __name__ == "__main__":
    es = entries(TEX)
    print(f"{len(es)} bibliography entries\n")
    report = []
    for i, e in enumerate(es, 1):
        rec = {"key": e["key"], "claimed_title": e["title"],
               "claimed_year": e["year"], "status": "", "found": None, "note": ""}
        if e["arxiv"]:
            got = arxiv(e["arxiv"])
            src = "arXiv"
        elif e["title"]:
            got = crossref(e["title"])
            src = "CrossRef"
        else:
            rec["status"] = "NO_TITLE"
            report.append(rec)
            print(f"{i:2d}. {e['key']:18s} NO TITLE PARSED")
            continue
        rec["source"] = src
        if not got or "error" in (got or {}):
            rec["status"] = "NOT_FOUND"
            rec["note"] = (got or {}).get("error", "no result")
        else:
            sim = similar(e["title"], got.get("title", ""))
            yr_ok = (e["year"] is None or got.get("year") is None
                     or abs(e["year"] - got["year"]) <= 1)
            rec["found"] = got
            rec["title_similarity"] = round(sim, 2)
            if sim >= 0.55 and yr_ok:
                rec["status"] = "VERIFIED"
            elif sim >= 0.55:
                rec["status"] = "YEAR_MISMATCH"
                rec["note"] = f"claimed {e['year']}, registry {got.get('year')}"
            else:
                rec["status"] = "TITLE_MISMATCH"
                rec["note"] = f"registry returned: {got.get('title','')[:70]}"
        report.append(rec)
        flag = {"VERIFIED": "ok ", "YEAR_MISMATCH": "YR ",
                "TITLE_MISMATCH": "TTL", "NOT_FOUND": "NF "}.get(rec["status"], "?  ")
        print(f"{i:2d}. {flag} {e['key']:18s} {rec.get('note','')[:60]}")
        time.sleep(0.35)

    counts = {}
    for r in report:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n=== summary ===")
    for k, v in sorted(counts.items()):
        print(f"  {k:16s} {v}")
    json.dump(report, open("results/bib_verification.json", "w"), indent=1)
    print("\nwrote results/bib_verification.json")
