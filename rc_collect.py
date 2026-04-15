"""
rc_collect.py — Step 1: Harvest & classify RC papers
=====================================================
Run this once. Produces rc_papers.csv which you can
edit directly before plotting.

pip install requests tqdm
python rc_collect.py
"""

import os, json, csv, time, re
import requests
from collections import defaultdict
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

YOUR_EMAIL  = "...v@tu-berlin.de"
OA_BASE     = "https://api.openalex.org"
YEARS       = (2001, 2026)
CACHE_RAW   = "rc_papers_raw.json"    # skip re-download
CACHE_CLEAN = "rc_papers_clean.json"  # skip re-filter
OUTPUT_CSV  = "rc_papers.csv"         # ← the file you edit & plot from

SEARCH_QUERIES = [
    "reservoir computing",
    "echo state network",
    "liquid state machine",
]

# ── RC relevance filter ───────────────────────────────────────────────
RC_CORE_TERMS = [
    "reservoir computing", "echo state network", "echo state networks",
    "liquid state machine", "reservoir computer", "esn",
    "lsm reservoir", "recurrent reservoir", "reservoir layer",
    "reservoir dynamics", "reservoir neuron", "reservoir readout",
    "reservoir model", "physical reservoir", "deep reservoir",
    "next generation reservoir", "echo state property",
    "spectral radius", "fading memory property",
]

# ── Subfield taxonomy (first match wins) ──────────────────────────────
SUBFIELD_RULES = [
    ("Photonics & Optics",
     ["photon","optical","optic","laser","opto","delay line",
      "fiber","mach-zehnder","electro-optic","silicon photonic"]),

    ("Physical & Hardware RC",
     ["physical reservoir","spintronic","memristor","analog","analogue",
      "fpga","vlsi","neuromorphic","in-material",
      "mechanical reservoir","soft robot","morphological"]),

    ("Quantum RC",
     ["quantum reservoir","qubit","quantum computing",
      "quantum circuit","quantum system"]),

    ("Nonlinear Dynamics & Chaos",
     ["chaos","chaotic","lorenz","lyapunov","attractor",
      "nonlinear dynamic","bifurcation","edge of chaos",
      "dynamical system","mackey-glass","kuramoto"]),

    ("Neuroscience & Comp. Neuro",
     ["cortex","cortical","spiking","spike","neural microcircuit",
      "working memory","hippocampal","cerebellum","synaptic",
      "biological plausib","in vivo","electrophysi"]),

    ("Mathematics & Theory",
     ["universal approximation","fading memory","separation property",
      "approximation theory","ergodic","stability analysis",
      "convergence","random matrix","theorem","functional analysis"]),

    ("Signal Processing & Comms",
     ["signal processing","channel equalization","speech","audio",
      "noise cancell","wireless","telecommunication","modulation",
      "radar","sonar"]),

    ("Control & Robotics",
     ["control system","robot","reinforcement learning","motor control",
      "locomotion","adaptive control","autonomous"]),

    ("Climate & Earth Sciences",
     ["climate","weather","wind power","turbulence","ocean",
      "atmospher","geophysic","rainfall","flood","seismic","earthquake"]),

    ("ML / Next-Gen RC",
     ["deep learning","transformer","lstm","machine learning",
      "next generation reservoir","transfer learning","hyperparameter"]),

    ("General RC / ESN Methods",
     ["echo state","reservoir computing","readout","spectral radius",
      "reservoir design","leaky integrator","esn","liquid state"]),
]
DEFAULT_SUBFIELD = "Other / Interdisciplinary"

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def oa_search(query, year_from, year_to, per_page=200, max_pages=30):
    params = {
        "search":   query,
        "filter":   f"publication_year:{year_from}-{year_to}",
        "per-page": per_page,
        "select":   ("id,title,publication_year,cited_by_count,"
                     "primary_topic,abstract_inverted_index"),
        "mailto":   YOUR_EMAIL,
    }
    results, cursor, page = [], "*", 0
    pbar = tqdm(desc=f"  '{query}'", unit=" papers")
    while cursor and page < max_pages:
        params["cursor"] = cursor
        try:
            r = requests.get(f"{OA_BASE}/works", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"\n  error: {e} — retry in 5s"); time.sleep(5); continue
        batch = data.get("results", [])
        if not batch: break
        results.extend(batch)
        pbar.update(len(batch))
        cursor = data.get("meta", {}).get("next_cursor")
        page += 1
        time.sleep(0.12)
    pbar.close()
    return results


def reconstruct_abstract(inv):
    if not inv: return ""
    words = {pos: w for w, positions in inv.items() for pos in positions}
    return " ".join(words[i] for i in sorted(words))


def normalise(work):
    topic  = work.get("primary_topic") or {}
    field  = topic.get("field",  {}).get("display_name", "")
    domain = topic.get("domain", {}).get("display_name", "")
    return {
        "paperId":       work.get("id","").replace("https://openalex.org/",""),
        "title":         work.get("title") or "",
        "year":          work.get("publication_year"),
        "citationCount": work.get("cited_by_count", 0),
        "abstract":      reconstruct_abstract(work.get("abstract_inverted_index")),
        "fieldsOfStudy": "; ".join(f for f in [field, domain] if f),
    }


def dedup_id(papers):
    seen, out = set(), []
    for p in papers:
        if p["paperId"] not in seen:
            seen.add(p["paperId"]); out.append(p)
    return out


def dedup_title(papers):
    def norm(t): return re.sub(r"[^a-z0-9]","",t.lower())[:60]
    seen, out = set(), []
    for p in papers:
        k = norm(p.get("title",""))
        if k and k not in seen:
            seen.add(k); out.append(p)
    return out


def rc_relevant(paper):
    hay = (paper.get("title","") + " " + paper.get("abstract","")).lower()
    if any(t in hay for t in RC_CORE_TERMS): return True
    title = paper.get("title","").lower()
    if "reservoir" in title and any(w in title for w in
       ["computing","network","neural","learning","layer","computer","model"]):
        return True
    return False


def classify(paper):
    fos = paper.get("fieldsOfStudy", "")
    if isinstance(fos, list):
        fos = " ".join(fos)
    text = (paper.get("title","") + " " +
            paper.get("abstract","") + " " +
            fos).lower()
    for label, kws in SUBFIELD_RULES:
        if any(kw in text for kw in kws):
            return label
    return DEFAULT_SUBFIELD

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  RC Collect — harvest, filter, classify")
    print("=" * 60)

    # 1. Harvest
    if os.path.exists(CACHE_RAW):
        print(f"\nRaw cache found — loading '{CACHE_RAW}' (delete to re-harvest)")
        with open(CACHE_RAW) as f: papers = json.load(f)
        print(f"  -> {len(papers):,} papers")
    else:
        raw = []
        print("\nQuerying OpenAlex...")
        for q in SEARCH_QUERIES:
            works = oa_search(q, YEARS[0], YEARS[1])
            raw.extend(works)
            print(f"  -> {len(works):,} results"); time.sleep(1)
        papers = dedup_title(dedup_id([normalise(w) for w in raw]))
        print(f"\n{len(papers):,} unique papers after dedup")
        with open(CACHE_RAW, "w") as f: json.dump(papers, f)
        print(f"Cached -> '{CACHE_RAW}'")

    # 2. Filter
    if os.path.exists(CACHE_CLEAN):
        print(f"\nClean cache found — loading '{CACHE_CLEAN}'")
        with open(CACHE_CLEAN) as f: papers = json.load(f)
        print(f"  -> {len(papers):,} papers")
    else:
        before  = len(papers)
        papers  = [p for p in papers if rc_relevant(p)]
        removed = before - len(papers)
        no_abs  = sum(1 for p in papers if not p.get("abstract"))
        print(f"\nRC relevance filter:")
        print(f"  Before : {before:,}")
        print(f"  After  : {len(papers):,}  ({removed:,} removed)")
        print(f"  No abstract : {no_abs:,} ({100*no_abs/max(len(papers),1):.1f}%)")
        with open(CACHE_CLEAN, "w") as f: json.dump(papers, f)
        print(f"Cached -> '{CACHE_CLEAN}'")

    # 3. Classify
    print("\nClassifying subfields...")
    for p in tqdm(papers):
        p["subfield"] = classify(p)

    # citation velocity (citations per year since publication)
    for p in papers:
        age = max(2026 - (p.get("year") or 2026), 1)
        p["cites_per_year"] = round(p.get("citationCount", 0) / age, 2)

    # 4. Save CSV
    # landmark column is empty — YOU fill it in (put any text e.g. "yes" or a reason)
    fields = [
        "paperId", "title", "year", "citationCount", "cites_per_year",
        "subfield", "fieldsOfStudy", "landmark",   # <- landmark column, all empty
        "abstract"
    ]
    for p in papers:
        p.setdefault("landmark", "")   # empty = not a landmark

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for p in sorted(papers, key=lambda x: x.get("year") or 0):
            row = dict(p)
            row["abstract"] = (p.get("abstract") or "")[:300]
            w.writerow(row)

    print(f"\n✓ Saved {len(papers):,} papers -> '{OUTPUT_CSV}'")
    print(f"\nSubfield breakdown:")
    sf = defaultdict(int)
    for p in papers: sf[p["subfield"]] += 1
    mx = max(sf.values()) if sf else 1
    for label, n in sorted(sf.items(), key=lambda x: -x[1]):
        print(f"  {label:<45} {n:>5}  {'█' * (n*35//mx)}")

    print(f"""
Next steps:
  1. Open '{OUTPUT_CSV}' in Excel / any spreadsheet
  2. Find papers you want as landmarks — put anything in the 'landmark' column
     e.g. "ESN founding paper" or just "yes"
  3. You can also fix any subfield misclassifications directly in the CSV
  4. Run:  python rc_plot.py
""")