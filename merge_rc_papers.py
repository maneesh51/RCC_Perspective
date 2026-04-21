"""
merge_rc_papers.py — Merge extra OpenAlex CSVs into rc_papers.csv
=================================================================
Usage:
    python merge_rc_papers.py rc_papers.csv Cocitations_NGRC.csv [more.csv ...]

Output:
    rc_papers_merged.csv  — deduplicated, classified, ready for rc_plot.py

The extra CSVs are raw OpenAlex exports (columns like `id`, `display_name`,
`cited_by_count`, `doi`, `publication_year`, `abstract_inverted_index` …).
The script normalises them to the same schema as rc_papers.csv, applies the
same RC-relevance filter and subfield classifier, then merges & deduplicates.
"""

import sys, csv, re, json, time
from collections import defaultdict
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── same taxonomy as rc_collect.py ────────────────────────────────────
RC_CORE_TERMS = [
    "reservoir computing", "echo state network", "echo state networks",
    "liquid state machine", "reservoir computer", "recurrent reservoir",
    "reservoir readout", "physical reservoir computing",
    "deep reservoir computing", "next generation reservoir computing",
    "echo state property", "fading memory property",
]
RC_EXCLUDE_TERMS = [
    "petroleum reservoir", "oil reservoir", "gas reservoir",
    "groundwater reservoir", "water reservoir", "geological reservoir",
    "hydrocarbon reservoir", "porous reservoir", "reservoir simulation",
    "reservoir characterization", "reservoir management", "reservoir pressure",
    "reservoir fluid", "reservoir rock", "reservoir permeability",
    "reservoir porosity", "dam reservoir", "slope erosion", "soil", "erosion",
    "surface reservoir", "reservoir network", "reservoir storage",
    "reservoir inflow", "reservoir operation", "reservoir routing",
    "reservoir level", "underground reservoir", "aquifer reservoir",
    "carbonate reservoir", "sandstone reservoir", "shale reservoir",
    "tight reservoir", "fractured reservoir",
]
# Score-based classification: ALL rules are evaluated, scores accumulated.
# Title match = 3pts, abstract/FoS match = 1pt. Highest score wins.
# General RC / ESN Methods is the fallback only (never scored directly).
# Rule order no longer matters for correctness, but kept logical for readability.
SUBFIELD_RULES = [
    ("Next-Gen RC",
     # high-weight specific terms — even 1 title hit beats generic RC keywords
     ["next generation reservoir","next-generation reservoir",
      "ng-rc","ngrc","next gen rc","nextrc"]),

    ("Photonics & Optics",
     ["photon","optical","optic","laser","opto","delay line","delay-line",
      "fiber","mach-zehnder","electro-optic","silicon photonic",
      "photonic","optoelectronic","wavelength","waveguide","microring"]),

    ("Physical & Hardware RC",
     ["physical reservoir","spintronic","memristor","analog","analogue",
      "fpga","vlsi","neuromorphic","in-material","mechanical reservoir",
      "soft robot","morphological","physical computing","hardware reservoir",
      "resistive switching","phase change","volatile memristor"]),

    ("Quantum RC",
     ["quantum reservoir","qubit","quantum computing",
      "quantum circuit","quantum system","quantum machine learning",
      "quantum noise","open quantum"]),

    ("Neuroscience & Comp. Neuro",
     ["cortex","cortical","spiking","spike","neural microcircuit",
      "working memory","hippocampal","cerebellum","synaptic",
      "biological plausib","in vivo","electrophysi","neuroscience",
      "neuronal network","neocortex","dendritic","basal ganglia"]),

    ("Mathematics & Theory",
     ["universal approximation","fading memory","separation property",
      "approximation theory","ergodic","stability analysis",
      "convergence","random matrix","theorem","functional analysis",
      "reproducing kernel","echo state property","contraction",
      "lyapunov","rademacher","generalization bound","capacity"]),

    ("Nonlinear Dynamics & Chaos",
     ["chaos","chaotic","lorenz","attractor","nonlinear dynamics",
      "bifurcation","edge of chaos","dynamical system","mackey-glass",
      "kuramoto","nonlinearity","oscillator","nonlinear system",
      "transient dynamics","strange attractor","phase space",
      "time series prediction","synchronization","limit cycle"]),

    ("Signal Processing & Comms",
     ["signal processing","channel equalization","speech","audio",
      "noise cancell","wireless","telecommunication","modulation",
      "radar","sonar","equaliz","distortion compensation"]),

    ("Control & Robotics",
     ["control system","robot","reinforcement learning","motor control",
      "locomotion","adaptive control","autonomous","model predictive",
      "pid","trajectory","actuator","manipulator"]),

    ("Climate & Earth Sciences",
     ["climate","weather","wind power","turbulence","ocean",
      "atmospher","geophysic","rainfall","flood","seismic","earthquake",
      "wind speed","sea surface","precipitation","hydro"]),

    # Fallback — matched only when nothing else scores
    ("General RC / ESN Methods",
     ["echo state","reservoir computing","ridge-regression","spectral radius",
      "reservoir design","leaky integrator","esn","liquid state"]),
]
DEFAULT_SUBFIELD = "General RC / ESN Methods"

OUTPUT_FIELDS = ["paperId","title","year","citationCount","cites_per_year",
                 "subfield","fieldsOfStudy","doi","landmark","abstract"]

# ── helpers ───────────────────────────────────────────────────────────

def reconstruct_abstract(inv_str):
    """Reconstruct abstract from OpenAlex inverted-index JSON string."""
    if not inv_str or inv_str.strip() in ("", "{}"):
        return ""
    try:
        inv = json.loads(inv_str)
        words = {pos: w for w, positions in inv.items() for pos in positions}
        return " ".join(words[i] for i in sorted(words))
    except Exception:
        return ""

def rc_relevant(title, abstract, strict=True):
    """
    strict=True  (default, used for the base harvest):
        Requires at least one RC core term OR esn/lsm abbreviation.
    strict=False (used for hand-picked citation exports):
        Only excludes obvious non-RC domains (petroleum, hydrology…).
        Since these CSVs come from citing specific RC seed papers,
        provenance already guarantees relevance.
    """
    hay = (title + " " + abstract).lower()
    # hard exclusion always applies
    if any(t in hay for t in RC_EXCLUDE_TERMS):
        return False
    if strict:
        if any(t in hay for t in RC_CORE_TERMS):
            return True
        if re.search(r"\b(esn|lsm)\b", hay):
            return True
        return False
    return True  # non-strict: exclusion was the only gate

def classify(title, abstract, fields_of_study=""):
    """
    Score-based classifier: every rule accumulates points across all matches.
    Title hits are worth 3x, abstract/FoS hits 1x.
    The label with the highest score wins; ties broken by rule order.
    General RC / ESN Methods only wins if nothing else scores at all.
    """
    title_l    = title.lower()
    body_l     = (abstract + " " + fields_of_study).lower()

    scores = {}
    for label, kws in SUBFIELD_RULES:
        if label == DEFAULT_SUBFIELD:
            continue  # scored separately as fallback
        s = 0
        for kw in kws:
            if kw in title_l:
                s += 3          # strong signal: keyword in title
            elif kw in body_l:
                s += 1          # weaker signal: keyword in abstract/FoS
        if s > 0:
            scores[label] = scores.get(label, 0) + s

    if not scores:
        return DEFAULT_SUBFIELD

    return max(scores, key=lambda l: scores[l])

def dedup_key(doi, title):
    if doi and doi.strip():
        return doi.strip().lower()
    return re.sub(r"[^a-z0-9]", "", title.lower())[:80]

def cites_per_year(citations, year):
    try:
        age = max(2026 - int(year), 1)
        return round(int(citations) / age, 2)
    except Exception:
        return 0.0

# ── load rc_papers.csv (already normalised) ───────────────────────────

def load_rc_papers(path):
    papers = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            papers.append(row)
    print(f"  Loaded {len(papers):,} rows from '{path}'")
    return papers

# ── load a raw OpenAlex export CSV ────────────────────────────────────

def load_oa_export(path):
    """
    Handles OpenAlex CSV exports which may have columns like:
      id, display_name, cited_by_count, doi, publication_year,
      abstract_inverted_index, primary_topic.display_name, ...
    Returns normalised dicts compatible with rc_papers schema.
    """
    papers = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        # detect column name variants
        def col(*candidates):
            for c in candidates:
                if c in cols: return c
            return None

        id_col     = col("id")
        title_col  = col("display_name", "title")
        year_col   = col("publication_year", "year")
        cit_col    = col("cited_by_count", "citationCount")
        doi_col    = col("doi")
        abs_col    = col("abstract_inverted_index", "abstract")
        topic_col  = col("primary_topic.display_name", "primary_topic",
                         "fieldsOfStudy")

        for row in reader:
            paper_id  = (row.get(id_col) or "").replace("https://openalex.org/","")
            title     = row.get(title_col) or ""
            year      = row.get(year_col) or ""
            citations = row.get(cit_col) or "0"
            doi       = row.get(doi_col) or ""
            raw_abs   = row.get(abs_col) or ""
            fos       = row.get(topic_col) or ""

            # abstract may already be plain text or still inverted-index JSON
            if raw_abs.strip().startswith("{"):
                abstract = reconstruct_abstract(raw_abs)
            else:
                abstract = raw_abs

            papers.append({
                "paperId":       paper_id,
                "title":         title,
                "year":          year,
                "citationCount": citations,
                "cites_per_year": cites_per_year(citations, year),
                "subfield":      classify(title, abstract, fos),
                "fieldsOfStudy": fos,
                "doi":           doi,
                "landmark":      "",
                "abstract":      abstract[:300],
                "_raw_abstract": abstract,  # full text for relevance filter
            })

    print(f"  Loaded {len(papers):,} rows from '{path}'")
    return papers

# ── main ──────────────────────────────────────────────────────────────


# ── optional: fetch missing abstracts from OpenAlex ───────────────────

def enrich_abstracts(papers, email="", batch=50, pause=0.15):
    """
    For papers that have a valid OpenAlex ID but no abstract,
    fetch the abstract_inverted_index in batches from the API.
    Pass email for the polite pool (faster rate limits).
    Set email="" to skip enrichment silently.
    """
    if not HAS_REQUESTS or not email:
        return papers

    need = [p for p in papers
            if not p.get("abstract","").strip()
            and re.match(r"W\d+", p.get("paperId",""))]

    if not need:
        print("  No abstracts to enrich.")
        return papers

    print(f"  Fetching abstracts for {len(need):,} papers via OpenAlex…")
    id_map = {p["paperId"]: p for p in need}
    ids    = list(id_map.keys())
    fetched = 0

    for i in range(0, len(ids), batch):
        chunk = ids[i:i+batch]
        pipe  = "|".join(chunk)
        params = {
            "filter": f"openalex_id:{pipe}",
            "select": "id,abstract_inverted_index",
            "per-page": batch,
            "mailto": email,
        }
        try:
            r = requests.get("https://api.openalex.org/works",
                             params=params, timeout=30)
            r.raise_for_status()
            for w in r.json().get("results", []):
                wid = w.get("id","").replace("https://openalex.org/","")
                inv = w.get("abstract_inverted_index")
                if inv and wid in id_map:
                    ab = reconstruct_abstract(json.dumps(inv))
                    id_map[wid]["abstract"] = ab[:300]
                    fetched += 1
        except Exception as e:
            print(f"    warning: batch {i//batch} failed ({e})")
        time.sleep(pause)

    print(f"  Enriched {fetched:,} abstracts.")

    # reclassify enriched papers with better signal
    for p in need:
        p["subfield"] = classify(
            p.get("title",""), p.get("abstract",""), p.get("fieldsOfStudy",""))

    return papers

def main():
    # ── HARDCODED FILE PATHS ──────────────────────────────────────────
    # Set BASE_CSV to your main rc_papers.csv produced by rc_collect.py.
    # Add every extra OpenAlex export CSV to EXTRA_CSVS — one string per file.
    BASE_CSV = "rc_papers.csv"
    EXTRA_CSVS = [
        "Cocitations_Jaeger_2004.csv",
        "Cocitations_NGRC.csv",
        "Cocitations_RCtoRNN_Lukosevicius_Jaeger_2009.csv",
        "Cocitations_Real_time_Maass_2002.csv",
    ]
    # ─────────────────────────────────────────────────────────────────

    # command-line args still work as an override if you prefer
    if len(sys.argv) >= 3:
        base_path   = sys.argv[1]
        extra_paths = list(sys.argv[2:])
    else:
        base_path   = BASE_CSV
        extra_paths = EXTRA_CSVS

    # load base
    base_papers = load_rc_papers(base_path)

    # build dedup set from base
    seen = set()
    for p in base_papers:
        k = dedup_key(p.get("doi",""), p.get("title",""))
        if k: seen.add(k)

    # load & filter extras
    new_papers = []
    for path in extra_paths:
        extras = load_oa_export(path)
        before = len(extras)

        # RC relevance filter — non-strict for hand-picked citation exports:
        # these CSVs already come from citing known RC seed papers, so we only
        # exclude obvious non-RC domains (petroleum, hydrology, etc.)
        filtered = []
        for p in extras:
            abs_text = p.pop("_raw_abstract", p.get("abstract",""))
            if rc_relevant(p["title"], abs_text, strict=False):
                filtered.append(p)

        removed = before - len(filtered)
        print(f"  RC filter: {before:,} → {len(filtered):,} ({removed:,} removed)")

        # dedup against base + previously added
        added = 0
        for p in filtered:
            k = dedup_key(p.get("doi",""), p.get("title",""))
            if k and k not in seen:
                seen.add(k)
                new_papers.append(p)
                added += 1

        print(f"  New unique papers added from '{path}': {added:,}")

    # recalculate cites_per_year for base papers too (in case year was missing)
    for p in base_papers:
        if not p.get("cites_per_year"):
            p["cites_per_year"] = cites_per_year(
                p.get("citationCount", 0), p.get("year", 2026))

    # ── OPTIONAL: fetch missing abstracts for newly added papers ────────
    # Set your email to enable (uses OpenAlex polite pool, ~50 req/s).
    # Leave as "" to skip — classification will use title + topic only.
    ENRICH_EMAIL = ""   # e.g. "you@example.com"
    new_papers = enrich_abstracts(new_papers, email=ENRICH_EMAIL)
    # ─────────────────────────────────────────────────────────────────

    # reclassify ALL papers with the current (fixed) taxonomy
    # ensures base papers with stale subfield labels are corrected too
    print("\nReclassifying all papers with updated taxonomy...")
    for p in base_papers + new_papers:
        p["subfield"] = classify(
            p.get("title", ""),
            p.get("abstract", ""),
            p.get("fieldsOfStudy", ""),
        )

    all_papers = base_papers + new_papers
    all_papers.sort(key=lambda x: int(x.get("year") or 0))

    # write merged CSV
    out_path = "rc_papers_merged.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_papers)

    print(f"\n✓ Merged {len(all_papers):,} papers → '{out_path}'")
    print(f"  (base: {len(base_papers):,}  +  new: {len(new_papers):,})")

    # subfield breakdown
    sf = defaultdict(int)
    for p in all_papers: sf[p.get("subfield", DEFAULT_SUBFIELD)] += 1
    mx = max(sf.values()) if sf else 1
    print("\nSubfield breakdown:")
    for label, n in sorted(sf.items(), key=lambda x: -x[1]):
        print(f"  {label:<45} {n:>5}  {'█' * (n*35//mx)}")

if __name__ == "__main__":
    main()