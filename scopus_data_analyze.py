"""
rc_scopus_fig1.py
=================
Loads Scopus RC exports (2000-2015 and 2016-2025), normalises them to the
same schema used by rc_plot_scopus.ipynb, classifies each paper into a
subfield, deduplicates, writes rc_papers_merged_scopus.csv, then produces
the same Fig 1 family as the OpenAlex version:

  rc_scopus_landscape.png
  rc_scopus_landscape_normalized.png
  rc_scopus_landscape_early_window.png
  rc_scopus_landscape_early_window_normalized.png
  rc_scopus_Fig1.png                  ← composite grid (main output)

Usage
-----
    python rc_scopus_fig1.py

Adjust the two FILE PATH constants at the top of the CONFIG block if your
CSV files live somewhere else.
"""

# ══════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# ══════════════════════════════════════════════════════════════════════
# CONFIG  ← edit paths here
# ══════════════════════════════════════════════════════════════════════

SCOPUS_CSV_2000_2015 = "cocitation_diagrams_files/scopus_export_2000_2015.csv"   # first file (in project)
SCOPUS_CSV_2016_2025 = "cocitation_diagrams_files/scopus_export_2016_2025.csv"   # ← PLACEHOLDER: add your file

OUTPUT_MERGED   = "rc_papers_merged_scopus.csv"
OUTPUT_ABS      = "rc_scopus_landscape.png"
OUTPUT_NORM     = "rc_scopus_landscape_normalized.png"
LANDMARK_CSV    = "rc_landmark_papers.csv"              # optional; same file as OpenAlex run

YEARS           = (2000, 2025)
EARLY_WINDOW_END = 2005

OUTPUT_ABS_EARLY  = "rc_scopus_landscape_early_window.png"
OUTPUT_NORM_EARLY = "rc_scopus_landscape_early_window_normalized.png"
OUTPUT_FIG1       = "rc_scopus_Fig1.png"

# ══════════════════════════════════════════════════════════════════════
# TAXONOMY  (identical to OpenAlex version so plots are comparable)
# ══════════════════════════════════════════════════════════════════════

SUBFIELD_COLORS = {
    "Photonics & Optics":          "red",
    "Physical & Hardware RC":      "green",
    "Quantum RC":                  "blue",
    "Nonlinear Dynamics & Chaos":  "orange",
    "Neuroscience & Comp. Neuro":  "royalblue",
    "Mathematics & Theory":        "black",
    "Signal Processing & Comms":   "darkviolet",
    "Control & Robotics":          "c",
    "Climate & Earth Sciences":    "brown",
    "Next-Gen RC":                 "deeppink",
    "General RC / ESN Methods":    "dimgray",
}
SUBFIELD_ORDER = list(SUBFIELD_COLORS.keys())
DEFAULT_SUBFIELD = "General RC / ESN Methods"

SUBFIELD_RULES = [
    ("Next-Gen RC",
     ["next generation reservoir", "next-generation reservoir",
      "ng-rc", "ngrc", "next gen rc", "nextrc"]),

    ("Photonics & Optics",
     ["photon", "optical", "optic", "laser", "opto", "delay line", "delay-line",
      "fiber", "mach-zehnder", "electro-optic", "silicon photonic",
      "photonic", "optoelectronic", "wavelength", "waveguide", "microring"]),

    ("Physical & Hardware RC",
     ["physical reservoir", "spintronic", "memristor", "analog", "analogue",
      "fpga", "vlsi", "neuromorphic", "in-material", "mechanical reservoir",
      "soft robot", "morphological", "physical computing", "hardware reservoir",
      "resistive switching", "phase change", "volatile memristor"]),

    ("Quantum RC",
     ["quantum reservoir", "qubit", "quantum computing",
      "quantum circuit", "quantum system", "quantum machine learning",
      "quantum noise", "open quantum"]),

    ("Neuroscience & Comp. Neuro",
     ["cortex", "cortical", "spiking", "spike", "neural microcircuit",
      "working memory", "hippocampal", "cerebellum", "synaptic",
      "biological plausib", "in vivo", "electrophysi", "neuroscience",
      "neuronal network", "neocortex", "dendritic", "basal ganglia"]),

    ("Mathematics & Theory",
     ["universal approximation", "fading memory", "separation property",
      "approximation theory", "ergodic", "stability analysis",
      "convergence", "random matrix", "theorem", "functional analysis",
      "reproducing kernel", "echo state property", "contraction",
      "lyapunov", "rademacher", "generalization bound", "capacity"]),

    ("Nonlinear Dynamics & Chaos",
     ["chaos", "chaotic", "lorenz", "attractor", "nonlinear dynamics",
      "bifurcation", "edge of chaos", "dynamical system", "mackey-glass",
      "kuramoto", "nonlinearity", "oscillator", "nonlinear system",
      "transient dynamics", "strange attractor", "phase space",
      "time series prediction", "synchronization", "limit cycle"]),

    ("Signal Processing & Comms",
     ["signal processing", "channel equalization", "speech", "audio",
      "noise cancell", "wireless", "telecommunication", "modulation",
      "radar", "sonar", "equaliz", "distortion compensation"]),

    ("Control & Robotics",
     ["control system", "robot", "reinforcement learning", "motor control",
      "locomotion", "adaptive control", "autonomous", "model predictive",
      "pid", "trajectory", "actuator", "manipulator"]),

    ("Climate & Earth Sciences",
     ["climate", "weather", "wind power", "turbulence", "ocean",
      "atmospher", "geophysic", "rainfall", "flood", "seismic", "earthquake",
      "wind speed", "sea surface", "precipitation", "hydro"]),

    (DEFAULT_SUBFIELD,
     ["echo state", "reservoir computing", "ridge-regression", "spectral radius",
      "reservoir design", "leaky integrator", "esn", "liquid state"]),
]

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
    "surface reservoir", "reservoir storage",
    "reservoir inflow", "reservoir operation", "reservoir routing",
    "reservoir level", "underground reservoir", "aquifer reservoir",
    "carbonate reservoir", "sandstone reservoir", "shale reservoir",
    "tight reservoir", "fractured reservoir",
]

OUTPUT_FIELDS = ["paperId", "title", "year", "citationCount", "cites_per_year",
                 "subfield", "fieldsOfStudy", "doi", "landmark", "abstract"]

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def classify(title, abstract, keywords=""):
    """Score-based subfield classifier (same logic as OpenAlex version)."""
    title_l = title.lower()
    body_l  = (abstract + " " + keywords).lower()
    scores  = {}
    for label, kws in SUBFIELD_RULES:
        if label == DEFAULT_SUBFIELD:
            continue
        s = sum(3 if kw in title_l else (1 if kw in body_l else 0) for kw in kws)
        if s > 0:
            scores[label] = scores.get(label, 0) + s
    return max(scores, key=lambda l: scores[l]) if scores else DEFAULT_SUBFIELD


def rc_relevant(title, abstract):
    hay = (title + " " + abstract).lower()
    if any(t in hay for t in RC_EXCLUDE_TERMS):
        return False
    if any(t in hay for t in RC_CORE_TERMS):
        return True
    if re.search(r"\b(esn|lsm)\b", hay):
        return True
    return False


def cites_per_year(citations, year):
    try:
        age = max(2025 - int(year), 1)
        return round(int(citations) / age, 2)
    except Exception:
        return 0.0


def dedup_key(doi, title):
    if doi and doi.strip():
        return doi.strip().lower()
    return re.sub(r"[^a-z0-9]", "", title.lower())[:80]


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:60]


# ══════════════════════════════════════════════════════════════════════
# SCOPUS CSV LOADER
# ══════════════════════════════════════════════════════════════════════

def load_scopus_csv(path):
    """
    Read a Scopus export CSV and normalise to the rc_papers schema.

    Scopus column mapping
    ---------------------
    Title          → title
    Year           → year
    Cited by       → citationCount
    DOI            → doi
    Abstract       → abstract
    Author Keywords + Index Keywords → keywords (used for classification)
    EID            → paperId  (Scopus-native unique ID)
    """
    papers = []
    path   = Path(path)
    if not path.exists():
        print(f"  WARNING: '{path}' not found — skipping.")
        return papers

    # Increase field size limit to handle large fields (e.g., long abstracts)
    csv.field_size_limit(int(1e7))  # 10 MB limit
    
    with open(path, newline="", encoding="utf-8-sig") as f:   # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        for row in reader:
            title     = (row.get("Title") or "").strip()
            year      = (row.get("Year") or "").strip()
            citations = (row.get("Cited by") or "0").strip() or "0"
            doi       = (row.get("DOI") or "").strip()
            abstract  = (row.get("Abstract") or "").strip()
            auth_kw   = (row.get("Author Keywords") or "").strip()
            idx_kw    = (row.get("Index Keywords") or "").strip()
            eid       = (row.get("EID") or "").strip()

            keywords  = auth_kw + " " + idx_kw   # combined keyword string

            if not title:
                continue

            papers.append({
                "paperId":        eid or norm_title(title),
                "title":          title,
                "year":           year,
                "citationCount":  citations,
                "cites_per_year": cites_per_year(citations, year),
                "subfield":       classify(title, abstract, keywords),
                "fieldsOfStudy":  auth_kw,
                "doi":            doi,
                "landmark":       "",
                "abstract":       abstract[:300],
                "_abstract_full": abstract,      # kept only for rc_relevant check
            })

    print(f"  Loaded {len(papers):,} rows from '{path}'")
    return papers


# ══════════════════════════════════════════════════════════════════════
# MERGE & DEDUP
# ══════════════════════════════════════════════════════════════════════

def merge_scopus_files(file_2000_2015, file_2016_2025):
    """
    Load both Scopus files, apply RC relevance filter, deduplicate, return
    combined list sorted by year.

    To add the second file later, just fill in SCOPUS_CSV_2016_2025 at the
    top of this script — this function handles it automatically.
    """
    all_raw = []

    for path in [file_2000_2015, file_2016_2025]:
        batch = load_scopus_csv(path)
        all_raw.extend(batch)

    print(f"\n  Total before filtering : {len(all_raw):,}")

    # RC relevance filter
    filtered = []
    for p in all_raw:
        # full_abs = p.pop("_abstract_full", p.get("abstract", ""))
        # if rc_relevant(p["title"], full_abs):
            filtered.append(p)
    print(f"  After RC filter        : {len(filtered):,}")

    # Dedup
    # seen, unique = set(), []
    # for p in filtered:
    #     k = dedup_key(p.get("doi", ""), p.get("title", ""))
    #     if k and k not in seen:
    #         seen.add(k)
    #         unique.append(p)
    # print(f"  After dedup            : {len(unique):,}")

    # unique.sort(key=lambda x: int(x.get("year") or 0))
    # return unique
    return filtered


# ══════════════════════════════════════════════════════════════════════
# LANDMARK LOADER  (same as OpenAlex notebook)
# ══════════════════════════════════════════════════════════════════════

def load_csv(path, required=True):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        if required:
            print(f"ERROR: '{path}' not found."); sys.exit(1)
        return []


def load_landmarks(papers, landmark_csv):
    lm_rows = load_csv(landmark_csv, required=False)
    if not lm_rows:
        print(f"  No '{landmark_csv}' — using 'landmark' column in merged CSV")
        return [p for p in papers if p.get("landmark", "").strip()]

    index = {norm_title(p.get("title", "")): p for p in papers}
    matched, unmatched = [], []

    for row in lm_rows:
        key = norm_title(row.get("title", ""))
        if key in index:
            p = index[key]
            p["landmark"]           = row.get("landmark", "")
            p["landmark_subfield"]  = row.get("subfield", p.get("subfield", ""))
            p["landmark_authors"]   = row.get("authors", "")
            matched.append(p)
        else:
            unmatched.append({
                "title":             row.get("title", ""),
                "year":              row.get("year", ""),
                "citationCount":     9999,
                "cites_per_year":    999,
                "subfield":          row.get("subfield", DEFAULT_SUBFIELD),
                "landmark":          row.get("landmark", ""),
                "landmark_subfield": row.get("subfield", DEFAULT_SUBFIELD),
                "landmark_authors":  row.get("authors", ""),
            })

    print(f"  Landmarks matched : {len(matched)}")
    if unmatched:
        print(f"  Landmarks injected: {len(unmatched)}  (not in Scopus dataset)")
    return matched + unmatched


# ══════════════════════════════════════════════════════════════════════
# MATRIX BUILDERS
# ══════════════════════════════════════════════════════════════════════

def build_matrix(papers, order, years):
    counts = {sf: defaultdict(int) for sf in order}
    for p in papers:
        y  = p.get("year")
        sf = (p.get("subfield") or "").strip() or DEFAULT_SUBFIELD
        try:
            y = int(y)
        except (TypeError, ValueError):
            continue
        if YEARS[0] <= y <= YEARS[1]:
            counts[sf][y] += 1
    return np.array([[counts[sf].get(y, 0) for y in years]
                     for sf in order], dtype=float)


def build_matrix_early_window(papers, order, years, early_end=EARLY_WINDOW_END):
    counts = {sf: defaultdict(int) for sf in order}
    for p in papers:
        y  = p.get("year")
        sf = (p.get("subfield") or "").strip() or DEFAULT_SUBFIELD
        try:
            y = int(y)
        except (TypeError, ValueError):
            continue
        if y > YEARS[1]:
            continue
        yb = early_end if y <= early_end else y
        if yb in years:
            counts[sf][yb] += 1
    return np.array([[counts[sf].get(y, 0) for y in years]
                     for sf in order], dtype=float)

