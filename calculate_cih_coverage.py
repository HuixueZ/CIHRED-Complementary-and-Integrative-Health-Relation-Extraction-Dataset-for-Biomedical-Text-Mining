#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_cih_coverage.py
=========================
Compute CIH (Complementary and Integrative Health) lexicon coverage over
biomedical NER datasets: BC5CDR (PubTator .txt) and BioRED / any BioC-JSON file.

Supported datasets
------------------
  - BC5CDR  : PubTator flat-text format (.txt)
  - BioRED  : BioC-JSON format (.json)
  - DocRED-Bio / other BioC-JSON datasets

Coverage metrics computed
-------------------------
  1. Free-text coverage   – how many documents contain ≥1 CIH variant mention
  2. Concept coverage     – how many CIH concepts are hit across the corpus
  3. NER term coverage    – what fraction of gold-annotated entity spans map to
                            a CIH concept (fuzzy / hybrid matching)

Usage
-----
  See README.md or run:  python calculate_cih_coverage.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional: rapidfuzz gives faster fuzzy matching; fall back to difflib
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    import difflib


# ===========================================================================
# 1.  TEXT NORMALISATION
# ===========================================================================

_RE_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_RE_WS    = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_WS.sub(" ", text)
    return text.strip()


# ===========================================================================
# 2.  DATASET LOADERS
# ===========================================================================

DocRecord = Dict  # {pmid, title, abstract, text, entities: [(term, type)]}


def load_bc5cdr(path: str) -> List[DocRecord]:
    """
    Load a BC5CDR PubTator file (blocks separated by blank lines).
    Entity types retained: Chemical, Disease.
    """
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read().strip()

    docs: List[DocRecord] = []
    for block in content.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        pmid = title = abstract = ""
        entities: List[Tuple[str, str]] = []
        for line in block.splitlines():
            if "|t|" in line:
                pmid, title = line.split("|t|", 1)
                pmid = pmid.strip(); title = title.strip()
            elif "|a|" in line:
                parts = line.split("|a|", 1)
                pmid = pmid or parts[0].strip()
                abstract = parts[1].strip()
            elif "\t" in line and "|t|" not in line and "|a|" not in line:
                cols = line.split("\t")
                if len(cols) >= 6 and cols[4] in ("Chemical", "Disease"):
                    entities.append((cols[3], cols[4]))
        if pmid:
            docs.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "text": f"{title} {abstract}".strip(),
                "entities": entities,
            })
    return docs


def load_bioc_json(path: str) -> List[DocRecord]:
    """
    Load a BioC-style JSON file (BioRED, DocRED-Bio, etc.).
    Entity type is read from annotation infons['type'].
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    docs: List[DocRecord] = []
    for doc in data.get("documents", []):
        doc_id = str(doc.get("id", "")).strip()
        passages = doc.get("passages", []) or []
        text_parts: List[str] = []
        entities: List[Tuple[str, str]] = []

        for passage in passages:
            ptxt = passage.get("text", "")
            if ptxt:
                text_parts.append(ptxt)
            for ann in passage.get("annotations", []) or []:
                term = ann.get("text", "")
                infons = ann.get("infons", {}) or {}
                etype = str(infons.get("type", "")).strip() if isinstance(infons, dict) else ""
                if term:
                    entities.append((term, etype))

        if doc_id:
            docs.append({
                "pmid": doc_id,
                "title": "",
                "abstract": "",
                "text": " ".join(text_parts).strip(),
                "entities": entities,
            })
    return docs


def load_dataset(path: str) -> List[DocRecord]:
    """Auto-detect format by file extension (.json → BioC, .txt → BC5CDR)."""
    ext = os.path.splitext(path.lower())[1]
    if ext == ".json":
        return load_bioc_json(path)
    return load_bc5cdr(path)


# ===========================================================================
# 3.  CIH LEXICON / MAPPING LOADER
# ===========================================================================

def load_cih_mapping(path: str) -> Dict[str, List[str]]:
    """
    Load the CIH lexicon JSON.

    Accepted formats:
      { concept: [variant, ...] }
      { category: { variant_key: { "term": "...", ... }, ... }, ... }
    Returns: { concept: [variant_str, ...] }
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    mapping: Dict[str, List[str]] = {}
    for concept, sub in raw.items():
        variants: List[str] = []
        if isinstance(sub, list):
            variants = [v for v in sub if isinstance(v, str) and v.strip()]
        elif isinstance(sub, dict):
            for vkey, meta in sub.items():
                if isinstance(meta, dict) and isinstance(meta.get("term"), str):
                    t = meta["term"].strip()
                    if t:
                        variants.append(t)
                elif isinstance(vkey, str) and vkey.strip():
                    variants.append(vkey.strip())
        if variants:
            mapping[concept] = sorted(set(variants))
    return mapping


# ===========================================================================
# 4.  VARIANT EXPANSION & REGEX INDEX
# ===========================================================================

def _expand_variant(variant: str) -> List[str]:
    """
    Generate surface-form expansions for a CIH variant string.
    Covers hyphenation, slash, plural, and a few domain expansions
    (electroacupuncture → acupuncture, LLLT → light therapy, etc.).
    """
    v = variant.strip()
    forms = {v}
    base = re.sub(r"[-/]", " ", v).strip()
    forms.update({base, re.sub(r"\s+", " ", base), base.replace(" ", "")})

    if base.endswith(" therapy"):
        forms.add(base[:-8].strip())
    else:
        forms.add(base + " therapy")
    if not base.endswith("s"):
        forms.add(base + "s")

    nb = base.lower()
    if "electroacupuncture" in nb or "electro acupuncture" in nb:
        forms.add("acupuncture")
    if any(k in nb for k in ("photobiomodulation", "low level laser", "lllt")):
        forms.update({"light therapy", "phototherapy"})

    return sorted({normalize(f) for f in forms if f.strip()})


def _make_regex(norm_variant: str) -> re.Pattern:
    parts = norm_variant.split()
    if not parts:
        return re.compile(r"$^")
    pattern = r"\b" + r"\s+".join(map(re.escape, parts)) + r"\b"
    return re.compile(pattern, re.IGNORECASE)


def build_variant_index(
    mapping: Dict[str, List[str]]
) -> Tuple[List, Dict[str, re.Pattern], Dict[str, List[Tuple[str, str]]]]:
    """
    Build lookup structures for fast coverage computation.

    Returns
    -------
    flat      : [(concept, raw_variant, norm_variant), ...]
    vregex    : {norm_variant: compiled_regex}
    v2owners  : {norm_variant: [(concept, raw_variant), ...]}
    """
    flat: List[Tuple[str, str, str]] = []
    v2owners: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for concept, variants in mapping.items():
        for v in variants:
            for expanded in _expand_variant(v):
                flat.append((concept, v, expanded))
                v2owners[expanded].append((concept, v))

    vregex = {vn: _make_regex(vn) for vn in v2owners}
    return flat, vregex, v2owners


# ===========================================================================
# 5.  FUZZY / HYBRID MATCHING
# ===========================================================================

def _fuzzy_ratio(a: str, b: str) -> float:
    if _HAS_RAPIDFUZZ:
        return _fuzz.token_set_ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _jaccard(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _hybrid_score(a: str, b: str, alpha: float = 0.70) -> float:
    return alpha * _fuzzy_ratio(a, b) + (1 - alpha) * _jaccard(a, b)


def _token_subset_ratio(a_norm: str, b_norm: str) -> float:
    A, B = set(a_norm.split()), set(b_norm.split())
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def match_term_to_cih(
    term: str,
    flat: List,
    alpha: float = 0.70,
    threshold: float = 0.65,
    jaccard_floor: float = 0.30,
) -> Tuple[bool, Optional[str], Optional[str], float]:
    """
    Three-stage cascade matcher:
      1. Exact normalised match
      2. Substring / token-subset match (≥0.80)
      3. Hybrid fuzzy + Jaccard score (≥ threshold & jaccard_floor)

    Returns (matched, concept, raw_variant, score).
    """
    tnorm = normalize(term)

    # Stage 1: exact
    for concept, v_raw, v_norm in flat:
        if tnorm == v_norm:
            return True, concept, v_raw, 1.0

    # Stage 2: substring / token subset
    for concept, v_raw, v_norm in flat:
        if v_norm in tnorm or tnorm in v_norm or _token_subset_ratio(tnorm, v_norm) >= 0.80:
            return True, concept, v_raw, 0.95

    # Stage 3: fuzzy hybrid
    best_score, best_concept, best_v_raw, best_jaccard = -1.0, None, None, 0.0
    for concept, v_raw, v_norm in flat:
        score = _hybrid_score(tnorm, v_norm, alpha=alpha)
        jac   = _jaccard(tnorm, v_norm) if (" " in tnorm or " " in v_norm) else 1.0
        if score > best_score:
            best_score, best_concept, best_v_raw, best_jaccard = score, concept, v_raw, jac

    j_floor = jaccard_floor if (" " in tnorm or " " in (best_v_raw or "")) else 0.0
    if best_score >= threshold and best_jaccard >= j_floor:
        return True, best_concept, best_v_raw, best_score

    return False, None, None, best_score


# ===========================================================================
# 6.  COVERAGE METRICS
# ===========================================================================

def compute_free_text_coverage(
    docs: List[DocRecord],
    vregex: Dict[str, re.Pattern],
    v2owners: Dict[str, List[Tuple[str, str]]],
) -> Tuple[Counter, Counter, int, float, Counter]:
    """
    Scan free text of each document for CIH variant regex hits.

    Returns
    -------
    concept_hits  : Counter {concept: total_regex_hits}
    variant_hits  : Counter {norm_variant: total_regex_hits}
    docs_with_hit : int
    doc_coverage_pct : float
    doc_hits      : Counter {pmid: total_hits_in_doc}
    """
    concept_hits: Counter = Counter()
    variant_hits: Counter = Counter()
    doc_hits:     Counter = Counter()
    docs_with_hit = 0

    for doc in docs:
        doc_norm = normalize(doc["text"])
        any_hit = False
        for v_norm, rgx in vregex.items():
            n = len(rgx.findall(doc_norm))
            if n > 0:
                variant_hits[v_norm] += n
                any_hit = True
                for concept, _ in v2owners[v_norm]:
                    concept_hits[concept] += n
                doc_hits[doc["pmid"]] += n
        if any_hit:
            docs_with_hit += 1

    total = len(docs)
    doc_cov_pct = 100.0 * docs_with_hit / total if total else 0.0
    return concept_hits, variant_hits, docs_with_hit, doc_cov_pct, doc_hits


def compute_ner_term_coverage(
    docs: List[DocRecord],
    flat: List,
    alpha: float = 0.70,
    threshold: float = 0.65,
    jaccard_floor: float = 0.30,
) -> Tuple[float, int, int, List[Dict], Counter]:
    """
    Map each unique gold-annotated entity span to the CIH lexicon.

    Returns
    -------
    coverage_pct   : float
    matched        : int
    total_unique   : int
    term_rows      : list of dicts (for CSV export)
    concept_hits   : Counter
    """
    unique_terms: Dict[str, str] = {}
    for doc in docs:
        for term, _ in doc["entities"]:
            key = normalize(term)
            if key and key not in unique_terms:
                unique_terms[key] = term

    matched = 0
    concept_hits: Counter = Counter()
    term_rows: List[Dict] = []

    for term in unique_terms.values():
        ok, concept, variant, score = match_term_to_cih(
            term, flat, alpha=alpha, threshold=threshold, jaccard_floor=jaccard_floor
        )
        term_rows.append({
            "term":    term,
            "matched": "yes" if ok else "no",
            "concept": concept or "",
            "variant": variant or "",
            "score":   f"{score:.3f}",
        })
        if ok and concept:
            matched += 1
            concept_hits[concept] += 1

    total = len(unique_terms)
    coverage_pct = round(100.0 * matched / total, 2) if total else 0.0
    return coverage_pct, matched, total, term_rows, concept_hits


def compute_concept_coverage(
    concept_hits: Counter, mapping: Dict[str, List[str]]
) -> Tuple[float, int]:
    """Return (concept_coverage_pct, n_covered_concepts)."""
    covered = sum(1 for c in mapping if concept_hits.get(c, 0) > 0)
    pct = 100.0 * covered / len(mapping) if mapping else 0.0
    return round(pct, 2), covered


# ===========================================================================
# 7.  OUTPUT WRITERS
# ===========================================================================

_SUMMARY_FIELDS = [
    "total_docs", "docs_with_cih", "doc_coverage_pct",
    "total_concepts", "covered_concepts", "concept_coverage_pct",
    "ner_total_unique_terms", "ner_matched_terms", "ner_term_coverage_pct",
    "alpha", "threshold", "jaccard_floor",
]


def write_summary_csv(
    path: str,
    *,
    total_docs: int,
    docs_with_hit: int,
    doc_cov_pct: float,
    total_concepts: int,
    covered_concepts: int,
    concept_cov_pct: float,
    ner_total: object = "",
    ner_matched: object = "",
    ner_pct: object = "",
    alpha: float,
    threshold: float,
    jaccard_floor: float,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({
            "total_docs":              total_docs,
            "docs_with_cih":           docs_with_hit,
            "doc_coverage_pct":        round(doc_cov_pct, 2),
            "total_concepts":          total_concepts,
            "covered_concepts":        covered_concepts,
            "concept_coverage_pct":    round(concept_cov_pct, 2),
            "ner_total_unique_terms":  ner_total,
            "ner_matched_terms":       ner_matched,
            "ner_term_coverage_pct":   ner_pct,
            "alpha":                   alpha,
            "threshold":               threshold,
            "jaccard_floor":           jaccard_floor,
        })


def _write_counter_csv(path: str, counter: Counter, col1: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([col1, "count"])
        writer.writerows(counter.most_common())


def write_threshold_sweep(
    path: str,
    all_docs: List[DocRecord],
    flat: List,
    thresholds: List[float],
    alpha: float,
    jaccard_floor: float,
) -> None:
    """Write a CSV showing NER term coverage at several threshold values."""
    unique_terms: Dict[str, str] = {}
    for doc in all_docs:
        for term, _ in doc["entities"]:
            key = normalize(term)
            if key and key not in unique_terms:
                unique_terms[key] = term

    rows = []
    for th in thresholds:
        matched = sum(
            1 for t in unique_terms.values()
            if match_term_to_cih(t, flat, alpha=alpha, threshold=th, jaccard_floor=jaccard_floor)[0]
        )
        cov = round(100.0 * matched / len(unique_terms), 2) if unique_terms else 0.0
        rows.append({
            "threshold":         th,
            "term_coverage_pct": cov,
            "matched_terms":     matched,
            "total_terms":       len(unique_terms),
        })

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["threshold", "term_coverage_pct", "matched_terms", "total_terms"]
        )
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# 8.  MAIN ENTRY POINT
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calculate_cih_coverage",
        description=(
            "Compute CIH lexicon coverage over biomedical NER datasets.\n\n"
            "Accepts BC5CDR PubTator (.txt) and/or BioRED / BioC-JSON (.json) files.\n"
            "Outputs per-file and combined summary CSVs plus optional detail tables."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# BioRED (train + dev + test)
python calculate_cih_coverage.py \\
    --infile BioRED/Train.BioC.JSON BioRED/Dev.BioC.JSON BioRED/Test.BioC.JSON \\
    --mapping_json 2023cihlex_manual_filter_mapping.json \\
    --out_dir results/biored --map_ner

# BC5CDR corpus
python calculate_cih_coverage.py \\
    --infile CDR_TrainingSet.PubTator.txt TestSet.tmChem.PubTator.txt \\
    --mapping_json 2023cihlex_manual_filter_mapping.json \\
    --out_dir results/bc5cdr --map_ner --sweep

# Mixed input (auto-detected by extension)
python calculate_cih_coverage.py \\
    --infile Train.BioC.JSON CDR_TrainingSet.PubTator.txt \\
    --mapping_json 2023cihlex_manual_filter_mapping.json \\
    --out_dir results/mixed
        """,
    )

    # --- Input ---
    inp = parser.add_argument_group("Input")
    inp.add_argument(
        "--infile", nargs="+", metavar="FILE",
        help="One or more input files (.txt for BC5CDR, .json for BioC/BioRED). "
             "Format is auto-detected by extension.",
    )
    inp.add_argument(
        "--mapping_json", default="2023cihlex_manual_filter_mapping.json", metavar="JSON",
        help="Path to CIH lexicon/mapping JSON. (default: %(default)s)",
    )

    # --- Output ---
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--out_dir", default="cih_coverage_results", metavar="DIR",
        help="Directory to write output CSV files. Created if absent. (default: %(default)s)",
    )

    # --- Matching ---
    match = parser.add_argument_group("Matching parameters")
    match.add_argument(
        "--alpha", type=float, default=0.70, metavar="FLOAT",
        help="Weight for fuzzy ratio vs Jaccard in hybrid score. (default: %(default)s)",
    )
    match.add_argument(
        "--threshold", type=float, default=0.65, metavar="FLOAT",
        help="Minimum hybrid score to count as a match. (default: %(default)s)",
    )
    match.add_argument(
        "--jaccard_floor", type=float, default=0.30, metavar="FLOAT",
        help="Minimum Jaccard similarity required alongside threshold. (default: %(default)s)",
    )

    # --- Optional outputs ---
    opt = parser.add_argument_group("Optional outputs")
    opt.add_argument(
        "--map_ner", action="store_true",
        help="Map gold NER spans to CIH concepts and write per-file + combined CSV.",
    )
    opt.add_argument(
        "--sweep", action="store_true",
        help="Write sweep.csv showing NER term coverage across multiple thresholds.",
    )

    return parser


def process_files(
    input_paths: List[str],
    mapping: Dict[str, List[str]],
    flat: List,
    vregex: Dict,
    v2owners: Dict,
    out_dir: str,
    args: argparse.Namespace,
) -> List[DocRecord]:
    """Run per-file coverage and write summary CSVs. Returns all docs combined."""
    all_docs: List[DocRecord] = []

    for fp in input_paths:
        print(f"  Loading: {fp}")
        docs = load_dataset(fp)
        all_docs.extend(docs)

        ch, vh, dwh, dcp, dh = compute_free_text_coverage(docs, vregex, v2owners)
        concept_cov_pct, covered = compute_concept_coverage(ch, mapping)

        ner_pct = ner_matched = ner_total = ""
        if args.map_ner:
            ner_pct, ner_matched, ner_total, term_rows, _ = compute_ner_term_coverage(
                docs, flat,
                alpha=args.alpha, threshold=args.threshold, jaccard_floor=args.jaccard_floor,
            )
            base = os.path.splitext(os.path.basename(fp))[0]
            ner_path = os.path.join(out_dir, f"ner_terms__{base}.csv")
            with open(ner_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["term", "matched", "concept", "variant", "score"])
                writer.writeheader()
                writer.writerows(term_rows)
            print(f"    → NER mapping written: {ner_path}")

        base = os.path.splitext(os.path.basename(fp))[0]
        summary_path = os.path.join(out_dir, f"summary__{base}.csv")
        write_summary_csv(
            summary_path,
            total_docs=len(docs), docs_with_hit=dwh, doc_cov_pct=dcp,
            total_concepts=len(mapping), covered_concepts=covered, concept_cov_pct=concept_cov_pct,
            ner_total=ner_total, ner_matched=ner_matched, ner_pct=ner_pct,
            alpha=args.alpha, threshold=args.threshold, jaccard_floor=args.jaccard_floor,
        )
        print(
            f"    docs={len(docs)}  doc_cov={dcp:.1f}%  "
            f"concept_cov={concept_cov_pct:.1f}%"
            + (f"  ner_cov={ner_pct:.1f}%" if args.map_ner else "")
        )

    return all_docs


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # --- Validate inputs ---
    if not args.infile:
        parser.error("No input files provided. Use --infile FILE [FILE ...]")

    missing = [p for p in args.infile if not os.path.isfile(p)]
    if missing:
        parser.error(f"Files not found: {', '.join(missing)}")

    if not os.path.isfile(args.mapping_json):
        parser.error(f"CIH mapping JSON not found: {args.mapping_json}")

    # De-duplicate while preserving order
    seen: set = set()
    input_paths: List[str] = []
    for p in args.infile:
        if p not in seen:
            seen.add(p); input_paths.append(p)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load lexicon ---
    print(f"Loading CIH mapping: {args.mapping_json}")
    mapping = load_cih_mapping(args.mapping_json)
    flat, vregex, v2owners = build_variant_index(mapping)
    print(f"  {len(mapping)} concepts | {len(vregex)} expanded variants\n")

    # --- Per-file ---
    print("Processing files...")
    all_docs = process_files(input_paths, mapping, flat, vregex, v2owners, args.out_dir, args)

    # --- Combined ---
    print("\nComputing combined coverage...")
    ch, vh, dwh, dcp, dh = compute_free_text_coverage(all_docs, vregex, v2owners)
    concept_cov_pct, covered = compute_concept_coverage(ch, mapping)

    combined_summary = os.path.join(args.out_dir, "summary__combined.csv")
    write_summary_csv(
        combined_summary,
        total_docs=len(all_docs), docs_with_hit=dwh, doc_cov_pct=dcp,
        total_concepts=len(mapping), covered_concepts=covered, concept_cov_pct=concept_cov_pct,
        alpha=args.alpha, threshold=args.threshold, jaccard_floor=args.jaccard_floor,
    )

    _write_counter_csv(os.path.join(args.out_dir, "concept_hits.csv"),  ch, "concept")
    _write_counter_csv(os.path.join(args.out_dir, "variant_hits.csv"),  vh, "variant_norm")
    _write_counter_csv(os.path.join(args.out_dir, "doc_hits.csv"),       dh, "pmid")

    print(
        f"  Combined: docs={len(all_docs)}  doc_cov={dcp:.1f}%  "
        f"concept_cov={concept_cov_pct:.1f}%"
    )

    # --- Optional: NER term mapping (combined) ---
    if args.map_ner:
        ner_pct, ner_matched, ner_total, term_rows, _ = compute_ner_term_coverage(
            all_docs, flat,
            alpha=args.alpha, threshold=args.threshold, jaccard_floor=args.jaccard_floor,
        )
        ner_combined = os.path.join(args.out_dir, "ner_terms__combined.csv")
        with open(ner_combined, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["term", "matched", "concept", "variant", "score"])
            writer.writeheader()
            writer.writerows(term_rows)
        print(f"  NER term coverage: {ner_matched}/{ner_total} ({ner_pct:.1f}%)")

    # --- Optional: threshold sweep ---
    if args.sweep:
        sweep_path = os.path.join(args.out_dir, "sweep.csv")
        write_threshold_sweep(
            sweep_path, all_docs, flat,
            thresholds=[0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80],
            alpha=args.alpha, jaccard_floor=args.jaccard_floor,
        )
        print(f"  Threshold sweep written: {sweep_path}")

    # --- Summary ---
    print(f"\nDone. Results written to: {args.out_dir}/")
    print("  summary__combined.csv")
    print("  summary__<file>.csv   (one per input)")
    print("  concept_hits.csv, variant_hits.csv, doc_hits.csv")
    if args.map_ner:
        print("  ner_terms__<file>.csv, ner_terms__combined.csv")
    if args.sweep:
        print("  sweep.csv")


if __name__ == "__main__":
    main()
