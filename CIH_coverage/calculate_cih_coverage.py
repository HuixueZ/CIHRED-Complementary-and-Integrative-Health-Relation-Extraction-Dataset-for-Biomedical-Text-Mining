#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_cih_coverage.py
=========================
Compute CIH lexicon coverage over biomedical NER datasets.
Reproduces Table 4 from the paper:

  Dataset | Total Docs | Docs with CIH | Covered CIH Concepts | Concept Coverage % | Unique Annotation Terms | Annotated CIH Terms

Column definitions
------------------
  Total Docs              : number of unique documents across all splits
  Docs with CIH           : documents where ≥1 CIH-type annotation matched a CIH concept
  Covered CIH Concepts    : distinct CIH concept names hit by matched annotation terms
  Concept Coverage %      : Covered CIH Concepts / total CIH concepts × 100
  Unique Annotation Terms : unique CIH-type annotation spans (Energy/Manual/Mindbody/
                            CIH_intervention/Usual_Medical_Care) pooled across splits
  Annotated CIH Terms     : those spans matched to a CIH concept via fuzzy matching

Methodology (from calculate_coverage2.py)
------------------------------------------
  - Only CIH entity types are matched (not Chemical / Disease / Gene / Outcome_marker).
  - Matching uses difflib.get_close_matches per entity category (cutoff=0.86).
  - Normalization: lowercase, collapse whitespace, remove punct except - + /
  - Concept coverage counts distinct concept names hit (not lexicon variant strings).
  - All splits pooled before matching → one row per dataset.

Supported input formats
-----------------------
  .txt  : BC5CDR PubTator  (entity types: Chemical, Disease)
  .json : BioC-JSON        (BioRED: passages/annotations)
          CIHRED           (sentence list with top-level entities + doc_id)
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ===========================================================================
# 1.  CIH ENTITY TYPES TO MATCH AGAINST THE LEXICON
#     (all other types — Chemical, Disease, Gene, Outcome_marker — are skipped)
# ===========================================================================

CIH_ENTITY_TYPES = {
    "Energy_therapy",
    "Energy_therapies",
    "Manual_bodybased_therapy",
    "Manual_bodybased_therapies",
    "Mindbody_therapy",
    "Mindbody_therapies",
    "CIH_intervention",
    "CAM_therapies",
    "Usual_Medical_Care",
}


# ===========================================================================
# 2.  NORMALISATION  (from calculate_coverage2.py)
# ===========================================================================

def _norm(s: str) -> str:
    """Lowercase, collapse whitespace, remove punct except - + /"""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-+/]", "", s)
    return s


# ===========================================================================
# 3.  CIH LEXICON LOADER  →  {concept_name: [norm_variant, ...]}
# ===========================================================================

def load_cih_mapping(path: str) -> Dict[str, List[str]]:
    """
    Load CIH lexicon JSON and return {concept: [normalised_variant, ...]}.

    Accepts the nested format from 2023cihlex_manual_filter_mapping.json:
      { category: { variant_key: { "term": "..." } } }
    and the flat format:
      { concept: ["variant", ...] }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping: Dict[str, List[str]] = {}
    for concept, variants in raw.items():
        if isinstance(variants, dict):
            variant_set: Set[str] = set()
            for vk, vv in variants.items():
                if isinstance(vv, dict) and isinstance(vv.get("term"), str):
                    t = vv["term"].strip()
                    if t:
                        variant_set.add(_norm(t))
                elif isinstance(vk, str) and vk.strip():
                    variant_set.add(_norm(vk))
            if variant_set:
                mapping[concept] = sorted(variant_set)
        elif isinstance(variants, list):
            vs = [_norm(v) for v in variants if isinstance(v, str) and v.strip()]
            if vs:
                mapping[concept] = sorted(set(vs))
    return mapping


# ===========================================================================
# 4.  DATASET LOADERS
#     Each loader returns:
#       doc_terms : {doc_id: set(lowercased CIH-type annotation spans)}
#       all_entities: list of {text, type, doc_id}  ← for per-category matching
# ===========================================================================

def _make_record(doc_id: str, text: str, etype: str) -> dict:
    return {"doc_id": doc_id, "text": _norm(text), "type": etype}


def load_bc5cdr(path: str) -> Tuple[Dict[str, Set[str]], List[dict], Dict[str, str]]:
    """
    BC5CDR PubTator format.
    Returns doc_terms, entities, and doc_texts {doc_id: full_text_lowercased}.
    """
    doc_terms: Dict[str, Set[str]] = {}
    doc_texts: Dict[str, str] = {}
    entities: List[dict] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|t|" in line or "|a|" in line:
                parts = line.split("|", 2)
                pmid  = parts[0].strip()
                text  = parts[2].strip() if len(parts) > 2 else ""
                doc_terms.setdefault(pmid, set())
                doc_texts[pmid] = doc_texts.get(pmid, "") + " " + text.lower()
                continue
            if "\t" not in line:
                continue
            parts = line.split("\t")
            if len(parts) >= 5:
                pmid  = parts[0].strip()
                term  = parts[3].strip()
                etype = parts[4].strip()
                doc_terms.setdefault(pmid, set())
                if term:
                    if etype in CIH_ENTITY_TYPES:
                        doc_terms[pmid].add(_norm(term))
                    entities.append(_make_record(pmid, term, etype))
    return doc_terms, entities, doc_texts


def load_bioc_json(path: str) -> Tuple[Dict[str, Set[str]], List[dict], Dict[str, str]]:
    """
    BioC-JSON (BioRED) or CIHRED sentence-list format.
    Returns doc_terms, entities, and doc_texts {doc_id: full_text_lowercased}.
    """
    with open(path, "r") as f:
        data = json.load(f)

    doc_terms: Dict[str, Set[str]] = {}
    doc_texts: Dict[str, str] = {}
    entities:  List[dict] = []

    # --- BioRED: {"documents": [...]} ---
    if isinstance(data, dict):
        for doc in data.get("documents", []):
            doc_id = str(doc.get("id", "")).strip()
            doc_terms.setdefault(doc_id, set())
            doc_texts[doc_id] = ""
            for passage in doc.get("passages", []):
                doc_texts[doc_id] += " " + (passage.get("text", "") or "").lower()
                for ann in passage.get("annotations", []):
                    term  = ann.get("text", "").strip()
                    etype = ""
                    infons = ann.get("infons", {})
                    if isinstance(infons, dict):
                        etype = str(infons.get("type", "")).strip()
                    if term:
                        if etype in CIH_ENTITY_TYPES:
                            doc_terms[doc_id].add(_norm(term))
                        entities.append(_make_record(doc_id, term, etype))

    # --- CIHRED: [{doc_id, entities:[{text, type}]}, ...] ---
    elif isinstance(data, list):
        for record in data:
            if "doc_id" in record:
                doc_id = str(record["doc_id"]).strip()
                doc_terms.setdefault(doc_id, set())
                doc_texts[doc_id] = doc_texts.get(doc_id, "") + " " + (record.get("text", "") or "").lower()
                for e in record.get("entities", []):
                    term  = e.get("text", "").strip()
                    etype = e.get("type", "").strip()
                    if term:
                        if etype in CIH_ENTITY_TYPES:
                            doc_terms[doc_id].add(_norm(term))
                        entities.append(_make_record(doc_id, term, etype))
            else:
                doc_id = str(record.get("id", record.get("pmid", ""))).strip()
                doc_terms.setdefault(doc_id, set())
                doc_texts[doc_id] = ""
                for passage in record.get("passages", []):
                    doc_texts[doc_id] += " " + (passage.get("text", "") or "").lower()
                    for ann in passage.get("annotations", []):
                        term  = ann.get("text", "").strip()
                        etype = ""
                        infons = ann.get("infons", {})
                        if isinstance(infons, dict):
                            etype = str(infons.get("type", "")).strip()
                        if term and etype in CIH_ENTITY_TYPES:
                            doc_terms[doc_id].add(_norm(term))
                            entities.append(_make_record(doc_id, term, etype))

    return doc_terms, entities, doc_texts


def load_dataset(path: str) -> Tuple[Dict[str, Set[str]], List[dict], Dict[str, str]]:
    """Auto-detect format by extension."""
    ext = os.path.splitext(path.lower())[1]
    if ext == ".json":
        return load_bioc_json(path)
    return load_bc5cdr(path)


# ===========================================================================
# 5.  MATCHING  (from calculate_coverage2.py)
#     Match per entity category, track which concept each match belongs to.
# ===========================================================================

def match_entities_to_cih(
    entities: List[dict],
    mapping: Dict[str, List[str]],
    cutoff: float = 0.86,
) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    For each CIH-type annotation term, find the best matching CIH concept
    using difflib.get_close_matches, grouped by entity type category.

    Returns
    -------
    matched_terms       : set of normalised annotation spans that matched
    concepts_hit        : {entity_type: set(concept_names matched)}
    matched_by_cat      : {entity_type: set(matched_term_strings)}
    """
    # Group unique terms by entity type
    by_type: Dict[str, Set[str]] = defaultdict(set)
    for e in entities:
        by_type[e["type"]].add(e["text"])

    # Build flat concept->variants bank
    # all_variants: [(norm_variant, concept_name), ...]
    all_variants: List[Tuple[str, str]] = []
    for concept, variants in mapping.items():
        for v in variants:
            all_variants.append((v, concept))
    bank_strings = [b[0] for b in all_variants]

    matched_terms: Set[str] = set()
    concepts_hit: Dict[str, Set[str]] = defaultdict(set)
    matched_by_cat: Dict[str, Set[str]] = defaultdict(set)

    for etype, terms in by_type.items():
        for term in terms:
            best = difflib.get_close_matches(term, bank_strings, n=1, cutoff=cutoff)
            if best:
                idx = bank_strings.index(best[0])
                concept_name = all_variants[idx][1]
                matched_terms.add(term)
                concepts_hit[etype].add(concept_name)
                matched_by_cat[etype].add(term)

    return matched_terms, dict(concepts_hit), dict(matched_by_cat)


# ===========================================================================
# 6.  COMPUTE ALL SIX TABLE COLUMNS
# ===========================================================================

def _flat_fuzzy_match(all_terms: Set[str], cih_terms_flat: List[str], cutoff: float) -> Set[str]:
    """
    Original logic from calculate_coverage_biored.py:
    Pool ALL annotation spans globally, run flat difflib match against all
    CIH variant strings. No per-category grouping, no concept tracking.
    Used for BC5CDR and BioRED.
    """
    matched: Set[str] = set()
    for term in all_terms:
        if difflib.get_close_matches(term, cih_terms_flat, n=1, cutoff=cutoff):
            matched.add(term)
    return matched


def compute_coverage(
    doc_terms: Dict[str, Set[str]],
    entities: List[dict],
    mapping: Dict[str, List[str]],
    doc_texts: Optional[Dict[str, str]] = None,
    cutoff: float = 0.86,
    docs_with_cih_by_presence: bool = False,
    use_original_flat_match: bool = False,
) -> dict:
    """
    Compute all six columns for Table 4.

    doc_terms  : {doc_id: set(CIH-type annotation spans)} — for CIHRED presence count
    entities   : [{doc_id, text(normed), type}]           — ALL entity types loaded
    mapping    : {concept: [variants]}                    — CIH lexicon
    doc_texts  : {doc_id: full_text_lowercased}           — for BC5CDR/BioRED full-text scan

    docs_with_cih_by_presence:
        True  → Docs with CIH = docs that have any CIH-type entity annotation (CIHRED)
        False → Docs with CIH = docs where full text contains a CIH lexicon variant

    use_original_flat_match:
        True  → BC5CDR / BioRED:
                  - Docs with CIH + Covered Concepts = full-text substring scan
                    (case-insensitive match of every CIH variant against raw doc text)
                  - Unique Ann. Terms / Annotated CIH Terms = CIH-type spans only
        False → CIHRED:
                  - Per entity-type category matching, tracks concept names hit
    """
    total_docs = len(doc_terms)

    # ── Three annotation columns (both paths) ────────────────────────────────
    # unique_ann_terms     : unique spans of ALL entity types (Chemical, Disease, CIH…)
    # unique_ann_cih_terms : unique spans of CIH-type entities only (deduplicated)
    # total_ann_cih_terms  : total CIH-type spans including duplicates across all docs
    cih_entities = [e for e in entities if e["type"] in CIH_ENTITY_TYPES]
    unique_ann_terms:     Set[str] = set(e["text"] for e in entities)
    unique_ann_cih_terms: Set[str] = set(e["text"] for e in cih_entities)
    total_ann_cih_terms:  int      = len(cih_entities)

    if use_original_flat_match:
        # ── BC5CDR / BioRED: full-text substring scan ─────────────────────────
        # For each doc, check if any CIH lexicon variant appears in the raw text
        cih_terms_flat: List[str] = [v for variants in mapping.values() for v in variants]
        # Build concept lookup: variant → concept name
        variant_to_concept: Dict[str, str] = {
            v: concept
            for concept, variants in mapping.items()
            for v in variants
        }

        docs_with_cih    = 0
        covered_concepts: Set[str] = set()

        texts = doc_texts or {}
        for doc_id, text in texts.items():
            norm_text = _norm(text)
            for variant in cih_terms_flat:
                if variant in norm_text:
                    docs_with_cih += 1
                    covered_concepts.add(variant_to_concept[variant])
                    break   # only need one hit to count the doc; keep scanning for concepts

        # Second pass to get ALL covered concepts across all docs
        covered_concepts = set()
        for doc_id, text in texts.items():
            norm_text = _norm(text)
            for variant in cih_terms_flat:
                if variant in norm_text:
                    covered_concepts.add(variant_to_concept[variant])

        total_concepts = len(mapping)
        concept_coverage_pct = round(
            100.0 * len(covered_concepts) / total_concepts, 2
        ) if total_concepts else 0.0

        return {
            "total_docs":              total_docs,
            "docs_with_cih":           docs_with_cih,
            "covered_cih_concepts":    len(covered_concepts),
            "concept_coverage_pct":    concept_coverage_pct,
            "unique_annotation_terms":     len(unique_ann_terms),
            "unique_ann_cih_terms":        len(unique_ann_cih_terms),
            "annotated_cih_terms":         total_ann_cih_terms,
        }

    else:
        # ── CIHRED: per entity-type category matching, tracks concept names ───
        matched_terms, concepts_hit_by_cat, _ = match_entities_to_cih(
            cih_entities, mapping, cutoff=cutoff
        )

        if docs_with_cih_by_presence:
            docs_with_cih = sum(1 for terms in doc_terms.values() if terms)
        else:
            docs_with_cih = sum(
                1 for terms in doc_terms.values()
                if terms & matched_terms
            )

        all_concepts_hit: Set[str] = set()
        for concepts in concepts_hit_by_cat.values():
            all_concepts_hit.update(concepts)

        total_concepts = len(mapping)
        concept_coverage_pct = round(
            100.0 * len(all_concepts_hit) / total_concepts, 2
        ) if total_concepts else 0.0

        return {
            "total_docs":              total_docs,
            "docs_with_cih":           docs_with_cih,
            "covered_cih_concepts":    len(all_concepts_hit),
            "concept_coverage_pct":    concept_coverage_pct,
            "unique_annotation_terms":     len(unique_ann_terms),
            "unique_ann_cih_terms":        len(unique_ann_cih_terms),
            "annotated_cih_terms":         total_ann_cih_terms,
        }


# ===========================================================================
# 7.  ARG PARSER
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calculate_cih_coverage",
        description=(
            "Reproduce Table 4: CIH concept coverage across datasets.\n\n"
            "Only CIH entity types are matched (Energy_therapy, Manual_bodybased_therapy,\n"
            "Mindbody_therapy, CIH_intervention, Usual_Medical_Care).\n"
            "Matching: difflib.get_close_matches per category, cutoff=0.86.\n\n"
            "Use --dataset NAME FILE [FILE ...] to group splits under one label."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Reproduce all three rows of Table 4:
python calculate_cih_coverage.py \\
    --dataset BC5CDR  CDR_Data/CDR_TrainingSet.PubTator.txt CDR_Data/TestSet.tmChem.PubTator.txt CDR_DevelopmentSet.PubTator.txt \\
    --dataset BioRED  BioRED/Train.BioC.JSON BioRED/Dev.BioC.JSON BioRED/Test.BioC.JSON \\
    --dataset CIHRED  CIHRED/train.json CIHRED/valid.json CIHRED/test.json \\
    --mapping_json 2023cihlex_manual_filter_mapping.json \\
    --out_dir results
        """,
    )
    inp = parser.add_argument_group("Input")
    inp.add_argument(
        "--dataset", nargs="+", action="append", metavar="NAME_OR_FILE",
        help="First token = dataset name; rest = file paths. Repeat for each dataset.",
    )
    inp.add_argument(
        "--mapping_json", default="2023cihlex_manual_filter_mapping.json", metavar="JSON",
        help="CIH lexicon JSON (https://github.com/zhang-informatics/CIH). (default: %(default)s)",
    )
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--out_dir", default="cih_coverage_results", metavar="DIR",
        help="Output directory. Created if absent. (default: %(default)s)",
    )
    match = parser.add_argument_group("Matching")
    match.add_argument(
        "--cutoff", type=float, default=0.86, metavar="FLOAT",
        help="difflib cutoff (default: %(default)s, matching calculate_coverage2.py)",
    )
    match.add_argument(
        "--presence_datasets", nargs="*", default=["CIHRED"], metavar="NAME",
        help=(
            "Dataset names where 'Docs with CIH' is counted by CIH entity presence "
            "rather than lexicon match. Use for corpora explicitly selected to contain "
            "CIH (e.g. CIHRED). (default: CIHRED)"
        ),
    )
    return parser


# ===========================================================================
# 8.  MAIN
# ===========================================================================

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.dataset:
        parser.error("No datasets provided. Example: --dataset CIHRED train.json valid.json test.json")
    if not os.path.isfile(args.mapping_json):
        parser.error(f"CIH mapping JSON not found: {args.mapping_json}")

    dataset_groups: List[Tuple[str, List[str]]] = []
    for group in args.dataset:
        if len(group) < 2:
            parser.error(f"--dataset needs a name and at least one file. Got: {group}")
        name, *files = group
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            parser.error(f"Files not found for '{name}': {', '.join(missing)}")
        dataset_groups.append((name, files))

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading CIH mapping: {args.mapping_json}")
    mapping = load_cih_mapping(args.mapping_json)
    print(f"  {len(mapping)} CIH concepts loaded\n")

    summary_rows: List[dict] = []

    for dataset_name, file_paths in dataset_groups:
        print(f"[{dataset_name}]  pooling {len(file_paths)} split(s)...")

        pooled_doc_terms: Dict[str, Set[str]] = {}
        pooled_entities: List[dict] = []
        pooled_doc_texts: Dict[str, str] = {}

        for fp in file_paths:
            dt, ents, texts = load_dataset(fp)
            print(f"  {os.path.basename(fp):45s}  {len(dt):>5,} docs  {len(ents):>6,} CIH entities")
            pooled_doc_terms.update(dt)
            pooled_entities.extend(ents)
            for doc_id, text in texts.items():
                pooled_doc_texts[doc_id] = pooled_doc_texts.get(doc_id, "") + " " + text

        print(f"  {'Pooled:':45s}  {len(pooled_doc_terms):>5,} docs  {len(pooled_entities):>6,} CIH entities")

        by_presence  = dataset_name in args.presence_datasets
        flat_match   = dataset_name not in args.presence_datasets
        metrics = compute_coverage(
            pooled_doc_terms, pooled_entities, mapping,
            doc_texts=pooled_doc_texts,
            cutoff=args.cutoff,
            docs_with_cih_by_presence=by_presence,
            use_original_flat_match=flat_match,
        )

        print(
            f"  Docs with CIH         : {metrics['docs_with_cih']} / {metrics['total_docs']}\n"
            f"  Covered concepts      : {metrics['covered_cih_concepts']} / {len(mapping)} "
            f"({metrics['concept_coverage_pct']:.2f}%)\n"
            f"  Unique Ann.           : {metrics['unique_annotation_terms']}  "
            f"(all entity types, deduplicated)\n"
            f"  Unique Ann. CIH Terms : {metrics['unique_ann_cih_terms']}  "
            f"(CIH-type spans, deduplicated)\n"
            f"  Ann. CIH Terms        : {metrics['annotated_cih_terms']}  "
            f"(all CIH-type spans incl. duplicates)\n"
        )

        # Report any docs that have no CIH-type entities
        no_cih_docs = [doc_id for doc_id, terms in pooled_doc_terms.items() if not terms]
        # if no_cih_docs:
        #     print(f"  ⚠️  {len(no_cih_docs)} doc(s) with NO CIH-type entities:")
        #     for doc_id in sorted(no_cih_docs):
        #         print(f"      - {doc_id}")
        #     print()

        # Per-term detail CSV
        all_terms: Set[str] = set(e["text"] for e in pooled_entities)
        matched_terms, _, _ = match_entities_to_cih(pooled_entities, mapping, cutoff=args.cutoff)
        detail_path = os.path.join(args.out_dir, f"terms__{dataset_name}.csv")
        with open(detail_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["term", "matched_to_cih"])
            for term in sorted(all_terms):
                writer.writerow([term, "yes" if term in matched_terms else "no"])

        summary_rows.append({"dataset": dataset_name, **metrics})

    # Write dataset_summary.csv
    summary_path = os.path.join(args.out_dir, "dataset_summary.csv")
    fields = [
        "dataset", "total_docs", "docs_with_cih",
        "covered_cih_concepts", "concept_coverage_pct",
        "unique_annotation_terms", "unique_ann_cih_terms", "annotated_cih_terms",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print paper Table 4
    print("=" * 115)
    print(f"{'Dataset':<10} {'Total Docs':>11} {'Docs w/ CIH':>12} {'Covered Concepts':>17} "
          f"{'Concept Cov%':>13} {'Unique Ann.':>12} {'Uniq Ann. CIH':>14} {'Ann. CIH Terms':>15}")
    print("-" * 115)
    for r in summary_rows:
        print(
            f"{r['dataset']:<10} "
            f"{r['total_docs']:>11,} "
            f"{r['docs_with_cih']:>12,} "
            f"{r['covered_cih_concepts']:>17,} "
            f"{r['concept_coverage_pct']:>12.2f}% "
            f"{r['unique_annotation_terms']:>12,} "
            f"{r['unique_ann_cih_terms']:>14,} "
            f"{r['annotated_cih_terms']:>15,}"
        )
    print("=" * 115)
    print(f"\nResults written to: {args.out_dir}/")
    print(f"  dataset_summary.csv      ← Table 4")
    print(f"  terms__<dataset>.csv     ← per-term match detail")


if __name__ == "__main__":
    main()