"""
Analyze existing NER prediction results
Diagnose extraction issues and evaluate performance
"""

import json
import re
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import pandas as pd


# ============================================================================
# CONSTANTS
# ============================================================================

# ALLOWED_TYPES = [
#     "Energy_therapies",
#     "Manual_bodybased_therapies",
#     "Mindbody_therapies",
#     "CAM_therapies",
#     "Usual_Medical_Care",
# ]
ALLOWED_TYPES = [
    "Energy_therapies",
    "Manual_bodybased_therapies",
    "Mindbody_therapies",
    "CAM_therapies",
    "Usual_Medical_Care",
    'Mindbody_therapy',
    'Manual_bodybased_therapy',
    'Energy_therapy',
    'CIH_intervention'
]

TYPE_ALIASES = {
    "Manual_therapies": "Manual_bodybased_therapies",
    
    "Mind_body_therapies": "Mindbody_therapies",
    "CAM therapies": "CAM_therapies",
    "Cam therapies": "CAM_therapies",
    "Usual Medical Care": "Usual_Medical_Care",
    "Manual_bodytested_therapies": "Manual_bodybased_therapies",  # Common model error
    "Mindbody_tested_therapies": "Mindbody_therapies",  # Common model error
    'Mindbody_therapy': "Mindbody_therapies",
    'Manual_bodybased_therapy':"Manual_bodybased_therapies",
    'Energy_therapy': "Energy_therapies",
    'CIH_intervention': "CAM_therapies"
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_type(t):
    """Normalize entity type names."""
    if not t:
        return None
    t = t.strip().replace(" ", "_")
    t = TYPE_ALIASES.get(t, t)
    return t if t in ALLOWED_TYPES else None


def char_similarity(a, b):
    """Calculate character-level similarity."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def word_overlap(a, b):
    """Check if two strings have word overlap."""
    s1, s2 = set(a.lower().split()), set(b.lower().split())
    return len(s1 & s2) > 0


# ============================================================================
# ENTITY EXTRACTION
# ============================================================================

def extract_entities_from_prediction(text, prompt):
    """
    Extract entities from the model's prediction text.
    Handles messy outputs with extra text, corrections, etc.
    """
    # Remove the input prompt from the output
    if prompt in text:
        text = text.replace(prompt, "").strip()
    
    # Try to extract only the entities line (between "### Entities:" and any subsequent "###" or newlines)
    entities_match = re.search(r'###\s*Entities:\s*([^\n###]+)', text, flags=re.IGNORECASE)
    if entities_match:
        text = entities_match.group(1).strip()
    else:
        # Fallback: remove common prefixes
        text = re.sub(r'^(###\s*)?Entities?:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(###\s*)?Response:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(###\s*)?Output:\s*', '', text, flags=re.IGNORECASE)
    
    # Stop at common delimiters that indicate end of entity list
    stop_patterns = [
        r'\n###',           # New section marker
        r'\nWait',          # Model corrections
        r'\nNote:',         # Model notes
        r'\nCorrection:',   # Model corrections
        r'\n\n',            # Double newline
    ]
    
    for pattern in stop_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
            break
    
    entities = []
    
    # Handle "None" case
    if text.strip().lower() in ['none', 'no entities', 'no therapies', '']:
        return entities
    
    # Pattern to match: entity_name [entity_type]
    pattern = r'([^[\],]+)\s*\[([^\]]+)\]'
    matches = re.findall(pattern, text)
    
    for entity_text, entity_type in matches:
        entity_text = entity_text.strip()
        entity_type = entity_type.strip()
        
        # Skip if entity text is empty or contains obvious errors
        if not entity_text or entity_type.lower() == 'entity_type':
            continue
            
        # Normalize the entity type
        normalized_type = normalize_type(entity_type)
        if normalized_type:
            entities.append({
                'text': entity_text,
                'type': normalized_type
            })
        else:
            # Keep with original type for analysis
            entities.append({
                'text': entity_text,
                'type': entity_type
            })
    
    return entities


# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def diagnose_predictions(all_results, num_samples=10):
    """
    Print diagnostic information about predictions to help debug extraction issues.
    """
    print("\n" + "=" * 80)
    print("PREDICTION DIAGNOSTICS")
    print("=" * 80)
    
    print(f"\nShowing {num_samples} sample predictions:\n")
    
    for i, result in enumerate(all_results[:num_samples]):
        print(f"\n{'─' * 80}")
        print(f"Sample {i+1} - ID: {result.get('id', 'N/A')}")
        print('─' * 80)
        
        print(f"\nText (first 200 chars):")
        print(f"  {result['text'][:200]}...")
        
        print(f"\nTrue entities ({len(result.get('true_entities', []))}):")
        for ent in result.get('true_entities', []):
            print(f"  ✓ {ent['text']:40s} [{ent['type']}]")
        
        print(f"\nRaw model output (first 400 chars):")
        raw = result['raw_output']
        if result['prompt'] in raw:
            actual_output = raw.replace(result['prompt'], "").strip()
        else:
            actual_output = raw
        
        # Show the actual output with line breaks preserved
        output_lines = actual_output[:400].split('\n')
        for line in output_lines:
            if line.strip():
                print(f"  {line}")
        if len(actual_output) > 400:
            print("  ...")
        
        print(f"\nExtracted entities ({len(extract_entities_from_prediction(result['raw_output'], result['prompt']))}):")
        extracted = extract_entities_from_prediction(result['raw_output'], result['prompt'])
        if extracted:
            for ent in extracted:
                print(f"  → {ent['text']:40s} [{ent['type']}]")
        else:
            print("  (none extracted)")
        
        # Show match status
        print(f"\nMatch analysis:")
        true_texts = {e['text'].lower().strip() for e in result.get('true_entities', [])}
        pred_texts = {e['text'].lower().strip() for e in extracted}
        
        matched = true_texts & pred_texts
        missed = true_texts - pred_texts
        extra = pred_texts - matched
        
        if matched:
            print(f"  ✓ Matched: {len(matched)}")
        if missed:
            print(f"  ✗ Missed: {len(missed)} - {list(missed)[:3]}")
        if extra:
            print(f"  ⚠ Extra: {len(extra)} - {list(extra)[:3]}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("EXTRACTION STATISTICS")
    print("=" * 80)
    
    total_true = sum(len(r.get('true_entities', [])) for r in all_results)
    total_predicted = sum(len(extract_entities_from_prediction(r['raw_output'], r['prompt'])) 
                         for r in all_results)
    
    print(f"\nDataset overview:")
    print(f"  Total documents: {len(all_results)}")
    print(f"  Total true entities: {total_true}")
    print(f"  Total predicted entities: {total_predicted}")
    print(f"  Average true entities per doc: {total_true/len(all_results):.2f}")
    print(f"  Average predicted entities per doc: {total_predicted/len(all_results):.2f}")
    
    # Check for common extraction issues
    empty_predictions = sum(1 for r in all_results 
                           if len(extract_entities_from_prediction(r['raw_output'], r['prompt'])) == 0)
    print(f"\nExtraction issues:")
    print(f"  Documents with no predictions: {empty_predictions} ({empty_predictions/len(all_results)*100:.1f}%)")
    
    # Check for type distribution
    all_true_types = []
    all_pred_types = []
    invalid_types = []
    
    for r in all_results:
        for ent in r.get('true_entities', []):
            all_true_types.append(ent['type'])
        
        for ent in extract_entities_from_prediction(r['raw_output'], r['prompt']):
            pred_type = ent['type']
            all_pred_types.append(pred_type)
            if pred_type not in ALLOWED_TYPES:
                invalid_types.append(pred_type)
    
    print(f"\nTrue entity type distribution:")
    if all_true_types:
        type_counts = pd.Series(all_true_types).value_counts()
        for entity_type, count in type_counts.items():
            print(f"  {entity_type:30s}: {count:5d}")
    
    print(f"\nPredicted entity type distribution:")
    if all_pred_types:
        type_counts = pd.Series(all_pred_types).value_counts()
        for entity_type, count in type_counts.items():
            valid_marker = "✓" if entity_type in ALLOWED_TYPES else "✗"
            print(f"  {valid_marker} {entity_type:30s}: {count:5d}")
    
    if invalid_types:
        print(f"\n⚠ Invalid types found (need normalization):")
        invalid_counts = pd.Series(invalid_types).value_counts()
        for entity_type, count in invalid_counts.items():
            print(f"  {entity_type:30s}: {count:5d}")
    
    print("=" * 80)


# ============================================================================
# EVALUATION
# ============================================================================

def clean_entities(ents):
    """Keep only allowed types, strip text."""
    cleaned = []
    for e in ents:
        t = e.get("type", "").strip()
        if t not in ALLOWED_TYPES:
            continue
        text = " ".join(e.get("text", "").split())
        if text:
            cleaned.append({"text": text, "type": t})
    return cleaned


def match_type(gold, pred, mode="exact", fuzzy_thresh=0.8):
    """Return True if gold and pred match under the given mode."""
    if gold["type"] != pred["type"]:
        return False

    gtext, ptext = gold["text"], pred["text"]

    if mode == "exact":
        return gtext.lower().strip() == ptext.lower().strip()
    elif mode == "overlap":
        return word_overlap(gtext, ptext)
    elif mode == "fuzzy":
        return char_similarity(gtext, ptext) >= fuzzy_thresh
    else:
        raise ValueError("mode must be 'exact','overlap','fuzzy'")


def evaluate_mode(all_true, all_pred, mode="exact", fuzzy_thresh=0.8):
    """Evaluate predictions against ground truth."""
    per_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for gold_list, pred_list in zip(all_true, all_pred):
        gold = clean_entities(gold_list)
        pred = clean_entities(pred_list)

        matched_g = set()
        matched_p = set()

        for i, p in enumerate(pred):
            for j, g in enumerate(gold):
                if j in matched_g or i in matched_p:
                    continue
                if match_type(g, p, mode=mode, fuzzy_thresh=fuzzy_thresh):
                    matched_g.add(j)
                    matched_p.add(i)
                    per_type_counts[g["type"]]["tp"] += 1
                    break

        for i, p in enumerate(pred):
            if i not in matched_p:
                per_type_counts[p["type"]]["fp"] += 1
        for j, g in enumerate(gold):
            if j not in matched_g:
                per_type_counts[g["type"]]["fn"] += 1

    # Per-class metrics
    per_class = {}
    total_tp = total_fp = total_fn = 0
    for t in ALLOWED_TYPES:
        c = per_type_counts[t]
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        per_class[t] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": tp + fn, "tp": tp, "fp": fp, "fn": fn
        }

    # Micro
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0

    # Macro / Weighted
    supports = [per_class[t]["support"] for t in ALLOWED_TYPES]
    total_support = sum(supports)
    macro_p = sum(per_class[t]["precision"] for t in ALLOWED_TYPES) / len(ALLOWED_TYPES)
    macro_r = sum(per_class[t]["recall"] for t in ALLOWED_TYPES) / len(ALLOWED_TYPES)
    macro_f1 = sum(per_class[t]["f1"] for t in ALLOWED_TYPES) / len(ALLOWED_TYPES)
    weighted_p = (sum(per_class[t]["precision"] * per_class[t]["support"] for t in ALLOWED_TYPES) / total_support) if total_support else 0
    weighted_r = (sum(per_class[t]["recall"] * per_class[t]["support"] for t in ALLOWED_TYPES) / total_support) if total_support else 0
    weighted_f1 = (sum(per_class[t]["f1"] * per_class[t]["support"] for t in ALLOWED_TYPES) / total_support) if total_support else 0

    return {
        "mode": mode,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
        "per_class": per_class,
        "totals": {"tp": total_tp, "fp": total_fp, "fn": total_fn}
    }


def print_summary(res):
    """Print evaluation summary."""
    print(f"\n{'=' * 80}")
    print(f"{res['mode'].upper()} MATCHING")
    print('=' * 80)
    
    print(f"\nOverall Metrics:")
    print(f"  Micro    - P: {res['micro']['precision']:.4f}  R: {res['micro']['recall']:.4f}  F1: {res['micro']['f1']:.4f}")
    print(f"  Macro    - P: {res['macro']['precision']:.4f}  R: {res['macro']['recall']:.4f}  F1: {res['macro']['f1']:.4f}")
    print(f"  Weighted - P: {res['weighted']['precision']:.4f}  R: {res['weighted']['recall']:.4f}  F1: {res['weighted']['f1']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  {'Type':30s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
    print("  " + "-" * 96)
    
    for t in ALLOWED_TYPES:
        m = res["per_class"][t]
        print(f"  {t:30s} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1']:10.4f} "
              f"{m['support']:10d} {m['tp']:6d} {m['fp']:6d} {m['fn']:6d}")
    
    print(f"\n  Totals: TP={res['totals']['tp']}, FP={res['totals']['fp']}, FN={res['totals']['fn']}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze NER prediction results")
    parser.add_argument("--results_file", type=str, required=True,
                        help="JSONL file with predictions")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to show in diagnostics")
    parser.add_argument("--skip_diagnostics", action="store_true",
                        help="Skip diagnostic output and go straight to evaluation")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NER RESULTS ANALYSIS")
    print("=" * 80)
    print(f"Results file: {args.results_file}")
    
    # Load results
    print("\nLoading results...")
    all_results = []
    with open(args.results_file, 'r') as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))
    
    print(f"Loaded {len(all_results)} predictions")
    
    # Run diagnostics
    if not args.skip_diagnostics:
        diagnose_predictions(all_results, num_samples=args.num_samples)
    
    # Prepare data for evaluation
    print("\nPreparing evaluation data...")
    all_true = []
    all_pred = []
    
    for r in all_results:
        true_ents = r.get("true_entities", [])
        true_ents = true_ents if isinstance(true_ents, list) else []
        
        pred_ents = extract_entities_from_prediction(r['raw_output'], r['prompt'])
        
        all_true.append(true_ents)
        all_pred.append(pred_ents)
    
    # Evaluate with different matching modes
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    res_exact = evaluate_mode(all_true, all_pred, mode="exact")
    res_overlap = evaluate_mode(all_true, all_pred, mode="overlap")
    res_fuzzy = evaluate_mode(all_true, all_pred, mode="fuzzy", fuzzy_thresh=0.8)
    
    print_summary(res_exact)
    print_summary(res_overlap)
    print_summary(res_fuzzy)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()