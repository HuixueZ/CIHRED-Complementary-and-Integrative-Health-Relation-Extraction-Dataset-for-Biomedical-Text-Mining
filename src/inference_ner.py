"""
NER Model Inference and Evaluation Script
Performs inference on test data and evaluates entity extraction performance.
"""

import os
import json
import re
import argparse
from collections import defaultdict
from difflib import SequenceMatcher

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model



# ============================================================================
# CONSTANTS
# ============================================================================

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
      'Mindbody_therapy': "Mindbody_therapies",
    'Manual_bodybased_therapy':"Manual_bodybased_therapies",
    'Energy_therapy': "Energy_therapies",
    'CIH_intervention':"CAM_therapies"
}

EXAMPLE = """Text: LLLT at 50 mW was more efficient in the modulation of matrix MMPs and tissue repair.
Entities: LLLT at 50 mW [Energy_therapy]"""



# ============================================================================
# RAG HELPER FUNCTIONS (inlined - no external dependency)
# ============================================================================

def get_similar_examples(row_idx, source_df, topk_indices, topk=1):
    """Get top-k similar examples as formatted string."""
    example_texts = []
    for sim_idx in topk_indices[row_idx][:topk]:
        sim_row = source_df.iloc[sim_idx]
        formatted = format_entities(sim_row)
        example_texts.append(f"Text: {sim_row['text']}\nEntities: {formatted}")
    return "\n".join(example_texts)


def add_similar_examples(df, source_df, topk_indices, topk=1):
    """Add similar examples to dataframe for RAG."""
    examples = []
    for idx in tqdm(range(len(df)), desc="Adding similar examples"):
        example_block = get_similar_examples(idx, source_df, topk_indices, topk=topk)
        examples.append(example_block)
        if idx == 0:
            print(f"Sample RAG example:\n{example_block}\n")
    df["similar_examples"] = examples
    return df

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_ner_data_to_df(path):
    """
    Load NER JSON data to DataFrame, preserving document structure with therapy entities.
    """
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        therapy_entities = [
            entity for entity in item.get("entities", [])
            if entity.get("type") in ALLOWED_TYPES
        ]
        
        if therapy_entities:
            records.append({
                "id": item.get("id", ""),
                "text": item.get("text", ""),
                "entities": therapy_entities,
                "all_entities": item.get("entities", [])
            })
    
    return pd.DataFrame(records)


class DataFrameDataset(Dataset):
    """Custom Dataset for DataFrame-based data."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if isinstance(row['entities'], str):
            try:
                row['entities'] = json.loads(row['entities'])
            except:
                row['entities'] = []
        return row


def collate_fn(batch):
    """Collate function for DataLoader."""
    return pd.DataFrame(batch)


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

def normalize_type(t):
    """Normalize entity type names."""
    if not t:
        return None
    t = t.strip().replace(" ", "_")
    t = TYPE_ALIASES.get(t, t)
    return t if t in ALLOWED_TYPES else None


def format_entities(row):
    """Format ONLY therapy entities as 'entity_text [entity_type]'."""
    ents = row.get("entities", [])
    if isinstance(ents, str):
        try:
            ents = json.loads(ents)
        except Exception:
            ents = []

    out = []
    for e in ents:
        etype = e.get("type", "").strip()
        if etype in ALLOWED_TYPES:
            etext = " ".join(e.get("text", "").split())
            if etext:
                out.append(f"{etext} [{etype}]")
    return ", ".join(out) if out else "None"


def formatting_func_for_inference(row, rag=False):
    """Generate prompt for inference."""
    example = row.get("similar_examples", EXAMPLE) if (rag and "similar_examples" in row) else EXAMPLE

    text = (
        "You are an expert NER system for CIH (Complementary and Integrative Health) therapy extraction.\n"
        "Extract ONLY therapy/treatment entities and assign ONE of the following types:\n"
            "  - Energy_therapy\n"
            "  - Manual_bodybased_therapy\n"
            "  - Mindbody_therapy\n"
            "  - CIH_intervention\n"
            "  - Usual_Medical_Care\n\n"
        "RULES:\n"
        "- Use ONLY the valid types above.\n"
        "- Ignore other entity types such as gene, chemical, biomarker, or outcome.\n"
        "- Output format: therapy_name [therapy_type], therapy_name [therapy_type], ...\n"
        "- If no therapies are found, output EXACTLY: None\n\n"
        f"Example: {example}\n\n"
        f"### Text: {row['text']}\n"
        f"### Entities:"
    )
    return text


# ============================================================================
# ENTITY EXTRACTION AND MATCHING
# ============================================================================

def extract_entities_from_prediction(text, prompt):
    """
    Extract entities from the model's prediction text.
    Expected format: "therapy_name [therapy_type], therapy_name [therapy_type], ..."
    """
    if prompt in text:
        text = text.replace(prompt, "").strip()
    
    text = re.sub(r'^(###\s*)?Entities?:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(###\s*)?Response:\s*', '', text, flags=re.IGNORECASE)
    
    entities = []
    
    if text.strip().lower() in ['none', 'no entities', 'no therapies', '']:
        return entities
    
    pattern = r'([^[,]+)\s*\[([^\]]+)\]'
    matches = re.findall(pattern, text)
    
    for entity_text, entity_type in matches:
        entities.append({
            'text': entity_text.strip(),
            'type': entity_type.strip()
        })
    
    return entities


def find_all_char_spans(hay, needle):
    """Find all occurrences of needle in hay (case-insensitive)."""
    needle = needle.strip()
    if not needle:
        return []
    spans = []
    start = 0
    while True:
        i = hay.lower().find(needle.lower(), start)
        if i < 0:
            break
        spans.append((i, i + len(needle) - 1))
        start = i + 1
    return spans


def attach_char_spans_to_preds(doc_text, pred_entities):
    """Attach character spans to predicted entities."""
    results = []
    for e in pred_entities:
        spans = find_all_char_spans(doc_text, e["text"])
        if not spans:
            results.append({**e, "start_char": None, "end_char": None})
        else:
            for (sc, ec) in spans:
                results.append({**e, "start_char": sc, "end_char": ec})
    return results


def attach_char_spans_to_gold(doc_text, gold_entities):
    """Attach character spans to gold entities."""
    out = []
    for e in gold_entities:
        t = e.get("text", "").strip()
        typ = normalize_type(e.get("type", "").strip())
        spans = find_all_char_spans(doc_text, t)
        if not spans:
            out.append({"text": t, "type": typ, "start_char": None, "end_char": None})
        else:
            sc, ec = spans[0]
            out.append({"text": t, "type": typ, "start_char": sc, "end_char": ec})
    return out


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def clean_entities(ents):
    """Keep only allowed types, strip text."""
    cleaned = []
    for e in ents:
        t = e.get("type", "")
        if t:
            t=t.strip()
        if t not in ALLOWED_TYPES:
            continue
        text = " ".join(e.get("text", "").split())
        if text:
            cleaned.append({"text": text, "type": t})
    return cleaned


def char_similarity(a, b):
    """Calculate character-level similarity."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def word_overlap(a, b):
    """Check if two strings have word overlap."""
    s1, s2 = set(a.lower().split()), set(b.lower().split())
    return len(s1 & s2) > 0


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
    """
    Evaluate predictions against ground truth.
    
    Args:
        all_true: list of list of gold entities per doc
        all_pred: list of list of pred entities per doc
        mode: 'exact' | 'overlap' | 'fuzzy'
    """
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
    print(f"\n=== {res['mode'].upper()} ===")
    print("Micro:", res["micro"])
    print("Macro:", res["macro"])
    print("Weighted:", res["weighted"])
    for t, m in res["per_class"].items():
        print(f"  {t:25s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"(support={m['support']}, TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(base_model, lora_weights=None):
    """Load model and tokenizer with optional LoRA weights."""
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map='auto'
    )

    # Load LoRA weights if provided
    if lora_weights:
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )

    return model, tokenizer


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(model, tokenizer, dataloader, save_file_name, use_rag=False):
    """Run inference on the dataset."""
    all_results = []
    model.eval()
    
    os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
    
    with open(save_file_name, "w") as fw:
        for batch_idx, batch_df in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference"):
            prompts = []
            for _, row in batch_df.iterrows():
                prompt = formatting_func_for_inference(row, use_rag)
                prompts.append(prompt)
            
            try:
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for i, (output, prompt) in enumerate(zip(decoded_outputs, prompts)):
                    row = batch_df.iloc[i]
                    
                    entities = row['entities']
                    if isinstance(entities, str):
                        try:
                            entities = json.loads(entities)
                        except:
                            entities = []
                    elif not isinstance(entities, list):
                        entities = []
                    
                    result_dict = {
                        "id": row.get('id', ''),
                        "text": row['text'],
                        "true_entities": entities,
                        "raw_output": output,
                        "prompt": prompt
                    }
                    
                    all_results.append(result_dict)
                    fw.write(json.dumps(result_dict, default=str) + "\n")
                    fw.flush()
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NER Model Inference and Evaluation")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path or name of the base model")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing LoRA checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint to use (e.g., 'checkpoint-3500')")
    
    # Data arguments
    parser.add_argument("--test_data", type=str, default="test.json",
                        help="Path to test data JSON file")
    parser.add_argument("--train_data", type=str, default="splits_all_claud_by_doc/ner_train.json",
                        help="Path to train data JSON file (for RAG)")
    
    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--rag", action="store_true",
                        help="Use RAG with similar examples")
    parser.add_argument("--topk", type=int, default=1,
                        help="Number of similar examples for RAG")
    parser.add_argument("--embeddings_path", type=str, default="embeddings/similar_test_qwen.npy",
                        help="Path to embeddings for RAG")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="prediction2",
                        help="Directory to save predictions")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Custom output filename (without .json extension)")
    
    args = parser.parse_args()
    
    # Construct LoRA weights path if checkpoint is provided
    lora_weights = None
    if args.checkpoint_dir and args.checkpoint:
        lora_weights = os.path.join(args.checkpoint_dir, args.checkpoint)
    
    # Construct output filename
    if args.output_name:
        save_file_name = os.path.join(args.output_dir, f"{args.output_name}.json")
    elif args.checkpoint:
        save_file_name = os.path.join(args.output_dir, f"predictions_{args.checkpoint}.json")
    else:
        save_file_name = os.path.join(args.output_dir, "predictions.json")
    
    print("=" * 60)
    print("NER Inference Configuration")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Weights: {lora_weights if lora_weights else 'None'}")
    print(f"Test Data: {args.test_data}")
    print(f"Batch Size: {args.batch_size}")
    print(f"RAG: {args.rag}")
    print(f"Output: {save_file_name}")
    print("=" * 60)
    
    # Load data
    print("\nLoading test data...")
    try:
        dfall_test = load_ner_data_to_df(args.test_data)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    if args.rag:
        print("Loading train data for RAG...")
        dfall_train = load_ner_data_to_df(args.train_data)
        embeddings = np.load(args.embeddings_path)
        dfall_test = add_similar_examples(dfall_test, dfall_train, embeddings, topk=args.topk)
    
    dfall_test = dfall_test.reset_index(drop=True)
    
    print(f"Test samples: {len(dfall_test)}")
    print("\nEntity type distribution:")
    all_entity_types = []
    for entities in dfall_test['entities']:
        for entity in entities:
            all_entity_types.append(entity['type'])
    print(pd.Series(all_entity_types).value_counts())
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.base_model, lora_weights)
    
    # Create dataloader
    dataset = DataFrameDataset(dfall_test)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Run inference
    print("\nStarting inference...")
    all_results = run_inference(model, tokenizer, dataloader, save_file_name, args.rag)
    
    print(f"\nInference completed! Results saved to: {save_file_name}")
    
    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    all_true_llm = []
    all_pred_llm = []
    
    for r in all_results:
        doc_text = r["text"]
        
        # Gold entities with spans
        gold = r.get("true_entities", [])
        gold = gold if isinstance(gold, list) else []
        gold_spans = attach_char_spans_to_gold(doc_text, gold)
        
        # Predicted entities with spans
        pred_items = extract_entities_from_prediction(r["raw_output"], r["prompt"])
        pred_items = [{"text": p["text"], "type": normalize_type(p.get("type", "UNKNOWN"))} 
                      for p in pred_items]
        pred_spans = attach_char_spans_to_preds(doc_text, pred_items)
        
        all_true_llm.append(gold_spans)
        all_pred_llm.append(pred_spans)
    
    # Evaluate with different matching modes
    res_exact = evaluate_mode(all_true_llm, all_pred_llm, mode="exact")
    res_overlap = evaluate_mode(all_true_llm, all_pred_llm, mode="overlap")
    res_fuzzy = evaluate_mode(all_true_llm, all_pred_llm, mode="fuzzy", fuzzy_thresh=0.8)
    
    print_summary(res_exact)
    print_summary(res_overlap)
    print_summary(res_fuzzy)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()