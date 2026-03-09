import argparse
import json
import io
import re
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================================
# Constants
# ============================================================================
DEFAULT_EXAMPLE = """Context: In sentence 'It exemplifies the upswing of biofield (energy field) therapies that have been taking place recently for a number of clinical conditions, including cancer, pain, arthritis, movement restriction, and energy psychology.', what is the relationship between biofield (energy field) therapies and pain?
            Response:TREATS."""

# ============================================================================
# Data Loading
# ============================================================================
SUBJECT=['CIH_intervention','Energy_therapy','Manual_bodybased_therapy','Mindbody_therapy','Usual_Medical_Care']
OBJECT=['Gene','Chemical','Outcome_marker', 'chronic_pain', 'mental_disorders', 'other_diseases']

def print_scores(labels, preds, title):
    print(f"\n{title}")
    print(f"Precision (macro):  {precision_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(f"Recall (macro):     {recall_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(f"F1 (macro):         {f1_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(f"Precision (micro):  {precision_score(labels, preds, average='micro', zero_division=0):.4f}")
    print(f"Recall (micro):     {recall_score(labels, preds, average='micro', zero_division=0):.4f}")
    print(f"F1 (micro):         {f1_score(labels, preds, average='micro', zero_division=0):.4f}")
    print(f"F1 (weighted):      {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")

    #print_scores(all_labels, all_predictions, "Macro / Micro / Weighted (with None)")

def load_json_or_jsonl(path):
    """Load RE split JSON to DataFrame"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for sent in data:
        if len(sent.get('entities', [])) <= 0:
            continue
        
        sent_text = sent["text"]
        
        # If triples exist, use them directly
        if len(sent.get('triples', [])) > 0:
            for t in sent['triples']:
                records.append({
                    "text": sent_text,
                    "subject": t["subject"],
                    "predicate": t["predicate"],
                    "object": t["object"]
                })
        else:
            # Generate negative examples from entities
            subject_entities = []
            object_entities = []
            
            for entity in sent['entities']:
                if entity['type'] in SUBJECT:
                    subject_entities.append(entity['text'])  # Extract text
                elif entity['type'] in OBJECT:
                    object_entities.append(entity['text'])  # Extract text
            
            # Create negative examples (None relations)
            if len(subject_entities) > 0 and len(object_entities) > 0:
                predicate = 'NONE'
                for subj in subject_entities:
                    for obj in object_entities:
                        records.append({
                            "text": sent_text,
                            "subject": subj,
                            "predicate": predicate,
                            "object": obj
                        })
    
    return pd.DataFrame(records)




# ============================================================================
# Dataset and Collation
# ============================================================================
class DataFrameDataset(Dataset):
    """Simple wrapper around a DataFrame for DataLoader compatibility."""
    
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


def collate_fn(batch):
    """Collate function that returns a DataFrame from batch items."""
    return pd.DataFrame(batch)


# ============================================================================
# Prompt Construction
# ============================================================================
def format_sentence(row):
    """Format sentence with context"""
    return f"In sentence '{row['text']}', what is the relationship between {row['subject']} and {row['object']}?"




def build_prompt(formatted_sentence, example=DEFAULT_EXAMPLE):
    # label=row['predicate'].replace('TC_','').replace('TD_','')
    # if label=='FTPC':
    #     label='FACILITATE'
    # if rag:
    #     example=row['similar_examples']
    # else:
    #     example=EXAMPLE
    text= (
       # "Identify the relationship between the HEAD (disease) and TAIL (outcome marker) in the medical text below. "
         "You are an expert information extractor. Your task is to identify the relationship between a CIH intervention and a disease/biomarker mentioned in the sentence.​\n"
"You should ONLY select one of the relations in ('TREATS','AFFECTS','PREVENTS','FACILITATE','INHIBITS', 'STIMULATES', 'ASSOCIATED_WITH').\n"
"You SHOULD output one of the list relation above."
f"Example: {example}\n"
#f"Examples: {EXAMPLE}"
f"### Context:{formatted_sentence}. ")
#f"### Response: {label}")
    return text


# ============================================================================
# Output Parsing
# ============================================================================
def extract_prediction(text):
    prompt= ("You are an expert information extractor. Your task is to identify the relationship between a CIH intervention and a disease/biomarker mentioned in the sentence.​\n"
            "You should ONLY select one of the relations in ('TREATS','AFFECTS','PREVENTS','FACILITATE','INHIBITS', 'STIMULATES', 'ASSOCIATED_WITH').\n"
            "You SHOULD output one of the list relation above.")
    # Remove the input prompt from the output
    response_indicate='### Response'
    if response_indicate in text:
        text=text.split('### Response')[1].strip()

    if prompt in text:
        text = text.replace(prompt, "").strip()
    
    # Define valid relations
    valid_relations = ['TREATS', 'AFFECTS', 'PREVENTS', 'FACILITATE', 'INHIBITS', 'STIMULATES', 'ASSOCIATED_WITH']
    
    # Look for exact matches first
    for relation in valid_relations:
        if relation in text.upper():
            return relation
    
    # If no exact match, look for partial matches
    text_upper = text.upper()
    for relation in valid_relations:
        if relation[:4] in text_upper:  # Match first 4 characters
            return relation
    
    # Default fallback
    return "None"


def triples_to_pipe_lines(triples):
    """
    Convert list of triple dicts to pipe-separated string format.
    
    Args:
        triples: List of dicts with subject, predicate, object
        
    Returns:
        Formatted string with one triple per line
    """
    if not isinstance(triples, list) or len(triples) == 0:
        return "None"
    
    lines = []
    for t in triples:
        subj = str(t.get("subject", "")).strip()
        pred = str(t.get("predicate", "")).strip()
        obj = str(t.get("object", "")).strip()
        
        # Clean predicate
        if pred:
            pred = pred.replace("TD_", "").replace("TC_", "")
        
        if pred.upper() == "NONE" or not subj or not obj:
            continue
        
        lines.append(f"{subj} | {pred} | {obj}")
    
    return "None" if not lines else "\n".join(lines)


# ============================================================================
# Model Loading
# ============================================================================
def load_model_and_tokenizer(model_id, lora_weights=None, load_in_4bit=True):
    """
    Load the model and tokenizer with optional LoRA weights.
    
    Args:
        model_id: HuggingFace model identifier
        lora_weights: Optional path to LoRA weights
        load_in_4bit: Whether to use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device_map = {"": 0}
    
    
    print(f"Loading model from {model_id}...")
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=device_map,
           # llm_int8_enable_fp32_cpu_offload=True 
        )
        print(model.hf_device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_map
        )
    
    if lora_weights:
        from peft import PeftModel
        print(f"Loading LoRA weights from {lora_weights}...")
        model = PeftModel.from_pretrained(model, lora_weights)
    
    model.eval()
    return model, tokenizer


# ============================================================================
# Inference
# ============================================================================
def run_inference(model, tokenizer, dataloader, output_path, args):
    """
    Run inference on the test dataset and save results.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        dataloader: DataLoader containing test data
        output_path: Path to save results
        args: Command line arguments
    """
    print(f"\nStarting inference...")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Output path: {output_path}\n")

    all_predictions = []
    all_labels = []
    all_results=[]
    
    with open(output_path, "w", encoding="utf-8") as fw:
        for batch_idx, batch_df in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
            # Build prompts for batch
            prompts = []
            for _, row in batch_df.iterrows():
                formatted = format_sentence(row)
                prompt = build_prompt(formatted, example=args.example)
                prompts.append(prompt)
            
            try:
                # Tokenize
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_length
                ).to(model.device)
                
            
    
                # Generate
                with torch.no_grad():
                    gen_kwargs = dict(
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=args.repetition_penalty,
                    )
                    if args.do_sample and args.temperature > 0:
                        gen_kwargs["temperature"] = args.temperature
                    outputs = model.generate(**inputs, **gen_kwargs)
                
                # Decode
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Process each output
                for i, (output, prompt) in enumerate(zip(decoded_outputs, prompts)):
                    row = batch_df.iloc[i]
                    
                    # Parse prediction
                    predicted_relation = extract_prediction(output)
                    
                    # Extract clean output
                    clean_output = output
                    if '### Response:' in output:
                        try:
                            clean_output = output.split('### Response:')[1].strip()
                        except IndexError:
                            pass
                    
                    true_label = row['predicate'].replace('TC_', '').replace('TD_', '')
                    if true_label == 'FTPC':
                        true_label = 'FACILITATE'

                    if predicted_relation=='NONE':
                        predicted_relation='None'
                    if true_label=='NONE':
                        true_label='None'
                   

                    # Save result
                    result = {
                        "sentence": row['text'],
                        "subject":row['subject'],
                        "object": row['object'],
                        'true_label': true_label,
                        #"ground_truth_triples": row['triples'],
                        "predicted": predicted_relation,
                        "raw_output": clean_output
                    }
                     # Clean label
                   

                    all_predictions.append(predicted_relation)
                    all_labels.append(true_label)
                    all_results.append(result)
                    
                    fw.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fw.flush()
                    
            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                continue
    
    print(f"\n✓ Inference completed! Results saved to: {output_path}")


    ##### evaluate
    # Calculate metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    # Overall accuracy
    correct = sum(1 for true, pred in zip(all_labels, all_predictions) if true == pred)
    accuracy = correct / len(all_labels)
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{len(all_labels)})")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, zero_division=0))
    
    # Macro scores
    precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    print(f"\nMacro Metrics:")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall: {recall_macro:.4f}")
    print(f"F1 Score: {f1_macro:.4f}")
    
    # Micro scores
    precision_micro = precision_score(all_labels, all_predictions, average='micro', zero_division=0)
    recall_micro = recall_score(all_labels, all_predictions, average='micro', zero_division=0)
    f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    
    print(f"\nMicro Metrics:")
    print(f"Precision: {precision_micro:.4f}")
    print(f"Recall: {recall_micro:.4f}")
    print(f"F1 Score: {f1_micro:.4f}")


     # Micro scores
    precision_weight = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall_weight = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1_weight = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    print(f"Weighted Metrics:")
    print(f"Precision: {precision_weight:.4f}")
    print(f"Recall: {recall_weight:.4f}")
    print(f"F1 Score: {f1_weight:.4f}")
    
    # Show some examples
    print("\nSample Predictions:")
    print("-" * 80)
    for i in range(min(5, len(all_results))):
        result = all_results[i]
        print(f"True: {result['true_label']} | Predicted: {result['predicted']}")
        print(f"Context: {result['sentence'][:100]}...")
        print(f"Raw output: {result['raw_output'][:150]}...")
        print("-" * 80)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS (Excluding 'None')")
    print("="*50)

    filtered_labels = []
    filtered_preds = []
    for true, pred in zip(all_labels, all_predictions):
        if true.lower() != "none":
            filtered_labels.append(true)
            filtered_preds.append(pred)

    if len(filtered_labels) == 0:
        print("⚠️ No non-'None' examples found; skipping this evaluation.")
    else:
        correct_no_none = sum(1 for t, p in zip(filtered_labels, filtered_preds) if t == p)
        accuracy_no_none = correct_no_none / len(filtered_labels)
        print(f"Overall Accuracy (no None): {accuracy_no_none:.4f} ({correct_no_none}/{len(filtered_labels)})")
        print("\nDetailed Classification Report (no None):")
        print(classification_report(filtered_labels, filtered_preds, zero_division=0))
        print_scores(filtered_labels, filtered_preds, "Macro / Micro / Weighted (no None)")
    

    # Macro scores
    precision_macro = precision_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
    recall_macro = recall_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
    f1_macro = f1_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
    
    print(f"\nMacro Metrics:")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall: {recall_macro:.4f}")
    print(f"F1 Score: {f1_macro:.4f}")
    
    # Micro scores
    precision_micro = precision_score(filtered_labels, filtered_preds, average='micro', zero_division=0)
    recall_micro = recall_score(filtered_labels, filtered_preds, average='micro', zero_division=0)
    f1_micro = f1_score(filtered_labels, filtered_preds, average='micro', zero_division=0)
    
    print(f"\nMicro Metrics:")
    print(f"Precision: {precision_micro:.4f}")
    print(f"Recall: {recall_micro:.4f}")
    print(f"F1 Score: {f1_micro:.4f}")


     # Micro scores
    precision_weight = precision_score(filtered_labels, filtered_preds, average='weighted', zero_division=0)
    recall_weight = recall_score(filtered_labels, filtered_preds, average='weighted', zero_division=0)
    f1_weight = f1_score(filtered_labels, filtered_preds, average='weighted', zero_division=0)
    
    print(f"Weighted Metrics:")
    print(f"Precision: {precision_weight:.4f}")
    print(f"Recall: {recall_weight:.4f}")
    print(f"F1 Score: {f1_weight:.4f}")

    # === Sample predictions ===
    print("\nSample Predictions:")
    print("-" * 80)
    for i in range(min(5, len(all_results))):
        result = all_results[i]
        print(f"True: {result['true_label']} | Predicted: {result['predicted']}")
        print(f"Context: {result['sentence'][:100]}...")
        print(f"Raw output: {result['raw_output'][:150]}...")
        print("-" * 80)


# ============================================================================
# Main
# ============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run triple extraction inference on test data"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        #default="skumar9/Llama-medx_v3.2",
        #default='meta-llama/Meta-Llama-3-8B',
        default="/projects/standard/zhan1386/zhou1742/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct",
       # default='Qwen/Qwen2.5-14B-Instruct',
        #default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to LoRA weights (optional)"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    
    # Data arguments
    parser.add_argument(
        "--test_path",
        type=str,
        #required=True,
        default='test.json',
        help="Path to test JSON/JSONL file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="inference_results2.jsonl",
        help="Path to save inference results"
    )
    
    # Inference arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=2048,
        help="Maximum input sequence length"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (only used if --do_sample)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for generation"
    )
    parser.add_argument(
        "--example",
        type=str,
        default=DEFAULT_EXAMPLE,
        help="Few-shot example to include in prompt"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_path}...")
    df_test = load_json_or_jsonl(args.test_path)
    print(f"Loaded {len(df_test)} test examples")
    
    # Create dataset and dataloader
    dataset = DataFrameDataset(df_test)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        lora_weights=args.lora_weights,
        load_in_4bit=not args.no_4bit
    )
    
    # Run inference
    run_inference(model, tokenizer, dataloader, args.output_path, args)


if __name__ == "__main__":
    main()