"""
NER Model Training Script with LoRA/QLoRA
Fine-tune language models for CIH therapy entity extraction.
"""

import os
import json
import argparse

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


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
}

EXAMPLE = """Text: LLLT at 50 mW was more efficient in the modulation of matrix MMPs and tissue repair.
Entities: LLLT at 50 mW [Energy_therapy]"""


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


def clean_text(s):
    """Minimal text cleaning."""
    return " ".join((s or "").strip().split())


# ============================================================================
# ENTITY FORMATTING
# ============================================================================

def format_entities(row):
    """
    Format ONLY therapy entities as 'entity_text [entity_type]'.
    Ignore other entity types (e.g., gene, chemical).
    """
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


def format_entities_simple(row):
    """Format entities as simple comma-separated list without types."""
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
                out.append(etext)
    return ", ".join(out) if out else "None"


# ============================================================================
# RAG FUNCTIONS
# ============================================================================

def get_similar_examples(row_idx, source_df, topk_indices, topk=1):
    """Get top-k similar examples as formatted string."""
    example_texts = []
    for sim_idx in topk_indices[row_idx][:topk]:
        sim_row = source_df.iloc[sim_idx]
        formatted_entities = format_entities(sim_row)
        example_texts.append(f"Text: {sim_row['text']}\nEntities: {formatted_entities}")
    return "\n".join(example_texts)


def add_similar_examples(df, source_df, topk_indices, topk=1):
    """Add similar examples to dataframe for RAG."""
    examples = []
    for idx in tqdm(range(len(df)), desc="Adding similar examples"):
        example_block = get_similar_examples(idx, source_df, topk_indices, topk=topk)
        examples.append(example_block)
        if idx == 0:
            print(f"Sample RAG example:\n{example_block}\n")
    df['similar_examples'] = examples
    return df


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

def formatting_func(row, rag=False, include_types=True):
    """
    Create training prompt with optional RAG examples.
    
    Args:
        row: DataFrame row with 'text' and 'entities'
        rag: Whether to include similar examples
        include_types: Whether to include entity types in output format
    """
    example = row.get("similar_examples", EXAMPLE) if (rag and "similar_examples" in row) else EXAMPLE
    formatted_entities = format_entities(row) if include_types else format_entities_simple(row)

    if include_types:
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
            f"### Entities: {formatted_entities}"
        )
    else:
        text = (
            "You are an expert NER system for CIH (Complementary and Integrative Health) therapy extraction.\n"
            "RULES:\n"
            "- Output format: therapy_name, therapy_name, ...\n"
            "- If no therapies are found, output EXACTLY: None\n\n"
            f"Example: {example}\n\n"
            f"### Text: {row['text']}\n"
            f"### Entities: {formatted_entities}"
        )
    
    return {"text": text}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ner_json_to_df(path):
    """Load NER JSON file to DataFrame."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        records.append({
            "id": item.get("id", ""),
            "text": item.get("text", ""),
            "entities": item.get("entities", [])
        })
    return pd.DataFrame(records)


def prepare_data(df, rag=False, include_types=True):
    """Convert DataFrame to Hugging Face dataset with formatting."""
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset.map(lambda x: formatting_func(x, rag, include_types))


def add_generation_data(dfall_train, generation_file, num_samples=100):
    """Add synthetic generation data if available."""
    if os.path.exists(generation_file):
        print(f"Adding synthetic data from {generation_file}")
        generate_train = pd.read_csv(generation_file)
        select_sample = generate_train.sample(min(num_samples, len(generate_train)))
        augmented_train = pd.concat([dfall_train, select_sample], ignore_index=True)
        print(f"Added {len(select_sample)} synthetic samples")
        return augmented_train
    else:
        print(f"Synthetic data file not found: {generation_file}")
    return dfall_train


# ============================================================================
# DATA ANALYSIS
# ============================================================================

def analyze_entity_distribution(df, split_name="Dataset"):
    """Analyze and print entity type distribution."""
    print(f"\n{'='*60}")
    print(f"{split_name} Entity Analysis")
    print('='*60)
    
    therapy_entity_types = []
    all_entity_types = set()
    
    for _, row in df.iterrows():
        entities = row['entities']
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except:
                entities = []
        
        for entity in entities:
            entity_type = entity.get('type', '')
            all_entity_types.add(entity_type)
            if entity_type in ALLOWED_TYPES:
                therapy_entity_types.append(entity_type)
    
    print(f"\nAll entity types found: {sorted(all_entity_types)}")
    
    if therapy_entity_types:
        entity_counts = pd.Series(therapy_entity_types).value_counts()
        print(f"\nCIH Therapy entity distribution:")
        for entity_type, count in entity_counts.items():
            print(f"  {entity_type:30s}: {count:5d}")
        print(f"\nTotal therapy entities: {len(therapy_entity_types)}")
    else:
        print("WARNING: No therapy entities found!")
    
    print(f"Total documents: {len(df)}")
    print('='*60)
    
    return therapy_entity_types


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

def get_lora_config(r=64, lora_alpha=16, lora_dropout=0.1):
    """Get LoRA configuration."""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def get_bnb_config():
    """Get BitsAndBytes quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


def get_training_config(output_dir, num_train_epochs=25, batch_size=4, 
                       gradient_accumulation_steps=4, learning_rate=2e-5,
                       max_seq_length=600, save_steps=500):
    """Get training configuration."""
    return SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        logging_steps=10,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        fp16=True,
        eval_strategy="no",
        save_strategy='steps',
        save_steps=save_steps,
        save_total_limit=10,
        load_best_model_at_end=False,
        max_seq_length=max_seq_length,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train NER model for CIH therapy extraction")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face model ID or path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from (e.g., 'ckp/model/checkpoint-1000')")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default='train.json',
                        help="Path to training JSON file")
    parser.add_argument("--test_file", type=str, default='valid.json',
                        help="Path to test JSON file (for validation)")
    
    # Augmentation arguments
    parser.add_argument("--add_generation", action="store_true",
                        help="Add synthetic generation data")
    parser.add_argument("--generation_file", type=str, 
                        default="generate_train_ner.csv",
                        help="Path to synthetic data CSV")
    parser.add_argument("--num_synthetic", type=int, default=100,
                        help="Number of synthetic samples to add")
    
    # RAG arguments
    parser.add_argument("--rag", action="store_true",
                        help="Enable RAG with similar examples")
    parser.add_argument("--topk", type=int, default=1,
                        help="Number of similar examples for RAG")
    parser.add_argument("--train_embeddings", type=str, 
                        default="embeddings/similar_ner_doc_train_qwen.npy",
                        help="Path to training embeddings for RAG")
    parser.add_argument("--test_embeddings", type=str,
                        default="embeddings/similar_ner_doc_test_qwen.npy",
                        help="Path to test embeddings for RAG")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=600,
                        help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    
    # Output format
    parser.add_argument("--include_types", action="store_true", default=True,
                        help="Include entity types in output format")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NER Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training file: {args.train_file}")
    print(f"Test file: {args.test_file}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"RAG enabled: {args.rag}")
    if args.rag:
        print(f"RAG top-k: {args.topk}")
    print(f"Add synthetic data: {args.add_generation}")
    print("=" * 60)
    
    # ========================================================================
    # Load and prepare data
    # ========================================================================
    print("\nLoading training data...")
    dfall_train = load_ner_json_to_df(args.train_file)
    
    dfall_test = None
    if args.test_file:
        print("Loading test data...")
        dfall_test = load_ner_json_to_df(args.test_file)
    
    # Add synthetic data if requested
    if args.add_generation:
        dfall_train = add_generation_data(dfall_train, args.generation_file, 
                                         args.num_synthetic)
    
    dfall_train = dfall_train.reset_index(drop=True)
    if dfall_test is not None:
        dfall_test = dfall_test.reset_index(drop=True)
    
    # Analyze entity distribution
    analyze_entity_distribution(dfall_train, "Training Data")
    if dfall_test is not None:
        analyze_entity_distribution(dfall_test, "Test Data")
    
    # ========================================================================
    # Add RAG examples if enabled
    # ========================================================================
    if args.rag:
        print("\nLoading RAG embeddings...")
        
        if not os.path.exists(args.train_embeddings):
            print(f"ERROR: Training embeddings not found: {args.train_embeddings}")
            print("Disabling RAG")
            args.rag = False
        elif dfall_test is not None and not os.path.exists(args.test_embeddings):
            print(f"ERROR: Test embeddings not found: {args.test_embeddings}")
            print("Disabling RAG")
            args.rag = False
        else:
            train_knn = np.load(args.train_embeddings)
            print(f"Loaded train embeddings: {train_knn.shape}")
            
            if train_knn.shape[0] != len(dfall_train):
                print(f"ERROR: Embedding rows ({train_knn.shape[0]}) != train rows ({len(dfall_train)})")
                print("Disabling RAG")
                args.rag = False
            else:
                dfall_train = add_similar_examples(dfall_train, dfall_train, 
                                                  train_knn, topk=args.topk)
                
                if dfall_test is not None:
                    test_knn = np.load(args.test_embeddings)
                    print(f"Loaded test embeddings: {test_knn.shape}")
                    
                    if test_knn.shape[0] != len(dfall_test):
                        print(f"WARNING: Test embedding rows ({test_knn.shape[0]}) != test rows ({len(dfall_test)})")
                    else:
                        dfall_test = add_similar_examples(dfall_test, dfall_train,
                                                         test_knn, topk=args.topk)
    
    # ========================================================================
    # Prepare datasets
    # ========================================================================
    print("\nPreparing datasets...")
    train_dataset = prepare_data(dfall_train, rag=args.rag, 
                                 include_types=args.include_types)
    
    eval_dataset = None
    if dfall_test is not None:
        eval_dataset = prepare_data(dfall_test, rag=args.rag,
                                   include_types=args.include_types)
    
    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Print a sample
    print("\nSample training example:")
    print("-" * 60)
    print(train_dataset[0]['text'][:500] + "...")
    print("-" * 60)
    
    # ========================================================================
    # Load model and tokenizer
    # ========================================================================
    print("\nLoading model and tokenizer...")
    
    bnb_config = get_bnb_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"Model loaded: {args.model_id}")
    print(f"Model parameters: {base_model.num_parameters():,}")
    
    # ========================================================================
    # Configure training
    # ========================================================================
    lora_config = get_lora_config(r=args.lora_r, lora_alpha=args.lora_alpha,
                                 lora_dropout=args.lora_dropout)
    
    output_dir = args.output_dir  # pass full path via --output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = get_training_config(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps
    )
    
    # ========================================================================
    # Create trainer
    # ========================================================================
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    
    # ========================================================================
    # Start training
    # ========================================================================
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train()
    
    # ========================================================================
    # Save final model
    # ========================================================================
    print("\nSaving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()