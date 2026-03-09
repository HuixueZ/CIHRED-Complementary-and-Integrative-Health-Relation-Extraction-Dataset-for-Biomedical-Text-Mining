import torch
import json
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import argparse

EXAMPLE="""Context: In sentence 'It exemplifies the upswing of biofield (energy field) therapies that have been taking place recently for a number of clinical conditions, including cancer, pain, arthritis, movement restriction, and energy psychology.', what is the relationship between biofield (energy field) therapies and pain?
            Response:TREATS."""

# Function to get top-k examples as formatted string
def get_similar_examples(row_idx, source_df, topk_indices, topk=1):
    example_texts = []
    for sim_idx in topk_indices[row_idx][:topk]:
        sim_row = source_df.iloc[sim_idx]
        label = sim_row['predicate'].replace('TC_','').replace('TD_','')
        if label=='FTPC':
            label='FACILITATE'
        formatted_sentence = format_sentence(sim_row)
        example_texts.append(f"Context: {formatted_sentence}\nResponse: {label}")
    return "\n".join(example_texts)

def add_similar_examples(df, source_df,topk_indices, topk=1):
    examples = []
    for idx in tqdm(range(len(df)), desc="Adding similar examples"):
        example_block = get_similar_examples(idx, source_df, topk_indices, topk=topk)
        examples.append(example_block)
        if idx==0:
            print("examples:",examples)
    df['similar_examples'] = examples
    return df


def formatting_func(row,rag):
    label=row['predicate'].replace('TC_','').replace('TD_','')
    if label=='FTPC':
        label='FACILITATE'
    if rag:
        example=row['similar_examples']
    else:
        example=EXAMPLE
    text= (
       # "Identify the relationship between the HEAD (disease) and TAIL (outcome marker) in the medical text below. "
         "You are an expert information extractor. Your task is to identify the relationship between a CIH intervention and a disease/biomarker mentioned in the sentence.​\n"
"You should ONLY select one of the relations in ('TREATS','AFFECTS','PREVENTS','FACILITATE','INHIBITS', 'STIMULATES', 'ASSOCIATED_WITH').\n"
"You SHOULD output one of the list relation above."
f"Example: {example}\n"
#f"Examples: {EXAMPLE}"
f"### Context:{row['formatted_sentence']}. "
f"### Response: {label}")
    return {"text" : text}

SUBJECT=['CIH_intervention','Energy_therapy','Manual_bodybased_therapy','Mindbody_therapy','Usual_Medical_Care']
OBJECT=['Gene','Chemical','Outcome_marker', 'chronic_pain', 'mental_disorders', 'other_diseases','Disease']
def load_re_split_to_df(path):
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
                    "sentence": sent_text,
                    "subject": t["subject"],
                    "predicate": t["predicate"],
                    "object": t["object"]
                })
        # else:
        #     # Generate negative examples from entities
        #     subject_entities = []
        #     object_entities = []
            
        #     for entity in sent['entities']:
        #         if entity['type'] in SUBJECT:
        #             subject_entities.append(entity['text'])  # Extract text
        #         elif entity['type'] in OBJECT:
        #             object_entities.append(entity['text'])  # Extract text
            
        #     # Create negative examples (None relations)
        #     if len(subject_entities) > 0 and len(object_entities) > 0:
        #         predicate = 'NONE'
        #         for subj in subject_entities:
        #             for obj in object_entities:
        #                 records.append({
        #                     "sentence": sent_text,
        #                     "subject": subj,
        #                     "predicate": predicate,
        #                     "object": obj
        #                 })
    
    return pd.DataFrame(records)

def format_sentence(row):
    """Format sentence with context"""
    return f"In sentence '{row['sentence']}', what is the relationship between {row['subject']} and {row['object']}?"

def prepare_data(df,rag):
    """Convert DataFrame to Hugging Face dataset"""
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset.map(lambda x: formatting_func(x, rag))

def add_generation_data(dfall_train,num1=60,num2=60):
    generate_train=pd.read_csv(os.environ.get("GENERATION_CSV", "generate_train.csv"))
    select_sample1=generate_train[generate_train['predicate']=='TC_ASSOCIATED_WITH'].sample(num1)
    select_sample2=generate_train[generate_train['predicate']=='TD_PREVENTS'].sample(num2)
    generate_train=pd.concat([select_sample1,select_sample2])
    generate_train=generate_train[['sentence','subject','predicate','object']]
    augmentated_train=pd.concat([dfall_train,generate_train])
    return augmentated_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # #add_model_args(parser)
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B") ##model_id="skumar9/Llama-medx_v3.2"
    parser.add_argument("--output_dir", type=str, default="re_by_doc_DeepSeek_8b_rag")
    parser.add_argument("--add_generation", action="store_true")
    parser.add_argument("--rag", default=0,type=int)
    
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    args = parser.parse_args()

    print("=" * 20)
    print("preparing data")

    # Load data
    #ragllm_mc/CIH/splits_all_v3_by_doc_recheck
    dfall_train = load_re_split_to_df("train.json")
    dfall_test = load_re_split_to_df("valid.json")
    if args.add_generation:
        dfall_train=add_generation_data(dfall_train)

    dfall_train = dfall_train.reset_index(drop=True)
    dfall_test = dfall_test.reset_index(drop=True)

    print(dfall_train['predicate'].value_counts())

    # Format sentences
    dfall_train['formatted_sentence'] = dfall_train.apply(format_sentence, axis=1)
    dfall_test['formatted_sentence'] = dfall_test.apply(format_sentence, axis=1)

    if args.rag:
        dfall_train = add_similar_examples(dfall_train,dfall_train, np.load('embeddings/similar_doc_train_qwen.npy'), topk=args.topk)
        dfall_test = add_similar_examples(dfall_test,dfall_train, np.load('embeddings/similar_doc_test_qwen.npy'), topk=args.topk)

    if args.rag:
        rag=True
    else:
        rag=False
    # Prepare datasets
    train = prepare_data(dfall_train,rag)
    dev = prepare_data(dfall_test,rag)

    # Model configuration
    model_id = args.model_id
    output_dir = args.output_dir  # pass full path via --output_dir

    #output_dir = "ckp/re_by_triple_llamamex_augmentated2"

    print("=" * 20)
    print(model_id)
    print(output_dir)
    print("=" * 20)

    # LoRA configuration
    qlora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
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

    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    device_map = {"": 0}
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map=device_map
    )
    base_model.config.use_cache = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Training configuration
    training_args = SFTConfig(
    per_device_train_batch_size=2,  # Reduced batch size
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,  # Increased to maintain effective batch size
    gradient_checkpointing=True,
    logging_steps=10,
    learning_rate=2e-5,
    num_train_epochs=args.num_train_epochs,  # Reduced epochs for initial testing
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    output_dir=output_dir,
    optim="paged_adamw_8bit",
    fp16=True,
    eval_strategy="no",
    save_strategy='steps',
    save_steps=500,
    save_total_limit=10,
    load_best_model_at_end=False,
    max_seq_length=1024,  # Reduced sequence length
    remove_unused_columns=False,
    dataloader_pin_memory=False,
        )

    # Create trainer
    supervised_finetuning_trainer = SFTTrainer(
    model=base_model,
    train_dataset=train,
    eval_dataset=dev,
    args=training_args,
    processing_class=tokenizer,
    peft_config=qlora_config,
    )

    # Start training
    supervised_finetuning_trainer.train()

    # Save the final model
    supervised_finetuning_trainer.save_model(output_dir)
    #tokenizer.save_pretrained(output_dir)