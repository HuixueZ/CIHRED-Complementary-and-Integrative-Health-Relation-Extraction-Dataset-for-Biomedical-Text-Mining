"""
NER Training with Label Collapsing
Collapse multiple therapy types into single CIH_intervention label

Usage:
    python ner_train_collapsed_labels.py --mode train_and_evaluate
"""

import sys
import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
import os
import re
import random
import math
import argparse
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report, accuracy_score as seq_accuracy
from seqeval.metrics import precision_score as seq_precision, recall_score as seq_recall, f1_score as seq_f1
from seqeval.scheme import IOB2

def log(msg):
    print(msg, flush=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

log("="*80)
log("NER TRAINING WITH COLLAPSED LABELS")
log("="*80)

# PATHS
DATA_DIR = os.environ.get("DATA_DIR", ".")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints_collapsed/")

# MODEL
#MODEL_NAME ="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# "dmis-lab/biobert-base-cased-v1.2"
PRE_TRAIN_MODELS={
    "BERT":'bert-base-uncased',
    "BIOBERT":"pucpr/biobertpt-all",
    "clinicalBERT":"emilyalsentzer/Bio_ClinicalBERT",
    "PubmedBERT":"cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "BLUEBERT":"bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "scibert": "allenai/scibert_scivocab_uncased"
    }
MODEL_NAME =PRE_TRAIN_MODELS["BERT"]

# TRAINING
LEARNING_RATE = 3e-5
BATCH_SIZE = 16
EPOCHS = 15
MAX_LENGTH = 128
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# DATA BALANCING
DOWNSAMPLE_RATIO = 0.3
UPSAMPLE_TARGET = 400

# LOSS
FOCAL_GAMMA = 2.5
FOCAL_ALPHA = 0.25
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True
WEIGHT_SCALING = 'sqrt'

# EARLY STOPPING
PATIENCE = 3
MIN_DELTA = 0.001

# LOGGING
LOG_INTERVAL = 50

# DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LABEL COLLAPSING CONFIGURATION
# ============================================================================

# These labels will be collapsed into a single label
THERAPY_LABELS = {
    'CIH_intervention',
    'Energy_therapy',
    'Manual_bodybased_therapy',
    'Mindbody_therapy',
    'Usual_Medical_Care'
}

# The new label name for all therapy types
COLLAPSED_LABEL = 'CIH_intervention'

# Whether to keep other labels (Chemical, Gene, etc.) or only focus on therapies
KEEP_OTHER_LABELS = False  # Set to True if you want to keep Chemical, Gene, etc.

log(f"\nLabel Collapsing Configuration:")
log(f"  Collapsing these labels into '{COLLAPSED_LABEL}':")
for label in sorted(THERAPY_LABELS):
    log(f"    - {label}")
log(f"  Keep other labels: {KEEP_OTHER_LABELS}")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def tokenize_with_spans(text):
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    return [(m.group(0), m.start(), m.end()) for m in pattern.finditer(text)]

def remove_abbreviation_tokens(tokens, labels):
    clean_tokens, clean_labels = [], []
    for t, l in zip(tokens, labels):
        if not re.fullmatch(r'[A-Z]{2,5}', t):
            clean_tokens.append(t)
            clean_labels.append(l)
    return clean_tokens, clean_labels

def bio_labels_for_entities(text, entities):
    tokens = tokenize_with_spans(text)
    labels = []
    all_tokens = []
    
    for token, start, end in tokens:
        label = "O"
        for ent in entities:
            ent_start, ent_end, ent_type = ent["start"], ent["end"], ent["type"]
            if start < ent_end and end > ent_start:
                if start == ent_start:
                    label = f"B-{ent_type}"
                else:
                    label = f"I-{ent_type}"
                break
        labels.append(label)
        all_tokens.append(token)
    
    all_tokens, labels = remove_abbreviation_tokens(all_tokens, labels)
    return [all_tokens, labels]

def collapse_labels(data):
    """
    Collapse multiple therapy labels into single CIH_intervention label
    
    Before: ['O', 'B-Mindbody_therapy', 'I-Mindbody_therapy', 'O', 'B-Energy_therapy']
    After:  ['O', 'B-CIH_intervention', 'I-CIH_intervention', 'O', 'B-CIH_intervention']
    """
    log("\nCollapsing labels...")
    
    original_label_counts = Counter()
    collapsed_label_counts = Counter()
    
    collapsed_data = []
    
    for tokens, labels in data:
        new_labels = []
        for label in labels:
            original_label_counts[label] += 1
            
            if label == 'O':
                new_labels.append('O')
            else:
                # Split BIO prefix and entity type
                parts = label.split('-', 1)
                if len(parts) == 2:
                    bio_prefix, entity_type = parts
                    
                    # Check if this is a therapy label
                    if entity_type in THERAPY_LABELS:
                        # Collapse to CIH_intervention
                        new_label = f"{bio_prefix}-{COLLAPSED_LABEL}"
                        new_labels.append(new_label)
                        collapsed_label_counts[new_label] += 1
                    else:
                        # Keep or remove based on KEEP_OTHER_LABELS
                        if KEEP_OTHER_LABELS:
                            new_labels.append(label)
                            collapsed_label_counts[label] += 1
                        else:
                            new_labels.append('O')
                            collapsed_label_counts['O'] += 1
                else:
                    new_labels.append('O')
        
        collapsed_data.append([tokens, new_labels])
    
    log("\n  Original label distribution:")
    for label, count in sorted(original_label_counts.items()):
        if label != 'O' and count > 0:
            log(f"    {label}: {count}")
    
    log("\n  Collapsed label distribution:")
    for label, count in sorted(collapsed_label_counts.items()):
        if label != 'O' and count > 0:
            log(f"    {label}: {count}")
    
    log(f"\n✓ Collapsed {len(data)} sequences")
    return collapsed_data

def load_data():
    log("\n" + "="*80)
    log("LOADING DATA")
    log("="*80)
    
    try:
        log(f"Loading from: {DATA_DIR}")
        
        with open(os.path.join(DATA_DIR, 'train.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        log(f"✓ Loaded train.json: {len(train_data)} samples")
        
        with open(os.path.join(DATA_DIR, 'valid.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
        log(f"✓ Loaded valid.json: {len(valid_data)} samples")
        
        with open(os.path.join(DATA_DIR, 'test.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        log(f"✓ Loaded test.json: {len(test_data)} samples")
        
        log("\nConverting to BIO format...")
        train_bio = [bio_labels_for_entities(item['text'], item['entities']) 
                     for item in train_data if 'text' in item and 'entities' in item]
        valid_bio = [bio_labels_for_entities(item['text'], item['entities']) 
                     for item in valid_data if 'text' in item and 'entities' in item]
        test_bio = [bio_labels_for_entities(item['text'], item['entities']) 
                    for item in test_data if 'text' in item and 'entities' in item]
        
        log(f"✓ Converted to {len(train_bio)} train, {len(valid_bio)} valid, {len(test_bio)} test sequences")
        
        # COLLAPSE LABELS HERE
        train_bio = collapse_labels(train_bio)
        valid_bio = collapse_labels(valid_bio)
        test_bio = collapse_labels(test_bio)
        
        return train_bio, valid_bio, test_bio
        
    except Exception as e:
        log(f"\n✗ ERROR LOADING DATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# DATA BALANCING
# ============================================================================

def downsample_o_only(dataset, keep_ratio=DOWNSAMPLE_RATIO):
    entity_sents = [row for row in dataset if any(tag != 'O' for tag in row[1])]
    o_only_sents = [row for row in dataset if all(tag == 'O' for tag in row[1])]
    
    keep_n = int(len(o_only_sents) * keep_ratio)
    keep_n = max(keep_n, min(100, len(o_only_sents)))
    
    downsampled = entity_sents + random.sample(o_only_sents, keep_n)
    random.shuffle(downsampled)
    
    log(f"  Entity sentences: {len(entity_sents)}")
    log(f"  O-only: {len(o_only_sents)} -> keeping {keep_n}")
    log(f"  Total: {len(downsampled)}")
    
    return downsampled

def upsample_rare(dataset, target=UPSAMPLE_TARGET):
    entity_counts = Counter()
    entity_samples = defaultdict(list)
    
    for idx, (tokens, tags) in enumerate(dataset):
        sample_tags = set(tag for tag in tags if tag != 'O')
        for tag in sample_tags:
            entity_counts[tag] += 1
            entity_samples[tag].append(idx)
    
    rare_tags = {tag: count for tag, count in entity_counts.items() if count < target}
    
    if rare_tags:
        log(f"  Upsampling {len(rare_tags)} rare classes:")
        for tag, count in sorted(rare_tags.items()):
            log(f"    {tag}: {count} -> {target}")
    
    upsampled_indices = list(range(len(dataset)))
    for tag, current_count in rare_tags.items():
        need = target - current_count
        if need > 0 and entity_samples[tag]:
            additional = random.choices(entity_samples[tag], k=need)
            upsampled_indices.extend(additional)
    
    random.shuffle(upsampled_indices)
    upsampled_dataset = [dataset[i] for i in upsampled_indices]
    
    log(f"  Dataset: {len(dataset)} -> {len(upsampled_dataset)}")
    return upsampled_dataset

def analyze_dataset(dataset, name="Dataset"):
    log(f"\n{name} Statistics:")
    total = len(dataset)
    with_entity = sum(1 for _, tags in dataset if any(t != 'O' for t in tags))
    only_o = total - with_entity
    tag_counter = Counter()
    
    for _, tags in dataset:
        for tag in tags:
            if tag != 'O':
                tag_counter[tag] += 1
    
    log(f"  Total: {total}")
    log(f"  With entities: {with_entity} ({with_entity/total*100:.1f}%)")
    log(f"  Only O: {only_o} ({only_o/total*100:.1f}%)")
    
    if tag_counter:
        log(f"  Label distribution:")
        for tag, count in sorted(tag_counter.items()):
            log(f"    {tag}: {count}")

# ============================================================================
# LABEL MAPPINGS
# ============================================================================

def create_label_mappings(train_data):
    log("\nCreating label mappings...")
    all_labels = [label for _, labels in train_data for label in labels]
    unique_labels = sorted(set(all_labels))
    
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    
    log(f"✓ Found {len(unique_labels)} unique labels:")
    for label in unique_labels:
        log(f"  {label}: {label_to_id[label]}")
    
    return label_to_id, id_to_label

def get_class_weights(train_labels, method=WEIGHT_SCALING):
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = {}
    
    for label, count in counts.items():
        raw_weight = total / (len(counts) * count)
        if method == 'sqrt':
            weights[label] = math.sqrt(raw_weight)
        elif method == 'log':
            weights[label] = math.log(raw_weight + 1)
        elif method == 'clip':
            weights[label] = min(raw_weight, 10.0)
        else:
            weights[label] = raw_weight
    
    return weights

# ============================================================================
# DATASET CLASS
# ============================================================================

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []
    
    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        
        if len(tokenized_sentence) >= 125:
            return tokenized_sentence, labels
        
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)
    
    return tokenized_sentence, labels

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id, max_length=MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data[idx][0]
        word_labels = self.data[idx][1]
        
        t_sen, t_labl = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
        
        sen_code = self.tokenizer.encode_plus(
            t_sen, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', return_attention_mask=True, truncation=True,
        )
        
        labels = [-100] * self.max_length
        for i, tok in enumerate(t_labl):
            if i + 1 >= self.max_length:
                break
            if self.label_to_id.get(tok) is not None:
                labels[i + 1] = self.label_to_id.get(tok)
        
        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['labels'] = torch.as_tensor(labels)
        return item

# ============================================================================
# LOSS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, 
                 label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, labels):
        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        labels_flat = labels.view(-1)
        
        valid_mask = labels_flat != -100
        valid_logits = logits_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]
        
        if len(valid_labels) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        log_probs = torch.nn.functional.log_softmax(valid_logits, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.class_weights is not None:
            weights = self.class_weights[valid_labels]
        else:
            weights = 1.0
        
        ce_loss = torch.nn.functional.cross_entropy(
            valid_logits, valid_labels, reduction='none', label_smoothing=self.label_smoothing
        )
        focal_loss = self.alpha * focal_weight * ce_loss * weights
        return focal_loss.mean()

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=ids, attention_mask=mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        
        if batch_idx % LOG_INTERVAL == 0:
            log(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {total_loss/num_batches:.4f}")
    
    return total_loss / num_batches

def evaluate(model, dataloader, criterion, device, id_to_label):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_true_labels_list, all_pred_labels_list = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            
            labels_flat = labels.view(-1)
            preds_flat = preds.view(-1)
            valid_mask = labels_flat != -100
            all_labels.extend(labels_flat[valid_mask].cpu().numpy())
            all_preds.extend(preds_flat[valid_mask].cpu().numpy())
            
            for seq_idx in range(ids.shape[0]):
                seq_labels = labels[seq_idx].cpu().numpy()
                seq_preds = preds[seq_idx].cpu().numpy()
                seq_mask = mask[seq_idx].cpu().numpy()
                
                active_labels = []
                active_preds = []
                for i in range(len(seq_labels)):
                    if seq_mask[i] == 1 and seq_labels[i] != -100:
                        active_labels.append(id_to_label[seq_labels[i]])
                        active_preds.append(id_to_label[seq_preds[i]])
                
                if active_labels:
                    all_true_labels_list.append(active_labels)
                    all_pred_labels_list.append(active_preds)
    
    avg_loss = total_loss / len(dataloader)
    token_acc = accuracy_score(all_labels, all_preds)
    
    try:
        seq_p = seq_precision(all_true_labels_list, all_pred_labels_list, 
                             mode='strict', scheme=IOB2, zero_division=0)
        seq_r = seq_recall(all_true_labels_list, all_pred_labels_list,
                          mode='strict', scheme=IOB2, zero_division=0)
        seq_f = seq_f1(all_true_labels_list, all_pred_labels_list,
                       mode='strict', scheme=IOB2, zero_division=0)
    except:
        seq_p = seq_r = seq_f = 0.0
    
    return {
        'loss': avg_loss, 'token_acc': token_acc,
        'precision': seq_p, 'recall': seq_r, 'f1': seq_f,
        'all_true_labels': all_true_labels_list,
        'all_pred_labels': all_pred_labels_list
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        log("\n" + "="*80)
        log("STARTING NER TRAINING WITH COLLAPSED LABELS")
        log("="*80)
        
        # Set seed
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Load data (with collapsed labels)
        train_data, valid_data, test_data = load_data()
        
        # Analyze
        analyze_dataset(train_data, "Original Train")
        analyze_dataset(valid_data, "Original Valid")
        
        # Balance
        log("\n" + "="*80)
        log("BALANCING DATA")
        log("="*80)
        train_balanced = downsample_o_only(train_data)
        train_balanced = upsample_rare(train_balanced)
        analyze_dataset(train_balanced, "Balanced Train")
        
        # Create mappings
        label_to_id, id_to_label = create_label_mappings(train_balanced)
        
        # Load tokenizer
        log("\n" + "="*80)
        log("LOADING TOKENIZER AND MODEL")
        log("="*80)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        log("✓ Tokenizer loaded")
        
        # Create datasets
        train_dataset = NERDataset(train_balanced, tokenizer, label_to_id)
        valid_dataset = NERDataset(valid_data, tokenizer, label_to_id)
        test_dataset = NERDataset(test_data, tokenizer, label_to_id)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        log(f"✓ Created dataloaders: {len(train_loader)} train, {len(valid_loader)} valid, {len(test_loader)} test batches")
        
        # Load model
        model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_to_id))
        model.to(DEVICE)
        log(f"✓ Model loaded with {len(label_to_id)} labels")
        
        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, 
                                                    num_training_steps=total_steps)
        
        if USE_CLASS_WEIGHTS:
            all_train_labels = [label for _, labels in train_balanced for label in labels]
            class_weights_dict = get_class_weights(all_train_labels)
            class_weights = torch.tensor(
                [class_weights_dict.get(id_to_label[i], 1.0) for i in range(len(id_to_label))],
                dtype=torch.float32
            ).to(DEVICE)
        else:
            class_weights = None
        
        criterion = FocalLoss(class_weights=class_weights)
        
        # Train
        log("\n" + "="*80)
        log("TRAINING")
        log("="*80)
        
        best_f1 = 0.0
        patience_counter = 0
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        for epoch in range(1, EPOCHS + 1):
            log(f"\nEpoch {epoch}/{EPOCHS}")
            log("-" * 80)
            
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE, epoch)
            val_metrics = evaluate(model, valid_loader, criterion, DEVICE, id_to_label)
            
            log(f"\nEpoch {epoch} Results:")
            log(f"  Train Loss: {train_loss:.4f}")
            log(f"  Val Loss: {val_metrics['loss']:.4f}")
            log(f"  Val Token Acc: {val_metrics['token_acc']:.4f}")
            log(f"  Val Precision: {val_metrics['precision']:.4f}")
            log(f"  Val Recall: {val_metrics['recall']:.4f}")
            log(f"  Val F1: {val_metrics['f1']:.4f}")
            
            if val_metrics['f1'] > best_f1 + MIN_DELTA:
                best_f1 = val_metrics['f1']
                patience_counter = 0
                
                timestamp = datetime.now().strftime('%m%d_%H%M')
                save_dir = os.path.join(CHECKPOINT_DIR, 
                                        f"collapsed_f1_{best_f1:.4f}_epoch{epoch}_{timestamp}")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                log(f"  ✓ SAVED: {save_dir}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    log(f"\nEarly stopping at epoch {epoch}")
                    break
        
        # Final evaluation
        log("\n" + "="*80)
        log("FINAL TEST EVALUATION")
        log("="*80)
        
        test_metrics = evaluate(model, test_loader, criterion, DEVICE, id_to_label)
        
        log(f"\nTest Results:")
        log(f"  Loss: {test_metrics['loss']:.4f}")
        log(f"  Token Acc: {test_metrics['token_acc']:.4f}")
        log(f"  Precision: {test_metrics['precision']:.4f}")
        log(f"  Recall: {test_metrics['recall']:.4f}")
        log(f"  F1: {test_metrics['f1']:.4f}")
        
        try:
            log("\nClassification Report:")
            log(classification_report(
                test_metrics['all_true_labels'],
                test_metrics['all_pred_labels'],
                mode='strict',
                scheme=IOB2,
                digits=4
            ))
        except:
            pass
        
        log("\n" + "="*80)
        log("TRAINING COMPLETE!")
        log("="*80)
        log(f"Best Validation F1: {best_f1:.4f}")
        log(f"Final Test F1: {test_metrics['f1']:.4f}")
        
    except Exception as e:
        log(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()