"""
Complete NER Training Script - Updated for Your Data Format
Ready to use for training and evaluation
Author: Improved version based on your original code
Date: 2024

Usage:
    python complete_ner_training.py --mode train
    python complete_ner_training.py --mode evaluate
    python complete_ner_training.py --mode train_and_evaluate
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
import os
import re
from difflib import SequenceMatcher
import random
import math
import argparse
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, 
    BertForTokenClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score
)

from seqeval.metrics import (
    classification_report,
    accuracy_score as seq_accuracy,
    precision_score as seq_precision,
    recall_score as seq_recall,
    f1_score as seq_f1
)
from seqeval.scheme import IOB2

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for NER training"""
    
    # Paths - CHANGE THESE TO YOUR PATHS
    DATA_DIR = os.environ.get("DATA_DIR", ".")
    CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints/")
    
    # Model settings

    PRE_TRAIN_MODELS={
    "BERT":'bert-base-uncased',
    "BIOBERT":"pucpr/biobertpt-all",
    "clinicalBERT":"emilyalsentzer/Bio_ClinicalBERT",
    "PubmedBERT":"cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "BLUEBERT":"bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "scibert": "allenai/scibert_scivocab_uncased"
    }
    MODEL_NAME =PRE_TRAIN_MODELS["BLUEBERT"]
    # "dmis-lab/biobert-base-cased-v1.2"  # BioBERT performs best
    # Alternative models:
    # "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"  # PubMedBERT
    # "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"  # BlueBERT
    
    # Training hyperparameters
    LEARNING_RATE = 3e-5
    BATCH_SIZE = 16
    EPOCHS = 15
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    MAX_LENGTH = 128
    
    # Data balancing
    DOWNSAMPLE_RATIO = 0.3  # Keep 30% of O-only sentences
    UPSAMPLE_TARGET = 400   # Target instances for rare classes
    
    # Loss function
    FOCAL_GAMMA = 2.5
    FOCAL_ALPHA = 0.25
    LABEL_SMOOTHING = 0.1
    USE_CLASS_WEIGHTS = True
    WEIGHT_SCALING = 'sqrt'  # 'sqrt', 'log', 'clip', or 'none'
    
    # Early stopping
    PATIENCE = 3
    MIN_DELTA = 0.001
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Logging
    LOG_INTERVAL = 50
    SAVE_BEST_ONLY = True
    
    # Entity type filtering - specify which entity types to use
    # Set to None to use all entity types, or specify a list
    # Examples:
    # ENTITY_TYPES = ['Mindbody_therapy', 'Manual_bodybased_therapy', 'Energy_therapy', 'CIH_intervention', 'Usual_Medical_Care']
    # ENTITY_TYPES = None  # Use all types
    ENTITY_TYPES = None  # Use all entity types found in data


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def tokenize_with_spans(text):
    """Tokenize text into words and punctuation with spans"""
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    return [(m.group(0), m.start(), m.end()) for m in pattern.finditer(text)]


def remove_abbreviation_tokens(tokens, labels):
    """Remove abbreviation tokens"""
    clean_tokens, clean_labels = [], []
    for t, l in zip(tokens, labels):
        if not re.fullmatch(r'[A-Z]{2,5}', t):
            clean_tokens.append(t)
            clean_labels.append(l)
    return clean_tokens, clean_labels


def bio_labels_for_entities(text, entities):
    """
    Convert text and entities to BIO format
    Input format matches your data structure:
    {
        'text': 'sentence text',
        'entities': [
            {'text': 'entity text', 'type': 'entity_type', 'start': 0, 'end': 10},
            ...
        ]
    }
    """
    tokens = tokenize_with_spans(text)
    labels = []
    all_tokens = []
    
    for token, start, end in tokens:
        label = "O"
        for ent in entities:
            ent_start, ent_end, ent_type = ent["start"], ent["end"], ent["type"]
            # Check if token overlaps with entity span
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


def load_data(config):
    """Load and preprocess data from your format"""
    print("Loading data...")
    
    # Load JSON files with your naming convention
    with open(os.path.join(config.DATA_DIR, 'train.json'), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(os.path.join(config.DATA_DIR, 'valid.json'), 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
    
    with open(os.path.join(config.DATA_DIR, 'test.json'), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"  Loaded {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test samples")
    
    # Convert to BIO format
    print("Converting to BIO format...")
    train_bio = []
    valid_bio = []
    test_bio = []
    
    for item in train_data:
        if 'text' in item and 'entities' in item:
            bio_result = bio_labels_for_entities(item['text'], item['entities'])
            train_bio.append(bio_result)
    
    for item in valid_data:
        if 'text' in item and 'entities' in item:
            bio_result = bio_labels_for_entities(item['text'], item['entities'])
            valid_bio.append(bio_result)
    
    for item in test_data:
        if 'text' in item and 'entities' in item:
            bio_result = bio_labels_for_entities(item['text'], item['entities'])
            test_bio.append(bio_result)
    
    print(f"  Converted to {len(train_bio)} train, {len(valid_bio)} valid, {len(test_bio)} test sequences")
    
    # Filter entity types if specified
    if config.ENTITY_TYPES is not None:
        print(f"\nFiltering for entity types: {config.ENTITY_TYPES}")
        train_bio = filter_entity_types(train_bio, config.ENTITY_TYPES)
        valid_bio = filter_entity_types(valid_bio, config.ENTITY_TYPES)
        test_bio = filter_entity_types(test_bio, config.ENTITY_TYPES)
    
    return train_bio, valid_bio, test_bio


def filter_entity_types(data, allowed_types):
    """Filter labels to only include specified entity types"""
    filtered_data = []
    for tokens, labels in data:
        new_labels = []
        for label in labels:
            if label == 'O':
                new_labels.append('O')
            else:
                # Extract entity type from BIO label (e.g., "B-Mindbody_therapy" -> "Mindbody_therapy")
                parts = label.split('-', 1)
                if len(parts) == 2:
                    bio_prefix, entity_type = parts
                    if entity_type in allowed_types:
                        new_labels.append(label)
                    else:
                        new_labels.append('O')
                else:
                    new_labels.append('O')
        filtered_data.append([tokens, new_labels])
    return filtered_data


# ============================================================================
# DATA BALANCING
# ============================================================================

def downsample_o_only(dataset, keep_ratio=0.3, min_keep=100):
    """Smart downsampling that keeps some O-only sentences"""
    entity_sents = [row for row in dataset if any(tag != 'O' for tag in row[1])]
    o_only_sents = [row for row in dataset if all(tag == 'O' for tag in row[1])]
    
    keep_n = max(int(len(o_only_sents) * keep_ratio), min_keep)
    keep_n = min(keep_n, len(o_only_sents))
    
    downsampled = entity_sents + random.sample(o_only_sents, keep_n)
    random.shuffle(downsampled)
    
    print(f"  Entity sentences: {len(entity_sents)}")
    print(f"  O-only sentences: {len(o_only_sents)} (keeping {keep_n})")
    print(f"  Total after balancing: {len(downsampled)}")
    
    return downsampled


def upsample_rare(dataset, target_per_class=400):
    """Upsample rare entities"""
    entity_counts = Counter()
    entity_samples = defaultdict(list)
    
    for idx, (tokens, tags) in enumerate(dataset):
        sample_tags = set(tag for tag in tags if tag != 'O')
        for tag in sample_tags:
            entity_counts[tag] += 1
            entity_samples[tag].append(idx)
    
    rare_tags = {tag: count for tag, count in entity_counts.items() 
                 if count < target_per_class}
    
    if rare_tags:
        print("  Upsampling rare entity classes:")
        for tag, count in sorted(rare_tags.items()):
            print(f"    {tag}: {count} -> {target_per_class}")
    
    upsampled_indices = list(range(len(dataset)))
    for tag, current_count in rare_tags.items():
        need = target_per_class - current_count
        if need > 0 and entity_samples[tag]:
            additional = random.choices(entity_samples[tag], k=need)
            upsampled_indices.extend(additional)
    
    random.shuffle(upsampled_indices)
    upsampled_dataset = [dataset[i] for i in upsampled_indices]
    
    print(f"  Dataset size: {len(dataset)} -> {len(upsampled_dataset)}")
    return upsampled_dataset


def analyze_dataset(dataset, name="Dataset"):
    """Analyze and print dataset statistics"""
    total_sentences = len(dataset)
    with_entity = 0
    only_O = 0
    tag_counter = Counter()
    
    for tokens, tags in dataset:
        if all(tag == 'O' for tag in tags):
            only_O += 1
        else:
            with_entity += 1
            tag_counter.update(tags)
    
    print(f"\n{name} Statistics:")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Sentences with only 'O': {only_O} ({only_O / total_sentences * 100:.1f}%)")
    print(f"  Sentences with entities: {with_entity} ({with_entity / total_sentences * 100:.1f}%)")
    
    if tag_counter:
        print(f"  Entity distribution:")
        for tag, count in sorted(tag_counter.items()):
            if tag != 'O':
                print(f"    {tag}: {count}")


# ============================================================================
# LABEL MAPPING
# ============================================================================

def create_label_mappings(train_data):
    """Create label to ID and ID to label mappings"""
    all_labels = [label for _, labels in train_data for label in labels]
    unique_labels = sorted(set(all_labels))
    
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    
    print(f"\nFound {len(unique_labels)} unique labels:")
    for label in unique_labels:
        print(f"  {label}: {label_to_id[label]}")
    
    return label_to_id, id_to_label


# ============================================================================
# CLASS WEIGHTS
# ============================================================================

def get_class_weights(train_labels, method='sqrt'):
    """Calculate class weights with different scaling methods"""
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
        else:  # 'none'
            weights[label] = raw_weight
    
    print(f"\nClass weights (method={method}):")
    for label, weight in sorted(weights.items()):
        print(f"  {label}: {weight:.4f}")
    
    return weights


# ============================================================================
# DATASET CLASS
# ============================================================================

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """Tokenize while preserving label alignment"""
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
    """Dataset class for NER"""
    
    def __init__(self, data, tokenizer, label_to_id, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data[idx][0]
        word_labels = self.data[idx][1]
        
        t_sen, t_labl = tokenize_and_preserve_labels(
            sentence, word_labels, self.tokenizer
        )
        
        sen_code = self.tokenizer.encode_plus(
            t_sen,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
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
# LOSS FUNCTION
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, class_weights=None, alpha=0.25, gamma=2.0, 
                 label_smoothing=0.1):
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
            valid_logits, valid_labels,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        focal_loss = self.alpha * focal_weight * ce_loss * weights
        
        return focal_loss.mean()


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, config):
    """Train for one epoch"""
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        
        if batch_idx % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {avg_loss:.4f}")
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device, id_to_label):
    """Evaluate the model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    all_tokens_list = []
    all_true_labels_list = []
    all_pred_labels_list = []
    
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
            
            # Token-level metrics
            labels_flat = labels.view(-1)
            preds_flat = preds.view(-1)
            valid_mask = labels_flat != -100
            
            all_labels.extend(labels_flat[valid_mask].cpu().numpy())
            all_preds.extend(preds_flat[valid_mask].cpu().numpy())
            
            # Sequence-level metrics
            batch_size = ids.shape[0]
            for seq_idx in range(batch_size):
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
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    
    # Token-level
    token_acc = accuracy_score(all_labels, all_preds)
    
    # Sequence-level
    try:
        seq_acc = seq_accuracy(all_true_labels_list, all_pred_labels_list)
        seq_p = seq_precision(all_true_labels_list, all_pred_labels_list, 
                             mode='strict', scheme=IOB2, zero_division=0)
        seq_r = seq_recall(all_true_labels_list, all_pred_labels_list,
                          mode='strict', scheme=IOB2, zero_division=0)
        seq_f = seq_f1(all_true_labels_list, all_pred_labels_list,
                       mode='strict', scheme=IOB2, zero_division=0)
    except:
        seq_acc = seq_p = seq_r = seq_f = 0.0
    
    return {
        'loss': avg_loss,
        'token_acc': token_acc,
        'seq_acc': seq_acc,
        'precision': seq_p,
        'recall': seq_r,
        'f1': seq_f,
        'all_true_labels': all_true_labels_list,
        'all_pred_labels': all_pred_labels_list
    }


def train_model(config, train_loader, valid_loader, model, optimizer, 
                scheduler, criterion, id_to_label):
    """Main training loop"""
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    best_f1 = 0.0
    patience_counter = 0
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            criterion, config.DEVICE, config
        )
        
        # Validate
        val_metrics = evaluate(
            model, valid_loader, criterion, config.DEVICE, id_to_label
        )
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Token Acc: {val_metrics['token_acc']:.4f}")
        print(f"  Val Seq Acc: {val_metrics['seq_acc']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}")
        print(f"  Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1 + config.MIN_DELTA:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            
            timestamp = datetime.now().strftime('%m%d_%H%M')
            save_dir = os.path.join(
                config.CHECKPOINT_DIR,
                f"best_model_epoch{epoch+1}_f1_{best_f1:.4f}_{timestamp}"
            )
            
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            
            # Save config and metrics
            with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_metrics['loss'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                }, f, indent=2)
            
            print(f"  ✓ Saved best model to: {save_dir}")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{config.PATIENCE})")
            
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    print(f"\nTraining completed! Best F1: {best_f1:.4f}")
    return best_f1


# ============================================================================
# DETAILED EVALUATION
# ============================================================================

def detailed_evaluation(model, test_loader, criterion, device, id_to_label):
    """Comprehensive evaluation with detailed metrics"""
    print("\n" + "="*80)
    print("DETAILED EVALUATION")
    print("="*80)
    
    metrics = evaluate(model, test_loader, criterion, device, id_to_label)
    
    print(f"\nTest Results:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Token Accuracy: {metrics['token_acc']:.4f}")
    print(f"  Sequence Accuracy: {metrics['seq_acc']:.4f}")
    print(f"  Precision (Strict): {metrics['precision']:.4f}")
    print(f"  Recall (Strict): {metrics['recall']:.4f}")
    print(f"  F1 (Strict): {metrics['f1']:.4f}")
    
    # Classification report
    try:
        print("\nDetailed Classification Report (Strict):")
        print(classification_report(
            metrics['all_true_labels'],
            metrics['all_pred_labels'],
            mode='strict',
            scheme=IOB2,
            digits=4
        ))
    except Exception as e:
        print(f"Could not generate classification report: {e}")
    
    # Lenient evaluation
    try:
        lenient_p = seq_precision(metrics['all_true_labels'], 
                                 metrics['all_pred_labels'], zero_division=0,scheme=IOB2)
        lenient_r = seq_recall(metrics['all_true_labels'],
                              metrics['all_pred_labels'], zero_division=0,scheme=IOB2)
        lenient_f = seq_f1(metrics['all_true_labels'],
                          metrics['all_pred_labels'], zero_division=0,scheme=IOB2)
        
        print(f"\nLenient Evaluation:")
        print(f"  Precision: {lenient_p:.4f}")
        print(f"  Recall: {lenient_r:.4f}")
        print(f"  F1: {lenient_f:.4f}")
    except:
        pass
    
    return metrics


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='NER Training and Evaluation')
    parser.add_argument('--mode', type=str, default='train_and_evaluate',
                       choices=['train', 'evaluate', 'train_and_evaluate'],
                       help='Mode: train, evaluate, or both')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing train.json, valid.json, test.json')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    config = Config()
    # Allow CLI to override data/checkpoint paths
    if hasattr(args, 'data_dir') and args.data_dir:
        config.DATA_DIR = args.data_dir
    if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir:
        config.CHECKPOINT_DIR = args.checkpoint_dir

    print("="*80)
    print("NER TRAINING SCRIPT")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    
    # Load data
    train_data, valid_data, test_data = load_data(config)
    
    # Analyze original data
    analyze_dataset(train_data, "Original Train")
    analyze_dataset(valid_data, "Original Valid")
    analyze_dataset(test_data, "Original Test")
    
    # Balance training data
    print("\nBalancing training data...")
    train_data_balanced = downsample_o_only(
        train_data, 
        keep_ratio=config.DOWNSAMPLE_RATIO
    )
    train_data_balanced = upsample_rare(
        train_data_balanced,
        target_per_class=config.UPSAMPLE_TARGET
    )
    
    analyze_dataset(train_data_balanced, "Balanced Train")
    
    # Create label mappings
    label_to_id, id_to_label = create_label_mappings(train_data_balanced)
    
    # Setup tokenizer
    print(f"\nLoading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=False)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NERDataset(train_data_balanced, tokenizer, label_to_id, config.MAX_LENGTH)
    valid_dataset = NERDataset(valid_data, tokenizer, label_to_id, config.MAX_LENGTH)
    test_dataset = NERDataset(test_data, tokenizer, label_to_id, config.MAX_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Valid batches: {len(valid_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    if args.mode in ['train', 'train_and_evaluate']:
        # Setup model
        print(f"\nLoading model: {config.MODEL_NAME}")
        model = BertForTokenClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=len(label_to_id)
        )
        model.to(config.DEVICE)
        
        print(f"Model loaded with {len(label_to_id)} labels")
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        total_steps = len(train_loader) * config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Setup loss function
        if config.USE_CLASS_WEIGHTS:
            all_train_labels = [label for _, labels in train_data_balanced for label in labels]
            class_weights_dict = get_class_weights(all_train_labels, config.WEIGHT_SCALING)
            class_weights = torch.tensor(
                [class_weights_dict.get(id_to_label[i], 1.0) for i in range(len(id_to_label))],
                dtype=torch.float32
            ).to(config.DEVICE)
        else:
            class_weights = None
        
        criterion = FocalLoss(
            class_weights=class_weights,
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA,
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        # Train
        best_f1 = train_model(
            config, train_loader, valid_loader, model,
            optimizer, scheduler, criterion, id_to_label
        )
        
        # Evaluate on test set
        if args.mode == 'train_and_evaluate':
            print("\n" + "="*80)
            print("EVALUATING ON TEST SET")
            print("="*80)
            test_metrics = detailed_evaluation(
                model, test_loader, criterion, config.DEVICE, id_to_label
            )
    
    elif args.mode == 'evaluate':
        if args.model_path is None:
            raise ValueError("Must provide --model_path for evaluation mode")
        
        print(f"\nLoading model from: {args.model_path}")
        model = BertForTokenClassification.from_pretrained(args.model_path)
        model.to(config.DEVICE)
        
        criterion = FocalLoss()
        
        test_metrics = detailed_evaluation(
            model, test_loader, criterion, config.DEVICE, id_to_label
        )
    
    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()