# CIHRED Model Training & Inference

Scripts for **NER** (named entity recognition) and **RE** (relation extraction) on the CIHRED dataset.
There are two NER paradigms: **BERT-based token classification** and **LLM-based generative (LoRA/QLoRA)**.

---

## Requirements

```bash
pip install torch transformers peft trl datasets seqeval scikit-learn pandas tqdm
```

GPU required for all LLM scripts. BERT scripts run on CPU but are much faster on GPU.

---

## 1. BERT-based NER (`complete_ner_training.py`)

Standard BERT token classification with BIO labels, focal loss, and class balancing.

**Supported base models** (set inside `Config.PRE_TRAIN_MODELS`):

| Key | Model |
|---|---|
| `BLUEBERT` | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` *(default)* |
| `PubmedBERT` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` |
| `scibert` | `allenai/scibert_scivocab_uncased` |
| `BIOBERT` | `pucpr/biobertpt-all` |
| `clinicalBERT` | `emilyalsentzer/Bio_ClinicalBERT` |
| `BERT` | `bert-base-uncased` |

To change model, edit `MODEL_NAME = PRE_TRAIN_MODELS["BLUEBERT"]` in the `Config` class.

### Train + evaluate

```bash
python complete_ner_training.py \
    --mode train_and_evaluate \
    --data_dir path/to/data/ \
    --checkpoint_dir path/to/save/ckp/
```

Or set via environment variables:

```bash
export DATA_DIR=path/to/data/
export CHECKPOINT_DIR=path/to/save/ckp/
python complete_ner_training.py --mode train_and_evaluate
```

### Evaluate only (from saved checkpoint)

```bash
python complete_ner_training.py \
    --mode evaluate \
    --model_path path/to/saved/checkpoint/ \
    --data_dir path/to/data/
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train_and_evaluate` | `train`, `evaluate`, or `train_and_evaluate` |
| `--data_dir` | `.` | Directory with `train.json`, `valid.json`, `test.json` |
| `--checkpoint_dir` | `checkpoints/` | Where to save best model checkpoints |
| `--model_path` | — | Required for `evaluate` mode |
| `--seed` | `42` | Random seed |

---

## 2. BERT NER with collapsed labels (`ner_train_collapsed_labels.py`)

Same as above but collapses all therapy subtypes (`Energy_therapy`, `Mindbody_therapy`, etc.)
into a single `CIH_intervention` label. Useful when type distinction is not needed.

```bash
export DATA_DIR=path/to/data/
export CHECKPOINT_DIR=path/to/collapsed_ckp/
python ner_train_collapsed_labels.py
```

> No CLI arguments — edit constants `MODEL_NAME`, `LEARNING_RATE`, `EPOCHS`, etc. at the top of the file.

---

## 3. LLM-based NER training — with entity types (`train_ner.py`)

Fine-tunes a causal LLM (e.g., Qwen, LLaMA) with QLoRA (4-bit) for NER.
Output format: `therapy_name [therapy_type], ...`

```bash
python train_ner.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --output_dir /path/to/output/ckp \
    --train_file train.json \
    --test_file valid.json \
    --num_train_epochs 25 \
    --batch_size 4 \
    --learning_rate 2e-5
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model_id` | *(required)* | HuggingFace model ID or local path |
| `--output_dir` | *(required)* | Full path to save checkpoints |
| `--train_file` | `train.json` | Training data |
| `--test_file` | `valid.json` | Validation data |
| `--num_train_epochs` | `25` | Training epochs |
| `--batch_size` | `4` | Per-device batch size |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |
| `--learning_rate` | `2e-5` | Learning rate |
| `--max_seq_length` | `600` | Max token length |
| `--save_steps` | `500` | Checkpoint interval |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha |
| `--add_generation` | off | Add synthetic data |

---

## 4. LLM-based NER training — CIH-only, no types (`train_ner_CIH.py`)

Same as `train_ner.py` but the output format uses `<start> ... <end>` tags and **no entity type labels**.
Used when only entity spans are needed, not type classification.

```bash
python train_ner_CIH.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --output_dir /path/to/output/ckp \
    --train_file train.json \
    --test_file valid.json
```

Arguments are identical to `train_ner.py`.

---

## 5. LLM NER inference — with entity types (`inference_ner.py`)

Runs inference on test data using a fine-tuned model (optionally with LoRA weights).
Expects output format: `therapy_name [therapy_type], ...`

```bash
python inference_ner.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint_dir /path/to/ckp \
    --checkpoint checkpoint-3500 \
    --test_data test.json \
    --output_dir predictions/ \
    --batch_size 16
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--base_model` | *(required)* | Base model ID or path |
| `--checkpoint_dir` | — | Directory containing LoRA checkpoints |
| `--checkpoint` | — | Specific checkpoint folder name |
| `--test_data` | `test.json` | Test data |
| `--batch_size` | `16` | Inference batch size |
| `--output_dir` | `prediction2` | Output directory |
| `--output_name` | — | Custom output filename |

Evaluation (exact / overlap / fuzzy match) is printed automatically after inference.

---

## 6. LLM NER inference — CIH-only, no types (`inference_ner_CIH.py`)

Same as `inference_ner.py` but expects `<start> ... <end>` output format.
Matches entities by text only (type-agnostic evaluation).

```bash
python inference_ner_CIH.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint_dir /path/to/ckp \
    --checkpoint checkpoint-3500 \
    --test_data test.json \
    --output_dir predictions_cih/
```

Arguments are identical to `inference_ner.py`.

---

## 7. Analyze existing NER predictions (`evaluate_ner.py`)

Load a saved JSONL prediction file and run diagnostics + evaluation without re-running inference.

```bash
python evaluate_ner.py \
    --results_file predictions/predictions_checkpoint-3500.json \
    --num_samples 10
```

```bash
# Skip per-sample diagnostics and go straight to metrics
python evaluate_ner.py \
    --results_file predictions/predictions_checkpoint-3500.json \
    --skip_diagnostics
```

| Argument | Default | Description |
|---|---|---|
| `--results_file` | *(required)* | JSONL file from `inference_ner.py` |
| `--num_samples` | `10` | Number of samples shown in diagnostics |
| `--skip_diagnostics` | off | Skip diagnostics, print metrics only |

Reports exact, overlap, and fuzzy match metrics per entity type.

---

## 8. RE training (`train_re.py`)

Fine-tunes a causal LLM for relation extraction between CIH interventions and biomarkers.
Relation labels: `TREATS`, `AFFECTS`, `PREVENTS`, `FACILITATE`, `INHIBITS`, `STIMULATES`, `ASSOCIATED_WITH`.

```bash
python train_re.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --output_dir /path/to/output/ckp \
    --num_train_epochs 20
```

### With synthetic data augmentation

```bash
# Set GENERATION_CSV env var to your CSV file path
export GENERATION_CSV=path/to/generate_train.csv
python train_re.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --output_dir /path/to/output/ckp \
    --add_generation
```

| Argument | Default | Description |
|---|---|---|
| `--model_id` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Model |
| `--output_dir` | *(required)* | Full output path |
| `--num_train_epochs` | `20` | Epochs |
| `--add_generation` | off | Add synthetic augmentation data |

---

## 9. RE inference — all relations (`inference_re.py`)

Runs RE inference on test data, including `NONE` relations (generated from entity pairs with no annotated triple).

```bash
python inference_re.py \
    --test_path test.json \
    --output_path results/inference_results.jsonl \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --batch_size 16 \
    --max_new_tokens 32
```

### With LoRA checkpoint

```bash
python inference_re.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --lora_weights /path/to/ckp/checkpoint-3500 \
    --test_path test.json \
    --output_path results/inference_results.jsonl
```

| Argument | Default | Description |
|---|---|---|
| `--model_id` | Qwen2.5-32B-Instruct path | Model |
| `--lora_weights` | — | LoRA checkpoint path |
| `--test_path` | `test.json` | Test data |
| `--output_path` | `inference_results2.jsonl` | Output JSONL |
| `--batch_size` | `16` | Batch size |
| `--max_input_length` | `2048` | Max input tokens |
| `--max_new_tokens` | `32` | Max output tokens |
| `--do_sample` | off | Use sampling |
| `--temperature` | `0.0` | Temperature (only when `--do_sample`) |
| `--repetition_penalty` | `1.1` | Repetition penalty |
| `--no_4bit` | off | Disable 4-bit quantization |

---

## 10. RE inference — annotated triples only (`inference_re_not_none.py`)

Identical to `inference_re.py` but skips generating `NONE` negative examples —
only evaluates on sentences that have annotated triples.
Use this when you want metrics on confirmed positive relations only.

```bash
python inference_re_not_none.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --lora_weights /path/to/ckp/checkpoint-3500 \
    --test_path test.json \
    --output_path results/inference_no_none.jsonl
```
