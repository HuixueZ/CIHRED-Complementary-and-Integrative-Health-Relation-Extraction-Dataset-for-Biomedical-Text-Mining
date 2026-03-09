# CIHRED: Complementary and Integrative Health Relation Extraction Dataset for Biomedical Text Mining


---

## Overview

CIHRED is a manually annotated biomedical corpus for **named entity recognition (NER)** and **relation extraction (RE)** focused on complementary and integrative health (CIH) interventions. Built from 415 PubMed abstracts, it contains **7,906 entity mentions** and **1,186 relation annotations** capturing therapeutic and mechanistic links between CIH therapies and biomedical entities.

It is the first English-language corpus specifically designed for CIH NER and RE, providing a benchmark for systematic study of how CIH interventions — acupuncture, yoga, meditation, herbal medicine, and more — interact with biological systems and clinical outcomes.

---

## Dataset

The dataset is publicly available on Figshare:

**DOI: [10.6084/m9.figshare.30334387](https://doi.org/10.6084/m9.figshare.30334387)**

The dataset contains three JSON splits using a 60/20/20 document-level split:

| Split | Docs | Sentences | Entities | Relations |
|---|---|---|---|---|
| `train.json` | 248 | 3,120 | 4,776 | 688 |
| `valid.json` | 83 | 994 | 1,614 | 257 |
| `test.json` | 84 | 997 | 1,516 | 241 |
| **Total** | **415** | **5,111** | **7,906** | **1,186** |

### Data Format

Each file is a list of sentence-level objects:

```json
{
  "id": "33879465_0",
  "text": "Guided imagery significantly reduced anxiety and pain in patients.",
  "entities": [
    {
      "text": "Guided imagery",
      "type": "Mindbody_therapy",
      "start": 0,
      "end": 14,
      "id": "T1"
    },
    {
      "text": "anxiety",
      "type": "Outcome_marker",
      "start": 38,
      "end": 45,
      "id": "T2"
    }
  ],
  "triples": [
    {
      "subject": "Guided imagery",
      "predicate": "RELIEVES",
      "object": "anxiety",
      "subject_type": "Mindbody_therapy",
      "object_type": "Outcome_marker",
      "subject_id": "T1",
      "object_id": "T2"
    }
  ]
}
```

### Entity Types

| Entity Type | Description | Examples |
|---|---|---|
| `Mindbody_therapy` | Techniques enhancing mind–body connection | guided imagery, meditation, yoga |
| `Manual_bodybased_therapy` | Manipulation or movement of body parts | acupuncture, massage, chiropractic |
| `Energy_therapy` | Therapies using energy fields | qigong, therapeutic touch, Low-Level Laser Therapy |
| `Usual_Medical_Care` | Standard conventional treatments | physical therapy, Cognitive Behavioral Therapy |
| `CIH_intervention` | Broad/umbrella CIH terms | complementary medicine, traditional Chinese medicine |
| `Gene` | Genes, gene products, or gene activities | TMPRSS2, ALP, isoforms |
| `Chemical` | Biochemical substances and pharmacological agents | glucocorticoids, nerve growth factor |
| `Disease` | Clinical conditions | chronic pain, depression |
| `Outcome_marker` | Clinical signs, symptoms, or measurement outcomes | fatigue, quality of life, salivary T/C ratio |

### Relation Types

**Treatment–Disease (TD):**

| Relation | Description |
|---|---|
| `TREATS` | Therapy manages or cures a disease |
| `PREVENTS` | Therapy reduces risk or occurrence of a disease |
| `RELIEVES` | Therapy reduces symptoms without claiming cure |
| `AFFECTS` | Therapy influences a disease without specifying direction or intent |

**Treatment–Gene/Chemical (TC):**

| Relation | Description |
|---|---|
| `ASSOCIATED_WITH` | Therapy associated with a gene or chemical |
| `INHIBITS` | Therapy decreases or blocks gene/chemical activity |
| `STIMULATES` | Therapy promotes or enhances gene/chemical activity |

---

## Repository Structure

```
.
├── CIH_coverage/          # Scripts for analyzing CIH lexicon coverage
└── src/                   # NER and RE training and inference scripts
```

---

## Requirements

```bash
pip install torch transformers peft trl datasets seqeval scikit-learn pandas tqdm
```

GPU required for all LLM scripts. BERT-based scripts run on CPU but are much faster on GPU.

---

## Running the Models

### BERT-based NER (`complete_ner_training.py`)

Standard BERT token classification with BIO labels, focal loss, and class balancing. Supported encoders: PubMedBERT *(default)*, ClinicalBERT, BioBERT, BlueBERT, SciBERT, BERT.

```bash
python src/complete_ner_training.py \
    --mode train_and_evaluate \
    --data_dir path/to/data/ \
    --checkpoint_dir path/to/checkpoints/
```

To evaluate from a saved checkpoint:

```bash
python src/complete_ner_training.py \
    --mode evaluate \
    --model_path path/to/checkpoint/ \
    --data_dir path/to/data/
```

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train_and_evaluate` | `train`, `evaluate`, or `train_and_evaluate` |
| `--data_dir` | `.` | Directory with `train.json`, `valid.json`, `test.json` |
| `--checkpoint_dir` | `checkpoints/` | Where to save best model |
| `--model_path` | — | Required for `evaluate` mode |
| `--seed` | `42` | Random seed |

---

### BERT NER with collapsed labels (`ner_train_collapsed_labels.py`)

Same as above but collapses all CIH therapy subtypes into a single `CIH_intervention` label.

```bash
export DATA_DIR=path/to/data/
export CHECKPOINT_DIR=path/to/checkpoints/
python src/ner_train_collapsed_labels.py
```

---

### LLM-based NER — with entity types (`train_ner.py`)

Fine-tunes a causal LLM (e.g., Qwen, LLaMA) with QLoRA (4-bit) for NER. Output format: `therapy_name [therapy_type], ...`

```bash
python src/train_ner.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --output_dir /path/to/output/ckp \
    --train_file train.json \
    --test_file valid.json \
    --num_train_epochs 25 \
    --batch_size 4 \
    --learning_rate 2e-5
```

| Argument | Default | Description |
|---|---|---|
| `--model_id` | *(required)* | HuggingFace model ID or local path |
| `--output_dir` | *(required)* | Full path to save checkpoints |
| `--train_file` | `train.json` | Training data |
| `--test_file` | `valid.json` | Validation data |
| `--num_train_epochs` | `25` | Training epochs |
| `--batch_size` | `4` | Per-device batch size |
| `--learning_rate` | `2e-5` | Learning rate |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha |

---

### LLM-based NER — CIH spans only (`train_ner_CIH.py`)

Same as `train_ner.py` but uses `<start> ... <end>` tag format with no entity type labels.

```bash
python src/train_ner_CIH.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --output_dir /path/to/output/ckp \
    --train_file train.json \
    --test_file valid.json
```

---

### LLM NER inference — with entity types (`inference_ner.py`)

```bash
python src/inference_ner.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint_dir /path/to/ckp \
    --checkpoint checkpoint-3500 \
    --test_data test.json \
    --output_dir predictions/ \
    --batch_size 16
```

| Argument | Default | Description |
|---|---|---|
| `--base_model` | *(required)* | Base model ID or path |
| `--checkpoint_dir` | — | Directory containing LoRA checkpoints |
| `--checkpoint` | — | Specific checkpoint folder |
| `--test_data` | `test.json` | Test data |
| `--batch_size` | `16` | Inference batch size |
| `--output_dir` | `prediction2` | Output directory |

Evaluation (exact / overlap / fuzzy match) is printed automatically after inference.

---

### LLM NER inference — CIH spans only (`inference_ner_CIH.py`)

Same as `inference_ner.py` but expects `<start> ... <end>` output format.

```bash
python src/inference_ner_CIH.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint_dir /path/to/ckp \
    --checkpoint checkpoint-3500 \
    --test_data test.json \
    --output_dir predictions_cih/
```

---

### Analyze existing NER predictions (`evaluate_ner.py`)

```bash
python src/evaluate_ner.py \
    --results_file predictions/predictions_checkpoint-3500.json \
    --num_samples 10
```

| Argument | Default | Description |
|---|---|---|
| `--results_file` | *(required)* | JSONL prediction file |
| `--num_samples` | `10` | Samples shown in diagnostics |
| `--skip_diagnostics` | off | Print metrics only |

---

### RE training (`train_re.py`)

Fine-tunes a causal LLM for relation extraction. Relation labels: `TREATS`, `AFFECTS`, `PREVENTS`, `RELIEVES`, `INHIBITS`, `STIMULATES`, `ASSOCIATED_WITH`.

```bash
python src/train_re.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --output_dir /path/to/output/ckp \
    --num_train_epochs 20
```

| Argument | Default | Description |
|---|---|---|
| `--model_id` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Model |
| `--output_dir` | *(required)* | Full output path |
| `--num_train_epochs` | `20` | Epochs |
| `--add_generation` | off | Add synthetic augmentation data |

---

### RE inference — all relations (`inference_re.py`)

Includes `NONE` relations generated from entity pairs without an annotated triple.

```bash
python src/inference_re.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --lora_weights /path/to/ckp/checkpoint-3500 \
    --test_path test.json \
    --output_path results/inference_results.jsonl \
    --batch_size 16 \
    --max_new_tokens 32
```

| Argument | Default | Description |
|---|---|---|
| `--model_id` | *(required)* | Model ID or path |
| `--lora_weights` | — | LoRA checkpoint path |
| `--test_path` | `test.json` | Test data |
| `--output_path` | `inference_results2.jsonl` | Output JSONL |
| `--batch_size` | `16` | Batch size |
| `--max_new_tokens` | `32` | Max output tokens |
| `--repetition_penalty` | `1.1` | Repetition penalty |
| `--no_4bit` | off | Disable 4-bit quantization |

---

### RE inference — annotated triples only (`inference_re_not_none.py`)

Same as `inference_re.py` but skips negative example generation — evaluates on sentences with annotated triples only.

```bash
python src/inference_re_not_none.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --lora_weights /path/to/ckp/checkpoint-3500 \
    --test_path test.json \
    --output_path results/inference_no_none.jsonl
```

---

## Baseline Results

### NER — CIH entity detection (micro-F1)

| Model | Strict F1 | Lenient F1 |
|---|---|---|
| GLiNER | **0.7491** | **0.8616** |
| SciBERT | 0.6646 | 0.7544 |
| BioBERT | 0.6874 | 0.7542 |
| PubMedBERT | 0.6486 | 0.7428 |
| Qwen-14B | 0.5889 | 0.7279 |
| Qwen-32B | 0.5955 | 0.6861 |
| LLaMA-3-8B | 0.5345 | 0.6620 |

### RE — without negative labels (micro-F1)

| Model | F1 |
|---|---|
| PubMedBERT | **0.5955** |
| BlueBERT | 0.5843 |
| SciBERT | 0.5730 |
| DeepSeek-R1-8B | 0.5141 |
| ClinicalBERT | 0.5056 |
| BioBERT | 0.4831 |

`

Dataset DOI: [10.6084/m9.figshare.30334387](https://doi.org/10.6084/m9.figshare.30334387)
