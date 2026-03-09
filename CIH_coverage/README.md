# CIH Lexicon Coverage Calculator

Compute **Complementary and Integrative Health (CIH)** lexicon coverage over
biomedical NER benchmark datasets, including **BC5CDR** and **BioRED**.

---

## Overview

This script measures how well a CIH terminology lexicon covers the entity
mentions in existing biomedical NER corpora. Three complementary metrics
are reported:

| Metric | Description |
|---|---|
| **Document coverage** | % of documents containing ≥1 CIH variant mention (free-text regex) |
| **Concept coverage** | % of CIH concepts hit anywhere across the corpus |
| **NER term coverage** | % of gold-annotated entity spans that map to a CIH concept (fuzzy matching) |

---

## Requirements

**Python 3.8+**

Install dependencies:

```bash
pip install rapidfuzz          # optional — falls back to difflib if absent
```

No other third-party packages are required (`json`, `csv`, `re`, `difflib`,
`argparse` are all standard library).

---

## Input Files

### CIH Lexicon JSON (`--mapping_json`)

The CIH lexicon is sourced from the **CIH GitHub repository**:

> 🔗 https://github.com/zhang-informatics/CIH

Download `2023cihlex_manual_filter_mapping.json` from that repository and
place it in your working directory (or pass its full path via `--mapping_json`).

The file is a JSON in one of two accepted formats:

```jsonc
// Format A — flat list of variants per concept
{ "concept_name": ["variant 1", "variant 2", ...], ... }

// Format B — nested dict with "term" field (used in 2023cihlex_manual_filter_mapping.json)
{ "category": { "var_key": { "term": "variant string" }, ... }, ... }
```

Default filename: `2023cihlex_manual_filter_mapping.json`

### Dataset files (`--infile`)
| Extension | Format | Example corpora |
|---|---|---|
| `.txt` | BC5CDR PubTator blocks | `CDR_TrainingSet.PubTator.txt`, `TestSet.tmChem.PubTator.txt` |
| `.json` | BioC JSON | `Train.BioC.JSON`, `Dev.BioC.JSON`, `Test.BioC.JSON` (BioRED) |

Format is **auto-detected** by file extension.

---

## Usage

```bash
python calculate_cih_coverage.py --infile FILE [FILE ...] [OPTIONS]
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--infile FILE [FILE ...]` | *(required)* | Input dataset files (.txt or .json) |
| `--mapping_json JSON` | `2023cihlex_manual_filter_mapping.json` | CIH lexicon JSON file |
| `--out_dir DIR` | `cih_coverage_results` | Output directory (created if absent) |
| `--alpha FLOAT` | `0.70` | Fuzzy ratio weight in hybrid matching score *(default used in paper)* |
| `--threshold FLOAT` | `0.65` | Minimum hybrid score to count as a match *(default used in paper)* |
| `--jaccard_floor FLOAT` | `0.30` | Minimum Jaccard similarity required *(default used in paper)* |
| `--map_ner` | *(flag)* | Also map gold NER spans to CIH and write term CSV |
| `--sweep` | *(flag)* | Write threshold sweep CSV for diagnostics |

---

## Example Commands

### BioRED (train + dev + test splits)
```bash
python calculate_cih_coverage.py \
    --infile BioRED/Train.BioC.JSON BioRED/Dev.BioC.JSON BioRED/Test.BioC.JSON \
    --mapping_json 2023cihlex_manual_filter_mapping.json \
    --out_dir results/biored \
    --map_ner
```

### BC5CDR corpus
```bash
python calculate_cih_coverage.py \
    --infile CDR_Data/CDR_TrainingSet.PubTator.txt CDR_Data/TestSet.tmChem.PubTator.txt \
    --mapping_json 2023cihlex_manual_filter_mapping.json \
    --out_dir results/bc5cdr \
    --map_ner --sweep
```

### Mixed input (auto-detected by extension)
```bash
python calculate_cih_coverage.py \
    --infile Train.BioC.JSON CDR_TrainingSet.PubTator.txt \
    --mapping_json 2023cihlex_manual_filter_mapping.json \
    --out_dir results/mixed
```

### Adjust matching sensitivity
```bash
# Stricter matching
python calculate_cih_coverage.py --infile ... --threshold 0.75 --jaccard_floor 0.40

# More lenient matching
python calculate_cih_coverage.py --infile ... --threshold 0.55 --jaccard_floor 0.20
```

> **Note:** All experiments reported in the paper used the **default matching
> parameters** (`--alpha 0.70`, `--threshold 0.65`, `--jaccard_floor 0.30`).
> You do not need to set these flags to reproduce the results — simply omit
> them and the defaults will be applied automatically.

---

## Output Files

All files are written to `--out_dir` (default: `cih_coverage_results/`).

### Always produced

| File | Description |
|---|---|
| `summary__combined.csv` | Combined metrics across all input files |
| `summary__<filename>.csv` | Per-file metrics (one per input file) |
| `concept_hits.csv` | CIH concepts ranked by free-text hit count |
| `variant_hits.csv` | CIH variant strings ranked by hit count |
| `doc_hits.csv` | Documents ranked by total CIH mention count |

### With `--map_ner`

| File | Description |
|---|---|
| `ner_terms__combined.csv` | All unique NER spans with match result, concept, and score |
| `ner_terms__<filename>.csv` | Per-file NER term mapping |

### With `--sweep`

| File | Description |
|---|---|
| `sweep.csv` | NER term coverage (%) at thresholds 0.55 → 0.80 |

### Summary CSV columns

| Column | Description |
|---|---|
| `total_docs` | Number of documents in the (sub-)corpus |
| `docs_with_cih` | Documents with ≥1 CIH regex match |
| `doc_coverage_pct` | `docs_with_cih / total_docs × 100` |
| `total_concepts` | Total CIH concepts in lexicon |
| `covered_concepts` | CIH concepts hit in corpus |
| `concept_coverage_pct` | `covered_concepts / total_concepts × 100` |
| `ner_total_unique_terms` | Unique annotated spans (NER; only with `--map_ner`) |
| `ner_matched_terms` | Spans matched to CIH |
| `ner_term_coverage_pct` | `ner_matched / ner_total × 100` |
| `alpha`, `threshold`, `jaccard_floor` | Matching hyperparameters used |

---

## Matching Algorithm

NER term matching uses a **three-stage cascade**:

1. **Exact match** — normalised string equality (score = 1.0)
2. **Substring / token-subset** — one string contains the other, or ≥80%
   token overlap (score = 0.95)
3. **Hybrid fuzzy** — weighted combination of character-level fuzzy ratio
   (rapidfuzz `token_set_ratio`, or `difflib` fallback) and Jaccard token
   similarity; accepted if score ≥ `--threshold` and Jaccard ≥ `--jaccard_floor`

Variant strings are expanded before indexing to cover hyphenation, plurals,
and domain synonyms (e.g., *electroacupuncture* → *acupuncture*;
*LLLT* → *light therapy*, *phototherapy*).

---

## Citation / Paper Context

This script was used to assess the coverage of the 2023 CIH Lexicon
(`2023cihlex_manual_filter_mapping.json`) over the BC5CDR and BioRED
benchmark corpora as reported in:

> *[Your paper title here]*  
> [Authors], [Venue], [Year]
