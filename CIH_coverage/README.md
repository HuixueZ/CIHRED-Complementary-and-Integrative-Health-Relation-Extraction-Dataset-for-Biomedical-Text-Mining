# CIH Lexicon Coverage Calculator

Computes **Complementary and Integrative Health (CIH)** lexicon coverage over
biomedical NER datasets. Reproduces **Table 4** from the CIHRED paper.

---

## Requirements

Python 3.8+. No third-party packages required — uses only standard library
(`json`, `csv`, `re`, `difflib`, `argparse`).

---

## Input Files

### 1. CIH Lexicon (`--mapping_json`)

Download `2023cihlex_manual_filter_mapping.json` from the CIH GitHub repository:

> 🔗 https://github.com/zhang-informatics/CIH

Place it in your working directory or pass its full path via `--mapping_json`.

### 2. Dataset files (`--dataset`)

| Extension | Format | Datasets |
|---|---|---|
| `.txt` | BC5CDR PubTator | BC5CDR |
| `.json` | BioC-JSON | BioRED |
| `.json` | CIHRED sentence list (with `doc_id`) | CIHRED |

Format is auto-detected by file extension.

---

## Usage

```bash
python calculate_cih_coverage.py \
    --dataset NAME FILE [FILE ...] \
    [--dataset NAME FILE ...] \
    --mapping_json 2023cihlex_manual_filter_mapping.json \
    --out_dir results
```

Use `--dataset` once per dataset. All splits listed under a name are **pooled
before metrics are computed**, producing one row per dataset in the output table.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset NAME FILE [FILE ...]` | *(required, repeatable)* | Dataset name followed by split files |
| `--mapping_json JSON` | `2023cihlex_manual_filter_mapping.json` | CIH lexicon JSON |
| `--out_dir DIR` | `cih_coverage_results` | Output directory (created if absent) |
| `--cutoff FLOAT` | `0.86` | difflib similarity threshold for CIH term matching (CIHRED only) |
| `--presence_datasets NAME [...]` | `CIHRED` | Datasets where "Docs with CIH" is counted by entity presence rather than lexicon match |

---

## Reproduce Table 4

```bash
python calculate_cih_coverage.py \
    --dataset BC5CDR \
        CDR_Data/CDR_TrainingSet.PubTator.txt \
        CDR_Data/CDR_DevelopmentSet.PubTator.txt \
        CDR_Data/TestSet.tmChem.PubTator.txt \
    --dataset BioRED \
        BioRED/Train.BioC.JSON \
        BioRED/Dev.BioC.JSON \
        BioRED/Test.BioC.JSON \
    --dataset CIHRED \
        CIHRED/train.json \
        CIHRED/valid.json \
        CIHRED/test.json \
    --mapping_json 2023cihlex_manual_filter_mapping.json \
    --out_dir results
```

> All experiments in the paper use **default parameters** — no extra flags needed.

---

## Output

All files are written to `--out_dir` (default: `cih_coverage_results/`).

| File | Description |
|---|---|
| `dataset_summary.csv` | Main table — one row per dataset with all metrics |
| `terms__<dataset>.csv` | Per-term detail — each unique CIH-type span with match result |

### Column definitions

| Column | Description |
|---|---|
| `total_docs` | Unique documents across all pooled splits |
| `docs_with_cih` | BC5CDR/BioRED: docs where full text contains ≥1 CIH lexicon variant (substring match). CIHRED: docs with ≥1 CIH-type entity annotation |
| `covered_cih_concepts` | Distinct CIH concept names (CIHRED) or variant strings (BC5CDR/BioRED) found |
| `concept_coverage_pct` | `covered_cih_concepts / total_concepts × 100` |
| `unique_annotation_terms` | Unique annotation spans across **all** entity types (Chemical, Disease, Gene, CIH…) |
| `unique_ann_cih_terms` | Unique annotation spans of **CIH-type entities only** (deduplicated) |
| `annotated_cih_terms` | Total count of CIH-type annotation spans including duplicates across all docs |
