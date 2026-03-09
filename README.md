# CIHRED: Complementary and Integrative Health Relation Extraction for Biomedical Text Mining

This repository contains the codebase for processing, training, evaluation, and deployment support for the **CIHRED** project, a benchmark for extracting complementary and integrative health (CIH) entities and relations from biomedical text.

The project supports **named entity recognition (NER)** and **relation extraction (RE)** for CIH-related biomedical literature, with applications in structured knowledge extraction, evidence synthesis, and downstream biomedical NLP workflows.

---

## Overview

Complementary and integrative health (CIH) approaches such as acupuncture, meditation, yoga, herbal medicine, and related therapies are widely discussed in biomedical literature, but most of this information remains embedded in unstructured text. CIHRED was developed to support computational extraction of:

- CIH interventions
- diseases and outcomes
- genes and chemicals
- semantic relations between them

This repository provides the implementation used for:

- data preprocessing
- dataset formatting
- model training
- model evaluation
- baseline experiments
- deployment-oriented inference pipelines

---

## Repository Structure

```text
.
├── data/                  # Input data, processed files, or dataset splits
├── scripts/               # Utility scripts for preprocessing and experiments
├── src/                   # Core source code
│   ├── preprocessing/     # Data cleaning and formatting
│   ├── ner/               # NER training and inference
│   ├── re/                # Relation extraction training and inference
│   ├── evaluation/        # Metrics and evaluation scripts
│   └── deployment/        # Deployment or inference pipeline code
├── configs/               # Model and training configuration files
├── notebooks/             # Optional exploratory notebooks
├── outputs/               # Predictions, logs, checkpoints, or results
├── requirements.txt       # Python dependencies
└── README.md
