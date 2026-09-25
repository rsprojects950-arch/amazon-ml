# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen.svg)](https://github.com/microsoft/LightGBM)
[![RapidFuzz](https://img.shields.io/badge/RapidFuzz-3.14.6-orange.svg)](https://github.com/rapidfuzz/RapidFuzz)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.5.5-yellow.svg)](https://duckdb.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, high-performance Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution**. 

This repository implements a scalable, memory-efficient two-stage Entity Resolution (ER) pipeline that matches business entities across three independent, noisy data sources (**Source 1**, **Source 2**, and **Source 3**) covering **United States**, **India**, and **France**.

---

## 🌟 Key Performance Highlights

- **Validation Benchmark (Held-out 50,000 S1 Entities):**
  - **Macro $F_{0.5}$ Score:** **`0.9632`**
  - **Micro Precision:** **`98.98%`**
  - **Micro Recall:** **`85.33%`**
  - **Blocking Pair Recall:** **`94.58%`** (retaining $\le 35$ candidates per entity)
- **High Throughput:** Multi-key C++ vectorized inverted indexing and vectorized GBDT inference scoring at **>550–650 records/sec**.
- **Strict Invariant Guarantee:** 100% geographic isolation (zero cross-country leakage) and global 1-to-many bipartite constraint solver preventing duplicate assignments.

---

## 🏗️ System Architecture

The pipeline consists of three core stages designed to handle 10M+ records within memory and time constraints:

```
               ┌────────────────────────────────────────────────────────┐
               │    Raw Multilingual Records (S1, S2, S3 TSVs)          │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                               ▼ Country Partitioning
               ┌────────────────────────────────────────────────────────┐
               │  Geographic Isolation: Independent [US, India, France]  │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                               ▼ Stage 1: Candidate Generation
               ┌────────────────────────────────────────────────────────┐
               │         Multi-Key Inverted Index Blocking              │
               │  - Clean Name Exact & Stem Keys (5-7 chars)            │
               │  - Token Signatures (First/Last, DBA, Domains)          │
               │  - Address Keys (Street No. + Keywords, PIN/ZIP codes) │
               │  - Posting Cap Frequency Pruning                       │
               │  - Top-K Candidate Selection (K <= 35)                 │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                               ▼ Stage 2: Pairwise ML Classification
               ┌────────────────────────────────────────────────────────┐
               │           LightGBM Gradient Boosted Trees              │
               │  - 17 RapidFuzz C++ String Similarity Features         │
               │  - Word Jaccard, Token Sort/Set Ratios, Partial Ratios │
               │  - Exact Street Number Overlaps & Length Differentials │
               │  - Optimal Probability Thresholding (tau = 0.58)       │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                               ▼ Stage 3: Global Bipartite Assignment
               ┌────────────────────────────────────────────────────────┐
               │      1-to-Many Bipartite Constraint Enforcement        │
               │  - Global Greedy Ranking by Probability Confidence     │
               │  - Guarantees each S2/S3 entity <= 1 S1 parent         │
               │  - Resolves edge conflicts and maximizes Macro F0.5    │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                               ▼ Official Outputs
               ┌────────────────────────────────────────────────────────┐
               │  matching_results.tsv   &   candidate_pairs.tsv        │
               └────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
├── Documentation_template.md        # Comprehensive methodology & architecture report
├── README.md                        # Master repository documentation
├── .gitignore                       # Git ignore rules for datasets, caches, and outputs
├── utils/
│   └── validate_submission.py       # Official format and rule validation tool
└── code/
    └── business_entity_resolution/  # Self-contained competition package
        ├── README.md                # Package-level execution documentation
        ├── requirements.txt         # Pinned Python package dependencies
        ├── run.py                   # Master end-to-end execution script
        ├── models/
        │   └── lgb_matcher.joblib   # Trained LightGBM model weights artifact
        └── src/
            ├── __init__.py
            ├── blocking.py          # Multi-key inverted indexing & text normalization
            ├── features.py          # RapidFuzz similarity feature vectorizer
            ├── model.py             # LightGBM training, calibration & persistence
            └── pipeline.py          # Batch inference & bipartite constraint solver
```

---

## ⚡ Quickstart & Reproduction

### 1. Environment Setup

Python 3.8+ (tested on Python 3.13):

```bash
git clone https://github.com/rsprojects950-arch/amazon-ml.git
cd amazon-ml/code/business_entity_resolution
pip install -r requirements.txt
```

### 2. End-to-End Execution

Place dataset files under `dataset/train` and `dataset/test`, then run:

```bash
python run.py \
    --train-dir ../../dataset/train \
    --test-dir ../../dataset/test \
    --output-dir ../../output \
    --model-path models/lgb_matcher.joblib \
    --threshold 0.58 \
    --max-cands 35 \
    --skip-train
```

*Note: Pass `--skip-train` to immediately execute inference using the included pre-trained model artifact (`models/lgb_matcher.joblib`), or omit it to re-train the model on training data.*

### 3. Submission Validation

Validate the output files against official competition requirements:

```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

---

## 📊 Features & Engineering

The model computes 17 engineered features per candidate pair:
1. `name_ratio`: Full business name Levenshtein ratio.
2. `name_clean_ratio`: Cleaned business name ratio (legal forms stripped).
3. `name_token_sort_ratio`: Token sort ratio (robust to word order permutations).
4. `name_token_set_ratio`: Token set ratio (handles subset names).
5. `name_partial_ratio`: Substring alignment ratio.
6. `name_jaccard`: Word token intersection-over-union.
7. `name_len_diff`: Absolute difference in name lengths.
8. `addr_ratio`: Full address Levenshtein ratio.
9. `addr_clean_ratio`: Cleaned address ratio.
10. `addr_token_sort_ratio`: Token sort ratio for addresses.
11. `addr_token_set_ratio`: Token set ratio for addresses.
12. `addr_partial_ratio`: Substring alignment for addresses.
13. `addr_jaccard`: Word token intersection-over-union for address components.
14. `has_num_overlap`: Indicator whether street/building numbers overlap.
15. `num_match_exact`: Indicator whether all numbers match exactly.
16. `is_source2`: Binary indicator for Source 2 candidate.
17. `blocking_hits`: Multi-key inverted index co-occurrence count.

---

## 📜 Documentation

For complete mathematical formulations, exploratory data analysis findings, ablation studies, and scaling strategies, refer to [Documentation_template.md](Documentation_template.md).
