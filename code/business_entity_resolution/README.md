# Business Entity Resolution Pipeline

This repository contains the complete, self-contained, end-to-end solution for the **ML Challenge 2026: Business Entity Resolution Challenge**.

---

## 1. System Architecture Overview

The entity resolution solution is structured into a modular, high-throughput two-stage architecture:

1. **Candidate Generation (Multi-Key Inverted Index Blocking)**:
   - Partitions data strictly by country (US, India, France) to ensure zero cross-country leakage and minimal memory footprint.
   - Generates multi-resolution candidate keys:
     - Exact clean business name (legal suffixes stripped)
     - Character prefix keys (5-7 character stems)
     - Distinctive first and last name tokens
     - Domain/website stem extraction (e.g., stripping `.com`, `.in`, `.org`)
     - DBA alias expansion
     - Address street number paired with distinctive street words
     - PIN code (India 6-digit) and ZIP code (US/France 5-digit) composite keys
     - Bi-gram address tokens for numberless addresses
     - Combined name-stem + address-identifier cross-field keys
   - Inverted indexes use adaptive posting caps to prune high-frequency generic tokens and retain discriminative candidates.
   - Candidates are prioritized by weighted co-occurrence score, keeping the top $K \le 35$ candidates per entity.

2. **Pairwise Matching Model (LightGBM Gradient Boosted Decision Trees)**:
   - Uses `rapidfuzz` for high-speed C++ string similarity calculations.
   - Extracts 17 pairwise features spanning:
     - Full and clean name Levenshtein ratios, token sort ratio, token set ratio, partial ratio, and word Jaccard.
     - Address Levenshtein ratio, token sort ratio, token set ratio, partial ratio, and word Jaccard.
     - Street number overlap and exact number match.
     - Source origin indicators ($S2$ vs $S3$) and blocking hit counts.
   - Calibrated probability output evaluated with optimal thresholding ($0.58$) tailored to maximize Macro $F_{0.5}$.

3. **Global 1-to-Many Bipartite Constraint Enforcement**:
   - Source 1 is the deduplicated reference source; each Source 2 and Source 3 target entity belongs to at most one Source 1 entity.
   - All candidate predictions are globally ranked by confidence, assigning each target entity uniquely to the highest-scoring Source 1 entity, preventing false positive cross-merges.

---

## 2. Directory Structure

```
business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── blocking.py          # Multi-key inverted index blocking & normalization
│   ├── features.py          # RapidFuzz similarity and numerical feature extraction
│   ├── model.py             # LightGBM classifier training & persistence
│   └── pipeline.py          # Parallel worker execution & batch inference
├── models/
│   └── lgb_matcher.joblib   # Trained LightGBM model artifact
├── run.py                   # Master entrypoint script
├── requirements.txt         # Pinned python dependencies
└── README.md                # Reproduction documentation
```

---

## 3. Installation & Prerequisites

Python 3.8+ (tested on Python 3.13 on Windows 11 AMD64).

Install required dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. How to Reproduce End-to-End

To regenerate both output files (`matching_results.tsv` and `candidate_pairs.tsv`) from the training and test data:

```bash
python run.py \
    --train-dir ../../dataset/train \
    --test-dir ../../dataset/test \
    --output-dir ../../output \
    --model-path models/lgb_matcher.joblib \
    --threshold 0.58 \
    --max-cands 35 \
    --num-workers 10
```

### Command Line Arguments:
- `--train-dir`: Directory containing `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`.
- `--test-dir`: Directory containing `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`.
- `--output-dir`: Output destination where `matching_results.tsv` and `candidate_pairs.tsv` will be created.
- `--model-path`: Path to save/load the trained LightGBM model.
- `--threshold`: Probability decision threshold (default: `0.58`, tuned for Macro $F_{0.5}$).
- `--max-cands`: Maximum candidates retained per Source 1 entity (default: `35`).
- `--num-workers`: Number of parallel CPU worker processes (default: `10`).
- `--skip-train`: Skip model training if a pre-trained model file already exists.

---

## 5. Output Validation

Run the official validation script to verify format and rule compliance:

```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
