"""
ML Challenge 2026: Business Entity Resolution Master Pipeline.
Reproduces end-to-end: Data -> Blocking -> Matching -> Output TSVs.
"""

import os
import sys
import argparse
import time
import gc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import duckdb
import joblib
import re
from sklearn.model_selection import train_test_split

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.blocking import (
    clean_text, clean_business_name, extract_name_keys,
    extract_addr_keys, extract_combined_keys, extract_all_keys,
    ADDR_STOPWORDS
)
from src.features import FEATURE_NAMES, extract_pair_features
from src.model import train_lgbm_model, load_lgbm_model
from src.pipeline import (
    preprocess_target_records, build_inverted_index, worker_process_s1_batch, init_worker
)

def compute_f05_score(val_df, val_pairs, probs, threshold, gt_map):
    scored_pairs = []
    for (s1_id, tid), prob in zip(val_pairs, probs):
        if prob >= threshold:
            scored_pairs.append((prob, s1_id, tid))
            
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    assigned_targets = set()
    pred_map = defaultdict(set)
    for prob, s1_id, tid in scored_pairs:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            pred_map[s1_id].add(tid)
            
    f05_scores = []
    for s1_id in val_df['entity_id']:
        preds = pred_map.get(s1_id, set())
        trues = gt_map.get(s1_id, set())
        
        if not trues:
            if not preds:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
            continue
            
        tp = len(preds & trues)
        fp = len(preds - trues)
        fn = len(trues - preds)
        
        if tp == 0:
            f05_scores.append(0.0)
            continue
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        f05_scores.append(f05)
        
    return np.mean(f05_scores)

def train_matching_model(train_dir, model_save_path, sample_size=200000, num_workers=8):
    """
    Trains the LightGBM matching model on a representative sample of ground truth data.
    """
    print(f"\n[Training Stage] Training LightGBM model on {sample_size} S1 entities...")
    t0 = time.time()
    con = duckdb.connect()
    
    s1_path = os.path.join(train_dir, "train_source1.tsv").replace("\\", "/")
    s2_path = os.path.join(train_dir, "train_source2.tsv").replace("\\", "/")
    s3_path = os.path.join(train_dir, "train_source3.tsv").replace("\\", "/")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv").replace("\\", "/")
    
    # Sample balanced US and India records from S1
    half = sample_size // 2
    con.execute(f"""
        CREATE TABLE s1_train AS 
        SELECT * FROM (
            SELECT * FROM read_csv('{s1_path}', delim='\\t', header=True) WHERE country='US' LIMIT {half}
        )
        UNION ALL
        SELECT * FROM (
            SELECT * FROM read_csv('{s1_path}', delim='\\t', header=True) WHERE country='India' LIMIT {half}
        )
    """)
    s1_train_df = con.execute("SELECT entity_id, business_name, business_address, country FROM s1_train").fetchdf()
    
    # Ground truth mapping
    con.execute(f"""
        CREATE TABLE gt_train AS 
        SELECT gt.source1_entity_id, gt.matched_entity_ids
        FROM read_csv('{gt_path}', delim='\\t', header=True) gt
        JOIN s1_train ON gt.source1_entity_id = s1_train.entity_id
    """)
    gt_train_df = con.execute("SELECT * FROM gt_train").fetchdf()
    
    gt_map = {}
    for _, row in gt_train_df.iterrows():
        s1_id = row['source1_entity_id']
        m_str = row['matched_entity_ids']
        if pd.notna(m_str) and str(m_str).strip():
            gt_map[s1_id] = set(str(m_str).strip().split(','))
        else:
            gt_map[s1_id] = set()
            
    # Load targets using native DuckDB unnest
    con.execute("""
        CREATE TABLE true_target_ids AS 
        SELECT unnest(str_split(matched_entity_ids, ',')) as id 
        FROM gt_train 
        WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
    """)
    
    # Increase negative sampling significantly to 500k each
    targets_train = con.execute(f"""
        SELECT entity_id, business_name, business_address, country 
        FROM read_csv('{s2_path}', delim='\\t', header=True)
        WHERE entity_id IN (SELECT id FROM true_target_ids)
        UNION ALL
        SELECT entity_id, business_name, business_address, country 
        FROM read_csv('{s3_path}', delim='\\t', header=True)
        WHERE entity_id IN (SELECT id FROM true_target_ids)
        UNION ALL
        SELECT * FROM (
            SELECT entity_id, business_name, business_address, country 
            FROM read_csv('{s2_path}', delim='\\t', header=True)
            LIMIT 500000
        )
        UNION ALL
        SELECT * FROM (
            SELECT entity_id, business_name, business_address, country 
            FROM read_csv('{s3_path}', delim='\\t', header=True)
            LIMIT 500000
        )
    """).fetchdf().drop_duplicates(subset=['entity_id'])
    
    print(f"  Loaded {len(s1_train_df)} S1 records and {len(targets_train)} target records in {time.time()-t0:.2f}s")
    
    # Preprocess targets and build country inverted index
    target_records = preprocess_target_records(targets_train)
    inv_index = defaultdict(lambda: defaultdict(list))
    for tid, (name, addr, c_nm, nm_words, ad_words, nums, country) in target_records.items():
        keys = extract_all_keys(name, addr)
        for k in keys:
            idx = inv_index[country][k]
            cap = 40 if (k.startswith("w1:") or k.startswith("w_last:")) else 80
            if len(idx) < cap:
                idx.append(tid)
                
    # Split S1 into 80% train, 20% validation
    train_df, val_df = train_test_split(s1_train_df, test_size=0.2, random_state=42)
    
    def extract_features_for_df(df):
        X_list, y_list, pair_list = [], [], []
        for s1_row in df.itertuples(index=False):
            s1_id = s1_row.entity_id
            country = s1_row.country
            s1_name = str(s1_row.business_name) if pd.notna(s1_row.business_name) else ""
            s1_addr = str(s1_row.business_address) if pd.notna(s1_row.business_address) else ""
            s1_clean = clean_business_name(s1_name)
            s1_nm_words = set(s1_clean.split())
            s1_ad_words = set(w for w in clean_text(s1_addr).split() if len(w) >= 3 and w not in ADDR_STOPWORDS)
            s1_nums = set(re.findall(r'\b\d+\b', s1_addr))
            s1_tuple = (s1_name, s1_addr, s1_clean, s1_nm_words, s1_ad_words, s1_nums)
            
            country_idx = inv_index[country]
            s1_keys = extract_name_keys(s1_name) + extract_addr_keys(s1_addr) + extract_combined_keys(s1_name, s1_addr)
            hit_counts = defaultdict(int)
            for k in set(s1_keys):
                if k in country_idx:
                    wt = 3 if (k.startswith("cn:") or k.startswith("dom:")) else 1
                    for tid in country_idx[k]:
                        hit_counts[tid] += wt
                        
            if not hit_counts:
                continue
                
            sorted_all = sorted(hit_counts.keys(), key=lambda x: hit_counts[x], reverse=True)
            top_hit = hit_counts[sorted_all[0]]
            
            # No dynamic cutoff to preserve 1-hit true matches
            cands = sorted_all[:80]
            
            feats = []
            for rank, tid in enumerate(cands):
                t_data = target_records[tid]
                target_tuple = (t_data[0], t_data[1], t_data[2], t_data[3], t_data[4], t_data[5])
                hits = hit_counts[tid]
                is_s2 = 1.0 if tid.startswith('S2-') else 0.0
                feat = extract_pair_features(s1_tuple, target_tuple, hits, is_s2, rank)
                feats.append(feat)
                
            true_set = gt_map.get(s1_id, set())
            labels = [1 if tid in true_set else 0 for tid in cands]
            
            if feats:
                X_list.append(np.array(feats, dtype=np.float32))
                y_list.append(np.array(labels, dtype=np.int32))
                pair_list.extend([(s1_id, tid) for tid in cands])
            
        X = np.vstack(X_list) if X_list else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        y = np.concatenate(y_list) if y_list else np.empty((0,), dtype=np.int32)
        return X, y, pair_list

    print("  Extracting features for training set...")
    X_train, y_train, _ = extract_features_for_df(train_df)
    print(f"  Training feature matrix: {X_train.shape} with {int(y_train.sum())} positive pairs")
    
    print("  Extracting features for validation set...")
    X_val, y_val, val_pairs = extract_features_for_df(val_df)
    
    print("  Training initial model on 80% split...")
    clf = train_lgbm_model(X_train, y_train, save_path=None)
    
    print("  Predicting on validation set & Tuning Threshold...")
    probs = clf.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.58
    best_f05 = -1.0
    thresholds_to_test = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    for t in thresholds_to_test:
        f05 = compute_f05_score(val_df, val_pairs, probs, t, gt_map)
        print(f"    Threshold: {t:.2f} -> F0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t
            
    print(f"  Best Threshold Found: {best_threshold:.2f} (F0.5 = {best_f05:.4f})")
    
    print("  Retraining model on full dataset (Train + Val)...")
    X_full = np.vstack([X_train, X_val]) if X_val.shape[0] > 0 else X_train
    y_full = np.concatenate([y_train, y_val]) if y_val.shape[0] > 0 else y_train
    clf = train_lgbm_model(X_full, y_full, save_path=model_save_path)
    
    con.close()
    print(f"  Training stage completed in {time.time()-t0:.2f}s")
    return clf, best_threshold

def run_test_inference(test_dir, output_dir, model_path, threshold=0.58, max_cands=80, num_workers=10):
    """
    Executes inference over test_source1, test_source2, and test_source3 partitioned by country.
    """
    print(f"\n[Inference Stage] Starting country-partitioned inference with threshold {threshold:.2f}...")
    t_start = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")
    
    clf = load_lgbm_model(model_path)
    con = duckdb.connect()
    
    s1_path = os.path.join(test_dir, "test_source1.tsv").replace("\\", "/")
    s2_path = os.path.join(test_dir, "test_source2.tsv").replace("\\", "/")
    s3_path = os.path.join(test_dir, "test_source3.tsv").replace("\\", "/")
    
    # Discover all countries present in test_source1 (ordered: France first, then US, then India)
    countries = [row[0] for row in con.execute(f"SELECT DISTINCT country FROM read_csv('{s1_path}', delim='\\t', header=True)").fetchall()]
    countries = sorted(countries, key=lambda c: {'France': 0, 'US': 1, 'India': 2}.get(c, 99))
    print(f"Discovered test countries (execution order): {countries}", flush=True)
    
    # Initialize output files with headers
    with open(candidate_out, 'w', encoding='utf-8') as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(matching_out, 'w', encoding='utf-8') as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        
    total_processed_s1 = 0
    total_matches_found = 0
    
    for country in countries:
        c_t0 = time.time()
        print(f"\n--- Processing Country: {country} ---")
        
        # 1. Load target records for this country
        print(f"  Loading Source 2 and Source 3 targets for {country}...")
        targets_df = con.execute(f"""
            SELECT entity_id, business_name, business_address, country
            FROM read_csv('{s2_path}', delim='\\t', header=True) WHERE country = '{country}'
            UNION ALL
            SELECT entity_id, business_name, business_address, country
            FROM read_csv('{s3_path}', delim='\\t', header=True) WHERE country = '{country}'
        """).fetchdf()
        
        print(f"  Loaded {len(targets_df)} targets for {country}. Preprocessing records...")
        target_records = preprocess_target_records(targets_df)
        del targets_df
        gc.collect()
        
        # 2. Build inverted index for this country
        print(f"  Building Inverted Index for {country}...")
        inv_index = build_inverted_index(target_records)
        print(f"  Inverted Index built: {len(inv_index)} keys.")
        
        # 3. Load S1 entities for this country
        print(f"  Loading Source 1 records for {country}...")
        s1_country_df = con.execute(f"""
            SELECT entity_id, business_name, business_address 
            FROM read_csv('{s1_path}', delim='\\t', header=True) 
            WHERE country = '{country}'
        """).fetchdf()
        
        n_s1 = len(s1_country_df)
        print(f"  Loaded {n_s1} S1 records for {country}.")
        
        # 4. Process S1 entities in chunks using fast vectorized scoring
        chunk_size = 25000
        country_candidate_map = {}
        country_scored_pairs = []
        
        t_country_start = time.time()
        for chunk_start in range(0, n_s1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_s1)
            chunk_df = s1_country_df.iloc[chunk_start:chunk_end]
            
            chunk_pair_tuples = []
            chunk_feat_rows = []
            
            for r in chunk_df.itertuples(index=False):
                s1_id = r.entity_id
                s1_name = str(r.business_name) if pd.notna(r.business_name) else ""
                s1_addr = str(r.business_address) if pd.notna(r.business_address) else ""
                s1_clean = clean_business_name(s1_name)
                s1_nm_words = set(s1_clean.split())
                s1_ad_words = set(w for w in clean_text(s1_addr).split() if len(w) >= 3 and w not in ADDR_STOPWORDS)
                s1_nums = set(re.findall(r'\b\d+\b', s1_addr))
                s1_tuple = (s1_name, s1_addr, s1_clean, s1_nm_words, s1_ad_words, s1_nums)
                
                s1_keys = extract_name_keys(s1_name) + extract_addr_keys(s1_addr) + extract_combined_keys(s1_name, s1_addr)
                hit_counts = defaultdict(int)
                for k in set(s1_keys):
                    if k in inv_index:
                        wt = 3 if (k.startswith("cn:") or k.startswith("dom:")) else 1
                        for tid in inv_index[k]:
                            hit_counts[tid] += wt
                            
                if not hit_counts:
                    country_candidate_map[s1_id] = []
                    continue
                    
                sorted_all = sorted(hit_counts.keys(), key=lambda x: hit_counts[x], reverse=True)
                top_hit = hit_counts[sorted_all[0]]
                
                cands = sorted_all[:max_cands]
                country_candidate_map[s1_id] = cands
                
                for rank, tid in enumerate(cands):
                    t_data = target_records[tid]
                    target_tuple = (t_data[0], t_data[1], t_data[2], t_data[3], t_data[4], t_data[5])
                    hits = hit_counts[tid]
                    is_s2 = 1.0 if tid.startswith('S2-') else 0.0
                    feat = extract_pair_features(s1_tuple, target_tuple, hits, is_s2, rank)
                    chunk_pair_tuples.append((s1_id, tid))
                    chunk_feat_rows.append(feat)
                    
            if chunk_feat_rows:
                X_chunk = np.array(chunk_feat_rows, dtype=np.float32)
                probs = clf.predict_proba(X_chunk)[:, 1]
                for (s1_id, tid), prob in zip(chunk_pair_tuples, probs):
                    if prob >= threshold:
                        country_scored_pairs.append((prob, s1_id, tid))
                        
            elapsed = time.time() - t_country_start
            rate = chunk_end / elapsed if elapsed > 0 else 0
            eta = (n_s1 - chunk_end) / rate if rate > 0 else 0
            print(f"    Processed S1 {chunk_end}/{n_s1} records ({chunk_end*100.0/n_s1:.1f}%) | {rate:.1f} rec/s | ETA: {eta/60:.1f}m", flush=True)
                
        print(f"  All S1 records for {country} processed. Resolving 1-to-many constraint...")
        
        # Enforce 1-to-many constraint: sort by prob descending, assign each target to top S1
        country_scored_pairs.sort(key=lambda x: x[0], reverse=True)
        assigned_targets = set()
        country_matched_map = defaultdict(list)
        
        for prob, s1_id, tid in country_scored_pairs:
            if tid not in assigned_targets:
                assigned_targets.add(tid)
                country_matched_map[s1_id].append(tid)
                
        # Append country results to disk files
        print(f"  Writing {country} results to disk...")
        with open(candidate_out, 'a', encoding='utf-8') as fc:
            for s1_id in s1_country_df['entity_id']:
                cands = country_candidate_map.get(s1_id, [])
                fc.write(f"{s1_id}\t{','.join(cands)}\n")
                
        with open(matching_out, 'a', encoding='utf-8') as fm:
            for s1_id in s1_country_df['entity_id']:
                matches = country_matched_map.get(s1_id, [])
                fm.write(f"{s1_id}\t{','.join(matches)}\n")
                
        total_processed_s1 += n_s1
        total_matches_found += len(assigned_targets)
        print(f"  Country {country} completed in {time.time()-c_t0:.2f}s ({len(assigned_targets)} matches assigned).")
        
        # Clear country structures to release memory
        del target_records, inv_index, s1_country_df, country_candidate_map, country_scored_pairs, country_matched_map
        gc.collect()
        
    con.close()
    print(f"\n================ INFERENCE COMPLETE ================")
    print(f"Total S1 entities processed: {total_processed_s1}")
    print(f"Total target matches linked: {total_matches_found}")
    print(f"Total inference runtime: {time.time()-t_start:.2f}s")
    print(f"Output files:")
    print(f"  - {matching_out}")
    print(f"  - {candidate_out}")
    print("====================================================")

def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution End-to-End Pipeline")
    parser.add_argument("--train-dir", default="../../dataset/train", help="Path to training data directory")
    parser.add_argument("--test-dir", default="../../dataset/test", help="Path to test data directory")
    parser.add_argument("--output-dir", default="../../output", help="Path to output directory")
    parser.add_argument("--model-path", default="models/lgb_matcher.joblib", help="Path to save/load trained model")
    parser.add_argument("--threshold", type=float, default=0.58, help="Fallback probability threshold for matching")
    parser.add_argument("--max-cands", type=int, default=80, help="Max candidates per S1 entity")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel worker processes")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if model file exists")
    args = parser.parse_args()
    
    # 1. Train model and tune threshold if needed
    if not args.skip_train or not os.path.exists(args.model_path):
        clf, best_threshold = train_matching_model(args.train_dir, args.model_path, sample_size=200000, num_workers=args.num_workers)
    else:
        print(f"Found existing model at {args.model_path}, skipping training.")
        best_threshold = args.threshold
        
    print(f"Validation suggested threshold: {best_threshold}. Proceeding with this threshold.")
        
    # 2. Run inference on test data
    run_test_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=best_threshold,
        max_cands=args.max_cands,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
