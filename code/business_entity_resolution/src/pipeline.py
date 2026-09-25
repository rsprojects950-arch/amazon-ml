"""
End-to-End Execution Pipeline for Business Entity Resolution.
Supports country-by-country chunking, multi-processing with shared memory initialization,
and 1-to-many constraint resolution.
"""

import os
import re
import time
import gc
from collections import defaultdict
import numpy as np
import pandas as pd

from .blocking import (
    clean_text, clean_business_name, extract_all_keys, extract_name_keys,
    extract_addr_keys, extract_combined_keys, ADDR_STOPWORDS
)
from .features import extract_pair_features

# Global worker references initialized once per process pool
_worker_inv_index = None
_worker_target_records = None

def init_worker(inv_index, target_records):
    """Initializes global worker state once when processes are spawned."""
    global _worker_inv_index, _worker_target_records
    _worker_inv_index = inv_index
    _worker_target_records = target_records

def preprocess_target_records(targets_df):
    """Preprocess target records for fast feature extraction and key generation."""
    target_records = {}
    for row in targets_df.itertuples(index=False):
        tid = str(row.entity_id)
        nm = str(row.business_name) if pd.notna(row.business_name) else ""
        ad = str(row.business_address) if pd.notna(row.business_address) else ""
        c_nm = clean_business_name(nm)
        nm_words = set(c_nm.split())
        ad_words = set(w for w in clean_text(ad).split() if len(w) >= 3 and w not in ADDR_STOPWORDS)
        nums = set(re.findall(r'\b\d+\b', ad))
        country = str(row.country)
        target_records[tid] = (nm, ad, c_nm, nm_words, ad_words, nums, country)
    return target_records

def build_inverted_index(target_records, max_posting=80, single_word_cap=40):
    """Build key -> list of target_ids inverted index with posting caps."""
    inv_index = defaultdict(list)
    for tid, (name, addr, c_nm, nm_words, ad_words, nums, country) in target_records.items():
        keys = extract_all_keys(name, addr)
        for k in keys:
            idx = inv_index[k]
            cap = single_word_cap if (k.startswith("w1:") or k.startswith("w_last:")) else max_posting
            if len(idx) < cap:
                idx.append(tid)
    return dict(inv_index)

def worker_process_s1_batch(batch_s1, max_cands=35):
    """
    Worker function executed in parallel processes.
    Uses pre-initialized global worker memory.
    """
    global _worker_inv_index, _worker_target_records
    inv_index = _worker_inv_index
    target_records = _worker_target_records
    
    batch_results = []
    for s1_id, s1_name, s1_addr in batch_s1:
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
            batch_results.append((s1_id, [], []))
            continue
            
        cands = sorted(hit_counts.keys(), key=lambda x: hit_counts[x], reverse=True)[:max_cands]
        
        features_list = []
        for tid in cands:
            t_data = target_records[tid]
            target_tuple = (t_data[0], t_data[1], t_data[2], t_data[3], t_data[4], t_data[5])
            hits = hit_counts[tid]
            is_s2 = 1.0 if tid.startswith('S2-') else 0.0
            
            feat = extract_pair_features(s1_tuple, target_tuple, hits, is_s2)
            features_list.append((tid, feat))
            
        batch_results.append((s1_id, cands, features_list))
    return batch_results
