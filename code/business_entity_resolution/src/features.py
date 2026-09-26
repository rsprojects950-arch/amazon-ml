"""
Feature Extraction for Business Entity Resolution matching model.
Uses rapidfuzz for ultra-fast C++ string distance computations.
"""

import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES = [
    'name_ratio', 'clean_ratio', 'clean_exact', 'token_sort', 'token_set', 'partial', 'name_jacc',
    'has_addr', 'addr_ratio', 'addr_token_sort', 'addr_token_set', 'addr_partial', 'addr_jacc',
    'num_overlap', 'num_match', 'hits', 'is_s2',
    'num_conflict', 'name_len_diff', 'addr_len_diff', 'name_token_intersection', 'hits_norm',
    'cross_name_addr', 'cross_addr_name', 'exact_substring', 'acronym_match', 'hits_rank',
    'jaro_winkler', 'name_word_diff', 'addr_word_diff'
]

def extract_pair_features(s1_tuple, target_tuple, hits, is_s2, hits_rank):
    """
    Extract pairwise string and numeric similarity features between S1 and target.
    
    s1_tuple: (s1_name, s1_addr, s1_clean, s1_nm_words, s1_ad_words, s1_nums)
    target_tuple: (t_name, t_addr, t_clean, t_nm_words, t_ad_words, t_nums)
    """
    s1_name, s1_addr, s1_clean, s1_nm_words, s1_ad_words, s1_nums = s1_tuple
    t_name, t_addr, t_clean, t_nm_words, t_ad_words, t_nums = target_tuple
    
    # 1. Name similarities
    name_ratio = fuzz.ratio(s1_name, t_name) / 100.0
    clean_ratio = fuzz.ratio(s1_clean, t_clean) / 100.0
    clean_exact = 1.0 if (s1_clean and s1_clean == t_clean) else 0.0
    token_sort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
    token_set = fuzz.token_set_ratio(s1_name, t_name) / 100.0
    partial = fuzz.partial_ratio(s1_name, t_name) / 100.0
    
    name_len_diff = abs(len(s1_clean) - len(t_clean)) / max(len(s1_clean), len(t_clean), 1)
    name_token_intersection = float(len(s1_nm_words & t_nm_words))
    
    if s1_nm_words or t_nm_words:
        name_jacc = len(s1_nm_words & t_nm_words) / len(s1_nm_words | t_nm_words)
    else:
        name_jacc = 0.0
        
    # 2. Address similarities
    has_addr = 1.0 if t_addr else 0.0
    if has_addr:
        addr_ratio = fuzz.ratio(s1_addr, t_addr) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0
        addr_partial = fuzz.partial_ratio(s1_addr, t_addr) / 100.0
        addr_len_diff = abs(len(s1_addr) - len(t_addr)) / max(len(s1_addr), len(t_addr), 1)
        
        if s1_ad_words or t_ad_words:
            addr_jacc = len(s1_ad_words & t_ad_words) / len(s1_ad_words | t_ad_words)
        else:
            addr_jacc = 0.0
            
        if s1_nums and t_nums:
            num_overlap = len(s1_nums & t_nums) / len(s1_nums | t_nums)
            num_match = 1.0 if (s1_nums & t_nums) else 0.0
            num_conflict = 1.0 if len(s1_nums & t_nums) == 0 else 0.0
        else:
            num_overlap = 0.0
            num_match = 0.0
            num_conflict = 0.0
    else:
        addr_ratio, addr_token_sort, addr_token_set, addr_partial, addr_jacc = 0.0, 0.0, 0.0, 0.0, 0.0
        addr_len_diff, num_overlap, num_match, num_conflict = 1.0, 0.0, 0.0, 0.0
        
    # 3. Cross-field matching (Name vs Address)
    if t_addr:
        cross_name_addr = fuzz.token_set_ratio(s1_name, t_addr) / 100.0
    else:
        cross_name_addr = 0.0
        
    if s1_addr:
        cross_addr_name = fuzz.token_set_ratio(s1_addr, t_name) / 100.0
    else:
        cross_addr_name = 0.0
        
    # 4. Hard Positives (Substring & Acronyms)
    if s1_clean and t_clean and (s1_clean in t_clean or t_clean in s1_clean):
        exact_substring = 1.0
    else:
        exact_substring = 0.0
        
    s1_clean_list = [w for w in s1_clean.split() if w] if s1_clean else []
    t_clean_list = [w for w in t_clean.split() if w] if t_clean else []
    s1_acr = "".join(w[0] for w in s1_clean_list[:5])
    t_acr = "".join(w[0] for w in t_clean_list[:5])
    
    if (s1_acr and s1_acr == t_clean) or (t_acr and t_acr == s1_clean):
        acronym_match = 1.0
    else:
        acronym_match = 0.0
        
    query_size = len(s1_nm_words) + len(s1_ad_words)
    hits_norm = float(hits) / max(query_size, 1)
    
    jaro_winkler = JaroWinkler.normalized_similarity(s1_clean, t_clean) if (s1_clean and t_clean) else 0.0
    name_word_diff = float(abs(len(s1_clean_list) - len(t_clean_list)))
    
    s1_ad_list = [w for w in s1_addr.split() if w] if s1_addr else []
    t_ad_list = [w for w in t_addr.split() if w] if t_addr else []
    addr_word_diff = float(abs(len(s1_ad_list) - len(t_ad_list)))
        
    return [
        name_ratio, clean_ratio, clean_exact, token_sort, token_set, partial, name_jacc,
        has_addr, addr_ratio, addr_token_sort, addr_token_set, addr_partial, addr_jacc,
        num_overlap, num_match, float(hits), float(is_s2),
        num_conflict, name_len_diff, addr_len_diff, name_token_intersection, hits_norm,
        cross_name_addr, cross_addr_name, exact_substring, acronym_match, float(hits_rank),
        jaro_winkler, name_word_diff, addr_word_diff
    ]
