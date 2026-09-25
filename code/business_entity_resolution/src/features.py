"""
Feature Extraction for Business Entity Resolution matching model.
Uses rapidfuzz for ultra-fast C++ string distance computations.
"""

import re
import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
    'name_ratio', 'clean_ratio', 'clean_exact', 'token_sort', 'token_set', 'partial', 'name_jacc',
    'has_addr', 'addr_ratio', 'addr_token_sort', 'addr_token_set', 'addr_partial', 'addr_jacc',
    'num_overlap', 'num_match', 'hits', 'is_s2'
]

def extract_pair_features(s1_tuple, target_tuple, hits, is_s2):
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
    
    # Name word Jaccard
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
        
        if s1_ad_words or t_ad_words:
            addr_jacc = len(s1_ad_words & t_ad_words) / len(s1_ad_words | t_ad_words)
        else:
            addr_jacc = 0.0
            
        if s1_nums and t_nums:
            num_overlap = len(s1_nums & t_nums) / len(s1_nums | t_nums)
            num_match = 1.0 if (s1_nums & t_nums) else 0.0
        else:
            num_overlap = 0.0
            num_match = 0.0
    else:
        addr_ratio = 0.0
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_partial = 0.0
        addr_jacc = 0.0
        num_overlap = 0.0
        num_match = 0.0
        
    return [
        name_ratio, clean_ratio, clean_exact, token_sort, token_set, partial, name_jacc,
        has_addr, addr_ratio, addr_token_sort, addr_token_set, addr_partial, addr_jacc,
        num_overlap, num_match, float(hits), float(is_s2)
    ]
