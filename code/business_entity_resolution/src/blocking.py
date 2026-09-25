"""
Candidate Generation and Inverted Index Blocking for Business Entity Resolution.
"""

import re
import unicodedata
from collections import defaultdict

# Common legal suffixes to remove for clean name matching across US, India, and France
SUFFIXES = {
    # US / UK / International
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private', 'corp', 
    'corporation', 'co', 'company', 'services', 'lp', 'llp', 'gmbh', 'sa', 
    'group', 'holdings', 'enterprises', 'associates', 'solutions', 'technologies',
    'the', 'and', 'center', 'international', 'cie', 'centre',
    # France
    'sarl', 'sasu', 'sas', 'sci', 'eurl', 'freres', 'fils',
    # India
    'private limited', 'pvt ltd'
}

# Stopwords for addresses
ADDR_STOPWORDS = {
    # US / General
    'road', 'street', 'avenue', 'lane', 'drive', 'rd', 'st', 'ave', 'dr', 'ln',
    'near', 'opp', 'floor', 'plot', 'sector', 'nagar', 'city', 'town', 'block',
    'flat', 'bldg', 'building', 'dept', 'division', 'unit', 'suite', 'ste', 'apt',
    'house', 'no', 'hn', 'shop', 'shp', 'h', 'n', 'first', 'second', 'ground',
    'delhi', 'maharashtra', 'tamil', 'nadu', 'kerala', 'karnataka', 'india',
    'new', 'york', 'california', 'texas', 'florida', 'ohio', 'nc', 'ca', 'ny', 'tx', 'fl', 'oh',
    # France
    'rue', 'blvd', 'boulevard', 'france', 'paris', 'nord', 'gironde', 'route', 'allee', 'all',
    'cours', 'place', 'chemin', 'impasse'
}

def clean_text(text):
    """Normalize text: NFKD decomposition, lowercase, alphanumeric only."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9]', ' ', text)
    return text

def normalize_number(num_str):
    """Strip leading zeros from numeric strings (e.g. '0738' -> '738')."""
    s = num_str.lstrip('0')
    return s if s else '0'

def clean_business_name(name):
    """Remove legal suffixes from business name for invariant matching."""
    ct = clean_text(name)
    words = [w for w in ct.split() if w and w not in SUFFIXES]
    return " ".join(words) if words else ct

def extract_name_keys(name):
    """Generate multi-resolution blocking keys from business name."""
    if not name or not isinstance(name, str):
        return []
    keys = []
    lower_name = name.lower()
    
    # Check for DBA
    if 'dba:' in lower_name or 'dba ' in lower_name:
        parts = re.split(r'dba[:\s]+', lower_name)
        for p in parts:
            keys.extend(extract_name_keys(p))
            
    # Check for domain / website
    if any(ext in lower_name for ext in ['.com', '.in', '.org', '.net', '.co', '.fr']):
        m = re.search(r'([a-z0-9]{3,})\.(?:com|in|org|net|co|fr)', lower_name)
        if m:
            keys.append(f"dom:{m.group(1)}")
            
    ct = clean_text(name)
    words = [w for w in ct.split() if w]
    if not words:
        return keys
        
    clean_words = [w for w in words if w not in SUFFIXES]
    if not clean_words:
        clean_words = words
        
    full_clean = "".join(clean_words)
    if len(full_clean) >= 4:
        keys.append(f"cn:{full_clean}")
        if len(full_clean) >= 6:
            keys.append(f"cnp6:{full_clean[:6]}")
            
    if len(clean_words[0]) >= 4:
        keys.append(f"w1:{clean_words[0]}")
        
    if len(clean_words) >= 2 and len(clean_words[-1]) >= 4:
        keys.append(f"w_last:{clean_words[-1]}")
        keys.append(f"w2:{clean_words[0]}_{clean_words[1]}")
        
    return keys

def extract_addr_keys(addr):
    """Generate blocking keys from address tokens, numbers, and codes."""
    if not addr or not isinstance(addr, str):
        return []
    ct = clean_text(addr)
    tokens = [w for w in ct.split() if w]
    numbers = [normalize_number(t) for t in tokens if t.isdigit() and len(t) <= 6]
    words = [t for t in tokens if not t.isdigit() and len(t) >= 4 and t not in ADDR_STOPWORDS]
    
    keys = []
    if numbers and words:
        for w in words[:2]:
            keys.append(f"num_w:{numbers[0]}_{w}")
            
    if len(words) >= 2:
        keys.append(f"aw2:{words[0]}_{words[1]}")
        if len(words) >= 3:
            keys.append(f"aw2:{words[0]}_{words[2]}")
            
    for num in numbers:
        if len(num) == 6 and words:  # Indian PIN code
            keys.append(f"pin_w:{num}_{words[0]}")
        elif len(num) == 5 and words: # US/FR ZIP code
            keys.append(f"zip_w:{num}_{words[0]}")
            
    return keys

def extract_combined_keys(name, addr):
    """Generate cross-field keys linking business name stem with address identifier."""
    if not name or not addr:
        return []
    n_ct = clean_text(name)
    n_words = [w for w in n_ct.split() if w and w not in SUFFIXES]
    a_ct = clean_text(addr)
    a_tokens = [w for w in a_ct.split() if w]
    a_words = [t for t in a_tokens if not t.isdigit() and len(t) >= 4 and t not in ADDR_STOPWORDS]
    a_nums = [normalize_number(t) for t in a_tokens if t.isdigit() and len(t) <= 6]
    
    keys = []
    if n_words and a_words:
        keys.append(f"nw_aw:{n_words[0]}_{a_words[0]}")
    if n_words and a_nums:
        keys.append(f"nw_num:{n_words[0]}_{a_nums[0]}")
        
    return keys

def extract_all_keys(name, addr):
    """Extract full set of blocking keys for an entity."""
    k1 = extract_name_keys(name)
    k2 = extract_addr_keys(addr)
    k3 = extract_combined_keys(name, addr)
    return set(k1 + k2 + k3)
