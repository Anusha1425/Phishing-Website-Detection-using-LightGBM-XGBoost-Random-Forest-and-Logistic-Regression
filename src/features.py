
import re
from urllib.parse import urlparse

def extract_features(url):
    """
    Extracts 14 structural and lexical features from a given URL.
    This logic is shared between the training pipeline and real-time inference.
    """
    try:
        parsed = urlparse(url)
        features = {
            # Lexical Features
            'url_length': len(url),
            'num_dots': url.count('.'),
            'num_hyphen': url.count('-'),
            'num_at': url.count('@'),
            'num_question': url.count('?'),
            'num_equal': url.count('='),
            'num_and': url.count('&'),
            'num_percent': url.count('%'),
            'num_slash': url.count('/'),
            'num_www': url.count('www'),
            
            # Host-based & Protocol Features
            'num_http': url.count('http'),
            'num_https': url.count('https'),
            'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
            
            # Directory / Path Features
            'path_length': len(parsed.path)
        }
        return list(features.values())
    except Exception as e:
        # Fallback for malformed URLs
        return [0] * 14

def get_feature_names():
    return [
        'url_length', 'num_dots', 'num_hyphen', 'num_at', 'num_question', 
        'num_equal', 'num_and', 'num_percent', 'num_slash', 'num_www', 
        'num_http', 'num_https', 'has_ip', 'path_length'
    ]
