"""
Input validation module for detecting invalid inputs like Tagalog and gibberish.
"""

import re
import math
from collections import Counter


# Common Tagalog words and patterns
TAGALOG_WORDS = {
    'ang', 'ng', 'sa', 'na', 'ay', 'mga', 'ito', 'iyan', 'iyon', 'siya', 'kami', 'tayo', 'kayo',
    'ako', 'ikaw', 'sila', 'namin', 'ninyo', 'nila', 'ko', 'mo', 'niya', 'amin', 'inyo', 'kanila',
    'at', 'o', 'pero', 'kung', 'kapag', 'dahil', 'kaya', 'kasi', 'dapat', 'pwede', 'puwede',
    'maganda', 'mahal', 'mura', 'mabuti', 'masama', 'salamat', 'po', 'opo', 'hindi', 'oo',
    'kumusta', 'kamusta', 'paalam', 'sige', 'tulong', 'tulungan', 'bili', 'bumili', 'bumibili',
    'gusto', 'gusto ko', 'ayaw', 'mahal ko', 'mahal kita', 'mahal na mahal', 'salamat po',
    'pangit', 'maganda', 'masarap', 'masama', 'mabait', 'masungit', 'mabuti', 'masama'
}

TAGALOG_PATTERNS = [
    r'\b(ng|na|ang|sa|mga|ay|ito|iyan|iyon)\b',  # Common particles
    r'\b(salamat|po|opo|hindi|oo|kumusta|kamusta)\b',  # Common words
    r'\b(mahal|maganda|mura|mabuti|masama|pangit|masarap)\b',  # Common adjectives
]


def detect_tagalog(text):
    """
    Detect if text contains Tagalog language.
    
    Args:
        text: Input text string
        
    Returns:
        bool: True if Tagalog is detected
    """
    if not text or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    words = text_lower.split()
    
    # Check for Tagalog words
    tagalog_word_count = sum(1 for word in words if word in TAGALOG_WORDS)
    
    # If more than 20% of words are Tagalog, consider it Tagalog
    if len(words) > 0 and (tagalog_word_count / len(words)) > 0.2:
        return True
    
    # Check for Tagalog patterns
    for pattern in TAGALOG_PATTERNS:
        if re.search(pattern, text_lower):
            # If pattern found and text is short, likely Tagalog
            if len(words) < 20:
                return True
            # If multiple patterns found, likely Tagalog
            matches = len(re.findall(pattern, text_lower))
            if matches >= 2:
                return True
    
    return False


def detect_gibberish(text):
    """
    Detect if text is gibberish (random letters, very low linguistic quality).
    
    Args:
        text: Input text string
        
    Returns:
        bool: True if gibberish is detected
    """
    if not text or not isinstance(text, str):
        return False
    
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) == 0:
        return False
    
    # Check 1: Very short average word length (likely random letters)
    avg_word_length = sum(len(w) for w in words) / len(words)
    if avg_word_length < 3.0 and len(words) > 3:
        return True
    
    # Check 2: Very low lexical diversity (repeating same random letters)
    unique_words = len(set(words))
    lexical_diversity = unique_words / len(words) if len(words) > 0 else 0
    
    # Check 3: High repetition of very short "words" (like "asdf", "qwerty")
    short_words = [w for w in words if len(w) <= 4]
    if len(short_words) > 0:
        short_word_ratio = len(short_words) / len(words)
        # If most words are short and there's low diversity, likely gibberish
        if short_word_ratio > 0.7 and lexical_diversity < 0.5:
            return True
    
    # Check 4: Check for patterns of random letter sequences
    # Look for words that don't follow English patterns (too many consonants in a row)
    consonant_clusters = 0
    for word in words:
        # Count unusual consonant clusters (3+ consonants in a row)
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{3,}', word, re.IGNORECASE):
            consonant_clusters += 1
    
    if len(words) > 0 and (consonant_clusters / len(words)) > 0.5:
        return True
    
    # Check 5: Very low word entropy (repetitive patterns)
    if len(words) > 0:
        word_freq = Counter(words)
        total = len(words)
        import math
        entropy = -sum((count / total) * math.log2(count / total) 
                      for count in word_freq.values() if count > 0)
        
        # Very low entropy suggests gibberish
        if entropy < 1.0 and len(words) > 5:
            return True
    
    # Check 6: All words are very short (1-2 letters) - likely random typing
    if len(words) >= 5:
        very_short = sum(1 for w in words if len(w) <= 2)
        if very_short / len(words) > 0.8:
            return True
    
    # Check 7: Long words with unusual consonant patterns (like "kjashduikqweha")
    # Check for words longer than 8 characters with 3+ consecutive consonants
    long_words_with_clusters = 0
    for word in words:
        if len(word) > 8:
            # Check for 3+ consecutive consonants
            if re.search(r'[bcdfghjklmnpqrstvwxyz]{3,}', word, re.IGNORECASE):
                long_words_with_clusters += 1
    
    # If we have long words with unusual patterns, likely gibberish
    if len(words) > 0:
        long_word_ratio = sum(1 for w in words if len(w) > 8) / len(words)
        if long_word_ratio > 0.3 and long_words_with_clusters > 0:
            return True
    
    # Check 8: Single long word that looks like random typing (like "kjashduikqweha")
    if len(words) == 1 and len(words[0]) > 10:
        word = words[0]
        # Check for unusual patterns: too many consonants, no vowels in long stretches
        # Count vowels
        vowels = sum(1 for c in word if c in 'aeiou')
        vowel_ratio = vowels / len(word) if len(word) > 0 else 0
        
        # If very few vowels (< 20%) in a long word, likely gibberish
        if vowel_ratio < 0.2:
            return True
        
        # Check for 4+ consecutive consonants
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{4,}', word, re.IGNORECASE):
            return True
    
    return False


def validate_input(text):
    """
    Validate input text for language and gibberish.
    
    Args:
        text: Input text string
        
    Returns:
        tuple: (is_valid, error_message)
        - is_valid: bool, True if input is valid
        - error_message: str, error message if invalid, None if valid
    """
    if not text or not isinstance(text, str):
        return False, "Invalid input: text must be a non-empty string"
    
    text = text.strip()
    
    if len(text) == 0:
        return False, "Invalid input: text cannot be empty"
    
    # Check for Tagalog
    if detect_tagalog(text):
        return False, "Invalid data entry"
    
    # Check for gibberish
    if detect_gibberish(text):
        return False, "Invalid data entry"
    
    return True, None
