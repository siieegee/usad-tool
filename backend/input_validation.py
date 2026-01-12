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
    
    # Check 1: Repetitive character patterns (like "adadada", "ababab", "aaaa")
    for word in words:
        if len(word) >= 3:
            # Check for repeating 2-3 character patterns
            # "adadada" -> "ad" repeats
            # "ababab" -> "ab" repeats
            for pattern_len in [2, 3]:
                if len(word) >= pattern_len * 2:
                    pattern = word[:pattern_len]
                    # Check if pattern repeats throughout the word
                    repeats = word.count(pattern)
                    if repeats >= 3:  # Pattern appears 3+ times
                        return True
            
            # Check for single character repetition (like "aaaa", "bbbb")
            if len(word) >= 4:
                char_counts = Counter(word)
                max_char_count = max(char_counts.values())
                # If one character makes up >70% of the word, likely gibberish
                if max_char_count / len(word) > 0.7:
                    return True
    
    # Check 2: Low character variety within words (like "adadada" - only 2 unique chars)
    for word in words:
        if len(word) >= 4:
            unique_chars = len(set(word))
            char_variety = unique_chars / len(word)
            # If word has very few unique characters (< 40% variety), likely repetitive gibberish
            if char_variety < 0.4:
                return True
    
    # Check 3: Words that are too repetitive (same 2-3 chars repeating)
    for word in words:
        if len(word) >= 6:
            # Check if word is mostly made of 2-3 unique characters
            unique_chars = len(set(word))
            if unique_chars <= 3:
                return True
    
    # Check 4: Very short average word length (likely random letters)
    avg_word_length = sum(len(w) for w in words) / len(words)
    if avg_word_length < 3.0 and len(words) > 3:
        return True
    
    # Check 5: Very low lexical diversity (repeating same random letters)
    unique_words = len(set(words))
    lexical_diversity = unique_words / len(words) if len(words) > 0 else 0
    
    # Stricter: if diversity is very low, likely gibberish
    if lexical_diversity < 0.4 and len(words) > 2:
        return True
    
    # Check 6: High repetition of very short "words" (like "asdf", "qwerty")
    short_words = [w for w in words if len(w) <= 4]
    if len(short_words) > 0:
        short_word_ratio = len(short_words) / len(words)
        # If most words are short and there's low diversity, likely gibberish
        if short_word_ratio > 0.6 and lexical_diversity < 0.5:
            return True
    
    # Check 7: Check for patterns of random letter sequences
    # Look for words that don't follow English patterns (too many consonants in a row)
    consonant_clusters = 0
    for word in words:
        # Count unusual consonant clusters (3+ consonants in a row)
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{3,}', word, re.IGNORECASE):
            consonant_clusters += 1
    
    if len(words) > 0 and (consonant_clusters / len(words)) > 0.4:
        return True
    
    # Check 8: Very low word entropy (repetitive patterns)
    if len(words) > 0:
        word_freq = Counter(words)
        total = len(words)
        import math
        entropy = -sum((count / total) * math.log2(count / total) 
                      for count in word_freq.values() if count > 0)
        
        # Stricter: lower entropy threshold
        if entropy < 1.5 and len(words) > 3:
            return True
    
    # Check 9: All words are very short (1-2 letters) - likely random typing
    if len(words) >= 3:
        very_short = sum(1 for w in words if len(w) <= 2)
        if very_short / len(words) > 0.7:
            return True
    
    # Check 10: Long words with unusual consonant patterns (like "kjashduikqweha")
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
    
    # Check 11: Single long word that looks like random typing (like "kjashduikqweha")
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
