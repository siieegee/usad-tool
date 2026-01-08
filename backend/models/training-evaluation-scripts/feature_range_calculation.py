"""
Feature Range Calculation & Validation for USAD Tool

This script implements the complete procedure for calculating and validating
feature ranges for genuine product reviews, following the step-by-step methodology.

PHASES:
1. Data Preparation - Assemble baseline dataset of genuine reviews
2. Feature Extraction - Extract all 11 features from each review
3. Range Calculation - Calculate statistical measures and define normal ranges
4. Range Validation - Test ranges on validation dataset and calculate metrics
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import re
from collections import Counter
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Download NLTK resources
NLTK_RESOURCES = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng']
for resource in NLTK_RESOURCES:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

# Constants
TRAINING_DATA_DIR = "training-data"
BASELINE_DATA_DIR = "baseline-range-data"
VALIDATION_DATA_DIR = "baseline-range-validation"
RANGE_REPORTS_DIR = "range-reports"

# Feature names mapping
FEATURE_NAMES = {
    'review_length': 'Length of Review',
    'lexical_diversity': 'Word Variety',
    'avg_word_length': 'Average Word Length',
    'sentiment_polarity': 'Overall Tone',
    'sentiment_subjectivity': 'Opinion Level',
    'language_complexity': 'Language Complexity',
    'repetition_ratio': 'Word Repetition',
    'exclamation_count': 'Exclamation Marks',
    'question_count': 'Question Marks',
    'capital_usage': 'Capital Letter Usage',
    'punctuation_density': 'Punctuation Density'
}

FEATURE_ORDER = [
    'review_length', 'lexical_diversity', 'avg_word_length',
    'sentiment_polarity', 'sentiment_subjectivity', 'language_complexity',
    'repetition_ratio', 'exclamation_count', 'question_count',
    'capital_usage', 'punctuation_density'
]


def get_base_models_dir():
    """Get the base models directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def get_training_data_dir():
    """Get the training data directory"""
    base_dir = get_base_models_dir()
    data_dir = os.path.join(base_dir, TRAINING_DATA_DIR)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def get_baseline_data_dir():
    """Get the baseline data directory for range calculation"""
    base_dir = get_base_models_dir()
    data_dir = os.path.join(base_dir, BASELINE_DATA_DIR)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def get_validation_data_dir():
    """Get the validation data directory"""
    base_dir = get_base_models_dir()
    data_dir = os.path.join(base_dir, VALIDATION_DATA_DIR)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def get_range_reports_dir():
    """Get the range reports directory"""
    base_dir = get_base_models_dir()
    reports_dir = os.path.join(base_dir, RANGE_REPORTS_DIR)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    return reports_dir


# ============================================================================
# PHASE 1: DATA PREPARATION
# ============================================================================

def preprocess_text(text):
    """
    Preprocess review text: clean, tokenize, remove stopwords, lemmatize.
    
    Args:
        text: Raw review text string
        
    Returns:
        tuple: (original_text, processed_tokens)
    """
    if pd.isna(text) or not isinstance(text, str):
        return str(text), []
    
    original_text = text
    
    # Clean: lowercase, remove special chars (keep punctuation for some features)
    cleaned = text.lower()
    # Remove URLs, emails
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\S+@\S+', '', cleaned)
    # Remove HTML tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    # Handle contractions (basic)
    contractions = {
        "don't": "do not", "won't": "will not", "can't": "cannot",
        "n't": " not", "'re": " are", "'ve": " have", "'ll": " will"
    }
    for cont, exp in contractions.items():
        cleaned = cleaned.replace(cont, exp)
    
    # Tokenize
    tokens = word_tokenize(cleaned)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and w.isalnum()]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
                  for word, tag in pos_tags]
    
    return original_text, lemmatized


# ============================================================================
# PHASE 2: FEATURE EXTRACTION
# ============================================================================

def calculate_flesch_kincaid(text):
    """
    Calculate Flesch-Kincaid Readability Score.
    
    Formula: 206.835 - (1.015 * ASL) - (84.6 * ASW)
    where ASL = average sentence length, ASW = average syllables per word
    
    Args:
        text: Review text string
        
    Returns:
        float: Flesch-Kincaid score (0-100, higher = easier to read)
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    try:
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return 0.0
        
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum()]
        
        if len(words) == 0:
            return 0.0
        
        # Count syllables (approximate: count vowel groups)
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            vowels = 'aeiouy'
            count = 0
            prev_was_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    count += 1
                prev_was_vowel = is_vowel
            if word.endswith('e'):
                count -= 1
            return max(1, count)
        
        total_syllables = sum(count_syllables(w) for w in words)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Flesch-Kincaid formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Clamp to 0-100 range
        return max(0.0, min(100.0, score))
    
    except Exception as e:
        return 0.0


def extract_all_features(original_text, processed_tokens):
    """
    Extract all 11 features from a review.
    
    Args:
        original_text: Original review text
        processed_tokens: List of processed tokens
        
    Returns:
        dict: Dictionary of all 11 features
    """
    features = {}
    
    # 1. Length of Review (word count)
    features['review_length'] = len(processed_tokens)
    
    # 2. Word Variety (lexical diversity)
    if len(processed_tokens) > 0:
        unique_tokens = len(set(processed_tokens))
        features['lexical_diversity'] = unique_tokens / len(processed_tokens)
    else:
        features['lexical_diversity'] = 0.0
    
    # 3. Average Word Length
    if len(processed_tokens) > 0:
        features['avg_word_length'] = np.mean([len(w) for w in processed_tokens])
    else:
        features['avg_word_length'] = 0.0
    
    # 4. Overall Tone (sentiment polarity)
    if len(processed_tokens) > 0:
        joined_text = " ".join(processed_tokens)
        blob = TextBlob(joined_text)
        features['sentiment_polarity'] = blob.sentiment.polarity  # -1 to 1
    else:
        features['sentiment_polarity'] = 0.0
    
    # 5. Opinion Level (sentiment subjectivity)
    if len(processed_tokens) > 0:
        joined_text = " ".join(processed_tokens)
        blob = TextBlob(joined_text)
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity  # 0 to 1
    else:
        features['sentiment_subjectivity'] = 0.0
    
    # 6. Language Complexity (Flesch-Kincaid readability)
    features['language_complexity'] = calculate_flesch_kincaid(original_text)
    
    # 7. Word Repetition
    if len(processed_tokens) > 0:
        unique_count = len(set(processed_tokens))
        features['repetition_ratio'] = (len(processed_tokens) - unique_count) / len(processed_tokens)
    else:
        features['repetition_ratio'] = 0.0
    
    # 8. Exclamation Marks
    features['exclamation_count'] = original_text.count('!')
    
    # 9. Question Marks
    features['question_count'] = original_text.count('?')
    
    # 10. Capital Letter Usage (ratio of words in ALL CAPS)
    words = original_text.split()
    if len(words) > 0:
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        features['capital_usage'] = caps_words / len(words)
    else:
        features['capital_usage'] = 0.0
    
    # 11. Punctuation Density
    if len(processed_tokens) > 0:
        punct_chars = len([c for c in original_text if c in '!?.,-;:'])
        features['punctuation_density'] = punct_chars / len(processed_tokens)
    else:
        features['punctuation_density'] = 0.0
    
    return features


def extract_features_from_dataset(df, review_column='review', label_column=None):
    """
    Extract all features from a dataset.
    
    Args:
        df: DataFrame with reviews
        review_column: Name of column containing review text
        label_column: Optional name of label column
        
    Returns:
        DataFrame: DataFrame with extracted features
    """
    print(f"\nExtracting features from {len(df)} reviews...")
    
    results = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing review {idx+1}/{len(df)}...")
        
        review_text = row[review_column]
        original_text, processed_tokens = preprocess_text(review_text)
        features = extract_all_features(original_text, processed_tokens)
        
        result = {
            'review_id': idx + 1,
            'original_review': original_text,
            **features
        }
        
        if label_column and label_column in row:
            result['label'] = row[label_column]
        
        results.append(result)
    
    feature_df = pd.DataFrame(results)
    print(f"✓ Feature extraction complete: {len(feature_df)} reviews processed")
    
    return feature_df


# ============================================================================
# PHASE 3: RANGE CALCULATION
# ============================================================================

def calculate_statistics(feature_series):
    """
    Calculate statistical measures for a feature.
    
    Args:
        feature_series: pandas Series of feature values
        
    Returns:
        dict: Statistical measures
    """
    stats = {
        'mean': float(feature_series.mean()),
        'std': float(feature_series.std()),
        'min': float(feature_series.min()),
        'max': float(feature_series.max()),
        'p5': float(feature_series.quantile(0.05)),
        'p25': float(feature_series.quantile(0.25)),
        'p50': float(feature_series.quantile(0.50)),  # median
        'p75': float(feature_series.quantile(0.75)),
        'p95': float(feature_series.quantile(0.95))
    }
    return stats


def calculate_normal_ranges(feature_df, method='std', std_multiplier=1.5):
    """
    Calculate normal ranges for all features.
    
    Args:
        feature_df: DataFrame with extracted features
        method: 'std' for standard deviation method, 'percentile' for percentile method
        std_multiplier: Multiplier for standard deviation (default 1.5)
        
    Returns:
        dict: Dictionary of normal ranges for each feature
    """
    print(f"\nCalculating normal ranges using {method} method...")
    
    ranges = {}
    statistics = {}
    
    for feature in FEATURE_ORDER:
        if feature not in feature_df.columns:
            print(f"  Warning: Feature '{feature}' not found, skipping...")
            continue
        
        values = feature_df[feature]
        
        # Calculate statistics
        stats = calculate_statistics(values)
        statistics[feature] = stats
        
        # Calculate normal range
        if method == 'std':
            # Use mean ± (std_multiplier * std)
            lower = stats['mean'] - (std_multiplier * stats['std'])
            upper = stats['mean'] + (std_multiplier * stats['std'])
        else:  # percentile method
            # Use 5th to 95th percentile
            lower = stats['p5']
            upper = stats['p95']
        
        ranges[feature] = {
            'normal_min': float(lower),
            'normal_max': float(upper),
            'statistics': stats
        }
        
        print(f"  {FEATURE_NAMES[feature]}: [{lower:.2f}, {upper:.2f}]")
    
    return ranges, statistics


# ============================================================================
# PHASE 4: RANGE VALIDATION
# ============================================================================

def check_feature_against_range(value, normal_min, normal_max):
    """
    Check if a feature value is within normal range.
    
    Args:
        value: Feature value
        normal_min: Minimum of normal range
        normal_max: Maximum of normal range
        
    Returns:
        bool: True if within range (PASS), False if outside (WARNING)
    """
    return normal_min <= value <= normal_max


def validate_review_features(features_dict, ranges):
    """
    Validate all features of a review against normal ranges.
    
    Args:
        features_dict: Dictionary of feature values
        ranges: Dictionary of normal ranges
        
    Returns:
        dict: Validation results
    """
    results = {}
    warnings = []
    passes = []
    
    for feature in FEATURE_ORDER:
        if feature not in features_dict or feature not in ranges:
            continue
        
        value = features_dict[feature]
        normal_min = ranges[feature]['normal_min']
        normal_max = ranges[feature]['normal_max']
        
        is_normal = check_feature_against_range(value, normal_min, normal_max)
        
        results[feature] = {
            'value': float(value),
            'normal_min': normal_min,
            'normal_max': normal_max,
            'status': 'PASS' if is_normal else 'WARNING'
        }
        
        if is_normal:
            passes.append(feature)
        else:
            warnings.append(feature)
    
    # Calculate suspiciousness score
    total_features = len(results)
    warning_count = len(warnings)
    suspiciousness_score = warning_count / total_features if total_features > 0 else 0.0
    
    # Determine if review is suspicious (threshold: >50% features outside range)
    is_suspicious = suspiciousness_score > 0.5
    
    return {
        'results': results,
        'warnings': warnings,
        'passes': passes,
        'warning_count': warning_count,
        'total_features': total_features,
        'suspiciousness_score': suspiciousness_score,
        'is_suspicious': is_suspicious
    }


def validate_dataset(feature_df, ranges, label_column='label'):
    """
    Validate entire dataset against normal ranges.
    
    Args:
        feature_df: DataFrame with extracted features
        ranges: Dictionary of normal ranges
        label_column: Name of label column (if exists)
        
    Returns:
        DataFrame: DataFrame with validation results
    """
    print(f"\nValidating {len(feature_df)} reviews against normal ranges...")
    
    validation_results = []
    
    for idx, row in feature_df.iterrows():
        if idx % 100 == 0:
            print(f"  Validating review {idx+1}/{len(feature_df)}...")
        
        features_dict = {feat: row[feat] for feat in FEATURE_ORDER if feat in row}
        validation = validate_review_features(features_dict, ranges)
        
        result = {
            'review_id': row.get('review_id', idx + 1),
            'predicted_suspicious': validation['is_suspicious'],
            'warning_count': validation['warning_count'],
            'suspiciousness_score': validation['suspiciousness_score']
        }
        
        if label_column and label_column in row:
            result['actual_label'] = row[label_column]
            # Convert to binary: 'Anomalous' or 'Fraudulent' = 1, else 0
            actual_binary = 1 if str(row[label_column]).lower() in ['anomalous', 'fraudulent', 'fake', '1'] else 0
            result['actual_binary'] = actual_binary
        
        validation_results.append(result)
    
    validation_df = pd.DataFrame(validation_results)
    print(f"✓ Validation complete")
    
    return validation_df


def calculate_performance_metrics(validation_df, label_column='actual_binary'):
    """
    Calculate performance metrics from validation results.
    
    Args:
        validation_df: DataFrame with validation results
        label_column: Name of actual label column
        
    Returns:
        dict: Performance metrics
    """
    if label_column not in validation_df.columns:
        print(f"Warning: Label column '{label_column}' not found. Cannot calculate metrics.")
        return None
    
    y_true = validation_df[label_column].values
    y_pred = validation_df['predicted_suspicious'].astype(int).values
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'false_alarm_rate': float(false_alarm_rate),
        'detection_rate': float(detection_rate),
        'confusion_matrix': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    }
    
    return metrics


def print_metrics_report(metrics):
    """Print formatted metrics report."""
    print("\n" + "=" * 80)
    print("VALIDATION PERFORMANCE METRICS")
    print("=" * 80)
    
    print(f"\nPrimary Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:          {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall/Sensitivity: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:           {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    print(f"\nAdditional Metrics:")
    print(f"  Specificity:        {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"  False Alarm Rate:   {metrics['false_alarm_rate']:.4f} ({metrics['false_alarm_rate']*100:.2f}%)")
    print(f"  Detection Rate:     {metrics['detection_rate']:.4f} ({metrics['detection_rate']*100:.2f}%)")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Suspicious  Normal")
    print(f"Actual Fraudulent    {cm['tp']:5d}    {cm['fn']:5d}")
    print(f"       Genuine       {cm['fp']:5d}    {cm['tn']:5d}")
    
    print(f"\nInterpretation:")
    print(f"  True Positives (TP):  {cm['tp']:4d} - Correctly caught fraudulent reviews")
    print(f"  True Negatives (TN):  {cm['tn']:4d} - Correctly identified genuine reviews")
    print(f"  False Positives (FP): {cm['fp']:4d} - Genuine reviews wrongly flagged")
    print(f"  False Negatives (FN): {cm['fn']:4d} - Fraudulent reviews that slipped through")
    
    print(f"\nPerformance Assessment:")
    if metrics['accuracy'] >= 0.85 and metrics['precision'] >= 0.80 and metrics['recall'] >= 0.85:
        print(f"  ✓ EXCELLENT - Performance meets all criteria")
        print(f"  ✓ Ranges are APPROVED for production use")
    elif metrics['accuracy'] >= 0.80 and metrics['precision'] >= 0.75 and metrics['recall'] >= 0.80:
        print(f"  ✓ GOOD - Performance is acceptable")
        print(f"  ✓ Ranges are APPROVED for production use")
    else:
        print(f"  ✗ POOR - Performance below acceptable thresholds")
        print(f"  ✗ Ranges need REFINEMENT before production use")
        print(f"\n  Recommendations:")
        if metrics['precision'] < 0.80:
            print(f"    - Too many false positives - expand normal ranges (use larger std_multiplier)")
        if metrics['recall'] < 0.85:
            print(f"    - Too many false negatives - narrow normal ranges (use smaller std_multiplier)")
    
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def save_range_report(ranges, statistics, metrics, output_path):
    """Save range calculation and validation report to JSON."""
    report = {
        'feature_ranges': {},
        'feature_statistics': {},
        'validation_metrics': metrics,
        'methodology': {
            'range_method': 'std_deviation',
            'std_multiplier': 1.5,
            'feature_count': len(FEATURE_ORDER)
        }
    }
    
    for feature in FEATURE_ORDER:
        if feature in ranges:
            report['feature_ranges'][feature] = {
                'name': FEATURE_NAMES[feature],
                'normal_min': ranges[feature]['normal_min'],
                'normal_max': ranges[feature]['normal_max']
            }
        
        if feature in statistics:
            report['feature_statistics'][feature] = statistics[feature]
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Range report saved to: {output_path}")


def main():
    """
    Main execution function for feature range calculation and validation.
    
    This function implements the complete procedure:
    1. Load baseline dataset (genuine reviews)
    2. Extract features
    3. Calculate normal ranges
    4. Load validation dataset
    5. Validate ranges
    6. Calculate metrics
    7. Save report
    """
    print("=" * 80)
    print("FEATURE RANGE CALCULATION & VALIDATION FOR USAD")
    print("=" * 80)
    
    # Get directories
    baseline_dir = get_baseline_data_dir()
    validation_dir = get_validation_data_dir()
    reports_dir = get_range_reports_dir()
    
    # PHASE 1 & 2: Load baseline dataset and extract features
    print("\n" + "=" * 80)
    print("PHASE 1 & 2: DATA PREPARATION & FEATURE EXTRACTION")
    print("=" * 80)
    
    baseline_file = os.path.join(baseline_dir, 'baseline_genuine_reviews.csv')
    
    if not os.path.exists(baseline_file):
        print(f"\nError: Baseline dataset not found at: {baseline_file}")
        print("\nPlease create a CSV file with the following structure:")
        print("  - Column 'review': Review text")
        print("  - Optional: Column 'label': 'Genuine' or 'Normal'")
        print(f"\nExpected location: {baseline_file}")
        print("\nYou can use existing training data or create a new dataset.")
        return
    
    print(f"\nLoading baseline dataset from: {baseline_file}")
    baseline_df = pd.read_csv(baseline_file)
    print(f"✓ Loaded {len(baseline_df)} reviews")
    
    # Extract features
    baseline_features_df = extract_features_from_dataset(
        baseline_df, 
        review_column='review',
        label_column='label' if 'label' in baseline_df.columns else None
    )
    
    # Save baseline features
    baseline_features_file = os.path.join(baseline_dir, 'baseline_features.csv')
    baseline_features_df.to_csv(baseline_features_file, index=False)
    print(f"✓ Baseline features saved to: {baseline_features_file}")
    
    # PHASE 3: Calculate normal ranges
    print("\n" + "=" * 80)
    print("PHASE 3: RANGE CALCULATION")
    print("=" * 80)
    
    ranges, statistics = calculate_normal_ranges(
        baseline_features_df, 
        method='std', 
        std_multiplier=1.5
    )
    
    # Print statistics table
    print("\n" + "-" * 80)
    print("FEATURE STATISTICS TABLE")
    print("-" * 80)
    print(f"{'Feature':<25} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'5th %':<10} {'Median':<10} {'95th %':<10} {'Max':<10}")
    print("-" * 80)
    
    for feature in FEATURE_ORDER:
        if feature in statistics:
            stats = statistics[feature]
            name = FEATURE_NAMES[feature]
            print(f"{name:<25} {stats['mean']:<10.2f} {stats['std']:<10.2f} "
                  f"{stats['min']:<10.2f} {stats['p5']:<10.2f} {stats['p50']:<10.2f} "
                  f"{stats['p95']:<10.2f} {stats['max']:<10.2f}")
    
    # Print normal ranges table
    print("\n" + "-" * 80)
    print("NORMAL RANGES TABLE")
    print("-" * 80)
    print(f"{'Feature':<25} {'Normal Range':<30} {'Unusual Indicator'}")
    print("-" * 80)
    
    for feature in FEATURE_ORDER:
        if feature in ranges:
            name = FEATURE_NAMES[feature]
            normal_min = ranges[feature]['normal_min']
            normal_max = ranges[feature]['normal_max']
            print(f"{name:<25} [{normal_min:.2f}, {normal_max:.2f}]{'':<15} "
                  f"<{normal_min:.2f} or >{normal_max:.2f}")
    
    # PHASE 4: Validate on validation dataset
    print("\n" + "=" * 80)
    print("PHASE 4: RANGE VALIDATION")
    print("=" * 80)
    
    validation_file = os.path.join(validation_dir, 'validation_dataset.csv')
    
    if not os.path.exists(validation_file):
        print(f"\nWarning: Validation dataset not found at: {validation_file}")
        print("\nSkipping validation phase. To validate ranges:")
        print("  - Create a CSV file with 'review' and 'label' columns")
        print("  - Label should be 'Genuine'/'Normal' or 'Fraudulent'/'Anomalous'")
        print(f"  - Expected location: {validation_file}")
        
        # Save report without validation metrics
        report_path = os.path.join(reports_dir, 'feature_ranges_report.json')
        save_range_report(ranges, statistics, None, report_path)
        return
    
    print(f"\nLoading validation dataset from: {validation_file}")
    validation_df = pd.read_csv(validation_file)
    print(f"✓ Loaded {len(validation_df)} reviews")
    
    # Extract features from validation dataset
    validation_features_df = extract_features_from_dataset(
        validation_df,
        review_column='review',
        label_column='label' if 'label' in validation_df.columns else None
    )
    
    # Validate against ranges
    validation_results_df = validate_dataset(
        validation_features_df,
        ranges,
        label_column='label' if 'label' in validation_features_df.columns else None
    )
    
    # Calculate performance metrics
    if 'label' in validation_features_df.columns:
        # Convert labels to binary
        validation_results_df['actual_binary'] = validation_results_df.apply(
            lambda row: 1 if str(row['actual_label']).lower() in ['anomalous', 'fraudulent', 'fake', '1'] else 0,
            axis=1
        )
        
        metrics = calculate_performance_metrics(validation_results_df, 'actual_binary')
        
        if metrics:
            print_metrics_report(metrics)
            
            # Save validation results
            validation_results_file = os.path.join(validation_dir, 'validation_results.csv')
            validation_results_df.to_csv(validation_results_file, index=False)
            print(f"\n✓ Validation results saved to: {validation_results_file}")
            
            # Save complete report
            report_path = os.path.join(reports_dir, 'feature_ranges_report.json')
            save_range_report(ranges, statistics, metrics, report_path)
        else:
            print("\nWarning: Could not calculate performance metrics")
    else:
        print("\nWarning: No label column found in validation dataset. Skipping metrics calculation.")
        report_path = os.path.join(reports_dir, 'feature_ranges_report.json')
        save_range_report(ranges, statistics, None, report_path)
    
    print("\n" + "=" * 80)
    print("FEATURE RANGE CALCULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

