# Feature Range Calculation & Validation Guide

This guide explains how to calculate and validate feature ranges for the USAD (Unsupervised Anomaly Detection) tool using genuine product reviews.

## Overview

The feature range calculation process determines "normal" ranges for 11 linguistic features extracted from product reviews. These ranges help identify suspicious reviews that deviate from typical genuine review patterns.

## The 11 Features

1. **Length of Review** - Total word count
2. **Word Variety** - Lexical diversity (unique words / total words)
3. **Average Word Length** - Mean characters per word
4. **Overall Tone** - Sentiment polarity score (-1 to 1)
5. **Opinion Level** - Sentiment subjectivity score (0 to 1)
6. **Language Complexity** - Flesch-Kincaid readability score (0-100)
7. **Word Repetition** - Ratio of repeated words
8. **Exclamation Marks** - Count of "!" characters
9. **Question Marks** - Count of "?" characters
10. **Capital Letter Usage** - Ratio of words in ALL CAPS
11. **Punctuation Density** - Punctuation count / total words

## Step-by-Step Procedure

### Phase 1: Data Preparation

**Step 1: Assemble Baseline Dataset**

- Collect 500-1000+ genuine product reviews
- Reviews should be verified as authentic (not fraudulent)
- Use reviews from your target platform (Amazon, Shopee, etc.)
- Ensure reviews are in English

**Step 2: Prepare Your Data**

Create a CSV file with at least one column:
- `review`: The review text

Optional columns:
- `label`: Review label (e.g., "Genuine", "Normal", "OR")

**Example CSV structure:**
```csv
review,label
"This product is amazing! I love it so much. Highly recommend.",Genuine
"Great quality and fast shipping. Will buy again.",Genuine
...
```

### Phase 2: Feature Extraction

The script automatically extracts all 11 features from each review in your baseline dataset.

### Phase 3: Range Calculation

The script calculates:
- **Statistical measures**: Mean, standard deviation, min, max, percentiles
- **Normal ranges**: Using ±1.5 standard deviations or 5th-95th percentiles

### Phase 4: Range Validation

**Step 7: Prepare Validation Dataset**

Create a separate validation dataset with:
- 500-1000 genuine reviews (different from baseline)
- 200-400 fraudulent reviews (known fake reviews)

**Step 8-10: Validate & Refine**

The script validates ranges on the validation dataset and calculates:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Performance recommendations

## Usage

### Quick Start

1. **Prepare your datasets:**

```bash
cd backend/models/training-evaluation-scripts
python prepare_range_datasets.py
```

This script will:
- Extract genuine reviews from your existing training data
- Create `baseline-range-data/baseline_genuine_reviews.csv`
- Create `baseline-range-validation/validation_dataset.csv`

2. **Calculate and validate ranges:**

```bash
python feature_range_calculation.py
```

This will:
- Extract features from baseline dataset
- Calculate normal ranges
- Validate on validation dataset
- Generate performance metrics
- Save report to `range-reports/feature_ranges_report.json`

### Manual Dataset Preparation

If you have your own datasets, place them in the following locations:

**Baseline Dataset:**
- Location: `backend/models/baseline-range-data/baseline_genuine_reviews.csv`
- Required column: `review`
- Optional column: `label`

**Validation Dataset:**
- Location: `backend/models/baseline-range-validation/validation_dataset.csv`
- Required columns: `review`, `label`
- Label values: `Genuine`/`Normal` or `Fraudulent`/`Anomalous`

### Example: Using Existing Training Data

If you have existing training data in `training-data/processed_reviews.csv`:

```python
import pandas as pd
from prepare_range_datasets import prepare_baseline_dataset, prepare_validation_dataset

# Prepare baseline (genuine reviews only)
baseline_df = prepare_baseline_dataset(
    source_file='training-data/processed_reviews.csv',
    min_reviews=500
)

# Prepare validation (genuine + fraudulent)
validation_df = prepare_validation_dataset(
    source_file='training-data/processed_reviews.csv',
    genuine_count=700,
    fraudulent_count=300
)
```

## Output Files

### Feature Extraction Output

- `baseline-range-data/baseline_features.csv` - Extracted features from baseline dataset
- Contains all 11 features for each review

### Range Calculation Output

- `range-reports/feature_ranges_report.json` - Complete range report including:
  - Normal ranges for each feature
  - Statistical measures (mean, std dev, percentiles)
  - Validation performance metrics (if validation dataset provided)

### Validation Output

- `baseline-range-validation/validation_results.csv` - Validation results for each review:
  - `predicted_suspicious`: Whether review was flagged as suspicious
  - `warning_count`: Number of features outside normal range
  - `suspiciousness_score`: Ratio of features outside range

## Understanding the Results

### Normal Ranges

Each feature has a "normal range" defined as:
- **Standard Deviation Method**: `[mean - 1.5×std, mean + 1.5×std]`
- **Percentile Method**: `[5th percentile, 95th percentile]`

Reviews with feature values outside these ranges are flagged as "unusual."

### Performance Metrics

**Acceptable Performance:**
- Accuracy ≥ 85%
- Precision ≥ 80%
- Recall ≥ 85%

**If performance is poor:**
- **Too many false positives** → Expand normal ranges (increase std_multiplier)
- **Too many false negatives** → Narrow normal ranges (decrease std_multiplier)

### Example Range Report

```json
{
  "feature_ranges": {
    "review_length": {
      "name": "Length of Review",
      "normal_min": 20.0,
      "normal_max": 45.0
    },
    "lexical_diversity": {
      "name": "Word Variety",
      "normal_min": 0.72,
      "normal_max": 0.92
    },
    ...
  },
  "validation_metrics": {
    "accuracy": 0.94,
    "precision": 0.864,
    "recall": 0.95,
    "f1_score": 0.906
  }
}
```

## Customization

### Adjusting Range Calculation Method

Edit `feature_range_calculation.py`:

```python
# Use standard deviation method (default)
ranges, statistics = calculate_normal_ranges(
    baseline_features_df, 
    method='std', 
    std_multiplier=1.5  # Adjust this value
)

# Or use percentile method
ranges, statistics = calculate_normal_ranges(
    baseline_features_df, 
    method='percentile'
)
```

### Adjusting Suspiciousness Threshold

In `validate_review_features()`, change the threshold:

```python
# Default: >50% features outside range = suspicious
is_suspicious = suspiciousness_score > 0.5

# More strict: >30% features outside range
is_suspicious = suspiciousness_score > 0.3

# Less strict: >70% features outside range
is_suspicious = suspiciousness_score > 0.7
```

## Troubleshooting

### Error: Baseline dataset not found

**Solution:** Create the baseline dataset file:
```bash
# Create directory
mkdir -p backend/models/baseline-range-data

# Create CSV file with 'review' column
# See "Manual Dataset Preparation" section above
```

### Error: Not enough reviews

**Solution:** 
- Collect more genuine reviews (minimum 500 recommended)
- Or reduce `min_reviews` parameter in `prepare_baseline_dataset()`

### Poor validation performance

**Solutions:**
1. **Check data quality**: Ensure baseline dataset contains only genuine reviews
2. **Adjust ranges**: Try different `std_multiplier` values (1.4, 1.6, etc.)
3. **Use percentile method**: May be more robust to outliers
4. **Increase dataset size**: More data = better statistical estimates

### Missing features in output

**Solution:** Ensure your reviews contain enough text. Very short reviews (< 5 words) may have incomplete features.

## Integration with USAD Tool

Once ranges are calculated and validated, you can integrate them into the prediction pipeline:

1. Load the range report JSON file
2. Extract features from new reviews
3. Check each feature against normal ranges
4. Flag reviews with multiple features outside ranges

Example integration code:

```python
import json
from feature_range_calculation import extract_all_features, preprocess_text, validate_review_features

# Load ranges
with open('range-reports/feature_ranges_report.json') as f:
    report = json.load(f)

ranges = {}
for feat, data in report['feature_ranges'].items():
    ranges[feat] = {
        'normal_min': data['normal_min'],
        'normal_max': data['normal_max']
    }

# Check a new review
review_text = "This product is amazing!!!"
original_text, processed_tokens = preprocess_text(review_text)
features = extract_all_features(original_text, processed_tokens)
validation = validate_review_features(features, ranges)

if validation['is_suspicious']:
    print(f"Review flagged as suspicious ({validation['warning_count']} features outside range)")
```

## References

- Flesch-Kincaid Readability: https://en.wikipedia.org/wiki/Flesch–Kincaid_readability_tests
- TextBlob Sentiment Analysis: https://textblob.readthedocs.io/
- NLTK Documentation: https://www.nltk.org/

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in `feature_range_calculation.py`
3. Ensure your datasets meet the requirements

