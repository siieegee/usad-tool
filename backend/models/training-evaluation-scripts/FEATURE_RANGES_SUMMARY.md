# Feature Range Calculation & Validation - Implementation Summary

## What Was Created

This implementation provides a complete solution for calculating and validating feature ranges for the USAD tool, following the step-by-step procedure you provided.

### Files Created

1. **`feature_range_calculation.py`** (Main Script)
   - Implements all 4 phases of the procedure
   - Extracts all 11 features from reviews
   - Calculates statistical ranges
   - Validates ranges on test dataset
   - Generates performance metrics and reports

2. **`prepare_range_datasets.py`** (Helper Script)
   - Helps prepare baseline and validation datasets
   - Extracts genuine reviews from existing training data
   - Creates properly formatted CSV files

3. **`README_FEATURE_RANGES.md`** (Documentation)
   - Complete usage guide
   - Step-by-step instructions
   - Troubleshooting tips
   - Integration examples

## Features Implemented

### All 11 Features

✅ **1. Length of Review** - Word count from processed tokens
✅ **2. Word Variety** - Lexical diversity (unique/total ratio)
✅ **3. Average Word Length** - Mean characters per word
✅ **4. Overall Tone** - Sentiment polarity (-1 to 1) using TextBlob
✅ **5. Opinion Level** - Sentiment subjectivity (0 to 1) using TextBlob
✅ **6. Language Complexity** - Flesch-Kincaid readability score (0-100)
✅ **7. Word Repetition** - Repetition ratio
✅ **8. Exclamation Marks** - Count of "!" characters
✅ **9. Question Marks** - Count of "?" characters
✅ **10. Capital Letter Usage** - Ratio of ALL CAPS words
✅ **11. Punctuation Density** - Punctuation count / word count

### Phase Implementation

✅ **Phase 1: Data Preparation**
- Text preprocessing (cleaning, tokenization, lemmatization)
- Handles contractions, URLs, HTML tags
- Removes stopwords

✅ **Phase 2: Feature Extraction**
- Extracts all 11 features from each review
- Handles edge cases (empty reviews, missing data)
- Creates feature matrix

✅ **Phase 3: Range Calculation**
- Calculates mean, std dev, min, max, percentiles
- Defines normal ranges using ±1.5σ method
- Alternative percentile method (5th-95th)
- Generates statistics tables

✅ **Phase 4: Range Validation**
- Validates reviews against normal ranges
- Calculates suspiciousness scores
- Generates confusion matrix
- Calculates accuracy, precision, recall, F1-score
- Provides performance recommendations

## Key Functions

### Main Functions

- `preprocess_text()` - Cleans and preprocesses review text
- `extract_all_features()` - Extracts all 11 features
- `calculate_flesch_kincaid()` - Calculates readability score
- `calculate_normal_ranges()` - Calculates statistical ranges
- `validate_review_features()` - Validates single review
- `validate_dataset()` - Validates entire dataset
- `calculate_performance_metrics()` - Calculates classification metrics
- `main()` - Complete workflow execution

### Helper Functions

- `prepare_baseline_dataset()` - Creates baseline dataset
- `prepare_validation_dataset()` - Creates validation dataset

## Usage Workflow

### Step 1: Prepare Datasets

```bash
cd backend/models/training-evaluation-scripts
python prepare_range_datasets.py
```

This creates:
- `baseline-range-data/baseline_genuine_reviews.csv`
- `baseline-range-validation/validation_dataset.csv`

### Step 2: Calculate Ranges

```bash
python feature_range_calculation.py
```

This:
1. Loads baseline dataset
2. Extracts features
3. Calculates normal ranges
4. Validates on validation dataset
5. Generates report

### Step 3: Review Results

Check the generated report:
- `range-reports/feature_ranges_report.json`
- `baseline-range-validation/validation_results.csv`

## Output Structure

### Range Report JSON

```json
{
  "feature_ranges": {
    "review_length": {
      "name": "Length of Review",
      "normal_min": 20.0,
      "normal_max": 45.0
    },
    ...
  },
  "feature_statistics": {
    "review_length": {
      "mean": 32.5,
      "std": 8.2,
      "min": 12,
      "max": 156,
      "p5": 18,
      "p25": 26,
      "p50": 32,
      "p75": 40,
      "p95": 48
    },
    ...
  },
  "validation_metrics": {
    "accuracy": 0.94,
    "precision": 0.864,
    "recall": 0.95,
    "f1_score": 0.906,
    ...
  }
}
```

## Customization Options

### Range Calculation Method

```python
# Standard deviation method (default)
ranges, stats = calculate_normal_ranges(df, method='std', std_multiplier=1.5)

# Percentile method
ranges, stats = calculate_normal_ranges(df, method='percentile')
```

### Suspiciousness Threshold

Default: >50% features outside range = suspicious

Can be adjusted in `validate_review_features()` function.

## Performance Criteria

The script evaluates performance using these criteria:

**Excellent:**
- Accuracy ≥ 85%
- Precision ≥ 80%
- Recall ≥ 85%

**Good:**
- Accuracy ≥ 80%
- Precision ≥ 75%
- Recall ≥ 80%

**Poor:**
- Any metric < 80%

## Integration Example

```python
import json
from feature_range_calculation import (
    preprocess_text, 
    extract_all_features, 
    validate_review_features
)

# Load ranges
with open('range-reports/feature_ranges_report.json') as f:
    report = json.load(f)

ranges = {}
for feat, data in report['feature_ranges'].items():
    ranges[feat] = {
        'normal_min': data['normal_min'],
        'normal_max': data['normal_max']
    }

# Check new review
review = "This product is amazing!!!"
original, tokens = preprocess_text(review)
features = extract_all_features(original, tokens)
result = validate_review_features(features, ranges)

if result['is_suspicious']:
    print(f"Flagged: {result['warning_count']} features outside range")
```

## Dependencies

Required Python packages:
- pandas
- numpy
- textblob
- nltk
- scikit-learn

Install with:
```bash
pip install pandas numpy textblob nltk scikit-learn
```

## File Structure

```
backend/models/
├── training-evaluation-scripts/
│   ├── feature_range_calculation.py      # Main script
│   ├── prepare_range_datasets.py         # Helper script
│   ├── README_FEATURE_RANGES.md          # Full documentation
│   └── FEATURE_RANGES_SUMMARY.md         # This file
├── baseline-range-data/                  # Created by script
│   ├── baseline_genuine_reviews.csv
│   └── baseline_features.csv
├── baseline-range-validation/            # Created by script
│   ├── validation_dataset.csv
│   └── validation_results.csv
└── range-reports/                        # Created by script
    └── feature_ranges_report.json
```

## Next Steps

1. **Run the scripts** with your data
2. **Review the generated ranges** for reasonableness
3. **Validate performance** on your validation dataset
4. **Refine ranges** if needed (adjust std_multiplier)
5. **Integrate** into your prediction pipeline

## Notes

- The Flesch-Kincaid calculation uses an approximate syllable counting method
- All features handle edge cases (empty reviews, missing data)
- The script provides detailed progress output
- Performance metrics follow standard classification evaluation practices

## Support

For detailed usage instructions, see `README_FEATURE_RANGES.md`.

