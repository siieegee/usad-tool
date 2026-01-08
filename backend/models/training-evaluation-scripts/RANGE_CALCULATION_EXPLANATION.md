# Feature Range Calculation & Validation - Complete Explanation

This document provides a comprehensive explanation of how feature ranges are calculated and validated for the USAD (Unsupervised Anomaly Detection) tool.

---

## Table of Contents

1. [Overview](#overview)
2. [How to Calculate Feature Ranges](#how-to-calculate-feature-ranges)
3. [How to Validate Feature Ranges](#how-to-validate-feature-ranges)
4. [Step-by-Step Example](#step-by-step-example)
5. [Understanding the Results](#understanding-the-results)
6. [Troubleshooting & Refinement](#troubleshooting--refinement)

---

## Overview

Feature ranges define "normal" values for each linguistic feature extracted from product reviews. These ranges help identify suspicious reviews that deviate from typical genuine review patterns.

**Key Concepts:**
- **Normal Range**: The range of values that are considered typical for genuine reviews
- **Baseline Dataset**: A collection of verified genuine reviews used to calculate ranges
- **Validation Dataset**: A separate dataset with known genuine and fraudulent reviews used to test the ranges

---

## How to Calculate Feature Ranges

### Step 1: Prepare Baseline Dataset

**Purpose**: Collect genuine reviews to establish baseline statistics.

**Requirements:**
- Minimum 500 reviews (1000+ recommended for better accuracy)
- All reviews must be verified as GENUINE (not fraudulent)
- Reviews should be from your target platform (Amazon, Shopee, etc.)
- English-only reviews

**Data Format:**
```csv
review,label
"This product is amazing! I love it so much. Highly recommend.",Genuine
"Great quality and fast shipping. Will buy again.",Genuine
...
```

### Step 2: Extract Features from Each Review

For each review in the baseline dataset, extract all 11 features:

#### Feature Extraction Process:

1. **Preprocess the Review**
   - Convert to lowercase
   - Remove URLs, emails, HTML tags
   - Handle contractions
   - Tokenize and remove stopwords
   - Lemmatize words

2. **Calculate Each Feature**

   | # | Feature | Calculation Method |
   |---|---------|-------------------|
   | 1 | Length of Review | Count total words after preprocessing |
   | 2 | Word Variety | `unique_words / total_words` |
   | 3 | Average Word Length | `sum(word_lengths) / word_count` |
   | 4 | Overall Tone | TextBlob sentiment polarity (-1 to 1) |
   | 5 | Opinion Level | TextBlob sentiment subjectivity (0 to 1) |
   | 6 | Language Complexity | Flesch-Kincaid readability score (0-100) |
   | 7 | Word Repetition | `(total_words - unique_words) / total_words` |
   | 8 | Exclamation Marks | Count of "!" characters |
   | 9 | Question Marks | Count of "?" characters |
   | 10 | Capital Letter Usage | `words_in_ALL_CAPS / total_words` |
   | 11 | Punctuation Density | `punctuation_count / total_words` |

**Example Feature Extraction:**
```
Review: "This product is amazing! I love it so much. Highly recommend."

Features:
- Length of Review: 9 words
- Word Variety: 9/9 = 1.0
- Average Word Length: 4.44 characters
- Overall Tone: 0.65 (positive)
- Opinion Level: 0.75 (subjective)
- Language Complexity: 85.2
- Word Repetition: 0.0
- Exclamation Marks: 1
- Question Marks: 0
- Capital Letter Usage: 0.0
- Punctuation Density: 0.22
```

### Step 3: Calculate Statistical Measures

For each of the 11 features, calculate statistical measures across all reviews in the baseline dataset:

#### Basic Statistics:

1. **Mean (μ)**: Average value
   ```
   Mean = (sum of all values) / (number of reviews)
   ```

2. **Standard Deviation (σ)**: Measure of spread/variability
   ```
   σ = √[Σ(xi - μ)² / (n-1)]
   ```

3. **Minimum (Min)**: Smallest value observed

4. **Maximum (Max)**: Largest value observed

5. **Percentiles**: Values below which a certain percentage of data falls
   - 5th percentile (P5): 5% of values are below this
   - 25th percentile (P25): First quartile
   - 50th percentile (P50): Median
   - 75th percentile (P75): Third quartile
   - 95th percentile (P95): 95% of values are below this

**Example Calculation for "Length of Review":**

Given 750 reviews with word counts: [35, 42, 28, 31, 45, 38, 29, 41, 33, 39, ..., 38]

```
Mean (μ) = (35+42+28+...+38) / 750 = 18.25 words
Std Dev (σ) = √[Σ(xi - 18.25)² / 749] = 5.51 words

Percentiles:
- 5th percentile = 11 words
- 25th percentile = 14 words
- 50th percentile (median) = 17 words
- 75th percentile = 23 words
- 95th percentile = 28 words

Min = 2 words
Max = 38 words
```

### Step 4: Define Normal Ranges

There are two methods to define normal ranges:

#### Method 1: Standard Deviation Approach (Default)

**Formula:**
```
Normal Range = [μ - (k × σ), μ + (k × σ)]
```

Where:
- `μ` = Mean
- `σ` = Standard Deviation
- `k` = Multiplier (typically 1.5)

**Example - Length of Review:**
```
Mean (μ) = 18.25
Std Dev (σ) = 5.51
Multiplier (k) = 1.5

Normal Min = 18.25 - (1.5 × 5.51) = 18.25 - 8.27 = 9.99
Normal Max = 18.25 + (1.5 × 5.51) = 18.25 + 8.27 = 26.51

Normal Range = [9.99, 26.51] words
```

**Interpretation:**
- ✓ **Normal**: Reviews with 10-27 words are considered normal
- ✗ **Unusual**: Reviews with <10 words OR >27 words are flagged

#### Method 2: Percentile-Based Approach (Alternative)

**Formula:**
```
Normal Range = [5th percentile, 95th percentile]
```

**Example - Length of Review:**
```
5th percentile = 11 words
95th percentile = 28 words

Normal Range = [11, 28] words
```

**Advantages:**
- More robust to outliers
- Doesn't assume normal distribution
- Uses actual data distribution

### Step 5: Create Range Table

Repeat the calculation for all 11 features to create a complete range table:

| Feature | Normal Min | Normal Max | Interpretation |
|---------|-----------|------------|----------------|
| Length of Review | 9.99 | 26.51 | <10 or >27 words = unusual |
| Word Variety | 0.83 | 1.03 | <0.83 or >1.03 = unusual |
| Average Word Length | 4.26 | 6.03 | <4.26 or >6.03 chars = unusual |
| ... | ... | ... | ... |

---

## How to Validate Feature Ranges

### Step 1: Prepare Validation Dataset

**Purpose**: Test the calculated ranges on a separate dataset with known labels.

**Requirements:**
- **Different from baseline dataset** (must be separate!)
- 500-1000 genuine reviews (verified as authentic)
- 200-400 fraudulent reviews (known fake reviews)
- Each review must have a label: "Genuine" or "Fraudulent"

**Data Format:**
```csv
review,label
"This product is amazing! I love it so much.",Genuine
"AMAZING PRODUCT!!! BUY NOW!!! BEST DEAL EVER!!!",Fraudulent
...
```

### Step 2: Extract Features from Validation Dataset

For each review in the validation dataset:
1. Extract all 11 features (same process as baseline)
2. Store feature values

### Step 3: Check Features Against Normal Ranges

For each review, check each feature value against its corresponding normal range:

**Rule:**
```
If feature_value is within [normal_min, normal_max]:
    Status = PASS (Normal)
Else:
    Status = WARNING (Unusual)
```

**Example Validation for One Review:**

| Feature | Value | Normal Range | Status |
|---------|-------|-------------|--------|
| Length of Review | 15 | [9.99, 26.51] | ✓ PASS |
| Word Variety | 0.68 | [0.83, 1.03] | ✗ WARNING |
| Average Word Length | 4.5 | [4.26, 6.03] | ✓ PASS |
| Overall Tone | 0.50 | [-0.09, 0.62] | ✓ PASS |
| Opinion Level | 0.40 | [0.28, 0.81] | ✓ PASS |
| Language Complexity | 35 | [58.94, 98.58] | ✗ WARNING |
| Word Repetition | 0.25 | [-0.03, 0.17] | ✗ WARNING |
| Exclamation Marks | 3 | [-1.26, 2.03] | ✗ WARNING |
| Question Marks | 0 | [-0.31, 0.37] | ✓ PASS |
| Capital Letter Usage | 0.08 | [-0.08, 0.11] | ✓ PASS |
| Punctuation Density | 0.10 | [-0.06, 0.59] | ✓ PASS |

**Summary:**
- Passes: 7 features
- Warnings: 4 features
- Suspiciousness Score: 4/11 = 0.36 (36% features outside range)

**Decision Rule:**
```
If suspiciousness_score > 0.5 (50% features outside range):
    Review is flagged as SUSPICIOUS
Else:
    Review is flagged as NORMAL
```

In this example: 0.36 < 0.5, so review is **NORMAL** (but close to suspicious).

### Step 4: Calculate Performance Metrics

Compare predictions (SUSPICIOUS vs NORMAL) with actual labels (Fraudulent vs Genuine):

#### Confusion Matrix

| | Predicted: SUSPICIOUS | Predicted: NORMAL |
|---|---------------------|-------------------|
| **Actual: Fraudulent** | **TP** (True Positive) | **FN** (False Negative) |
| **Actual: Genuine** | **FP** (False Positive) | **TN** (True Negative) |

**Where:**
- **TP (True Positive)**: Correctly caught fraudulent reviews
- **TN (True Negative)**: Correctly identified genuine reviews
- **FP (False Positive)**: Genuine reviews wrongly flagged as suspicious
- **FN (False Negative)**: Fraudulent reviews that slipped through

#### Calculate Metrics

1. **Accuracy**: Overall correctness
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precision**: Of reviews flagged as suspicious, how many were actually fraudulent?
   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall (Sensitivity)**: Of all fraudulent reviews, how many did we catch?
   ```
   Recall = TP / (TP + FN)
   ```

4. **F1-Score**: Balanced measure of precision and recall
   ```
   F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
   ```

5. **Specificity**: Of all genuine reviews, how many did we correctly identify?
   ```
   Specificity = TN / (TN + FP)
   ```

6. **False Alarm Rate**: Rate of genuine reviews wrongly flagged
   ```
   False Alarm Rate = FP / (FP + TN)
   ```

**Example Calculation:**

Given:
- TP = 699 (correctly caught fraudulent)
- TN = 5 (correctly identified genuine)
- FP = 295 (genuine wrongly flagged)
- FN = 1 (fraudulent missed)

Total = 1000 reviews

```
Accuracy = (699 + 5) / 1000 = 704 / 1000 = 0.704 (70.4%)
Precision = 699 / (699 + 295) = 699 / 994 = 0.703 (70.3%)
Recall = 699 / (699 + 1) = 699 / 700 = 0.999 (99.9%)
F1-Score = 2 × (0.703 × 0.999) / (0.703 + 0.999) = 0.826 (82.6%)
Specificity = 5 / (5 + 295) = 5 / 300 = 0.017 (1.7%)
False Alarm Rate = 295 / (295 + 5) = 295 / 300 = 0.983 (98.3%)
```

### Step 5: Assess Performance

**Performance Criteria:**

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| Accuracy | ≥ 85% | ≥ 80% | ≥ 70% | < 70% |
| Precision | ≥ 80% | ≥ 75% | ≥ 60% | < 60% |
| Recall | ≥ 85% | ≥ 80% | ≥ 60% | < 60% |
| F1-Score | ≥ 80% | ≥ 70% | ≥ 60% | < 60% |

**Decision:**
- **If all metrics ≥ Excellent thresholds**: ✓ APPROVE ranges for production
- **If metrics ≥ Good thresholds**: ✓ APPROVE with monitoring
- **If metrics ≥ Acceptable thresholds**: ⚠ REFINE ranges before production
- **If any metric < Acceptable**: ✗ REFINE ranges and retest

---

## Step-by-Step Example

### Complete Workflow Example

**Scenario**: Calculate and validate ranges for "Length of Review" feature

#### Phase 1: Calculate Range

1. **Baseline Dataset**: 750 genuine reviews
2. **Extract Feature**: Count words in each review
   - Review 1: 35 words
   - Review 2: 42 words
   - Review 3: 28 words
   - ... (750 total)

3. **Calculate Statistics:**
   ```
   Mean = 18.25 words
   Std Dev = 5.51 words
   Min = 2 words
   Max = 38 words
   ```

4. **Calculate Normal Range (k=1.5):**
   ```
   Normal Min = 18.25 - (1.5 × 5.51) = 9.99 words
   Normal Max = 18.25 + (1.5 × 5.51) = 26.51 words
   ```

5. **Result**: Normal Range = [9.99, 26.51] words

#### Phase 2: Validate Range

1. **Validation Dataset**: 1000 reviews (700 genuine, 300 fraudulent)

2. **Test Each Review:**
   - Review A: 15 words → Within [9.99, 26.51] → PASS
   - Review B: 5 words → Below 9.99 → WARNING
   - Review C: 30 words → Above 26.51 → WARNING
   - ... (1000 total)

3. **Count Results:**
   - Genuine reviews: 650 PASS, 50 WARNING
   - Fraudulent reviews: 50 PASS, 250 WARNING

4. **Create Confusion Matrix:**
   - TP = 250 (fraudulent flagged as suspicious)
   - TN = 650 (genuine flagged as normal)
   - FP = 50 (genuine flagged as suspicious)
   - FN = 50 (fraudulent flagged as normal)

5. **Calculate Metrics:**
   ```
   Accuracy = (250 + 650) / 1000 = 90%
   Precision = 250 / (250 + 50) = 83.3%
   Recall = 250 / (250 + 50) = 83.3%
   F1-Score = 2 × (0.833 × 0.833) / (0.833 + 0.833) = 83.3%
   ```

6. **Assessment**: 
   - Accuracy: 90% ≥ 85% ✓ Excellent
   - Precision: 83.3% ≥ 80% ✓ Excellent
   - Recall: 83.3% ≥ 85% ⚠ Good (slightly below)
   - **Overall**: GOOD performance, ranges approved

---

## Understanding the Results

### What Do the Ranges Mean?

**Normal Range [9.99, 26.51] for Length of Review:**
- Reviews with 10-27 words are considered **normal** (typical of genuine reviews)
- Reviews with <10 words are **too short** (may indicate spam/fake)
- Reviews with >27 words are **too long** (may indicate copy-paste or unusual patterns)

### Why Some Ranges Have Negative Values?

Some features (like Exclamation Marks, Question Marks) can have negative normal minimums when using the standard deviation method. This happens when:
- Mean is close to zero
- Standard deviation is large
- Formula: `mean - (1.5 × std)` results in negative value

**Solution**: In practice, treat negative minimums as 0 (since counts can't be negative).

### Interpreting Validation Metrics

**High Precision (83.3%)**: 
- When we flag a review as suspicious, we're usually right
- Low false positive rate
- Good for avoiding false alarms

**Low Recall (1.67%)**:
- We're missing most fraudulent reviews
- High false negative rate
- Bad for catching fraud

**High Accuracy (70.4%)**:
- Overall, we're correct 70% of the time
- But this can be misleading if classes are imbalanced

---

## Troubleshooting & Refinement

### Problem: Too Many False Positives (Low Precision)

**Symptoms:**
- Many genuine reviews flagged as suspicious
- High FP count in confusion matrix
- Precision < 80%

**Solution:**
- **Expand normal ranges** (make thresholds less strict)
- Increase `std_multiplier` from 1.5 to 1.6 or 1.7
- Or use percentile method with wider range (e.g., 1st-99th percentile)

**Example:**
```
Original: Normal Range = [9.99, 26.51] (k=1.5)
Refined:  Normal Range = [8.50, 28.00] (k=1.7)
```

### Problem: Too Many False Negatives (Low Recall)

**Symptoms:**
- Many fraudulent reviews not caught
- High FN count in confusion matrix
- Recall < 85%

**Solution:**
- **Narrow normal ranges** (make thresholds stricter)
- Decrease `std_multiplier` from 1.5 to 1.3 or 1.4
- Or use percentile method with narrower range (e.g., 10th-90th percentile)

**Example:**
```
Original: Normal Range = [9.99, 26.51] (k=1.5)
Refined:  Normal Range = [11.50, 25.00] (k=1.3)
```

### Problem: Poor Overall Performance

**Symptoms:**
- Both precision and recall are low
- Accuracy < 70%

**Solutions:**
1. **Check data quality**: Ensure baseline dataset contains only genuine reviews
2. **Increase dataset size**: More data = better statistical estimates
3. **Try different method**: Switch from std deviation to percentile method
4. **Adjust suspiciousness threshold**: Change from 50% to 40% or 60%
5. **Feature selection**: Some features may not be useful - consider removing them

### Refinement Process

1. **Identify the problem** (low precision, low recall, or both)
2. **Adjust parameters** (std_multiplier, percentile range, or threshold)
3. **Recalculate ranges** on baseline dataset
4. **Re-validate** on validation dataset
5. **Compare metrics** with previous results
6. **Iterate** until performance meets criteria

**Example Refinement Cycle:**

```
Iteration 1: k=1.5 → Accuracy: 70%, Precision: 83%, Recall: 2%
Problem: Very low recall

Iteration 2: k=1.3 → Accuracy: 75%, Precision: 78%, Recall: 15%
Better recall, but still low

Iteration 3: k=1.2 → Accuracy: 78%, Precision: 75%, Recall: 25%
Improving, but recall still needs work

Iteration 4: Use percentile method (10th-90th) → Accuracy: 82%, Precision: 80%, Recall: 80%
✓ Good performance! Approved.
```

---

## Summary

### Calculation Process:
1. Collect genuine reviews (baseline dataset)
2. Extract 11 features from each review
3. Calculate statistics (mean, std dev, percentiles)
4. Define normal ranges using ±1.5σ or percentiles

### Validation Process:
1. Prepare separate validation dataset (genuine + fraudulent)
2. Extract features and check against normal ranges
3. Flag reviews with >50% features outside range as suspicious
4. Calculate performance metrics (accuracy, precision, recall, F1)
5. Assess performance and refine if needed

### Key Success Criteria:
- Accuracy ≥ 85%
- Precision ≥ 80%
- Recall ≥ 85%
- F1-Score ≥ 80%

If all criteria are met, ranges are approved for production use!
