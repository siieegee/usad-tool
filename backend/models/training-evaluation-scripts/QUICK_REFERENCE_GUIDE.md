# Quick Reference Guide: Range Calculation & Validation

## 🎯 Quick Overview

**Goal**: Calculate "normal" ranges for 11 features, then validate them on test data.

---

## 📊 How to Calculate Ranges

### Formula (Standard Deviation Method)

```
Normal Range = [Mean - (1.5 × Std Dev), Mean + (1.5 × Std Dev)]
```

### Step-by-Step

1. **Collect Baseline Data**
   - 500-1000+ genuine reviews
   - All verified as authentic

2. **Extract Features**
   - For each review, calculate all 11 features
   - Store in a table

3. **Calculate Statistics**
   - Mean (average)
   - Standard Deviation (spread)
   - Min, Max, Percentiles

4. **Define Normal Range**
   ```
   For each feature:
   Normal Min = Mean - (1.5 × Std Dev)
   Normal Max = Mean + (1.5 × Std Dev)
   ```

### Example

**Feature: Length of Review**

```
Baseline Data: [35, 42, 28, 31, 45, 38, ...] (750 reviews)

Step 1: Calculate Mean
Mean = 18.25 words

Step 2: Calculate Std Dev
Std Dev = 5.51 words

Step 3: Calculate Normal Range
Normal Min = 18.25 - (1.5 × 5.51) = 9.99
Normal Max = 18.25 + (1.5 × 5.51) = 26.51

Result: Normal Range = [9.99, 26.51] words
```

---

## ✅ How to Validate Ranges

### Process Flow

```
Validation Dataset (1000 reviews)
    ↓
Extract Features (same 11 features)
    ↓
Check Each Feature Against Normal Range
    ↓
Count Warnings (features outside range)
    ↓
Flag Review if >50% features are warnings
    ↓
Compare Predictions vs Actual Labels
    ↓
Calculate Performance Metrics
```

### Step-by-Step

1. **Prepare Validation Data**
   - Separate dataset (different from baseline!)
   - 700 genuine + 300 fraudulent reviews
   - Each review has a label

2. **Check Features**
   ```
   For each review:
     For each feature:
       If feature_value within [normal_min, normal_max]:
         Status = PASS
       Else:
         Status = WARNING
     
     Count warnings
     If warnings > 50% of features:
       Prediction = SUSPICIOUS
     Else:
       Prediction = NORMAL
   ```

3. **Calculate Metrics**
   ```
   Accuracy = (TP + TN) / Total
   Precision = TP / (TP + FP)
   Recall = TP / (TP + FN)
   F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
   ```

### Example Validation

**Review**: "AMAZING PRODUCT!!! BUY NOW!!!"

| Feature | Value | Normal Range | Status |
|---------|-------|-------------|--------|
| Length | 3 | [9.99, 26.51] | ✗ WARNING |
| Word Variety | 0.67 | [0.83, 1.03] | ✗ WARNING |
| Exclamation | 3 | [-1.26, 2.03] | ✗ WARNING |
| ... | ... | ... | ... |

**Result**: 7 out of 11 features = WARNING (64%)
**Decision**: 64% > 50% → **SUSPICIOUS** ✓

---

## 📈 Performance Criteria

| Metric | Target | Your Result | Status |
|--------|--------|-------------|--------|
| Accuracy | ≥ 85% | 70.4% | ⚠ Needs Improvement |
| Precision | ≥ 80% | 83.3% | ✓ Good |
| Recall | ≥ 85% | 1.67% | ✗ Poor |
| F1-Score | ≥ 80% | 3.27% | ✗ Poor |

**Assessment**: POOR - Ranges need refinement

---

## 🔧 Common Issues & Solutions

### Issue 1: Low Recall (Missing Fraudulent Reviews)

**Problem**: Too many false negatives (FN)

**Solution**: 
- Narrow normal ranges (make stricter)
- Decrease std_multiplier: 1.5 → 1.3
- Or use percentile method: 10th-90th

### Issue 2: Low Precision (Too Many False Alarms)

**Problem**: Too many false positives (FP)

**Solution**:
- Expand normal ranges (make looser)
- Increase std_multiplier: 1.5 → 1.7
- Or use percentile method: 1st-99th

### Issue 3: Both Metrics Low

**Problem**: Overall poor performance

**Solution**:
- Check data quality (baseline must be genuine only)
- Increase dataset size
- Try different method (percentile vs std dev)
- Adjust suspiciousness threshold (50% → 40% or 60%)

---

## 📝 Quick Formulas Reference

### Range Calculation

**Standard Deviation Method:**
```
Normal Min = Mean - (k × Std Dev)
Normal Max = Mean + (k × Std Dev)
where k = 1.5 (default)
```

**Percentile Method:**
```
Normal Min = 5th Percentile
Normal Max = 95th Percentile
```

### Performance Metrics

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
Specificity = TN / (TN + FP)
False Alarm Rate = FP / (FP + TN)
```

### Decision Rule

```
Suspiciousness Score = (Number of Warnings) / (Total Features)

If Suspiciousness Score > 0.5:
    Review is SUSPICIOUS
Else:
    Review is NORMAL
```

---

## 🎓 Key Concepts

**Normal Range**: Values typical of genuine reviews
- Values within range = Normal behavior
- Values outside range = Unusual behavior

**Baseline Dataset**: Used to calculate ranges
- Must contain only genuine reviews
- Minimum 500 reviews recommended

**Validation Dataset**: Used to test ranges
- Must be separate from baseline
- Contains both genuine and fraudulent reviews
- Used to measure performance

**Confusion Matrix**: Shows prediction accuracy
- TP: Correctly caught fraud
- TN: Correctly identified genuine
- FP: Wrongly flagged genuine as fraud
- FN: Missed fraud (slipped through)

---

## 🚀 Quick Start Checklist

### For Calculation:
- [ ] Collect 500+ genuine reviews
- [ ] Extract all 11 features
- [ ] Calculate mean and std dev for each feature
- [ ] Calculate normal ranges (mean ± 1.5×std)
- [ ] Document ranges in table

### For Validation:
- [ ] Prepare separate validation dataset
- [ ] Extract features from validation data
- [ ] Check each feature against normal ranges
- [ ] Count warnings per review
- [ ] Flag reviews with >50% warnings
- [ ] Calculate performance metrics
- [ ] Assess if performance meets criteria

### For Refinement:
- [ ] Identify problem (low precision/recall)
- [ ] Adjust std_multiplier or use percentile method
- [ ] Recalculate ranges
- [ ] Re-validate
- [ ] Compare metrics
- [ ] Iterate until performance is acceptable

---

## 📚 Related Files

- **Full Explanation**: `RANGE_CALCULATION_EXPLANATION.md`
- **Code Implementation**: `feature_range_calculation.py`
- **Visualization**: `visualize_feature_ranges.py`
- **Summary Table**: `create_final_summary_table.py`

---

## 💡 Pro Tips

1. **Always use separate datasets** for baseline and validation
2. **Start with std_multiplier = 1.5**, then adjust based on results
3. **Percentile method is more robust** to outliers
4. **Aim for balanced precision and recall** (both ≥ 80%)
5. **Document your methodology** for reproducibility
6. **Test on multiple validation sets** to ensure consistency

---

## ❓ FAQ

**Q: Why do some ranges have negative values?**
A: When mean is close to zero and std dev is large, the formula can produce negative values. In practice, treat as 0 (counts can't be negative).

**Q: How many features need to be outside range to flag a review?**
A: Default is >50% (more than 5.5 out of 11 features). You can adjust this threshold.

**Q: What if my baseline dataset is small (<500 reviews)?**
A: Results may be less reliable. Try to collect more data, or use percentile method which is more robust to small samples.

**Q: Can I use different multipliers for different features?**
A: Yes! Some features may need stricter/looser thresholds. You can customize per feature.

**Q: How often should I recalculate ranges?**
A: When you notice performance degradation, or when you have significantly more baseline data, or periodically (e.g., quarterly).
