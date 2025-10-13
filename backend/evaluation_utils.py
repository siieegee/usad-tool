import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
import json

# Set style to match evaluation.py
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def detailed_metrics(y_true, y_pred, dataset_name="Dataset"):
    """Calculate and display detailed metrics in the same format as evaluation.py"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='Anomalous', zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label='Anomalous', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='Anomalous', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=['Anomalous', 'Normal'])
    tn, fp, fn, tp = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n{'─'*80}")
    print(f"{dataset_name.upper()} PERFORMANCE")
    print(f"{'─'*80}")
    print(f"\nPrimary Metrics:")
    print(f"  Accuracy:           {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision:          {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall/Sensitivity: {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:           {f1:.4f} ({f1*100:.2f}%) ⭐")
    print(f"\nAdditional Metrics:")
    print(f"  Specificity:        {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  False Alarm Rate:   {false_alarm_rate:.4f} ({false_alarm_rate*100:.2f}%)")
    print(f"  Detection Rate:     {detection_rate:.4f} ({detection_rate*100:.2f}%)")

    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Anomalous  Normal")
    print(f"Actual Anomalous    {tp:5d}    {fn:5d}")
    print(f"       Normal       {fp:5d}    {tn:5d}")

    print(f"\nInterpretation:")
    print(f"  True Positives (TP):  {tp:4d} - Correctly caught fake reviews")
    print(f"  True Negatives (TN):  {tn:4d} - Correctly identified genuine reviews")
    print(f"  False Positives (FP): {fp:4d} - Genuine reviews wrongly flagged")
    print(f"  False Negatives (FN): {fn:4d} - Fake reviews that slipped through")

    print(f"\nPerformance Assessment:")
    if f1 >= 0.80:
        print(f"  EXCELLENT - F1 Score >= 0.80")
    elif f1 >= 0.70:
        print(f"  GOOD - F1 Score >= 0.70")
    elif f1 >= 0.60:
        print(f"  ACCEPTABLE - F1 Score >= 0.60")
    else:
        print(f"  POOR - F1 Score < 0.60 (needs improvement)")

    if prec < 0.60:
        print(f"  Warning: Low precision - many false alarms")
    if rec < 0.60:
        print(f"  Warning: Low recall - missing many anomalies")

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'false_alarm_rate': float(false_alarm_rate),
        'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
    }


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to Python native types"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def run_full_evaluation(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        threshold: float,
                        save_report_path: str = 'evaluation_report.json',
                        print_output: bool = True):
    """
    Run the comprehensive evaluation identical to backend/evaluation.py and save a JSON report.
    Expects columns: 'true_label', 'distance_to_centroid'.
    Optionally uses 'cluster' if present for cluster analysis.
    """
    if print_output:
        _print_header("COMPREHENSIVE MODEL EVALUATION")
        print("\nLoading evaluation data...")
        print(f"✓ Training samples: {len(train_df)}")
        print(f"✓ Test samples: {len(test_df)}")
        print(f"✓ Threshold: {threshold:.4f}")

    # Predictions
    train_preds = np.where(train_df['distance_to_centroid'] > threshold, 'Anomalous', 'Normal')
    test_preds = np.where(test_df['distance_to_centroid'] > threshold, 'Anomalous', 'Normal')

    # 1. Confusion Matrix & Basic Metrics
    if print_output:
        _print_header("1. CONFUSION MATRIX & CLASSIFICATION METRICS")
    train_metrics = detailed_metrics(train_df['true_label'], train_preds, "Training")
    test_metrics = detailed_metrics(test_df['true_label'], test_preds, "Test")

    # 2. ROC & AUC
    if print_output:
        _print_header("2. ROC CURVE & AUC ANALYSIS")
    train_binary = (train_df['true_label'] == 'Anomalous').astype(int)
    test_binary = (test_df['true_label'] == 'Anomalous').astype(int)
    train_roc_auc = roc_auc_score(train_binary, train_df['distance_to_centroid'])
    test_roc_auc = roc_auc_score(test_binary, test_df['distance_to_centroid'])
    if print_output:
        print(f"\nAUC-ROC Scores:")
        print(f"  Training:   {train_roc_auc:.4f}")
        print(f"  Test:       {test_roc_auc:.4f}")
        if test_roc_auc >= 0.90:
            print(f"  EXCELLENT discrimination ability")
        elif test_roc_auc >= 0.80:
            print(f"  GOOD discrimination ability")
        elif test_roc_auc >= 0.70:
            print(f"  ACCEPTABLE discrimination ability")
        else:
            print(f"  POOR discrimination ability")

    # 3. Precision-Recall
    if print_output:
        _print_header("3. PRECISION-RECALL ANALYSIS")
    train_ap = average_precision_score(train_binary, train_df['distance_to_centroid'])
    test_ap = average_precision_score(test_binary, test_df['distance_to_centroid'])
    if print_output:
        print(f"\nAverage Precision Scores:")
        print(f"  Training:   {train_ap:.4f}")
        print(f"  Test:       {test_ap:.4f}")

    # 4. Distance Distribution
    if print_output:
        _print_header("4. DISTANCE DISTRIBUTION ANALYSIS")
    train_normal_dist = train_df[train_df['true_label'] == 'Normal']['distance_to_centroid']
    train_anom_dist = train_df[train_df['true_label'] == 'Anomalous']['distance_to_centroid']
    test_normal_dist = test_df[test_df['true_label'] == 'Normal']['distance_to_centroid']
    test_anom_dist = test_df[test_df['true_label'] == 'Anomalous']['distance_to_centroid']
    if print_output:
        print(f"\nTraining Set Distance Statistics:")
        print(f"  Normal Reviews:")
        print(f"    Mean: {train_normal_dist.mean():.4f} ± {train_normal_dist.std():.4f}")
        print(f"    Range: [{train_normal_dist.min():.4f}, {train_normal_dist.max():.4f}]")
        print(f"  Anomalous Reviews:")
        print(f"    Mean: {train_anom_dist.mean():.4f} ± {train_anom_dist.std():.4f}")
        print(f"    Range: [{train_anom_dist.min():.4f}, {train_anom_dist.max():.4f}]")
    separation_train = (train_anom_dist.mean() / train_normal_dist.mean()) if train_normal_dist.mean() != 0 else float('inf')
    if print_output:
        print(f"\n  Separation Ratio: {separation_train:.4f}")
        if separation_train >= 2.0:
            print(f"  EXCELLENT class separation")
        elif separation_train >= 1.5:
            print(f"  GOOD class separation")
        elif separation_train >= 1.2:
            print(f"  ACCEPTABLE class separation")
        else:
            print(f"  POOR class separation (classes overlap too much)")
        print(f"\nTest Set Distance Statistics:")
        print(f"  Normal Reviews:")
        print(f"    Mean: {test_normal_dist.mean():.4f} ± {test_normal_dist.std():.4f}")
        print(f"  Anomalous Reviews:")
        print(f"    Mean: {test_anom_dist.mean():.4f} ± {test_anom_dist.std():.4f}")
    separation_test = (test_anom_dist.mean() / test_normal_dist.mean()) if test_normal_dist.mean() != 0 else float('inf')
    if print_output:
        print(f"\n  Separation Ratio: {separation_test:.4f}")

    # 5. Threshold Sensitivity (on test)
    if print_output:
        _print_header("5. THRESHOLD SENSITIVITY ANALYSIS")
    thresholds_to_test = np.linspace(test_df['distance_to_centroid'].min(),
                                     test_df['distance_to_centroid'].max(), 50)
    f1_scores, precisions, recalls = [], [], []
    for th in thresholds_to_test:
        preds = np.where(test_df['distance_to_centroid'] > th, 'Anomalous', 'Normal')
        f1_scores.append(f1_score(test_df['true_label'], preds, pos_label='Anomalous', zero_division=0))
        precisions.append(precision_score(test_df['true_label'], preds, pos_label='Anomalous', zero_division=0))
        recalls.append(recall_score(test_df['true_label'], preds, pos_label='Anomalous', zero_division=0))
    best_f1_idx = int(np.argmax(f1_scores))
    best_f1_threshold = float(thresholds_to_test[best_f1_idx])
    best_f1 = float(f1_scores[best_f1_idx])
    if print_output:
        print(f"\nCurrent Threshold: {threshold:.4f}")
        print(f"  Current F1: {test_metrics['f1_score']:.4f}")
        print(f"\nOptimal Threshold (by F1): {best_f1_threshold:.4f}")
        print(f"  Best F1: {best_f1:.4f}")
        print(f"  Improvement: {(best_f1 - test_metrics['f1_score']):.4f}")
        if abs(threshold - best_f1_threshold) / max(threshold, 1e-12) > 0.1:
            print(f"\n  Warning: Current threshold may not be optimal")
        else:
            print(f"\n  Threshold is near optimal")

    # 6. Cluster Analysis (if available)
    if print_output:
        _print_header("6. CLUSTER ANALYSIS")
    cluster_analysis = None
    if 'cluster' in train_df.columns:
        cluster_analysis = train_df.groupby('cluster').agg({
            'true_label': lambda x: (x == 'Anomalous').sum(),
            'distance_to_centroid': ['mean', 'std', 'count']
        }).round(4)
        cluster_analysis.columns = ['anomalous_count', 'mean_distance', 'std_distance', 'total_count']
        cluster_analysis['anomalous_ratio'] = (cluster_analysis['anomalous_count'] / cluster_analysis['total_count']).round(4)
        if print_output:
            print(f"\nCluster Statistics:")
            print(cluster_analysis)
            print(f"\nCluster Quality:")
            print(f"  Total clusters: {len(cluster_analysis)}")
            print(f"  Avg cluster size: {cluster_analysis['total_count'].mean():.0f}")
            print(f"  Largest cluster: {cluster_analysis['total_count'].max()}")
            print(f"  Smallest cluster: {cluster_analysis['total_count'].min()}")
    else:
        if print_output:
            print("No 'cluster' column found in training data. Skipping cluster analysis.")

    # 7. Final Summary
    if print_output:
        _print_header("7. FINAL SUMMARY & RECOMMENDATIONS")
        print(f"\nOVERALL MODEL PERFORMANCE:")
        print(f"  Test F1-Score:    {test_metrics['f1_score']:.4f}")
        print(f"  Test Precision:   {test_metrics['precision']:.4f}")
        print(f"  Test Recall:      {test_metrics['recall']:.4f}")
        print(f"  Test Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"  Test AUC-ROC:     {test_roc_auc:.4f}")
        print(f"  Separation Ratio: {separation_test:.4f}")
        print(f"\nMODEL GRADE:")
    score_sum = (test_metrics['f1_score'] + test_roc_auc + min(separation_test/2, 1.0)) / 3
    if score_sum >= 0.85:
        grade = "A (EXCELLENT)"
    elif score_sum >= 0.75:
        grade = "B (GOOD)"
    elif score_sum >= 0.65:
        grade = "C (ACCEPTABLE)"
    else:
        grade = "D (NEEDS IMPROVEMENT)"
    if print_output:
        print(f"  Overall Grade: {grade}")
        print(f"\nRECOMMENDATIONS:")
        if test_metrics['f1_score'] < 0.70:
            print(f"  - Consider retraining with different hyperparameters")
            print(f"  - Try adding more features or better feature engineering")
        if test_metrics['precision'] < 0.70:
            print(f"  - Too many false alarms - consider increasing threshold")
        if test_metrics['recall'] < 0.70:
            print(f"  - Missing too many anomalies - consider decreasing threshold")
        if separation_test < 1.5:
            print(f"  - Poor class separation - consider different clustering algorithm")
        if abs(train_metrics['f1_score'] - test_metrics['f1_score']) > 0.1:
            print(f"  - Large train/test gap suggests overfitting")
        if test_metrics['f1_score'] >= 0.70 and test_metrics['precision'] >= 0.70 and test_metrics['recall'] >= 0.70:
            print(f"  Model is performing well and ready for production!")

    # 8. Save report
    evaluation_report = {
        'training_metrics': train_metrics,
        'test_metrics': test_metrics,
        'roc_auc': {
            'train': float(train_roc_auc),
            'test': float(test_roc_auc)
        },
        'average_precision': {
            'train': float(train_ap),
            'test': float(test_ap)
        },
        'distance_statistics': {
            'train_separation': float(separation_train),
            'test_separation': float(separation_test)
        },
        'threshold_analysis': {
            'current_threshold': float(threshold),
            'optimal_threshold': float(best_f1_threshold),
            'current_f1': float(test_metrics['f1_score']),
            'current_accuracy': float(test_metrics['accuracy']),
            'optimal_f1': float(best_f1)
        },
        'overall_grade': grade
    }
    evaluation_report = convert_to_json_serializable(evaluation_report)

    with open(save_report_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)

    if print_output:
        print(f"\nEvaluation report saved to: {save_report_path}")
        print("=" * 80)

    return evaluation_report
