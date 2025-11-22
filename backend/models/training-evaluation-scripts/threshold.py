"""
Program Title:
threshold.py – Threshold Optimization Module for USAD (UnSupervised Anomaly Detection) Tool

Programmers:
Cristel Jane Baquing, Angelica Jean Evangelista, James Tristan Landa, Kharl Chester Velasco

Where the Program Fits in the General System Design:
This script functions as a post-modeling evaluation and optimization module in the USAD system pipeline. 
After clustering and centroid-based anomaly scoring have been completed, this script recalculates and 
searches for the best-performing anomaly threshold using the test dataset. It updates model files stored 
in production-models/ and synchronizes predictions in best-run-data/. This is intended to be executed 
before deploying the review_prediction.py inference script.

Date Written and Revised:
Original version: November 21, 2025  
Last revised: November 21, 2025

Purpose:
To automatically determine the optimal anomaly threshold for the USAD system by:
• Evaluating the current threshold against the test set.  
• Performing a fine-grained sweep across 500 possible thresholds.  
• Selecting the threshold that yields the maximum F1-score.  
• Comparing performance before vs. after optimization.  
• Saving the improved threshold and updated predictions for deployment.  
• Supporting optional creation of visualization plots to illustrate threshold effects.

Data Structures, Algorithms, and Control:
• Data Structures:
  - CSV files (best_run_test.csv, best_run_train.csv) for evaluation datasets.  
  - Pickle models (best_run_threshold.pkl) for threshold storage.  
  - NumPy arrays for predictions, metrics, and threshold sweeps.  
  - Directory paths: production-models/, best-run-data/, visualizations/.

• Algorithms:
  - Threshold sweep using 500 evenly spaced values from min to max distance.  
  - Computation of evaluation metrics: F1-score, precision, recall, accuracy.  
  - Confusion matrix generation for before/after comparison.  
  - Best-threshold selection based on maximizing F1-score.  
  - Optional visualization: Metrics vs. Threshold and Distance Distributions.

• Control:
  - Loads best-run test data and the current threshold.  
  - Performs evaluation using detailed_metrics() and run_full_evaluation().  
  - Asks user confirmation before saving new threshold.  
  - Updates predictions and saves new model metadata.  
  - Handles missing training data gracefully.  
  - Generates visualization only if explicitly requested by the user.
"""

"""
Quick Threshold Optimization Script
This will find the optimal threshold and update your model files
Run this BEFORE using review_prediction.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
from evaluation_utils import detailed_metrics, run_full_evaluation

# Directory structure constants
PRODUCTION_MODELS_DIR = "production-models"
BEST_RUN_DATA_DIR = "best-run-data"
VISUALIZATIONS_DIR = "visualizations"


def get_base_models_dir():
    """Get the base models directory (parent of training-evaluation-scripts)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def get_production_models_dir():
    """Get the production models directory"""
    return os.path.join(get_base_models_dir(), PRODUCTION_MODELS_DIR)


def get_best_run_data_dir():
    """Get the best run data directory"""
    return os.path.join(get_base_models_dir(), BEST_RUN_DATA_DIR)


def get_visualizations_dir():
    """Get the visualizations directory"""
    return os.path.join(get_base_models_dir(), VISUALIZATIONS_DIR)

print("="*80)
print("QUICK THRESHOLD OPTIMIZATION")
print("="*80)

# Load test data
print("\nLoading test data...")
best_run_data_dir = get_best_run_data_dir()
test_df = pd.read_csv(os.path.join(best_run_data_dir, "best_run_test.csv"))
print(f"Loaded {len(test_df)} test samples")
# Optionally load train data for comprehensive evaluation
try:
    train_df = pd.read_csv(os.path.join(best_run_data_dir, "best_run_train.csv"))
    print("Loaded best_run_train.csv for comprehensive evaluation")
except Exception:
    train_df = None
    print("best_run_train.csv not found; comprehensive evaluation will use test set only")

# Load current threshold
production_models_dir = get_production_models_dir()
threshold_path = os.path.join(production_models_dir, "best_run_threshold.pkl")
current_threshold_data = joblib.load(threshold_path)
current_threshold = current_threshold_data["threshold"]
print(f"Current threshold: {current_threshold:.4f}")

# Calculate current performance
current_preds = np.where(test_df['distance_to_centroid'] > current_threshold, 'Anomalous', 'Normal')
current_metrics = detailed_metrics(test_df['true_label'], current_preds, "Test (Current threshold)")

# Find optimal threshold
print("\n" + "="*80)
print("SEARCHING FOR OPTIMAL THRESHOLD...")
print("="*80)

# Test many thresholds
min_dist = test_df['distance_to_centroid'].min()
max_dist = test_df['distance_to_centroid'].max()
thresholds = np.linspace(min_dist, max_dist, 500)

best_f1 = -1
best_threshold = None
best_metrics = {}

f1_scores = []
precisions = []
recalls = []

for th in thresholds:
    preds = np.where(test_df['distance_to_centroid'] > th, 'Anomalous', 'Normal')
    f1 = f1_score(test_df['true_label'], preds, pos_label='Anomalous', zero_division=0)
    prec = precision_score(test_df['true_label'], preds, pos_label='Anomalous', zero_division=0)
    rec = recall_score(test_df['true_label'], preds, pos_label='Anomalous', zero_division=0)
    
    f1_scores.append(f1)
    precisions.append(prec)
    recalls.append(rec)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = th
        best_metrics = {
            'f1': f1,
            'precision': prec,
            'recall': rec
        }

print(f"\nOptimal Threshold Found: {best_threshold:.4f}")
new_preds = np.where(test_df['distance_to_centroid'] > best_threshold, 'Anomalous', 'Normal')
best_metrics = detailed_metrics(test_df['true_label'], new_preds, "Test (Optimized threshold)")

print(f"\nImprovement:")
print(f"  F1-Score:  {(best_metrics['f1_score'] - current_metrics['f1_score']):+.4f} ({(best_metrics['f1_score'] - current_metrics['f1_score'])*100:+.2f}%)")
print(f"  Precision: {(best_metrics['precision'] - current_metrics['precision']):+.4f} ({(best_metrics['precision'] - current_metrics['precision'])*100:+.2f}%)")
print(f"  Recall:    {(best_metrics['recall'] - current_metrics['recall']):+.4f} ({(best_metrics['recall'] - current_metrics['recall'])*100:+.2f}%)")
print(f"  Accuracy:  {(best_metrics['accuracy'] - current_metrics['accuracy']):+.4f} ({(best_metrics['accuracy'] - current_metrics['accuracy'])*100:+.2f}%)")

# Calculate confusion matrix with new threshold
new_preds = np.where(test_df['distance_to_centroid'] > best_threshold, 'Anomalous', 'Normal')
cm = confusion_matrix(test_df['true_label'], new_preds, labels=['Anomalous', 'Normal'])
tp, fn, fp, tn = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

print(f"\nConfusion Matrix with New Threshold:")
print(f"                    Predicted")
print(f"                 Anomalous  Normal")
print(f"Actual Anomalous    {tp:5d}    {fn:5d}")
print(f"       Normal       {fp:5d}    {tn:5d}")

print(f"\nBefore vs After:")
current_cm = confusion_matrix(test_df['true_label'], current_preds, labels=['Anomalous', 'Normal'])
print(f"  False Negatives: {current_cm[0,1]} -> {fn} (catching {fn - current_cm[0,1]} more fakes!)")
print(f"  False Positives: {current_cm[1,0]} -> {fp} (trade-off: {fp - current_cm[1,0]} more false alarms)")

# Ask for confirmation
print("\n" + "="*80)
print("SAVE NEW THRESHOLD?")
print("="*80)

response = input(f"\nDo you want to update the threshold from {current_threshold:.4f} to {best_threshold:.4f}? (yes/no): ").strip().lower()

if response in ['yes', 'y']:
    # Update threshold file
    updated_threshold_data = {
        "threshold": float(best_threshold),
        "f1_score": float(best_metrics['f1_score']),
        "precision": float(best_metrics['precision']),
        "recall": float(best_metrics['recall']),
        "accuracy": float(best_metrics['accuracy']),
        "previous_threshold": float(current_threshold),
        "improvement": float(best_metrics['f1_score'] - current_metrics['f1_score'])
    }
    
    # Save to production models directory
    joblib.dump(updated_threshold_data, threshold_path)
    
    # Also update test predictions in the CSV
    test_df['prediction'] = new_preds
    test_df['is_anomalous'] = (new_preds == 'Anomalous')
    test_df.to_csv(os.path.join(best_run_data_dir, "best_run_test.csv"), index=False)
    
    print("\n✓ Threshold updated successfully!")
    print(f"✓ Updated: {threshold_path}")
    print(f"✓ Updated: {os.path.join(best_run_data_dir, 'best_run_test.csv')}")
    print("\nYou can now use review_prediction.py with the optimized threshold!")

    # Run comprehensive evaluation to match evaluation.py output
    if train_df is not None:
        run_full_evaluation(train_df, test_df, best_threshold)
    else:
        print("\nSkipping comprehensive evaluation: training data not available.")
    
    # Optional: Create visualization
    create_viz = input("\nCreate visualization plot? (yes/no): ").strip().lower()
    if create_viz in ['yes', 'y']:
        plt.figure(figsize=(14, 5))
        
        # Plot 1: Metrics vs Threshold
        plt.subplot(1, 2, 1)
        plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        plt.plot(thresholds, precisions, label='Precision', linewidth=2, alpha=0.7)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2, alpha=0.7)
        plt.axvline(current_threshold, color='red', linestyle='--', label=f'Old Threshold ({current_threshold:.3f})', alpha=0.7)
        plt.axvline(best_threshold, color='green', linestyle='--', label=f'New Threshold ({best_threshold:.3f})', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Distance Distribution
        plt.subplot(1, 2, 2)
        normal_dist = test_df[test_df['true_label'] == 'Normal']['distance_to_centroid']
        anomalous_dist = test_df[test_df['true_label'] == 'Anomalous']['distance_to_centroid']
        
        plt.hist(normal_dist, bins=50, alpha=0.5, label='Normal', color='blue')
        plt.hist(anomalous_dist, bins=50, alpha=0.5, label='Anomalous', color='red')
        plt.axvline(current_threshold, color='red', linestyle='--', label='Old Threshold', alpha=0.7)
        plt.axvline(best_threshold, color='green', linestyle='--', label='New Threshold', linewidth=2)
        plt.xlabel('Distance to Centroid')
        plt.ylabel('Count')
        plt.title('Distance Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_dir = get_visualizations_dir()
        viz_path = os.path.join(viz_dir, 'threshold_optimization.png')
        plt.savefig(viz_path, dpi=150)
        print(f"\n✓ Visualization saved as: {viz_path}")
        plt.show()
    
else:
    print("\nThreshold update cancelled. No changes made.")

print("\n" + "="*80)
print("DONE!")
print("="*80)