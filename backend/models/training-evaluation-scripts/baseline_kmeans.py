"""
Program Title:
baseline_kmeans.py – Baseline SVD + KMeans Clustering Module for the USAD (UnSupervised Anomaly Detection) Tool

Programmers:
Cristel Jane Baquing, Angelica Jean Evangelista, James Tristan Landa, Kharl Chester Velasco

Where the Program Fits in the General System Design:
This module belongs to the Baseline Modeling Component of the USAD system. Before applying advanced
optimization algorithms (e.g., PSO, DBSCAN selection, enhanced distance metrics), this script generates 
a classical baseline anomaly detection model using TruncatedSVD for dimensionality reduction and KMeans 
for vector-space clustering. It also automatically prepares preprocessing outputs, feature matrices, 
and data splits by calling other pipeline modules if needed, ensuring the entire model-building flow 
remains synchronized.

Date Written and Revised:
Original version: November 22, 2025  
Last revised: November 22, 2025

Purpose:
To build and evaluate a baseline unsupervised anomaly detector using:
• TruncatedSVD for reducing TF-IDF vectors to dense embeddings.  
• ℓ₂-normalized feature spaces for cosine-based clustering.  
• KMeans for obtaining cluster assignments and centroid vectors.  
• Distance-to-centroid scoring for anomaly detection.  
• Automatic threshold optimization using training F1-maximization.  
• Integrated evaluation report generation for comparison with optimized models.

The baseline provides a reference point used to measure improvement when advanced optimized
models are applied later in the pipeline.

Data Structures, Algorithms, and Control:
• Data Structures:
  - Sparse matrices X_train.npz and X_test.npz loaded from feature-matrices/.  
  - train_data.csv and test_data.csv containing labels and metadata.  
  - Pandas DataFrames enriched with cluster assignments and centroid distances.  
  - Serialized model artifacts (centroids, SVD, threshold) saved under baseline-models/.  
  - Baseline evaluation CSVs saved to baseline-data/.  
  - JSON evaluation report in evaluation-reports/.

• Algorithms:
  - TruncatedSVD (Scikit-learn) for dimensionality reduction.  
  - ℓ₂ normalization for cosine compatibility.  
  - KMeans clustering (k=8 by default) for unsupervised grouping.  
  - Cosine distance formula: distance = 1 − dot(z, centroid).  
  - Threshold sweep across 200 steps to maximize F1 (Anomalous as positive class).  
  - Centroid nearest-distance inference for test predictions.  
  - Evaluation summary via shared run_full_evaluation() utility.

• Control:
  - Automatically executes text_preprocessing.py if processed_reviews.csv is missing.  
  - Automatically executes feature_extraction.py if feature matrices are missing.  
  - Runs the full baseline model pipeline end-to-end when executed as __main__.  
  - Saves all artifacts and produces evaluation report without user intervention.  
  - Returns clean, standardized DataFrames used for further model comparison.
"""

import os
import subprocess
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
import joblib

# Reuse evaluation utilities
from evaluation_utils import run_full_evaluation

# Directory structure constants
BASELINE_MODELS_DIR = "baseline-models"
BASELINE_DATA_DIR = "baseline-data"
FEATURE_MATRICES_DIR = "feature-matrices"
TRAINING_DATA_DIR = "training-data"
EVALUATION_REPORTS_DIR = "evaluation-reports"


def get_base_models_dir():
    """Get the base models directory (parent of training-evaluation-scripts)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def get_baseline_models_dir():
    """Get the baseline models directory"""
    base_dir = get_base_models_dir()
    models_dir = os.path.join(base_dir, BASELINE_MODELS_DIR)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return models_dir


def get_baseline_data_dir():
    """Get the baseline data directory"""
    base_dir = get_base_models_dir()
    data_dir = os.path.join(base_dir, BASELINE_DATA_DIR)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def get_feature_matrices_dir():
    """Get the feature matrices directory"""
    return os.path.join(get_base_models_dir(), FEATURE_MATRICES_DIR)


def get_training_data_dir():
    """Get the training data directory"""
    return os.path.join(get_base_models_dir(), TRAINING_DATA_DIR)


def get_evaluation_reports_dir():
    """Get the evaluation reports directory"""
    base_dir = get_base_models_dir()
    reports_dir = os.path.join(base_dir, EVALUATION_REPORTS_DIR)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    return reports_dir


def ensure_preprocessing():
    training_data_dir = get_training_data_dir()
    processed_path = os.path.join(training_data_dir, 'processed_reviews.csv')
    if not os.path.exists(processed_path):
        print('processed_reviews.csv not found. Running text_preprocessing.py...')
        subprocess.check_call(
            ['python', 'text_preprocessing.py'],
            cwd=os.path.dirname(__file__) or '.'
        )
    else:
        print('✓ processed_reviews.csv found')


def ensure_features():
    feature_dir = get_feature_matrices_dir()
    training_data_dir = get_training_data_dir()
    required = {
        'X_train.npz': feature_dir,
        'X_test.npz': feature_dir,
        'train_data.csv': training_data_dir,
        'test_data.csv': training_data_dir
    }
    required_paths = [os.path.join(dir_path, file) for file, dir_path in required.items()]
    if not all(os.path.exists(p) for p in required_paths):
        print('Feature files not found. Running feature_extraction.py...')
        subprocess.check_call(
            ['python', 'feature_extraction.py'],
            cwd=os.path.dirname(__file__) or '.'
        )
    else:
        print('✓ Feature files found:', ', '.join(required.keys()))


def run_baseline_kmeans(svd_components: int = 200, n_clusters: int = 8, random_state: int = 42):
    print('=' * 60)
    print('BASELINE: SVD + KMeans (no PSO/DBSCAN)')
    print('=' * 60)

    # Load features and metadata
    feature_dir = get_feature_matrices_dir()
    training_data_dir = get_training_data_dir()
    X_train = sparse.load_npz(os.path.join(feature_dir, 'X_train.npz'))
    X_test = sparse.load_npz(os.path.join(feature_dir, 'X_test.npz'))
    train_df = pd.read_csv(os.path.join(training_data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(training_data_dir, 'test_data.csv'))

    # Map labels to human-readable
    label_map = {'CG': 'Anomalous', 'OR': 'Normal'}
    if 'label' in train_df.columns:
        train_df['true_label'] = train_df['label'].map(label_map).fillna(train_df['label'])
    if 'label' in test_df.columns:
        test_df['true_label'] = test_df['label'].map(label_map).fillna(test_df['label'])

    # Dimensionality reduction
    print(f'Applying TruncatedSVD with {svd_components} components...')
    svd = TruncatedSVD(n_components=svd_components, random_state=random_state)
    Z_train = svd.fit_transform(X_train)
    Z_test = svd.transform(X_test)
    print(f'✓ Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}')

    # Normalize for cosine similarity
    Z_train = normalize(Z_train, norm='l2', axis=1)
    Z_test = normalize(Z_test, norm='l2', axis=1)

    # KMeans clustering on train
    print(f'Running KMeans with k={n_clusters}...')
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    train_clusters = km.fit_predict(Z_train)

    # Compute cosine distances to assigned centroid for train
    centroids = km.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    assigned_centroids = centroids[train_clusters]
    dot_products = np.sum(Z_train * assigned_centroids, axis=1)
    train_distances = 1.0 - dot_products
    train_df['cluster'] = train_clusters
    train_df['distance_to_centroid'] = train_distances

    # Threshold selection on train for best F1 (Anomalous as positive)
    print('Selecting decision threshold on training set (max F1)...')
    th_min, th_max = float(train_distances.min()), float(train_distances.max())
    thresholds = np.linspace(th_min, th_max, 200)
    best_f1, best_th = -1.0, th_min
    y_true_train = train_df['true_label'] if 'true_label' in train_df.columns else pd.Series(['Normal'] * len(train_df))
    for th in thresholds:
        preds = np.where(train_df['distance_to_centroid'] > th, 'Anomalous', 'Normal')
        f1 = f1_score(y_true_train, preds, pos_label='Anomalous', zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    print(f'✓ Best threshold: {best_th:.4f} (Train F1={best_f1:.4f})')

    # Apply to test: nearest centroid distance and prediction
    sim_matrix = Z_test.dot(centroids.T)
    nearest_idx = np.argmax(sim_matrix, axis=1)
    test_df['cluster'] = nearest_idx
    test_df['distance_to_centroid'] = 1.0 - np.max(sim_matrix, axis=1)

    # Save baseline artifacts
    print('Saving baseline artifacts...')
    baseline_models_dir = get_baseline_models_dir()
    baseline_data_dir = get_baseline_data_dir()
    
    joblib.dump(centroids, os.path.join(baseline_models_dir, 'baseline_centroids.pkl'))
    joblib.dump(svd, os.path.join(baseline_models_dir, 'baseline_svd.pkl'))
    joblib.dump(
        {'threshold': best_th, 'train_f1': float(best_f1)},
        os.path.join(baseline_models_dir, 'baseline_threshold.pkl')
    )
    train_df.to_csv(os.path.join(baseline_data_dir, 'baseline_train.csv'), index=False)
    test_df.to_csv(os.path.join(baseline_data_dir, 'baseline_test.csv'), index=False)
    print(f'✓ Saved: {os.path.join(baseline_models_dir, "baseline_centroids.pkl")}')
    print(f'✓ Saved: {os.path.join(baseline_models_dir, "baseline_svd.pkl")}')
    print(f'✓ Saved: {os.path.join(baseline_models_dir, "baseline_threshold.pkl")}')
    print(f'✓ Saved: {os.path.join(baseline_data_dir, "baseline_train.csv")}')
    print(f'✓ Saved: {os.path.join(baseline_data_dir, "baseline_test.csv")}')

    # Evaluation report (reusing shared formatting/util)
    print('Running evaluation...')
    reports_dir = get_evaluation_reports_dir()
    report_path = os.path.join(reports_dir, 'baseline_evaluation_report.json')
    run_full_evaluation(
        train_df, test_df, best_th,
        save_report_path=report_path,
        print_output=True
    )


if __name__ == '__main__':
    # Step 1: preprocessing
    ensure_preprocessing()
    # Step 2: features
    ensure_features()
    # Step 3: baseline model + evaluation
    run_baseline_kmeans()


