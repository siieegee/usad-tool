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


def ensure_preprocessing():
    if not os.path.exists('processed_reviews.csv'):
        print('processed_reviews.csv not found. Running text_preprocessing.py...')
        subprocess.check_call(['python', 'text_preprocessing.py'], cwd=os.path.dirname(__file__) or '.')
    else:
        print('✓ processed_reviews.csv found')


def ensure_features():
    required = ['X_train.npz', 'X_test.npz', 'train_data.csv', 'test_data.csv']
    if not all(os.path.exists(p) for p in required):
        print('Feature files not found. Running feature_extraction.py...')
        subprocess.check_call(['python', 'feature_extraction.py'], cwd=os.path.dirname(__file__) or '.')
    else:
        print('✓ Feature files found:', ', '.join(required))


def run_baseline_kmeans(svd_components: int = 200, n_clusters: int = 8, random_state: int = 42):
    print('=' * 60)
    print('BASELINE: SVD + KMeans (no PSO/DBSCAN)')
    print('=' * 60)

    # Load features and metadata
    X_train = sparse.load_npz('X_train.npz')
    X_test = sparse.load_npz('X_test.npz')
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')

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
    joblib.dump(centroids, 'baseline_centroids.pkl')
    joblib.dump(svd, 'baseline_svd.pkl')
    joblib.dump({'threshold': best_th, 'train_f1': float(best_f1)}, 'baseline_threshold.pkl')
    train_df.to_csv('baseline_train.csv', index=False)
    test_df.to_csv('baseline_test.csv', index=False)
    print('✓ Saved: baseline_centroids.pkl, baseline_svd.pkl, baseline_threshold.pkl, baseline_train.csv, baseline_test.csv')

    # Evaluation report (reusing shared formatting/util)
    print('Running evaluation...')
    run_full_evaluation(train_df, test_df, best_th, save_report_path='baseline_evaluation_report.json', print_output=True)


if __name__ == '__main__':
    # Step 1: preprocessing
    ensure_preprocessing()
    # Step 2: features
    ensure_features()
    # Step 3: baseline model + evaluation
    run_baseline_kmeans()


