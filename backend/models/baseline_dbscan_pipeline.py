import os
import subprocess
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
import joblib

from evaluation_utils import run_full_evaluation


def ensure_preprocessing():
    if not os.path.exists('processed_reviews.csv'):
        print('processed_reviews.csv not found. Running text_preprocessing.py...')
        subprocess.check.call(['python', 'text_preprocessing.py'], cwd=os.path.dirname(__file__) or '.')
    else:
        print('✓ processed_reviews.csv found')


def ensure_features():
    required = ['X_train.npz', 'X_test.npz', 'train_data.csv', 'test_data.csv']
    if not all(os.path.exists(p) for p in required):
        print('Feature files not found. Running feature_extraction.py...')
        subprocess.check_call(['python', 'feature_extraction.py'], cwd=os.path.dirname(__file__) or '.')
    else:
        print('✓ Feature files found:', ', '.join(required))


def compute_centroids(z_train: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(labels)
    # Exclude noise label -1 from centroid computation
    unique = unique[unique != -1]
    centroids = []
    for lbl in unique:
        members = z_train[labels == lbl]
        if members.shape[0] == 0:
            centroids.append(np.zeros(z_train.shape[1]))
        else:
            c = members.mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-10)
            centroids.append(c)
    if len(centroids) == 0:
        # Fallback to a single zero centroid to avoid downstream errors
        centroids = [np.zeros(z_train.shape[1])]
        unique = np.array([0])
    return np.vstack(centroids), unique


def run_baseline_dbscan(svd_components: int = 200, eps: float = 0.1, min_samples: int = 10, random_state: int = 42):
    print('=' * 60)
    print('BASELINE: SVD + DBSCAN')
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

    # DBSCAN clustering on train
    print(f'Running DBSCAN (eps={eps}, min_samples={min_samples})...')
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    db_labels = db.fit_predict(Z_train)
    train_df['cluster'] = db_labels
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_core = np.sum(db_labels != -1)
    n_noise = np.sum(db_labels == -1)
    print(f'✓ DBSCAN clusters: {n_clusters} | core points: {n_core} | noise: {n_noise}')

    # Compute centroids for non-noise clusters
    centroids, unique_labels = compute_centroids(Z_train, db_labels)

    # Distances for train: if noise, distance to nearest centroid; else distance to its cluster centroid
    if len(unique_labels) > 0:
        # Build mapping from cluster label to centroid index
        label_to_index = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        # For assigned clusters
        assigned_idx = np.array([label_to_index.get(lbl, -1) for lbl in db_labels])
        # Cosine similarity to all centroids
        sims_all = Z_train.dot(centroids.T)
        # Distance for core points
        core_mask = assigned_idx != -1
        core_dot = sims_all[np.arange(sims_all.shape[0]), np.clip(assigned_idx, 0, len(unique_labels)-1)]
        distances = np.empty(Z_train.shape[0], dtype=float)
        distances[core_mask] = 1.0 - core_dot[core_mask]
        # Distance for noise points: nearest centroid
        if np.any(~core_mask):
            nearest_sim = np.max(sims_all[~core_mask], axis=1)
            distances[~core_mask] = 1.0 - nearest_sim
    else:
        # No clusters formed: fall back to zero distances
        distances = np.zeros(Z_train.shape[0], dtype=float)

    train_df['distance_to_centroid'] = distances

    # Threshold selection on train (maximize F1 for Anomalous)
    print('Selecting decision threshold on training set (max F1)...')
    th_min, th_max = float(distances.min()), float(distances.max())
    thresholds = np.linspace(th_min, th_max, 200) if th_max > th_min else np.array([th_min])
    best_f1, best_th = -1.0, th_min
    y_true_train = train_df['true_label'] if 'true_label' in train_df.columns else pd.Series(['Normal'] * len(train_df))
    for th in thresholds:
        preds = np.where(train_df['distance_to_centroid'] > th, 'Anomalous', 'Normal')
        f1 = f1_score(y_true_train, preds, pos_label='Anomalous', zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    print(f'✓ Best threshold: {best_th:.4f} (Train F1={best_f1:.4f})')

    # Apply to test: assign nearest centroid and compute distances
    if len(unique_labels) > 0:
        sims_test = Z_test.dot(centroids.T)
        nearest_idx = np.argmax(sims_test, axis=1)
        test_df['cluster'] = unique_labels[nearest_idx]
        test_df['distance_to_centroid'] = 1.0 - np.max(sims_test, axis=1)
    else:
        test_df['cluster'] = -1
        test_df['distance_to_centroid'] = 0.0

    # Save artifacts
    print('Saving baseline DBSCAN artifacts...')
    joblib.dump(centroids, 'baseline_dbscan_centroids.pkl')
    joblib.dump(svd, 'baseline_dbscan_svd.pkl')
    joblib.dump({'threshold': best_th, 'train_f1': float(best_f1), 'eps': float(eps), 'min_samples': int(min_samples)}, 'baseline_dbscan_threshold.pkl')
    train_df.to_csv('baseline_dbscan_train.csv', index=False)
    test_df.to_csv('baseline_dbscan_test.csv', index=False)
    print('✓ Saved: baseline_dbscan_centroids.pkl, baseline_dbscan_svd.pkl, baseline_dbscan_threshold.pkl, baseline_dbscan_train.csv, baseline_dbscan_test.csv')

    # Evaluation report
    print('Running evaluation...')
    run_full_evaluation(train_df, test_df, best_th, save_report_path='baseline_dbscan_evaluation_report.json', print_output=True)


if __name__ == '__main__':
    ensure_preprocessing()
    ensure_features()
    run_baseline_dbscan()



