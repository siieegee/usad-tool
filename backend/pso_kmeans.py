import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import (silhouette_score, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import normalize
import joblib
from pyswarm import pso
import warnings
import os

warnings.filterwarnings('ignore')

# ========== SET RANDOM SEEDS FOR REPRODUCIBILITY ==========
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)

# ========== CONFIGURATION ==========
SVD_COMPONENTS = 200
DBSCAN_EPS = 0.1
DBSCAN_MIN_SAMPLES = 10
PSO_K_SWARMSIZE = 10
PSO_K_MAXITER = 8
PSO_CENTROID_SWARMSIZE = 20
PSO_CENTROID_MAXITER = 15
KMEANS_BATCH = 2048

print("=" * 60)
print("PSO-ENHANCED DBSCAN + K-MEANS CLUSTERING")
print("=" * 60)

# ========== LOAD DATA ==========
print("\nLoading data...")

# Check if files exist
required_files = ["X_train.npz", "X_test.npz", "train_data.csv", "test_data.csv"]
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Required file not found: {file}")

train_features = sparse.load_npz("X_train.npz")
test_features = sparse.load_npz("X_test.npz")
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

print(f"✓ Training features shape: {train_features.shape}")
print(f"✓ Test features shape: {test_features.shape}")
print(f"✓ Training samples: {len(train_df)}")
print(f"✓ Test samples: {len(test_df)}")

# Map labels
label_map = {'CG': 'Anomalous', 'OR': 'Normal'}
train_df['true_label'] = train_df['label'].map(label_map)
test_df['true_label'] = test_df['label'].map(label_map)

print(f"\nLabel distribution (train):")
print(train_df['true_label'].value_counts())

# ========== DIMENSIONALITY REDUCTION ==========
print(f"\n{'=' * 60}")
print(f"PHASE 1: DIMENSIONALITY REDUCTION")
print(f"{'=' * 60}")
print(f"Applying Truncated SVD ({SVD_COMPONENTS} components)...")

svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_SEED)
reduced_train = svd.fit_transform(train_features)
reduced_test = svd.transform(test_features)

print(f"✓ Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

# Normalize for cosine similarity
reduced_train = normalize(reduced_train, norm='l2', axis=1)
reduced_test = normalize(reduced_test, norm='l2', axis=1)
print(f"✓ Features normalized for cosine similarity")

# ========== DBSCAN CLUSTERING ==========
print(f"\n{'=' * 60}")
print(f"PHASE 2: DBSCAN CLUSTERING")
print(f"{'=' * 60}")
print(f"Running DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")

dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='cosine')
dbscan.fit(reduced_train)
dbscan_labels = dbscan.labels_

core_mask = dbscan_labels != -1
noise_mask = dbscan_labels == -1

n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_core_points = np.sum(core_mask)
n_noise_points = np.sum(noise_mask)

print(f"✓ DBSCAN clusters formed: {n_dbscan_clusters}")
print(f"✓ Core points: {n_core_points}")
print(f"✓ Noise/ambiguous points: {n_noise_points}")

# Initialize cluster assignments
train_df['cluster'] = -2  # Default value
train_df.loc[core_mask, 'cluster'] = dbscan_labels[core_mask]

# ========== PSO + K-MEANS ON NOISE POINTS ==========
print(f"\n{'=' * 60}")
print(f"PHASE 3: PSO-OPTIMIZED K-MEANS ON AMBIGUOUS POINTS")
print(f"{'=' * 60}")

subset_train = reduced_train[noise_mask]

if subset_train.shape[0] >= 3:
    print(f"Processing {subset_train.shape[0]} ambiguous points...")
    
    # Step 1: PSO to find optimal K
    print("\nStep 1: Finding optimal K using PSO...")
    
    # Re-seed before PSO
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    def k_objective(k):
        k = int(k[0])
        if k < 2:
            return 999999
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED, batch_size=KMEANS_BATCH)
        km.fit(subset_train)
        score = silhouette_score(subset_train, km.labels_, metric='cosine')
        return -score
    
    lb, ub = [2], [min(12, subset_train.shape[0] - 1)]
    best_k, best_score = pso(k_objective, lb, ub,
                             swarmsize=PSO_K_SWARMSIZE, maxiter=PSO_K_MAXITER)
    best_k = int(best_k[0])
    print(f"✓ Optimal K: {best_k}")
    print(f"✓ Silhouette score: {-best_score:.4f}")
    
    # Step 2: PSO to optimize centroids
    print("\nStep 2: Optimizing centroids using PSO...")
    
    # Re-seed before sampling
    np.random.seed(RANDOM_SEED)
    
    # Sample subset for efficiency
    centroid_sample_indices = np.random.choice(
        subset_train.shape[0], 
        min(2000, subset_train.shape[0]), 
        replace=False
    )
    sampled_subset = subset_train[centroid_sample_indices]
    n_features = sampled_subset.shape[1]
    
    # Re-seed before PSO centroid optimization
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    def centroid_objective(flat_centroids):
        centroids = flat_centroids.reshape((best_k, n_features))
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        sims = sampled_subset.dot(centroids.T)
        labels = np.argmax(sims, axis=1)
        sse = np.sum(1.0 - sims[np.arange(len(labels)), labels])
        return sse
    
    # Set bounds for PSO
    feature_min = sampled_subset.min(axis=0)
    feature_max = sampled_subset.max(axis=0)
    epsilon = 1e-6
    feature_max_fixed = np.where(feature_max == feature_min, feature_min + epsilon, feature_max)
    lb = np.tile(feature_min, best_k)
    ub = np.tile(feature_max_fixed, best_k)
    
    # Run PSO for centroid optimization
    best_centroids_flat, best_sse = pso(
        centroid_objective, lb, ub,
        swarmsize=PSO_CENTROID_SWARMSIZE,
        maxiter=PSO_CENTROID_MAXITER
    )
    
    best_centroids = best_centroids_flat.reshape((best_k, n_features))
    best_centroids = best_centroids / (np.linalg.norm(best_centroids, axis=1, keepdims=True) + 1e-10)
    print(f"✓ PSO centroid optimization complete")
    print(f"✓ Best SSE: {best_sse:.4f}")
    
    # Step 3: Final K-Means with PSO-optimized centroids
    print("\nStep 3: Running final K-Means with optimized centroids...")
    
    kmeans_final = MiniBatchKMeans(
        n_clusters=best_k, 
        init=best_centroids, 
        n_init=1,
        random_state=RANDOM_SEED, 
        batch_size=KMEANS_BATCH, 
        max_iter=200
    )
    kmeans_final.fit(subset_train)
    
    # Assign offset cluster labels
    offset = dbscan_labels.max() + 1
    train_df.loc[noise_mask, 'cluster'] = kmeans_final.labels_ + offset
    print(f"✓ K-Means clustering complete")
    print(f"✓ New clusters created: {best_k} (labels {offset} to {offset + best_k - 1})")
    
else:
    print(f"⚠ Only {subset_train.shape[0]} ambiguous points - skipping PSO + K-means")
    if subset_train.shape[0] > 0:
        offset = dbscan_labels.max() + 1
        train_df.loc[noise_mask, 'cluster'] = offset

# ========== COMPUTE CENTROIDS & DISTANCES ==========
print(f"\n{'=' * 60}")
print(f"PHASE 4: DISTANCE CALCULATION")
print(f"{'=' * 60}")

unique_labels = np.unique(train_df['cluster'])
centroids_dict = {}

print(f"Computing centroids for {len(unique_labels)} clusters...")

for label in unique_labels:
    cluster_members = reduced_train[train_df['cluster'] == label]
    if cluster_members.shape[0] == 0:
        centroids_dict[label] = np.zeros(reduced_train.shape[1])
    else:
        cent = cluster_members.mean(axis=0)
        centroids_dict[label] = cent / (np.linalg.norm(cent) + 1e-10)

all_centroids = np.vstack([centroids_dict[label] for label in unique_labels])
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
train_centroid_indices = train_df['cluster'].map(label_to_index).values

# Calculate cosine distances
dot_products = np.sum(reduced_train * all_centroids[train_centroid_indices], axis=1)
distances_train = 1.0 - dot_products
train_df['distance_to_centroid'] = distances_train

print(f"✓ Distance statistics:")
print(f"  Mean: {distances_train.mean():.4f}")
print(f"  Std: {distances_train.std():.4f}")
print(f"  Min: {distances_train.min():.4f}")
print(f"  Max: {distances_train.max():.4f}")

# ========== THRESHOLD SELECTION ==========
print(f"\n{'=' * 60}")
print(f"PHASE 5: THRESHOLD OPTIMIZATION")
print(f"{'=' * 60}")

best_f1, best_threshold = -1.0, None
threshold_range = np.linspace(distances_train.min(), distances_train.max(), 100)

print("Searching for optimal threshold...")
for th in threshold_range:
    preds = np.where(train_df['distance_to_centroid'] > th, 'Anomalous', 'Normal')
    f1 = f1_score(train_df['true_label'], preds, pos_label='Anomalous')
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = th

print(f"✓ Optimal threshold: {best_threshold:.4f}")
print(f"✓ Training F1 score: {best_f1:.4f}")

# ========== TEST SET EVALUATION ==========
print(f"\n{'=' * 60}")
print(f"PHASE 6: TEST SET EVALUATION")
print(f"{'=' * 60}")

# Assign test samples to nearest cluster
cosine_sim_matrix = reduced_test.dot(all_centroids.T)
nearest_indices = np.argmax(cosine_sim_matrix, axis=1)
test_df['cluster'] = [unique_labels[idx] for idx in nearest_indices]

# Calculate distances for test set
distances_test = 1.0 - np.max(cosine_sim_matrix, axis=1)
test_df['distance_to_centroid'] = distances_test

# Make predictions
train_preds = np.where(train_df['distance_to_centroid'] > best_threshold, 'Anomalous', 'Normal')
test_preds = np.where(test_df['distance_to_centroid'] > best_threshold, 'Anomalous', 'Normal')

# Print metrics function
def print_metrics(y_true, y_pred, dataset="Set"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='Anomalous', zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label='Anomalous', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='Anomalous', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=['Anomalous', 'Normal'])
    
    print(f"\n{dataset} Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Anom  Norm")
    print(f"  Actual Anom  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"         Norm  {cm[1][0]:4d}  {cm[1][1]:4d}")

print_metrics(train_df['true_label'], train_preds, dataset="TRAINING")
print_metrics(test_df['true_label'], test_preds, dataset="TEST")

# ========== SAVE RESULTS ==========
print(f"\n{'=' * 60}")
print(f"SAVING RESULTS")
print(f"{'=' * 60}")

train_df.to_csv("best_run_train.csv", index=False)
test_df.to_csv("best_run_test.csv", index=False)
joblib.dump(all_centroids, "best_run_centroids.pkl")
joblib.dump(svd, "best_run_svd.pkl")
joblib.dump({"threshold": best_threshold, "f1_score": best_f1}, "best_run_threshold.pkl")

print("✓ Saved: best_run_train.csv")
print("✓ Saved: best_run_test.csv")
print("✓ Saved: best_run_centroids.pkl")
print("✓ Saved: best_run_svd.pkl")
print("✓ Saved: best_run_threshold.pkl")

print(f"\n{'=' * 60}")
print("PIPELINE COMPLETED SUCCESSFULLY!")
print(f"{'=' * 60}")