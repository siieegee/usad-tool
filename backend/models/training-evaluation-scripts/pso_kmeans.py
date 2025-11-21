import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import normalize
import joblib
from pyswarm import pso
import warnings
from evaluation_utils import run_full_evaluation

warnings.filterwarnings('ignore')

# Directory structure constants
PRODUCTION_MODELS_DIR = "production-models"
BEST_RUN_DATA_DIR = "best-run-data"
FEATURE_MATRICES_DIR = "feature-matrices"
TRAINING_DATA_DIR = "training-data"


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


def get_feature_matrices_dir():
    """Get the feature matrices directory"""
    return os.path.join(get_base_models_dir(), FEATURE_MATRICES_DIR)


def get_training_data_dir():
    """Get the training data directory"""
    return os.path.join(get_base_models_dir(), TRAINING_DATA_DIR)

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

# Get directory paths
feature_dir = get_feature_matrices_dir()
training_data_dir = get_training_data_dir()

# Check if files exist
required_files = {
    "X_train.npz": feature_dir,
    "X_test.npz": feature_dir,
    "train_data.csv": training_data_dir,
    "test_data.csv": training_data_dir
}
for file, dir_path in required_files.items():
    file_path = os.path.join(dir_path, file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")

train_features = sparse.load_npz(os.path.join(feature_dir, "X_train.npz"))
test_features = sparse.load_npz(os.path.join(feature_dir, "X_test.npz"))
train_df = pd.read_csv(os.path.join(training_data_dir, "train_data.csv"))
test_df = pd.read_csv(os.path.join(training_data_dir, "test_data.csv"))

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

# Run comprehensive evaluation identical to evaluation.py
run_full_evaluation(train_df, test_df, best_threshold)

# ========== SAVE RESULTS ==========
print(f"\n{'=' * 60}")
print(f"SAVING RESULTS")
print(f"{'=' * 60}")

# Save to appropriate directories
best_run_data_dir = get_best_run_data_dir()
production_models_dir = get_production_models_dir()

train_df.to_csv(os.path.join(best_run_data_dir, "best_run_train.csv"), index=False)
test_df.to_csv(os.path.join(best_run_data_dir, "best_run_test.csv"), index=False)
joblib.dump(all_centroids, os.path.join(production_models_dir, "best_run_centroids.pkl"))
joblib.dump(svd, os.path.join(production_models_dir, "best_run_svd.pkl"))
joblib.dump(
    {"threshold": best_threshold, "f1_score": best_f1},
    os.path.join(production_models_dir, "best_run_threshold.pkl")
)

print(f"✓ Saved: {os.path.join(best_run_data_dir, 'best_run_train.csv')}")
print(f"✓ Saved: {os.path.join(best_run_data_dir, 'best_run_test.csv')}")
print(f"✓ Saved: {os.path.join(production_models_dir, 'best_run_centroids.pkl')}")
print(f"✓ Saved: {os.path.join(production_models_dir, 'best_run_svd.pkl')}")
print(f"✓ Saved: {os.path.join(production_models_dir, 'best_run_threshold.pkl')}")

print(f"\n{'=' * 60}")
print("PIPELINE COMPLETED SUCCESSFULLY!")
print(f"{'=' * 60}")