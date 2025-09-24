import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

from pyswarm import pso

import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

# -----------------------------
# Load pre-split features and data
# -----------------------------
train_features = sparse.load_npz("X_train.npz")
test_features = sparse.load_npz("X_test.npz")

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

print(f"Training features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")

# -----------------------------
# Dimensionality reduction using TruncatedSVD for sparse data
# -----------------------------
# This step makes clustering more effective on high-dimensional TF-IDF data
print("\nApplying Truncated SVD (300 components)...")
svd = TruncatedSVD(n_components=300, random_state=42)
reduced_train = svd.fit_transform(train_features)
reduced_test = svd.transform(test_features)

print("Reduced train shape:", reduced_train.shape)
print("Reduced test shape:", reduced_test.shape)

# Optional visualization with PCA after SVD
pca_2d = PCA(n_components=2, random_state=42)
visual_train = pca_2d.fit_transform(reduced_train)
plt.figure(figsize=(8,6))
sns.scatterplot(x=visual_train[:,0], y=visual_train[:,1], alpha=0.5)
plt.title("PCA Visualization of Reduced Training Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -----------------------------
# Use PSO to find optimal number of clusters (k)
# -----------------------------
sample_size = min(5000, reduced_train.shape[0])
sample_indices = np.random.choice(reduced_train.shape[0], sample_size, replace=False)

print("\nOptimizing number of clusters using PSO...")

def k_objective(k):
    k = int(k[0])
    if k < 2:
        return 999999
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
    km.fit(reduced_train)
    labels = km.labels_[sample_indices]
    score = silhouette_score(reduced_train[sample_indices], labels)
    return -score  # PSO minimizes, so negate the silhouette score

lb, ub = [2], [12]
best_k, best_score = pso(k_objective, lb, ub, swarmsize=20, maxiter=15)

best_k = int(best_k[0])
print(f"Optimal k found: {best_k} with silhouette score: {-best_score:.4f}")

# -----------------------------
# PSO to optimize initial centroids
# -----------------------------
print(f"\nRunning PSO to optimize centroids for k={best_k}...")
centroid_sample_indices = np.random.choice(reduced_train.shape[0], min(3000, reduced_train.shape[0]), replace=False)
sampled_train = reduced_train[centroid_sample_indices]

n_features = sampled_train.shape[1]

def centroid_objective(flat_centroids):
    centroids = flat_centroids.reshape((best_k, n_features))
    distances = np.linalg.norm(sampled_train[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    sse = np.sum((sampled_train - centroids[labels]) ** 2)
    return sse

feature_min = sampled_train.min(axis=0)
feature_max = sampled_train.max(axis=0)
epsilon = 1e-6
feature_max_fixed = np.where(feature_max == feature_min, feature_min + epsilon, feature_max)

lb = np.tile(feature_min, best_k)
ub = np.tile(feature_max_fixed, best_k)

best_centroids_flat, best_sse = pso(centroid_objective, lb, ub, swarmsize=30, maxiter=30)
best_centroids = best_centroids_flat.reshape((best_k, n_features))

print(f"PSO centroid optimization complete. Best SSE: {best_sse:.4f}")

# -----------------------------
# Final KMeans clustering
# -----------------------------
kmeans_final = MiniBatchKMeans(
    n_clusters=best_k,
    init=best_centroids,
    n_init=1,
    random_state=42,
    batch_size=2048,
    max_iter=200
)

kmeans_final.fit(reduced_train)
train_df['cluster'] = kmeans_final.labels_

# -----------------------------
# Distance-based anomaly detection using percentile threshold
# -----------------------------
distances_train = euclidean_distances(reduced_train, kmeans_final.cluster_centers_)
min_distances_train = distances_train[np.arange(reduced_train.shape[0]), train_df['cluster']]
train_df['distance_to_centroid'] = min_distances_train

# Use 95th percentile instead of mean + std
threshold = np.percentile(train_df['distance_to_centroid'], 50)
print(f"\nAnomaly detection threshold (95th percentile): {threshold:.4f}")

train_df['review_type'] = np.where(train_df['distance_to_centroid'] > threshold, 'Anomalous', 'Normal')

plt.figure(figsize=(8,6))
sns.scatterplot(x=visual_train[:,0], y=visual_train[:,1], hue=train_df['review_type'], palette=['green','red'], alpha=0.6)
plt.title("Anomaly Detection Visualization (Training Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -----------------------------
# Apply model to test set
# -----------------------------
test_df['cluster'] = kmeans_final.predict(reduced_test)
distances_test = euclidean_distances(reduced_test, kmeans_final.cluster_centers_)
min_distances_test = distances_test[np.arange(reduced_test.shape[0]), test_df['cluster']]
test_df['distance_to_centroid'] = min_distances_test
test_df['review_type'] = np.where(test_df['distance_to_centroid'] > threshold, 'Anomalous', 'Normal')

# -----------------------------
# Evaluate clustering metrics
# -----------------------------
sample_eval_indices = np.random.choice(reduced_train.shape[0], min(5000, reduced_train.shape[0]), replace=False)
labels_sample = train_df['cluster'].iloc[sample_eval_indices]

sil_score = silhouette_score(reduced_train[sample_eval_indices], labels_sample)
db_score = davies_bouldin_score(reduced_train[sample_eval_indices], labels_sample)
ch_score = calinski_harabasz_score(reduced_train[sample_eval_indices], labels_sample)

print("\nClustering Evaluation Metrics:")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")

# -----------------------------
# Save models and results
# -----------------------------
train_df.to_csv("clustered_reviews_train.csv", index=False)
test_df.to_csv("clustered_reviews_test.csv", index=False)
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df.to_csv("clustered_reviews.csv", index=False)

joblib.dump(kmeans_final, "kmeans_model.pkl")
joblib.dump(threshold, "anomaly_distance_threshold.pkl")
joblib.dump(svd, "svd_model.pkl")

print("\nSaved clustered datasets and models successfully!")
