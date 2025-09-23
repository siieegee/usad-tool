### Setup & Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances
import joblib

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pyswarm import pso

import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Load Pre-split Features & Data
# -------------------------
train_features = sparse.load_npz("X_train.npz")
test_features = sparse.load_npz("X_test.npz")

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

print(f"Training set shape: {train_features.shape}")
print(f"Test set shape: {test_features.shape}")

# -------------------------
# PCA Visualization (Training Set)
# -------------------------
sample_size = 5000
if train_features.shape[0] > sample_size:
    sampled_indices = np.random.choice(train_features.shape[0], sample_size, replace=False)
    sampled_features = train_features[sampled_indices].toarray()
    sampled_df = train_df.iloc[sampled_indices].reset_index(drop=True)
else:
    sampled_features = train_features.toarray()
    sampled_df = train_df.copy()

pca = PCA(n_components=2)
features_2d_train = pca.fit_transform(sampled_features)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=features_2d_train[:, 0], y=features_2d_train[:, 1], alpha=0.5)
plt.title("PCA Visualization of Sampled Training Reviews")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -------------------------
# PSO to Find Optimal k
# -------------------------
# Precompute a fixed sample subset to stabilize silhouette score
silhouette_sample_indices = (
    np.random.choice(train_features.shape[0], min(5000, train_features.shape[0]), replace=False)
)

def objective_function(k):
    k = int(k[0])
    if k < 2:
        return 999999
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
    km.fit(train_features)
    score = silhouette_score(train_features[silhouette_sample_indices],
                            km.labels_[silhouette_sample_indices])
    return -score

lb, ub = [2], [10]
best_k, best_score = pso(objective_function, lb, ub, swarmsize=10, maxiter=5)
best_k = int(best_k[0])
print(f"Optimal k: {best_k}, silhouette score: {-best_score:.4f}")

# -------------------------
# Final KMeans Clustering
# -------------------------
kmeans_final = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=2048, max_iter=100)
kmeans_final.fit(train_features)
train_df['cluster'] = kmeans_final.labels_

# -------------------------
# Distance-based Anomaly Detection (Training Set)
# -------------------------
distances_train_matrix = euclidean_distances(train_features, kmeans_final.cluster_centers_)
min_distances_train = distances_train_matrix[np.arange(train_features.shape[0]), train_df['cluster']]
train_df['distance_to_centroid'] = min_distances_train

threshold = np.percentile(min_distances_train, 95)  # 95th percentile
train_df['review_type'] = np.where(train_df['distance_to_centroid'] > threshold, "Anomalous", "Normal")

# -------------------------
# Visualize PCA with Anomalies
# -------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=features_2d_train[:, 0],
    y=features_2d_train[:, 1],
    hue=train_df['review_type'].iloc[sampled_indices],
    palette=['green', 'red'],
    alpha=0.6
)
plt.title("PCA Visualization with Anomalies Highlighted (Training Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -------------------------
# Evaluation on Test Set
# -------------------------
test_df['cluster'] = kmeans_final.predict(test_features)

distances_test_matrix = euclidean_distances(test_features, kmeans_final.cluster_centers_)
min_distances_test = distances_test_matrix[np.arange(test_features.shape[0]), test_df['cluster']]
test_df['distance_to_centroid'] = min_distances_test
test_df['review_type'] = np.where(test_df['distance_to_centroid'] > threshold, "Anomalous", "Normal")

# PCA Visualization (Test Set)
if test_features.shape[0] > sample_size:
    sampled_indices_test = np.random.choice(test_features.shape[0], sample_size, replace=False)
    sampled_features_test = test_features[sampled_indices_test].toarray()
    sampled_test_df = test_df.iloc[sampled_indices_test].reset_index(drop=True)
else:
    sampled_features_test = test_features.toarray()
    sampled_test_df = test_df.copy()

features_2d_test = pca.transform(sampled_features_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=features_2d_test[:, 0],
    y=features_2d_test[:, 1],
    hue=sampled_test_df['review_type'],
    palette=['green', 'red'],
    alpha=0.6
)
plt.title("PCA Visualization with Anomalies Highlighted (Test Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -------------------------
# Extract Top Keywords per Cluster (Using Training Set)
# -------------------------
train_df['Review_Text_joined'] = train_df['processed_review'].apply(
    lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else str(tokens)
)

vectorizer = joblib.load("tfidf_vectorizer.pkl")
feature_names = vectorizer.get_feature_names_out()

# Slice centroids to exclude review_length feature
centroids = kmeans_final.cluster_centers_[:, :len(feature_names)]

def get_top_keywords_per_cluster(centroids, feature_names, n=10):
    top_keywords = {}
    for i, centroid in enumerate(centroids):
        top_idx = centroid.argsort()[::-1][:n]
        top_keywords[i] = [feature_names[idx] for idx in top_idx]
    return top_keywords

top_keywords = get_top_keywords_per_cluster(centroids, feature_names, n=10)

print("\nTop keywords per cluster:")
for cluster, keywords in top_keywords.items():
    print(f"Cluster {cluster}: {keywords}")

cluster_summary = pd.DataFrame({
    'cluster': range(best_k),
    'size': [sum(train_df['cluster'] == i) for i in range(best_k)],
    'top_keywords': [', '.join(top_keywords[i]) for i in range(best_k)]
})
print("\nCluster Summary:")
print(cluster_summary)

# -------------------------
# Save Results & Models
# -------------------------
train_df.to_csv("clustered_reviews_train.csv", index=False)
test_df.to_csv("clustered_reviews_test.csv", index=False)
print("Clustered datasets saved.")

joblib.dump(kmeans_final, "kmeans_model.pkl")
joblib.dump(threshold, "anomaly_distance_threshold.pkl")
print("Models saved successfully!")
