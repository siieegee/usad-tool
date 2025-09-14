### Setup & Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy.spatial.distance import euclidean

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from pyswarm import pso

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

### Load Saved Feature Matrix
final_features = sparse.load_npz("final_features_sparse.npz")
print("Loaded feature matrix shape:", final_features.shape)

# Load original dataset to map cluster labels back to reviews
df = pd.read_csv("processed_reviews.csv")
print("Dataset shape:", df.shape)

### 70/30 Train-Test Split
train_features, test_features, train_df, test_df = train_test_split(
    final_features, df, test_size=0.3, random_state=42, shuffle=True
)

print(f"Training set shape: {train_features.shape}")
print(f"Test set shape: {test_features.shape}")

### PCA Visualization (Training Set)
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

plt.figure(figsize=(8,6))
sns.scatterplot(x=features_2d_train[:,0], y=features_2d_train[:,1], alpha=0.5)
plt.title("PCA Visualization of Sampled Training Reviews")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

### PSO to Find Optimal Number of Clusters (Training Set)
def objective_function(k):
    k = int(k[0])
    if k < 2: return 999999
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
    km.fit(train_features)
    sample_size = min(5000, train_features.shape[0])
    sample_indices = np.random.choice(train_features.shape[0], sample_size, replace=False)
    score = silhouette_score(train_features[sample_indices], km.labels_[sample_indices])
    return -score

lb, ub = [2], [10]
best_k, best_score = pso(objective_function, lb, ub, swarmsize=10, maxiter=5)
best_k = int(best_k[0])
print(f"Optimal k: {best_k}, silhouette score: {-best_score:.4f}")

### Final KMeans Clustering (Training Set)
kmeans_final = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=2048, max_iter=100)
kmeans_final.fit(train_features)
train_df['cluster'] = kmeans_final.labels_

### Distance-based Anomaly Detection (Training Set)
distances_train = []
for i in range(train_features.shape[0]):
    cluster = train_df.iloc[i]['cluster']
    centroid = kmeans_final.cluster_centers_[cluster]
    dist = euclidean(train_features[i].toarray().ravel(), centroid)
    distances_train.append(dist)

train_df['distance_to_centroid'] = distances_train
threshold = np.percentile(distances_train, 95)  # 95th percentile threshold
train_df['review_type'] = train_df['distance_to_centroid'].apply(lambda x: 'Anomalous' if x > threshold else 'Normal')

### Visualize PCA with Anomalies (Training Set)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=features_2d_train[:,0],
    y=features_2d_train[:,1],
    hue=train_df['review_type'].iloc[sampled_indices],
    palette=['green','red'],
    alpha=0.6
)
plt.title("PCA Visualization with Anomalies Highlighted (Training Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

## --- Evaluation on Test Set ---

### Assign clusters to test set
test_clusters = kmeans_final.predict(test_features)
test_df['cluster'] = test_clusters

### Compute distances from test points to cluster centroids
distances_test = []
for i in range(test_features.shape[0]):
    cluster = test_df.iloc[i]['cluster']
    centroid = kmeans_final.cluster_centers_[cluster]
    dist = euclidean(test_features[i].toarray().ravel(), centroid)
    distances_test.append(dist)

test_df['distance_to_centroid'] = distances_test
test_df['review_type'] = test_df['distance_to_centroid'].apply(lambda x: 'Anomalous' if x > threshold else 'Normal')

### PCA Visualization (Test Set)
if test_features.shape[0] > sample_size:
    sampled_indices_test = np.random.choice(test_features.shape[0], sample_size, replace=False)
    sampled_features_test = test_features[sampled_indices_test].toarray()
    sampled_test_df = test_df.iloc[sampled_indices_test].reset_index(drop=True)
else:
    sampled_features_test = test_features.toarray()
    sampled_test_df = test_df.copy()

features_2d_test = pca.transform(sampled_features_test)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=features_2d_test[:,0],
    y=features_2d_test[:,1],
    hue=sampled_test_df['review_type'],
    palette=['green','red'],
    alpha=0.6
)
plt.title("PCA Visualization with Anomalies Highlighted (Test Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

### Extract Top Keywords per Cluster (Using Training Set)
train_df['Review_Text_joined'] = train_df['processed_review'].apply(
    lambda tokens: ' '.join(eval(tokens)) if isinstance(tokens, str) else ' '.join(tokens)
)

vectorizer = TfidfVectorizer()
vectorizer.fit(train_df['Review_Text_joined'])
feature_names = vectorizer.get_feature_names_out()

centroids = kmeans_final.cluster_centers_

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

### Save Results
train_df.to_csv("clustered_reviews_train.csv", index=False)
test_df.to_csv("clustered_reviews_test.csv", index=False)
print("Clustered datasets saved to clustered_reviews_train.csv and clustered_reviews_test.csv")

### Save Models for Prediction
joblib.dump(kmeans_final, "kmeans_model.pkl")
joblib.dump(threshold, "anomaly_distance_threshold.pkl")
print("Models saved successfully for prediction!")
