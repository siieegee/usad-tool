import os
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.preprocessing import StandardScaler

# Set seaborn style
sns.set(style="whitegrid")

# -----------------------------
# Load processed dataset
# -----------------------------
df = pd.read_csv('processed_reviews.csv')
print(f"Initial dataset shape: {df.shape}")
print(df.head())

# Convert string representation of tokens to list
df['processed_review'] = df['processed_review'].apply(lambda x: ast.literal_eval(x))

# Validate conversion
print("\nSample processed tokens:", df['processed_review'].iloc[0])
print("Type of processed_review:", type(df['processed_review'].iloc[0]))

# -----------------------------
# Additional feature: review length
# -----------------------------
df['review_length'] = df['processed_review'].apply(len)

plt.figure(figsize=(8,5))
plt.hist(df['review_length'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(df['review_length'].mean(), color='red', linestyle='dashed', linewidth=2,
            label=f'Mean: {df["review_length"].mean():.2f}')
plt.title("Distribution of Review Lengths")
plt.xlabel("Review Length (tokens)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# -----------------------------
# Train/Test split by label (balanced)
# -----------------------------
train_list = []
test_list = []

for label in df['label'].unique():
    subset = df[df['label'] == label]
    train_sub, test_sub = train_test_split(subset, test_size=0.3, random_state=42)
    train_list.append(train_sub)
    test_list.append(test_sub)

train_df = pd.concat(train_list).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_list).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nTraining class distribution:")
print(train_df['label'].value_counts())
print("\nTesting class distribution:")
print(test_df['label'].value_counts())

# -----------------------------
# TF-IDF Vectorization (more flexible)
# -----------------------------
# Lower min_df to keep rare fraud-related words
# Added validation to see total features retained
tfidf = TfidfVectorizer(
    max_features=15000,    # Increased to 15k for more coverage
    min_df=2,              # Lowered from 5 to 2 to keep rare terms
    ngram_range=(1,2)      # Unigrams + Bigrams
)

# Fit and transform
print("\nFitting TF-IDF on training data...")
tfidf.fit(train_df['processed_review'].apply(lambda x: ' '.join(x)))

X_train_tfidf = tfidf.transform(train_df['processed_review'].apply(lambda x: ' '.join(x)))
X_test_tfidf = tfidf.transform(test_df['processed_review'].apply(lambda x: ' '.join(x)))

print("TF-IDF feature count:", len(tfidf.get_feature_names_out()))
print("Train TF-IDF shape:", X_train_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)

# -----------------------------
# Scale review_length and combine with TF-IDF
# -----------------------------
scaler = StandardScaler()
train_len_scaled = scaler.fit_transform(train_df['review_length'].values.reshape(-1, 1))
test_len_scaled = scaler.transform(test_df['review_length'].values.reshape(-1, 1))

train_len_sparse = csr_matrix(train_len_scaled)
test_len_sparse = csr_matrix(test_len_scaled)

# Combine features into final sparse matrix
X_train = hstack([X_train_tfidf, train_len_sparse])
X_test = hstack([X_test_tfidf, test_len_sparse])

print("Final X_train shape:", X_train.shape)
print("Final X_test shape:", X_test.shape)

# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("Saved TF-IDF vectorizer -> tfidf_vectorizer.pkl")

joblib.dump(scaler, "review_length_scaler.pkl")
print("Saved review length scaler -> review_length_scaler.pkl")

save_npz("X_train.npz", X_train)
save_npz("X_test.npz", X_test)

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

pd.DataFrame(tfidf.get_feature_names_out(), columns=['term']).to_csv("tfidf_vocabulary.csv", index=False)
print("TF-IDF vocabulary saved -> tfidf_vocabulary.csv")

print("\nFeature extraction completed successfully!")
