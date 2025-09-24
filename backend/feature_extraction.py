### Setup & Imports
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
from scipy import sparse

# Set visualization style
sns.set(style="whitegrid")

### Load Dataset
df = pd.read_csv('processed_reviews.csv')
print(f"Initial dataset shape: {df.shape}")
print(df.head())

# Convert string representation of tokens back to list
df['processed_review'] = df['processed_review'].apply(lambda x: ast.literal_eval(x))
print("\nSample processed review tokens:", df['processed_review'].iloc[0])
print("Type:", type(df['processed_review'].iloc[0]))

# 1. Review Length Feature
df['review_length'] = df['processed_review'].apply(len)
print("\nReview length stats:")
print(df['review_length'].describe())

# 2. Combine tokens into single string for vectorizers
df['processed_text'] = df['processed_review'].apply(lambda tokens: ' '.join(tokens))

### ------------------------
### Visualization
### ------------------------

# Distribution of review lengths
plt.figure(figsize=(8,5))
plt.hist(df['review_length'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(df['review_length'].mean(), color='red', linestyle='dashed', linewidth=2,
            label=f'Mean: {df["review_length"].mean():.2f}')
plt.title("Distribution of Review Lengths")
plt.xlabel("Review Length (tokens)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Class distribution
if 'label' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='label', palette='coolwarm')
    plt.title('Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()


### Train/Test Split
train_list = []
test_list = []

for label in df['label'].unique():
    class_subset = df[df['label'] == label]
    train_sub, test_sub = train_test_split(
        class_subset,
        test_size=0.3,
        random_state=42
    )
    train_list.append(train_sub)
    test_list.append(test_sub)

train_df = pd.concat(train_list).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_list).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTraining set: {train_df.shape}, Test set: {test_df.shape}")
print("Training class distribution:\n", train_df['label'].value_counts())
print("Test class distribution:\n", test_df['label'].value_counts())

### TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=10000,   # top 10k terms
    min_df=5,             # ignore words appearing in <5 reviews
    ngram_range=(1,2)     # unigrams + bigrams
)

# Fit on training data
tfidf.fit(train_df['processed_text'])

# Transform train/test
X_train_tfidf = tfidf.transform(train_df['processed_text'])
X_test_tfidf = tfidf.transform(test_df['processed_text'])

print("Train TF-IDF shape:", X_train_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)

# Save TF-IDF vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("TF-IDF vectorizer saved as tfidf_vectorizer.pkl")

### Add Review Length Feature (Scaled)
scaler = StandardScaler()
train_len_scaled = scaler.fit_transform(train_df['review_length'].values.reshape(-1,1))
test_len_scaled = scaler.transform(test_df['review_length'].values.reshape(-1,1))

# Convert to sparse matrices
train_len_sparse = csr_matrix(train_len_scaled)
test_len_sparse = csr_matrix(test_len_scaled)

# Combine TF-IDF + review length
X_train = hstack([X_train_tfidf, train_len_sparse])
X_test = hstack([X_test_tfidf, test_len_sparse])

print("Final train feature matrix shape:", X_train.shape)
print("Final test feature matrix shape:", X_test.shape)

# Save scaler for future use
joblib.dump(scaler, "review_length_scaler.pkl")
print("Review length scaler saved as review_length_scaler.pkl")


### Save Features & Datasets

# Save TF-IDF vocabulary
vocab_path = "tfidf_vocabulary.csv"
pd.DataFrame(tfidf.get_feature_names_out(), columns=['term']).to_csv(vocab_path, index=False)
print(f"TF-IDF vocabulary saved to {vocab_path}")

# Save sparse matrices
save_npz("X_train.npz", X_train)
save_npz("X_test.npz", X_test)

# Save train/test CSVs
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Train/Test feature matrices and datasets saved successfully!")
