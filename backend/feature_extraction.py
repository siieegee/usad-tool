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
from scipy import sparse

### Load dataset
df = pd.read_csv('processed_reviews.csv')

print("Initial dataset shape:", df.shape)
print(df.head())

### Convert string representation of tokens back to list
df['processed_review'] = df['processed_review'].apply(lambda x: ast.literal_eval(x))

print("\nSample processed review tokens:", df['processed_review'].iloc[0])
print("Type:", type(df['processed_review'].iloc[0]))

## Feature Extraction

### Review Length
df['review_length'] = df['processed_review'].apply(len)

print("\nReview length stats:")
print(df['review_length'].describe())

### Prepare text for vectorizers
df['processed_text'] = df['processed_review'].apply(lambda tokens: ' '.join(tokens))

# Visualization: Distribution of review lengths
plt.figure(figsize=(8,5))
plt.hist(df['review_length'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(df['review_length'].mean(), color='red', linestyle='dashed', linewidth=2,
            label=f'Mean: {df['review_length'].mean():.2f}')
plt.title("Distribution of Review Lengths")
plt.xlabel("Review Length (number of tokens)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Visualization: Class Distribution
if 'label' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='label', hue='label', palette='coolwarm', legend=False)
    plt.title('Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

### Train/Test Split
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print(f"\nTraining set: {train_df.shape}, Test set: {test_df.shape}")

### TF-IDF Vectorizer (fit on training data only)
tfidf = TfidfVectorizer(
    max_features=10000,   # top 10,000 terms
    min_df=5,             # ignore words appearing in <5 reviews
    ngram_range=(1, 2)    # unigrams + bigrams
)

tfidf.fit(train_df['processed_text'])

# Transform train/test data
X_train_tfidf = tfidf.transform(train_df['processed_text'])
X_test_tfidf = tfidf.transform(test_df['processed_text'])

print("Train TF-IDF shape:", X_train_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)

# Save TF-IDF vectorizer for future use
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("TF-IDF vectorizer saved as tfidf_vectorizer.pkl")

### Combine Review Length Feature
train_len_sparse = csr_matrix(train_df['review_length'].values.reshape(-1, 1))
test_len_sparse = csr_matrix(test_df['review_length'].values.reshape(-1, 1))

X_train = hstack([X_train_tfidf, train_len_sparse])
X_test = hstack([X_test_tfidf, test_len_sparse])

print("Final train feature matrix shape:", X_train.shape)
print("Final test feature matrix shape:", X_test.shape)

### Save TF-IDF Vocabulary
vocab_path = "tfidf_vocabulary.csv"
pd.DataFrame(tfidf.get_feature_names_out(), columns=['term']).to_csv(vocab_path, index=False)
print(f"TF-IDF vocabulary saved to {vocab_path}")

### Save Sparse Matrices
sparse.save_npz("X_train.npz", X_train)
sparse.save_npz("X_test.npz", X_test)
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Train/Test feature matrices and datasets saved successfully!")
