### Setup & Imports
import os
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

# Join tokens into a single string
df['processed_text'] = df['processed_review'].apply(lambda tokens: ' '.join(tokens))

# Visualization: Distribution of review lengths
plt.figure(figsize=(8,5))
plt.hist(df['review_length'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(df['review_length'].mean(), color='red', linestyle='dashed', linewidth=2,
            label=f'Mean: {df["review_length"].mean():.2f}')
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

### N-grams (Unigrams + Bigrams) with Feature Limiting
vectorizer = CountVectorizer(
    ngram_range=(1, 2),    # unigrams and bigrams
    max_features=10000,    # keep only the top 10,000 terms
    min_df=5               # ignore words appearing in <5 reviews
)

# Keep this sparse to avoid memory issues
ngram_features = vectorizer.fit_transform(df['processed_text'])

print("N-gram sparse matrix shape:", ngram_features.shape)

### TF-IDF with Feature Limiting
tfidf = TfidfVectorizer(
    max_features=10000,   # top 10,000 terms only
    min_df=5,             # ignore words appearing in <5 reviews
    ngram_range=(1, 2)    # unigrams and bigrams
)

# Keep this sparse as well
tfidf_features = tfidf.fit_transform(df['processed_text'])

print("TF-IDF sparse matrix shape:", tfidf_features.shape)

# Save the fitted TF-IDF vectorizer for future use
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("TF-IDF vectorizer saved as tfidf_vectorizer.pkl")

### Combine Review Length + TF-IDF

# Convert review_length to sparse matrix
review_length_sparse = csr_matrix(df['review_length'].values.reshape(-1, 1))

# Combine TF-IDF features with review_length
final_features = hstack([tfidf_features, review_length_sparse])

print("Final combined feature matrix shape:", final_features.shape)

### Save TF-IDF Vocabulary for Future Use

# Save vocabulary to keep consistent features for model deployment
vocab_path = "tfidf_vocabulary.csv"
pd.DataFrame(tfidf.get_feature_names_out(), columns=['term']).to_csv(vocab_path, index=False)
print(f"TF-IDF vocabulary saved to {vocab_path}")

### Save sparse matrix
sparse.save_npz("final_features_sparse.npz", final_features)
print("Sparse feature matrix saved to final_features_sparse.npz")