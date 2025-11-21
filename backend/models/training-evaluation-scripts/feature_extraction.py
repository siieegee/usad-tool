import os
import pandas as pd
import numpy as np
import joblib
import re
import ast
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.model_selection import train_test_split

# Directory structure constants
PRODUCTION_MODELS_DIR = "production-models"
FEATURE_MATRICES_DIR = "feature-matrices"
TRAINING_DATA_DIR = "training-data"


def get_base_models_dir():
    """Get the base models directory (parent of training-evaluation-scripts)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def get_production_models_dir():
    """Get the production models directory"""
    base_dir = get_base_models_dir()
    models_dir = os.path.join(base_dir, PRODUCTION_MODELS_DIR)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return models_dir


def get_feature_matrices_dir():
    """Get the feature matrices directory"""
    base_dir = get_base_models_dir()
    matrices_dir = os.path.join(base_dir, FEATURE_MATRICES_DIR)
    if not os.path.exists(matrices_dir):
        os.makedirs(matrices_dir)
    return matrices_dir


def get_training_data_dir():
    """Get the training data directory"""
    base_dir = get_base_models_dir()
    data_dir = os.path.join(base_dir, TRAINING_DATA_DIR)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


# Load preprocessed reviews
print("Loading preprocessed data...")
training_data_dir = get_training_data_dir()
df = pd.read_csv(os.path.join(training_data_dir, 'processed_reviews.csv'))
print(f'Initial dataset shape: {df.shape}')
print(f'Available columns: {df.columns.tolist()}')

# Convert processed_review from string back to list
print("\nConverting processed_review from string to list...")
df['processed_review'] = df['processed_review'].apply(ast.literal_eval)
print(f"Sample processed_review type: {type(df['processed_review'].iloc[0])}")
print(f"Sample processed_review: {df['processed_review'].iloc[0][:10]}")  # First 10 tokens

# Verify required columns exist
assert 'original_review' in df.columns, "Missing 'original_review' column!"
assert 'processed_review' in df.columns, "Missing 'processed_review' column!"

# Use correct column name
review_col = 'original_review'
print(f"\nUsing '{review_col}' for original text features")

# ========== FEATURE ENGINEERING ==========
print("\nExtracting features...")

# Token-based features (from processed_review)
df['review_length'] = df['processed_review'].apply(len)  # Token count
df['lexical_diversity'] = df['processed_review'].apply(
    lambda x: len(set(x))/len(x) if len(x) > 0 else 0
)
df['avg_word_length'] = df['processed_review'].apply(
    lambda x: np.mean([len(w) for w in x]) if len(x) > 0 else 0
)

# Sentiment Scores (from processed_review)
df['sentiment_polarity'] = df['processed_review'].apply(
    lambda x: TextBlob(" ".join(x)).sentiment.polarity
)
df['sentiment_subjectivity'] = df['processed_review'].apply(
    lambda x: TextBlob(" ".join(x)).sentiment.subjectivity
)

# Repetition Ratio (from processed_review)
df['repetition_ratio'] = df['processed_review'].apply(
    lambda x: (len(x)-len(set(x)))/len(x) if len(x) > 0 else 0
)

# Punctuation features (from original_review)
df['exclamation_count'] = df[review_col].apply(lambda x: str(x).count('!'))
df['question_count'] = df[review_col].apply(lambda x: str(x).count('?'))

# Capital Letter Ratio (from original_review)
df['capital_ratio'] = df[review_col].apply(
    lambda x: sum(1 for c in str(x) if c.isupper())/len(str(x)) if len(str(x)) > 0 else 0
)

# Punctuation Density (from original_review)
df['punctuation_density'] = df[review_col].apply(
    lambda x: len(re.findall(r'\W', str(x)))/len(str(x)) if len(str(x)) > 0 else 0
)

# Word Entropy (from processed_review)
def calculate_entropy(tokens):
    if len(tokens) == 0:
        return 0
    freq = Counter(tokens)
    probs = np.array(list(freq.values()))/len(tokens)
    return -np.sum(probs * np.log2(probs + 1e-10))

df['word_entropy'] = df['processed_review'].apply(calculate_entropy)

# Select all feature columns
feature_cols = [
    'review_length', 'lexical_diversity', 'avg_word_length',
    'sentiment_polarity', 'sentiment_subjectivity', 'word_entropy',
    'repetition_ratio', 'exclamation_count', 'question_count',
    'capital_ratio', 'punctuation_density'
]
print(f"\nFeatures created: {feature_cols}")
print(f"Feature statistics:\n{df[feature_cols].describe()}")

# ========== TRAIN/TEST SPLIT ==========
print("\nSplitting data into train/test sets...")

# Split train/test by label
train_list, test_list = [], []
if 'label' in df.columns:
    print("Stratified split by label...")
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        train_sub, test_sub = train_test_split(subset, test_size=0.3, random_state=42)
        train_list.append(train_sub)
        test_list.append(test_sub)
    train_df = pd.concat(train_list).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_list).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nTrain label distribution:")
    print(train_df['label'].value_counts())
    print("\nTest label distribution:")
    print(test_df['label'].value_counts())
else:
    print("No label column found, random split...")
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")

# ========== TF-IDF VECTORIZATION ==========
print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=20000,      # Increased TF-IDF features
    min_df=2,                # Only keep terms in at least 2 docs
    ngram_range=(1, 3)       # Use unigrams, bigrams, and trigrams
)

# Fit on training data only
tfidf.fit(train_df['processed_review'].apply(lambda x: " ".join(x)))
X_train_tfidf = tfidf.transform(train_df['processed_review'].apply(lambda x: " ".join(x)))
X_test_tfidf = tfidf.transform(test_df['processed_review'].apply(lambda x: " ".join(x)))

print(f"TF-IDF vocabulary size: {len(tfidf.get_feature_names_out())}")
print(f"Train TF-IDF shape: {X_train_tfidf.shape}")
print(f"Test TF-IDF shape: {X_test_tfidf.shape}")

# ========== SCALE AND COMBINE FEATURES ==========
print("\nScaling and combining features...")

# Scale enhanced features
scaler = StandardScaler()
train_enhanced = scaler.fit_transform(train_df[feature_cols])
test_enhanced = scaler.transform(test_df[feature_cols])

# Convert to sparse format
train_enhanced_sparse = csr_matrix(train_enhanced)
test_enhanced_sparse = csr_matrix(test_enhanced)

# Combine TF-IDF and enhanced features
X_train = hstack([X_train_tfidf, train_enhanced_sparse])
X_test = hstack([X_test_tfidf, test_enhanced_sparse])

print(f"\nFinal X_train shape: {X_train.shape}")
print(f"Final X_test shape: {X_test.shape}")
print(f"Total features: {X_train.shape[1]} (TF-IDF: {X_train_tfidf.shape[1]}, Enhanced: {len(feature_cols)})")

# ========== SAVE OUTPUTS ==========
print("\nSaving feature extraction outputs...")

# Save to appropriate directories
production_models_dir = get_production_models_dir()
feature_dir = get_feature_matrices_dir()

joblib.dump(tfidf, os.path.join(production_models_dir, "tfidf_vectorizer.pkl"))
joblib.dump(scaler, os.path.join(production_models_dir, "enhanced_feature_scaler.pkl"))
save_npz(os.path.join(feature_dir, "X_train.npz"), X_train)
save_npz(os.path.join(feature_dir, "X_test.npz"), X_test)
train_df.to_csv(os.path.join(training_data_dir, "train_data.csv"), index=False)
test_df.to_csv(os.path.join(training_data_dir, "test_data.csv"), index=False)

print("\n✓ Feature extraction completed successfully!")
print(f"✓ Files saved: {os.path.join(feature_dir, 'X_train.npz')}, "
      f"{os.path.join(feature_dir, 'X_test.npz')}, "
      f"{os.path.join(training_data_dir, 'train_data.csv')}, "
      f"{os.path.join(training_data_dir, 'test_data.csv')}")
print(f"✓ Models saved: {os.path.join(production_models_dir, 'tfidf_vectorizer.pkl')}, "
      f"{os.path.join(production_models_dir, 'enhanced_feature_scaler.pkl')}")