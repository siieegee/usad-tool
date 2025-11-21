"""
Review prediction module for USAD Tool
"""
import os
import re
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from textblob import TextBlob
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize

# Download NLTK resources quietly
NLTK_RESOURCES = [
    'stopwords',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger_eng',
    'punkt',
    'punkt_tab'
]

for resource in NLTK_RESOURCES:
    nltk.download(resource, quiet=True)

# Constants
MODELS_DIR_NAME = "models"
PRODUCTION_MODELS_DIR = "production-models"
TFIDF_MODEL_FILE = "tfidf_vectorizer.pkl"
SCALER_MODEL_FILE = "enhanced_feature_scaler.pkl"
SVD_MODEL_FILE = "best_run_svd.pkl"
CENTROIDS_MODEL_FILE = "best_run_centroids.pkl"
THRESHOLD_MODEL_FILE = "best_run_threshold.pkl"
THRESHOLD_KEY = "threshold"
EPSILON = 1e-10  # Small value to prevent log(0)
MAX_CONFIDENCE = 100.0
NORMALIZATION_NORM = 'l2'


def get_models_directory():
    """
    Get the directory where production models are stored.

    Returns:
        str: Path to the production models directory
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, MODELS_DIR_NAME)
    return os.path.join(models_dir, PRODUCTION_MODELS_DIR)


def load_models():
    """
    Load all pre-trained models from disk.

    Returns:
        tuple: Tuple containing (tfidf_vectorizer, enhanced_scaler, svd,
                                 centroids, anomaly_threshold)

    Raises:
        FileNotFoundError: If any model file is missing
        ValueError: If threshold data is invalid
    """
    models_dir = get_models_directory()
    model_files = {
        'tfidf': TFIDF_MODEL_FILE,
        'scaler': SCALER_MODEL_FILE,
        'svd': SVD_MODEL_FILE,
        'centroids': CENTROIDS_MODEL_FILE,
        'threshold': THRESHOLD_MODEL_FILE
    }

    print("Loading models...")
    models = {}
    for model_name, model_file in model_files.items():
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )
        try:
            models[model_name] = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {model_name} model: {str(e)}"
            )

    # Extract threshold
    threshold_data = models['threshold']
    if not isinstance(threshold_data, dict):
        raise ValueError("Threshold data must be a dictionary")
    if THRESHOLD_KEY not in threshold_data:
        raise ValueError(
            f"Threshold data missing '{THRESHOLD_KEY}' key"
        )
    anomaly_threshold = threshold_data[THRESHOLD_KEY]

    print("✓ Models loaded successfully")
    print(f"✓ Anomaly threshold: {anomaly_threshold:.4f}")

    return (
        models['tfidf'],
        models['scaler'],
        models['svd'],
        models['centroids'],
        anomaly_threshold
    )


# Load models on module import
try:
    TFIDF_VECTORIZER, ENHANCED_SCALER, SVD, CENTROIDS, ANOMALY_THRESHOLD = \
        load_models()
except Exception as e:
    print(f"Error loading models: {e}")
    # Set to None so errors occur at prediction time, not import time
    TFIDF_VECTORIZER = None
    ENHANCED_SCALER = None
    SVD = None
    CENTROIDS = None
    ANOMALY_THRESHOLD = None


# Initialize NLP resources
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    """
    Map POS tag to format recognized by WordNetLemmatizer.

    Args:
        treebank_tag: Treebank POS tag

    Returns:
        wordnet constant: WordNet POS tag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_review(review_text):
    """
    Preprocess review text to match training pipeline.

    Args:
        review_text: Raw review text string

    Returns:
        tuple: (original_text, processed_tokens)
    """
    original_text = review_text

    # Clean text: lowercase, remove punctuation/emojis
    cleaned_text = review_text.lower()
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text, flags=re.UNICODE)

    # Tokenize
    tokens = word_tokenize(cleaned_text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in STOP_WORDS]

    # POS tagging and lemmatization
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = [
        LEMMATIZER.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]

    return original_text, lemmatized_tokens


def extract_features(original_text, processed_tokens):
    """
    Extract all features used in training pipeline.

    Args:
        original_text: Original review text
        processed_tokens: List of processed tokens

    Returns:
        dict: Dictionary of extracted features
    """
    features = {}

    # Token-based features
    num_tokens = len(processed_tokens)
    features['review_length'] = num_tokens

    # Lexical diversity: unique tokens / total tokens
    if num_tokens > 0:
        unique_tokens = len(set(processed_tokens))
        features['lexical_diversity'] = unique_tokens / num_tokens
        features['avg_word_length'] = np.mean(
            [len(w) for w in processed_tokens]
        )
    else:
        features['lexical_diversity'] = 0.0
        features['avg_word_length'] = 0.0

    # Sentiment scores
    joined_text = " ".join(processed_tokens)
    blob = TextBlob(joined_text)
    features['sentiment_polarity'] = blob.sentiment.polarity
    features['sentiment_subjectivity'] = blob.sentiment.subjectivity

    # Word entropy
    if num_tokens == 0:
        features['word_entropy'] = 0.0
    else:
        freq = Counter(processed_tokens)
        probs = np.array(list(freq.values())) / num_tokens
        features['word_entropy'] = -np.sum(
            probs * np.log2(probs + EPSILON)
        )

    # Repetition ratio
    if num_tokens > 0:
        unique_count = len(set(processed_tokens))
        features['repetition_ratio'] = \
            (num_tokens - unique_count) / num_tokens
    else:
        features['repetition_ratio'] = 0.0

    # Punctuation features
    features['exclamation_count'] = original_text.count('!')
    features['question_count'] = original_text.count('?')

    # Capital letter ratio
    text_length = len(original_text)
    if text_length > 0:
        capital_count = sum(1 for c in original_text if c.isupper())
        features['capital_ratio'] = capital_count / text_length
    else:
        features['capital_ratio'] = 0.0

    # Punctuation density
    if text_length > 0:
        punct_count = len(re.findall(r'\W', original_text))
        features['punctuation_density'] = punct_count / text_length
    else:
        features['punctuation_density'] = 0.0

    return features


def predict_review(review_text):
    """
    Predict if a review is Normal or Anomalous using the full pipeline.

    Args:
        review_text: Raw review text string

    Returns:
        dict: Prediction results with details

    Raises:
        RuntimeError: If models are not loaded
        ValueError: If review_text is empty or invalid
    """
    if not review_text or not isinstance(review_text, str):
        raise ValueError("Review text must be a non-empty string")

    if TFIDF_VECTORIZER is None:
        raise RuntimeError("Models not loaded. Cannot make predictions.")

    # Step 1: Preprocess the review
    original_text, processed_tokens = preprocess_review(review_text)

    # Step 2: Extract enhanced features
    enhanced_features = extract_features(original_text, processed_tokens)

    # Step 3: Order features to match training
    feature_order = [
        'review_length', 'lexical_diversity', 'avg_word_length',
        'sentiment_polarity', 'sentiment_subjectivity', 'word_entropy',
        'repetition_ratio', 'exclamation_count', 'question_count',
        'capital_ratio', 'punctuation_density'
    ]
    enhanced_array = np.array([
        [enhanced_features[f] for f in feature_order]
    ])

    # Step 4: Transform into TF-IDF features
    joined_tokens = " ".join(processed_tokens)
    tfidf_features = TFIDF_VECTORIZER.transform([joined_tokens])

    # Step 5: Scale enhanced features
    scaled_enhanced = ENHANCED_SCALER.transform(enhanced_array)
    scaled_enhanced_sparse = csr_matrix(scaled_enhanced)

    # Step 6: Combine TF-IDF and enhanced features
    combined_features = hstack([tfidf_features, scaled_enhanced_sparse])

    # Step 7: Apply SVD dimensionality reduction
    reduced_features = SVD.transform(combined_features)

    # Step 8: Normalize for cosine similarity
    normalized_features = normalize(
        reduced_features,
        norm=NORMALIZATION_NORM,
        axis=1
    )

    # Step 9: Find nearest cluster centroid
    cosine_similarities = normalized_features.dot(CENTROIDS.T)
    nearest_cluster_idx = np.argmax(cosine_similarities)
    max_similarity = cosine_similarities[0, nearest_cluster_idx]

    # Step 10: Calculate cosine distance
    cosine_distance = 1.0 - max_similarity

    # Step 11: Compare with threshold
    is_anomalous = cosine_distance > ANOMALY_THRESHOLD
    review_type = 'Anomalous' if is_anomalous else 'Normal'

    # Step 12: Calculate confidence (distance from threshold)
    distance_from_threshold = abs(cosine_distance - ANOMALY_THRESHOLD)
    confidence = min(
        distance_from_threshold / ANOMALY_THRESHOLD * MAX_CONFIDENCE,
        MAX_CONFIDENCE
    )

    return {
        "prediction": review_type,
        "cluster": int(nearest_cluster_idx),
        "distance": float(cosine_distance),
        "threshold": float(ANOMALY_THRESHOLD),
        "confidence": float(confidence),
        "processed_text": joined_tokens,
        "features": enhanced_features
    }

### Testing with sample reviews
if __name__ == "__main__":
    test_reviews = [
        "Love this! Well made, sturdy, and very comfortable. I love it! Very pretty",
        "Love it, a great upgrade from the original. I've had mine for a couple of years",
        "Panget",
        "Not impossible to put together by yourself. Only scratched one place in a not very noticeable place. Get many compliments on it and has lots of storage.",
        "AMAZING PRODUCT!!! BUY NOW!!! BEST DEAL EVER!!! ⭐⭐⭐⭐⭐",
        "good",
    ]
    
    print("\n" + "=" * 80)
    print("REVIEW PREDICTION RESULTS")
    print("=" * 80)
    
    for i, review in enumerate(test_reviews, 1):
        result = predict_review(review)
        
        print(f"\n{'─' * 80}")
        print(f"Review {i}:")
        print(f"{'─' * 80}")
        print(f"Original: {review}")
        print(f"Processed: {result['processed_text']}")
        print(f"\nPrediction: {result['prediction']} (Confidence: {result['confidence']:.1f}%)")
        print(f"Cluster: {result['cluster']}")
        print(f"Distance: {result['distance']:.4f} | Threshold: {result['threshold']:.4f}")
        
        # Show key features
        print(f"\nKey Features:")
        print(f"  Review Length: {result['features']['review_length']}")
        print(f"  Sentiment: {result['features']['sentiment_polarity']:.3f}")
        print(f"  Lexical Diversity: {result['features']['lexical_diversity']:.3f}")
        print(f"  Repetition Ratio: {result['features']['repetition_ratio']:.3f}")
    
    print("\n" + "=" * 80)