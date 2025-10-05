### Setup & Imports
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
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# === Dynamically locate backend folder ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

### Load pre-trained objects using absolute paths
print("Loading models...")
tfidf_vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
enhanced_scaler = joblib.load(os.path.join(BASE_DIR, "enhanced_feature_scaler.pkl"))
svd = joblib.load(os.path.join(BASE_DIR, "best_run_svd.pkl"))
centroids = joblib.load(os.path.join(BASE_DIR, "best_run_centroids.pkl"))
threshold_data = joblib.load(os.path.join(BASE_DIR, "best_run_threshold.pkl"))
anomaly_threshold = threshold_data["threshold"]

print(f"✓ Models loaded successfully")
print(f"✓ Anomaly threshold: {anomaly_threshold:.4f}")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

### POS tagging helper function
def get_wordnet_pos(treebank_tag):
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

### Preprocessing function
def preprocess_review(review_text):
    """
    Preprocess review text to match training pipeline.
    Returns both original text and processed tokens.
    """
    # Store original for feature extraction
    original_text = review_text
    
    # Clean text (lowercase, remove punctuation/emojis)
    cleaned_text = review_text.lower()
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text, flags=re.UNICODE)
    
    # Tokenize
    tokens = word_tokenize(cleaned_text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # POS tagging and lemmatization
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    
    return original_text, lemmatized_tokens

### Feature extraction
def extract_features(original_text, processed_tokens):
    """
    Extract all features used in training pipeline.
    Returns a dictionary of features.
    """
    features = {}
    
    # Token-based features
    features['review_length'] = len(processed_tokens)
    features['lexical_diversity'] = len(set(processed_tokens)) / len(processed_tokens) if len(processed_tokens) > 0 else 0
    features['avg_word_length'] = np.mean([len(w) for w in processed_tokens]) if len(processed_tokens) > 0 else 0
    
    # Sentiment Scores
    joined_text = " ".join(processed_tokens)
    blob = TextBlob(joined_text)
    features['sentiment_polarity'] = blob.sentiment.polarity
    features['sentiment_subjectivity'] = blob.sentiment.subjectivity
    
    # Word entropy
    if len(processed_tokens) == 0:
        features['word_entropy'] = 0
    else:
        freq = Counter(processed_tokens)
        probs = np.array(list(freq.values())) / len(processed_tokens)
        features['word_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Repetition Ratio
    features['repetition_ratio'] = (len(processed_tokens) - len(set(processed_tokens))) / len(processed_tokens) if len(processed_tokens) > 0 else 0
    
    # Punctuation features
    features['exclamation_count'] = original_text.count('!')
    features['question_count'] = original_text.count('?')
    
    # Capital Letter Ratio
    features['capital_ratio'] = sum(1 for c in original_text if c.isupper()) / len(original_text) if len(original_text) > 0 else 0
    
    # Punctuation Densit
    features['punctuation_density'] = len(re.findall(r'\W', original_text)) / len(original_text) if len(original_text) > 0 else 0
    
    return features

### Prediction function
def predict_review(review_text):
    """
    Predict if a review is Normal or Anomalous using the full pipeline.
    """
    # Step 1: Preprocess the review
    original_text, processed_tokens = preprocess_review(review_text)
    
    # Step 2: Extract enhanced features
    enhanced_features = extract_features(original_text, processed_tokens)
    
    # Order features to match training
    feature_order = [
        'review_length', 'lexical_diversity', 'avg_word_length',
        'sentiment_polarity', 'sentiment_subjectivity', 'word_entropy',
        'repetition_ratio', 'exclamation_count', 'question_count',
        'capital_ratio', 'punctuation_density'
    ]
    enhanced_array = np.array([[enhanced_features[f] for f in feature_order]])
    
    # Step 3: Transform into TF-IDF features
    joined_tokens = " ".join(processed_tokens)
    tfidf_features = tfidf_vectorizer.transform([joined_tokens])
    
    # Step 4: Scale enhanced features
    scaled_enhanced = enhanced_scaler.transform(enhanced_array)
    scaled_enhanced_sparse = csr_matrix(scaled_enhanced)
    
    # Step 5: Combine TF-IDF and enhanced features
    combined_features = hstack([tfidf_features, scaled_enhanced_sparse])
    
    # Step 6: Apply SVD dimensionality reduction
    reduced_features = svd.transform(combined_features)
    
    # Step 7: Normalize for cosine similarity
    normalized_features = normalize(reduced_features, norm='l2', axis=1)
    
    # Step 8: Find nearest cluster centroid
    cosine_similarities = normalized_features.dot(centroids.T)
    nearest_cluster_idx = np.argmax(cosine_similarities)
    max_similarity = cosine_similarities[0, nearest_cluster_idx]
    
    # Step 9: Calculate cosine distance
    cosine_distance = 1.0 - max_similarity
    
    # Step 10: Compare with threshold
    is_anomalous = cosine_distance > anomaly_threshold
    review_type = 'Anomalous' if is_anomalous else 'Normal'
    
    # Calculate confidence (distance from threshold)
    distance_from_threshold = abs(cosine_distance - anomaly_threshold)
    confidence = min(distance_from_threshold / anomaly_threshold * 100, 100)
    
    return {
        "prediction": review_type,
        "cluster": int(nearest_cluster_idx),
        "distance": float(cosine_distance),
        "threshold": float(anomaly_threshold),
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