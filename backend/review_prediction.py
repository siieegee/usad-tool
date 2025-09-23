### Setup & Imports
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from scipy.spatial.distance import euclidean

# Download NLTK resources quietly
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt', quiet=True)

# === Dynamically locate backend folder ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

### Load pre-trained objects using absolute paths
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
kmeans_final = joblib.load(os.path.join(BASE_DIR, "kmeans_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "review_length_scaler.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "anomaly_distance_threshold.pkl"))


# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

### POS tagging helper function (same as training)
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

### Preprocessing function (updated to match training)
def preprocess_review(review_text):
    # Clean text
    review_text_cleaned = review_text.lower()
    review_text_cleaned = re.sub(r'[^\w\s]', '', review_text_cleaned)
    
    # Tokenize
    tokens = word_tokenize(review_text_cleaned)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # POS tagging and lemmatization (same as training)
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    
    return ' '.join(lemmatized_tokens)

### Prediction function
def predict_review(review_text):
    # Step 1: Preprocess the review
    processed_text = preprocess_review(review_text)
    
    # Step 2: Transform into TF-IDF features
    tfidf_features = vectorizer.transform([processed_text])
    
    # Step 3: Add scaled review length feature
    review_length = len(processed_text.split())
    scaled_length = scaler.transform([[review_length]])[0][0]
    length_feature = csr_matrix([[scaled_length]])
    
    # Step 4: Combine features
    final_features = hstack([tfidf_features, length_feature])
    
    # Step 5: Predict cluster
    cluster_label = kmeans_final.predict(final_features)[0]
    centroid = kmeans_final.cluster_centers_[cluster_label]
    
    # Step 6: Calculate Euclidean distance
    distance = euclidean(final_features.toarray().ravel(), centroid)
    
    # Step 7: Compare with threshold
    is_anomalous = distance > threshold
    review_type = 'Anomalous' if is_anomalous else 'Normal'
    
    return {
        "prediction": review_type,
        "cluster": int(cluster_label),
        "distance": float(distance),
        "processed_text": processed_text
    }

### Testing with sample reviews
if __name__ == "__main__":
    test_reviews = [
        "Love this! Well made, sturdy, and very comfortable. I love it! Very pretty",
        "Love it, a great upgrade from the original. I've had mine for a couple of years",
        "Panget",
        "Not impossible to put together by yourself. Only scratched one place in a not very noticeable place. Get many compliments on it and has lots of storage.",
    ]
    
    print("Review Prediction Results:")
    print("=" * 80)
    
    for i, review in enumerate(test_reviews, 1):
        result = predict_review(review)
        
        print(f"\nReview {i}:")
        print(f"Original: {review}")
        print(f"Processed: {result['processed_text']}")
        print(f"Cluster: {result['cluster']}, Distance: {result['distance']:.4f}")
        print(f"Prediction: {result['prediction']}")
        print("-" * 40)