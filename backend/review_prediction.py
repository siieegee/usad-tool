### Setup & Imports
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from scipy.spatial.distance import euclidean

# Download NLTK resources quietly
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# === Dynamically locate backend folder ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

### Load pre-trained objects using absolute paths
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))  # Fitted TF-IDF vectorizer
kmeans_final = joblib.load(os.path.join(BASE_DIR, "kmeans_model.pkl"))    # Fitted MiniBatchKMeans model
threshold = joblib.load(os.path.join(BASE_DIR, "anomaly_distance_threshold.pkl"))  # Threshold for anomaly detection

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

### Preprocessing function
def preprocess_review(review_text):
    """
    Cleans and preprocesses the input review text by:
    1. Lowercasing
    2. Removing non-alphabetic characters
    3. Removing stopwords
    4. Lemmatizing words
    """
    review_text_cleaned = review_text.lower()
    review_text_cleaned = re.sub(r'[^a-z\s]', '', review_text_cleaned)

    tokens = [
        lemmatizer.lemmatize(word, pos='v')  # Use verb as default POS for better normalization
        for word in review_text_cleaned.split() if word not in stop_words
    ]

    return ' '.join(tokens)

### Prediction function
def predict_review(review_text):

    # Step 1: Preprocess the review
    processed_text = preprocess_review(review_text)
    
    # Step 2: Transform into TF-IDF features
    tfidf_features = vectorizer.transform([processed_text])
    
    # Step 3: Add review length as an additional feature
    review_length = len(processed_text.split())
    length_feature = csr_matrix([[review_length]])
    
    # Step 4: Combine features
    final_features = hstack([tfidf_features, length_feature])
    
    # Step 5: Predict cluster
    cluster_label = kmeans_final.predict(final_features)[0]
    centroid = kmeans_final.cluster_centers_[cluster_label]
    
    # Step 6: Calculate Euclidean distance
    distance = euclidean(final_features.toarray().ravel(), centroid)
    
    # Step 7: Compare with threshold to determine if anomalous
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
