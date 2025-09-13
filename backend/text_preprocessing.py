### Setup & Imports
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

### Load dataset
df = pd.read_csv('before_preprocess_reviews.csv')
df.head()

### Lowercase, Punctuation, Emoji Removal
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and emojis using regex
    text = re.sub(r'[^\w\s]', '', text)  # Keep only words and spaces
    
    return text

### Tokenization
def tokenize_text(text):
    return word_tokenize(text)

### Stopword Removal
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

### Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

### Preprocessing Function
def preprocess_text(text):
    # 1. Lowercase & clean
    text = clean_text(text)
    
    # 2. Tokenize
    tokens = tokenize_text(text)
    
    # 3. Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # 4. Lemmatize
    tokens = lemmatize_tokens(tokens)
    
    return tokens

### Apply to Dataset
df['processed_review'] = df['review'].apply(preprocess_text)

# Preview results
df[['review', 'processed_review']].head()
### Save Results
df.to_csv('processed_reviews.csv', index=False)
print("Processed dataset saved to processed_reviews.csv")