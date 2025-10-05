import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

### Load dataset
df = pd.read_csv('before_preprocess_reviews.csv', encoding='latin2', sep=';')
print(f"Loaded dataset shape: {df.shape}")
print(df.head())

# IMPORTANT: Keep original review for feature extraction later
df['original_review'] = df['review'].copy()

### Lowercase, Punctuation, Emoji Removal
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and emojis
    # Removes all non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    
    return text

### Tokenization
def tokenize_text(text):
    return word_tokenize(text)

### Stopword Removal
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

### Lemmatization with POS tagging
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Map POS tag to format recognized by WordNetLemmatizer"""
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

def lemmatize_tokens(tokens):
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return lemmatized

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
print("\nApplying preprocessing...")
df['processed_review'] = df['review'].apply(preprocess_text)

# Preview results
print("\nPreview of results:")
print(df[['original_review', 'review', 'processed_review']].head())

### Verify we have both columns
print("\nColumns in dataframe:")
print(df.columns.tolist())

### Save Results with BOTH original and processed reviews
df.to_csv('processed_reviews.csv', index=False)
print("\nProcessed dataset saved to processed_reviews.csv")
print(f"Dataset contains {len(df)} reviews")
print(f"Columns saved: {df.columns.tolist()}")