# Performance Issue Analysis: Slow Performance After 2 Minutes Idle

## Problem
The app becomes slow after being idle for 2 minutes.

## Root Causes Identified

### 1. **TextBlob Lazy Resource Loading** ⚠️ **MOST LIKELY**
- TextBlob creates new instances on each call
- It lazily loads NLP resources (sentiment analysis models) on first use
- After idle time, these resources may be garbage collected
- First request after idle triggers resource reloading (slow)

**Location**: `backend/review_prediction.py:279`
```python
blob = TextBlob(joined_text)  # Creates new instance, may reload resources
```

### 2. **NLTK Resource Re-initialization**
- NLTK resources are loaded on module import
- But they might be cached in memory and cleared by GC after idle
- First use after idle may trigger re-initialization

**Location**: `backend/review_prediction.py:89-90, 191-192`

### 3. **No Resource Pre-warming**
- FastAPI has no startup event to pre-warm resources
- First request after idle does all initialization work
- No keep-alive mechanism

### 4. **Memory Management / Garbage Collection**
- Python's GC may clean up cached resources after idle time
- Large models/objects may be swapped out of memory
- First request after idle triggers reloading from disk

### 5. **Modal Container Sleep** (if using Modal)
- Modal containers may sleep after idle time
- Wake-up time adds latency to first request

## Solutions

### Solution 1: Pre-warm Resources on Startup (RECOMMENDED)
Add FastAPI startup event to pre-initialize all resources.

### Solution 2: Cache TextBlob Resources
Pre-initialize TextBlob to load resources into memory.

### Solution 3: Keep-Alive Endpoint
Add a lightweight endpoint that keeps resources warm.

### Solution 4: Lazy Loading with Caching
Implement lazy loading with persistent caching.

## Recommended Fix
Implement Solution 1 + Solution 2 for best results.
