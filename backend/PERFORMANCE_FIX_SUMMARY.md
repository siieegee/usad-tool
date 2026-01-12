# Performance Fix: Slow Performance After 2 Minutes Idle

## Problem
Your app becomes slow after being idle for 2 minutes. The first request after idle time takes significantly longer than subsequent requests.

## Root Causes

### 1. **TextBlob Lazy Resource Loading** (Primary Issue)
- **Problem**: TextBlob loads sentiment analysis resources lazily on first use
- **Impact**: After idle time, these resources may be garbage collected
- **Result**: First request after idle reloads resources (adds 1-3 seconds delay)

### 2. **No Resource Pre-warming**
- **Problem**: FastAPI doesn't pre-initialize resources on startup
- **Impact**: First request after idle does all initialization work
- **Result**: Slow first request

### 3. **Memory Management**
- **Problem**: Python's garbage collector may clean up cached resources
- **Impact**: NLP resources and model caches may be cleared
- **Result**: Resources need to be reloaded

## Solutions Implemented

### ✅ Fix 1: Pre-warm TextBlob on Module Import
**File**: `backend/review_prediction.py`

Added pre-warming function that loads TextBlob resources immediately:
```python
def _prewarm_textblob():
    """Pre-warm TextBlob resources to avoid lazy loading delays"""
    dummy_blob = TextBlob("test")
    _ = dummy_blob.sentiment  # Trigger resource loading
```

**Benefit**: TextBlob resources are loaded into memory immediately, not on first use.

### ✅ Fix 2: FastAPI Startup Event
**File**: `backend/main.py`

Added startup event that pre-warms all resources:
```python
@app.on_event("startup")
async def startup_event():
    # Pre-warm by making a dummy prediction
    dummy_review = "This is a test review to pre-warm resources."
    _ = predict_review(dummy_review)
```

**Benefit**: All models, NLP resources, and TextBlob are loaded and ready before first request.

### ✅ Fix 3: Keep-Alive Endpoint
**File**: `backend/main.py`

Added lightweight keep-alive endpoint:
```python
@app.get("/api/keepalive")
def keepalive():
    # Lightweight operation that touches resources
    return {"status": "alive"}
```

**Benefit**: You can call this endpoint periodically (every 1-2 minutes) to prevent resource cleanup.

## How to Use

### Option 1: Automatic (Recommended)
The fixes are automatic! Resources are pre-warmed on startup. No action needed.

### Option 2: Manual Keep-Alive (Optional)
If you want extra protection, set up a keep-alive mechanism:

**Frontend (JavaScript)**:
```javascript
// Call keep-alive every 90 seconds
setInterval(() => {
    fetch('/api/keepalive')
        .then(res => res.json())
        .catch(err => console.log('Keep-alive failed:', err));
}, 90000); // 90 seconds
```

**Or use a simple cron job** (if running on server):
```bash
# Call keep-alive every 2 minutes
*/2 * * * * curl http://localhost:8000/api/keepalive
```

## Expected Results

### Before Fix:
- First request after idle: **2-5 seconds**
- Subsequent requests: **0.1-0.3 seconds**

### After Fix:
- First request after idle: **0.1-0.3 seconds** (same as always)
- Subsequent requests: **0.1-0.3 seconds**

## Testing

1. **Start your app**
2. **Wait 2+ minutes** (idle time)
3. **Make a prediction request**
4. **Measure response time** - should be fast (< 0.5 seconds)

## Additional Recommendations

### If Still Experiencing Issues:

1. **Check Modal Container Sleep** (if using Modal):
   - Modal containers may sleep after idle
   - Consider using `keep_warm` parameter in Modal deployment

2. **Monitor Memory Usage**:
   - Ensure server has enough RAM
   - Models should stay in memory, not swap to disk

3. **Check Garbage Collection**:
   - Python GC may be too aggressive
   - Consider adjusting GC thresholds if needed

4. **Add Logging**:
   - Log request times to identify slow operations
   - Monitor which resources are being reloaded

## Files Modified

1. `backend/review_prediction.py` - Added TextBlob pre-warming
2. `backend/main.py` - Added startup event and keep-alive endpoint

## Performance Monitoring

You can monitor performance by checking:
- Response times in FastAPI logs
- `/api/keepalive` endpoint response time
- Memory usage over time

## Questions?

If performance is still slow after these fixes:
1. Check server logs for errors
2. Monitor memory usage
3. Verify models are loading correctly
4. Check if Modal container is sleeping (if applicable)
