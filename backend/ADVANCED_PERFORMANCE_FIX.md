# Advanced Performance Fix: Modal Container Sleep Issue

## Problem Still Persisting
Even after initial fixes, the app is still slow after 2 minutes idle.

## Root Cause: Modal Container Sleep
If you're using **Modal** for deployment, containers automatically sleep after ~2 minutes of inactivity. This is the most likely cause.

## Solutions Implemented

### ✅ Fix 1: Modal `keep_warm` Parameter
**File**: `deploy.py`

Added `keep_warm=1` to keep at least 1 container always running:
```python
@app.function(
    keep_warm=1,  # Keep 1 container warm
    ...
)
```

**Benefit**: Prevents container from sleeping, eliminating cold start delays.

### ✅ Fix 2: Background Keep-Alive Task
**File**: `backend/main.py`

Added background task that runs every 90 seconds to keep resources active:
```python
async def background_keepalive():
    while True:
        await asyncio.sleep(90)
        _ = predict_review("keepalive")
```

**Benefit**: Automatically keeps resources warm without manual intervention.

### ✅ Fix 3: Enhanced Keep-Alive Endpoint
**File**: `backend/main.py`

Updated keep-alive endpoint to make actual predictions:
```python
@app.get("/api/keepalive")
def keepalive():
    _ = predict_review("keepalive check")
    return {"status": "alive"}
```

**Benefit**: Endpoint now actively uses all resources, preventing cleanup.

## How to Deploy

1. **Update Modal deployment**:
   ```bash
   modal deploy deploy.py
   ```

2. **The fixes are automatic** - no additional configuration needed.

## Alternative: If Not Using Modal

If you're running locally or on another platform:

1. **Use the keep-alive endpoint**:
   - Set up a cron job or scheduled task
   - Call `/api/keepalive` every 90 seconds

2. **Or use the background task** (already implemented):
   - The background task runs automatically
   - No manual setup needed

## Testing

1. Deploy with `keep_warm=1`
2. Wait 2+ minutes
3. Make a prediction request
4. Should be fast (< 0.5 seconds)

## If Still Slow

Check:
1. **Modal logs** - Are containers actually staying warm?
2. **Memory usage** - Are models being swapped to disk?
3. **Network latency** - Is the delay from network, not processing?

## Cost Consideration

`keep_warm=1` means 1 container is always running, which has a cost. However:
- It's usually minimal (Modal charges per second)
- Much better user experience
- Prevents slow first requests

If cost is a concern, you can:
- Use `keep_warm=0` and rely on background task
- Or use keep-alive endpoint with external cron job
