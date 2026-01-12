"""
Program Title:
main.py – FastAPI Backend Application for the USAD (UnSupervised Anomaly Detection) Review Prediction System

Programmers:
Cristel Jane Baquing, Angelica Jean Evangelista, James Tristan Landa, Kharl Chester Velasco

Where the Program Fits in the General System Design:
This module functions as the primary backend API layer of the USAD system. It provides REST endpoints 
for predicting whether a given review is Normal or Anomalous using the trained anomaly detection 
pipeline implemented in review_prediction.py. The FastAPI server serves as the bridge between the 
machine learning components and external clients such as the web-based frontend, third-party systems, 
or automated evaluation scripts.

Additionally, this module hosts static frontend assets, delivers model feature-basis data for UI 
visualization, supports Single Page Application routing, and exposes a health check for service 
monitoring.

Date Written and Revised:
Original version: November 22, 2025  
Last revised: November 22, 2025

Purpose:
To deploy a production-ready API for real-time anomaly detection of text reviews by:
• Accepting raw review text through a structured request model  
• Executing preprocessing, vectorization, feature projection, centroid distance scoring, and threshold-based classification  
• Returning detailed prediction metadata such as cluster ID, distance, threshold, processed text, and feature contributions  
• Serving feature basis information for frontend model explanation  
• Optionally hosting static frontend files and enabling SPA routing  
• Providing CORS support for external clients  

This API enables seamless integration of the USAD model into web interfaces, dashboards, and automated 
workflows.

Data Structures, Algorithms, and Control:
• Data Structures:
  - Pydantic models (ReviewRequest, ReviewResponse) for input/output validation  
  - JSON feature_basis.json for model interpretability  
  - Static frontend directory mounted for UI assets  

• Algorithms:
  - Delegated prediction logic via predict_review()  
  - Preprocessing and feature extraction defined in the model pipeline  
  - Cosine-distance scoring and threshold comparison from the trained model  
  - SPA routing logic for serving index.html as fallback  

• Control:
  - CORS middleware for controlled cross-origin requests  
  - /health endpoint for system monitoring  
  - /api/predict endpoint for main model inference  
  - /api/feature-basis endpoint for interpretability metadata  
  - Static file mounting for frontend delivery  
  - Catch-all route to support browser-based navigation in SPAs  

This module is the core operational gateway of the USAD review prediction platform.
"""

"""
FastAPI application for USAD Review Prediction API
"""
import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import functions from review_prediction.py
from .review_prediction import predict_review


# Constants
FRONTEND_DIR = "/root/frontend"
FEATURE_BASIS_FILE = "feature_basis.json"
INDEX_FILE = "index.html"
DEFAULT_ORIGINS = ["*"]  # In production, restrict this


# Initialize FastAPI app
app = FastAPI(title="USAD Review Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Pre-warm resources on application startup to avoid slow first request.
    This ensures all NLP resources and models are loaded and ready.
    """
    import time
    import asyncio
    from .review_prediction import predict_review
    
    print("Pre-warming application resources...")
    start_time = time.time()
    
    try:
        # Pre-warm by making a dummy prediction
        # This ensures all models, NLP resources, and TextBlob are loaded
        dummy_review = "This is a test review to pre-warm resources."
        _ = predict_review(dummy_review)
        
        # Also pre-warm TextBlob multiple times to ensure it's fully loaded
        from textblob import TextBlob
        for _ in range(3):
            blob = TextBlob("test warmup")
            _ = blob.sentiment
        
        elapsed = time.time() - start_time
        print(f"✓ Resources pre-warmed successfully in {elapsed:.2f}s")
        
        # Start background keep-alive task
        asyncio.create_task(background_keepalive())
    except Exception as e:
        print(f"⚠ Warning: Resource pre-warming failed: {e}")
        print("  Application will continue, but first request may be slow.")


async def background_keepalive():
    """
    Background task to keep resources warm by making periodic dummy predictions.
    This prevents resources from being garbage collected during idle time.
    """
    import asyncio
    from .review_prediction import predict_review
    
    while True:
        try:
            # Wait 90 seconds (before 2-minute idle threshold)
            await asyncio.sleep(90)
            
            # Make a lightweight prediction to keep resources active
            try:
                _ = predict_review("keepalive")
            except:
                pass  # Silently fail if prediction fails
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Keep-alive task error: {e}")
            await asyncio.sleep(90)  # Wait before retrying


# Request/Response models
class ReviewRequest(BaseModel):
    """Request model for review prediction"""
    review: str


class ReviewResponse(BaseModel):
    """Response model for review prediction"""
    prediction: str
    cluster: int
    distance: float
    threshold: float
    features: dict
    processed_text: str


# Health check endpoint
@app.get("/health")
def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Health status information
    """
    return {"status": "healthy"}


# Keep-alive endpoint to prevent resource cleanup during idle time
@app.get("/api/keepalive")
def keepalive():
    """
    Keep-alive endpoint to prevent resource cleanup during idle time.
    This endpoint makes a lightweight prediction to keep all resources active.
    
    Returns:
        dict: Keep-alive status
    """
    try:
        # Make a very lightweight prediction to keep all resources warm
        # This ensures models, NLP resources, and TextBlob stay in memory
        from .review_prediction import predict_review
        _ = predict_review("keepalive check")
        
        return {
            "status": "alive",
            "message": "Resources are active and warmed"
        }
    except Exception as e:
        return {
            "status": "alive",
            "message": f"Keep-alive check completed (warning: {str(e)})"
        }


# API endpoint
@app.post("/api/predict", response_model=ReviewResponse)
def predict(review_request: ReviewRequest):
    """
    Predict if a review is Normal or Anomalous.

    Args:
        review_request: ReviewRequest object containing the review text

    Returns:
        ReviewResponse: Prediction result with details

    Raises:
        HTTPException: If prediction fails
    """
    try:
        review_text = review_request.review
        if not review_text or not review_text.strip():
            return {
                "error": "Review text cannot be empty"
            }
        result = predict_review(review_text)
        return result
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}"
        }


@app.get("/api/feature-basis")
def get_feature_basis():
    """
    Get feature basis for frontend.

    Returns:
        dict: Feature basis data or empty dict if file not found
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        feature_basis_path = os.path.join(base_dir, FEATURE_BASIS_FILE)
        if os.path.exists(feature_basis_path):
            with open(feature_basis_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"normal": {}}
    except Exception as e:
        # Return empty basis on error
        return {"normal": {}}


# Serve static files (CSS, JS, images)
# Mount this BEFORE the root route
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# Serve index.html at root
@app.get("/")
def read_index():
    """
    Serve index.html at root.

    Returns:
        FileResponse: The index.html file if it exists,
                     otherwise a JSON response with API status
    """
    frontend_path = os.path.join(FRONTEND_DIR, INDEX_FILE)
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {
        "message": "USAD Review Prediction API",
        "status": "running",
        "note": "Frontend not found. API endpoints available at /api/predict"
    }


# Catch-all route for SPA (if using client-side routing)
@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    """
    Serve SPA routes and static files.

    Args:
        full_path: Requested file path

    Returns:
        FileResponse: Requested file or index.html as fallback,
                     otherwise error response
    """
    # If it's an API route, return not found
    if full_path.startswith("api/"):
        return {"error": "Not found"}

    # Try to serve the requested file
    file_path = os.path.join(FRONTEND_DIR, full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)

    # Fall back to index.html for SPA routing
    index_path = os.path.join(FRONTEND_DIR, INDEX_FILE)
    if os.path.exists(index_path):
        return FileResponse(index_path)

    return {"error": "Not found"}