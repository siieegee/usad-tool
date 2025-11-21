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