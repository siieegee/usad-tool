"""
FastAPI application for USAD Tool - Ready for Render deployment
"""
import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import prediction function
from .review_prediction import predict_review


# Constants
STATIC_DIR_NAME = "static"
MODELS_DIR_NAME = "models"
FEATURE_BASIS_FILE = "feature_basis.json"
INDEX_FILE = "index.html"
DEFAULT_ORIGINS = ["*"]  # In production, restrict this to specific origins


def get_static_directory():
    """
    Get the path to the static files directory.

    Returns:
        str: Path to the static directory
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        STATIC_DIR_NAME
    )


def get_models_directory():
    """
    Get the path to the models directory.

    Returns:
        str: Path to the models directory
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        MODELS_DIR_NAME
    )


# Initialize FastAPI app
app = FastAPI(
    title="USAD Tool API",
    description="UnSupervised Anomaly Detection for Product Reviews",
    version="1.0.0"
)

# Configure CORS - Allow all origins for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
STATIC_DIR = get_static_directory()
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    confidence: float
    processed_text: str
    features: dict


# API Routes
@app.get("/")
async def root():
    """
    Serve the main frontend page.

    Returns:
        FileResponse: The index.html file if it exists,
                     otherwise a JSON response with API status
    """
    index_path = os.path.join(STATIC_DIR, INDEX_FILE)
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "USAD Tool API",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for Render.

    Returns:
        dict: Health status information
    """
    return {
        "status": "healthy",
        "service": "usad-tool"
    }


@app.post("/api/predict", response_model=ReviewResponse)
async def predict(review_request: ReviewRequest):
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
        # Error handling - return appropriate error message
        return {
            "error": f"Prediction failed: {str(e)}"
        }


@app.get("/api/feature-basis")
async def get_feature_basis():
    """
    Get feature basis for frontend.

    Returns:
        dict: Feature basis data or empty dict if file not found
    """
    try:
        models_dir = get_models_directory()
        feature_basis_path = os.path.join(
            models_dir,
            FEATURE_BASIS_FILE
        )
        if os.path.exists(feature_basis_path):
            with open(feature_basis_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        # Fallback: return empty basis if file not found
        return {"normal": {}}
    except Exception as e:
        # Return empty basis on error
        return {"normal": {}}

