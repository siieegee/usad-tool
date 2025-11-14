"""
FastAPI application for USAD Tool - Ready for Render deployment
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json

# Import prediction function
from .review_prediction import predict_review

# Initialize FastAPI app
app = FastAPI(
    title="USAD Tool API",
    description="UnSupervised Anomaly Detection for Product Reviews",
    version="1.0.0"
)

# Configure CORS - Allow all origins for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you may want to restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Request/Response models
class ReviewRequest(BaseModel):
    review: str


class ReviewResponse(BaseModel):
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
    """Serve the main frontend page"""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "USAD Tool API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy", "service": "usad-tool"}


@app.post("/api/predict", response_model=ReviewResponse)
async def predict(review_request: ReviewRequest):
    """Predict if a review is Normal or Anomalous"""
    review_text = review_request.review
    result = predict_review(review_text)
    return result


@app.get("/api/feature-basis")
async def get_feature_basis():
    """Get feature basis for frontend"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_dir, "models", "feature_basis.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    # Fallback: empty basis
    return {"normal": {}}

