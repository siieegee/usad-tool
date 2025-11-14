# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import json
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import functions from review_prediction.py
from .review_prediction import predict_review

app = FastAPI(title="USAD Review Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for request body
class ReviewRequest(BaseModel):
    review: str

# Define the output model
class ReviewResponse(BaseModel):
    prediction: str
    cluster: int
    distance: float
    threshold: float
    features: dict
    processed_text: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# API endpoint
@app.post("/api/predict", response_model=ReviewResponse)
def predict(review_request: ReviewRequest):
    review_text = review_request.review
    result = predict_review(review_text)
    return result

@app.get("/api/feature-basis")
def get_feature_basis():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'feature_basis.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {"normal": {}}

# Serve static files (CSS, JS, images)
# Mount this BEFORE the root route
if os.path.exists("/root/frontend"):
    app.mount("/static", StaticFiles(directory="/root/frontend"), name="static")

# Serve index.html at root
@app.get("/")
def read_index():
    frontend_path = "/root/frontend/index.html"
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
    # If it's an API route, let FastAPI handle it
    if full_path.startswith("api/"):
        return {"error": "Not found"}
    
    # Try to serve the requested file
    file_path = f"/root/frontend/{full_path}"
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Fall back to index.html for SPA routing
    index_path = "/root/frontend/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return {"error": "Not found"}