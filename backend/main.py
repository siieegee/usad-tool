# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import functions from review_prediction.py
from .review_prediction import predict_review

app = FastAPI()

# Enable CORS for your frontend
origins = [
    "http://127.0.0.1:5500",  # VSCode Live Server
    "http://127.0.0.1:8000",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

# Serve the entire 'frontend' folder at /static
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html at root
@app.get("/")
def read_index():
    return FileResponse(os.path.join("frontend", "index.html"))

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
    # Fallback: empty basis
    return {"normal": {}}


    ### python -m uvicorn backend.main:app --reload