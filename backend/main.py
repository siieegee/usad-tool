# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    processed_text: str

# API endpoint
@app.post("/api/predict", response_model=ReviewResponse)
def predict(review_request: ReviewRequest):
    review_text = review_request.review
    result = predict_review(review_text)
    return result


    ### python -m uvicorn backend.main:app --reload