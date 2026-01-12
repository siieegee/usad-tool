# deploy.py
import modal

app = modal.App("usad-fastapi-app")

image = (
    modal.Image.debian_slim()
    .pip_install([
        # FastAPI
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-multipart",
        
        # Core ML & Data Processing
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        
        # NLP
        "nltk",
        "textblob",
        
        # Deep Learning
        "torch",
        
        # Optimization
        "pyswarm",
        
        # Visualization
        "matplotlib",
        "seaborn",
    ])
    # Download NLTK data
    .run_commands([
        "python -c \"import nltk; nltk.download('punkt', quiet=True)\"",
        "python -c \"import nltk; nltk.download('stopwords', quiet=True)\"",
        "python -c \"import nltk; nltk.download('wordnet', quiet=True)\"",
        "python -c \"import nltk; nltk.download('omw-1.4', quiet=True)\"",
        "python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True)\"",
        "python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\"",
    ])
    # Add backend files
    .add_local_dir("backend", remote_path="/root/backend")
    # Add frontend files
    .add_local_dir("frontend", remote_path="/root/frontend")
)

@app.function(
    image=image,
    secrets=[],
    env={
        "ANOMALY_THRESHOLD": "0.54",
    },
    timeout=600,
    min_containers=1,  # Keep 1 container warm to prevent cold starts
    # Uncomment if you need GPU:
    # gpu="T4",
)
@modal.asgi_app()
def fastapi_app():
    from backend.main import app as fastapi_app
    return fastapi_app