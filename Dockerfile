FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY ./backend/requirements.txt .
RUN pip install -r requirements.txt

# Download NLTK stopwords during build
RUN python -m nltk.downloader stopwords

# Copy backend and frontend
COPY ./backend /app/backend
COPY ./frontend /app/frontend

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
