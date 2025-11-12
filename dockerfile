FROM python:3.13-bookworm

WORKDIR /app

COPY ./backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend /app/backend

CMD ["uvicorn", "backend.main:app"]