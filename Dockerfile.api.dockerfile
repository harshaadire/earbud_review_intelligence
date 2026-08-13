FROM python:3.11-slim

WORKDIR /app

#installing all dependencies needed by troch and transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#copy requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#this only download at build up not everytime so no worry
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY api/ ./api/
COPY data/processed/aspect_sentiment_summary.csv ./data/processed/aspect_sentiment_summary.csv

EXPOSE 8000

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
