# FastAPI for the aspect-based sentiment analysis on earbuds review

#importing required modules
import os
from typing import List

import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt',quiet=True)
nltk.download('punkt_tab',quiet=True)

#initialising the path
project_root= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
summary_root = os.path.join(project_root,'data','processed','aspect_sentiment_summary.csv')

#declaring keywords
ASPECT_KEYWORDS = {
    "battery": ["battery", "charge", "charging", "backup", "battery life", "power", "charged"],
    "sound_quality": ["sound", "audio", "bass", "treble", "music", "noise cancel", "noise cancellation", "volume", "clarity"],
    "comfort_fit": ["comfort", "comfortable", "fit", "ear", "ears", "tight", "loose", "pain", "hurt", "lightweight"],
    "connectivity": ["connect", "connection", "bluetooth", "pairing", "pair", "disconnect", "range", "signal"],
    "durability": ["durable", "durability", "broke", "broken", "quality", "build", "sturdy", "crack", "stopped working", "defective"],
}

app = FastAPI(
    title="Earbud Review Intelligence API",
    description = "Aspect based sentiment analysis for earbud review",
    version="1.0.0"
)

#Loading only once at starup not for each and every request
device = 0 if torch.cuda.is_available() else -1
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model='distilbert-base-uncased-finetuned-sst-2-english',
    device=device
)

class ReviewRequest(BaseModel):
    review_text: str

class AspectResult(BaseModel):
    aspect: str
    sentence: str
    sentiment_label: str
    sentiment_score: float

class ReviewResponse(BaseModel):
    review_text: str
    aspect_found: list[AspectResult]

def get_sentence_aspects(sentence: str) -> list[str]:
    sentence_lower = sentence.lower()
    return [aspect for aspect,keywords in ASPECT_KEYWORDS.items()
            if any(kw in sentence_lower for kw in keywords)]

@app.get("/")
def health_check():
    return {"status":"OK","message":"Earbud Review Intelligence API is running"}

@app.post("/predict",response_model=ReviewResponse)
def predict(request: ReviewRequest):
    sentences = sent_tokenize(request.review_text)
    results = []

    for sentence in sentences:
        aspects = get_sentence_aspects(sentence)
        if not aspects:
            continue
        sentiment_analysis = sentiment_pipeline(sentence,truncation=True)[0]
        for aspect in aspects:
            results.append(AspectResult(
                aspect=aspect,
                sentence=sentence,
                sentiment_label=sentiment_analysis['label'],
                sentiment_score=round(sentiment_analysis['score'],4)
            ))

    return ReviewResponse(review_text=request.review_text,aspect_found=results)

@app.get("/summary")
def get_summary():
#Return the precomputed negative rate which is in the notebook analysis
    if not os.path.exists(summary_root):
        return {
            "error":"Summary file not found.Run the aspect sentiment analysis first"
        }
    df = pd.read_csv(summary_root)
    return df.to_dict(orient='records')




