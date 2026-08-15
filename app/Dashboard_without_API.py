"""
dashboard.py

Streamlit dashboard for the Earbuds Review Intelligence project.
Self-contained version: runs the aspect detection + sentiment model
directly in the Streamlit process, so it deploys free on Streamlit
Community Cloud with no separately-hosted API required.

(api/main.py still exists in this repo as a standalone FastAPI service —
 useful for local development or production-style deployment elsewhere.
 This file duplicates that logic in-process purely for free, simple hosting.)

Two views:
    1. Live prediction — paste any review, get aspect-level sentiment
    2. Aggregate insights — precomputed aspect negativity chart across the dataset

Run locally with:
    streamlit run app/dashboard.py
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "aspect_sentiment_summary.csv")

ASPECT_KEYWORDS = {
    "battery": ["battery", "charge", "charging", "backup", "battery life", "power", "charged"],
    "sound_quality": ["sound", "audio", "bass", "treble", "music", "noise cancel", "noise cancellation", "volume",
                      "clarity"],
    "comfort_fit": ["comfort", "comfortable", "fit", "ear", "ears", "tight", "loose", "pain", "hurt", "lightweight"],
    "connectivity": ["connect", "connection", "bluetooth", "pairing", "pair", "disconnect", "range", "signal"],
    "durability": ["durable", "durability", "broke", "broken", "quality", "build", "sturdy", "crack", "stopped working",
                   "defective"],
}


@st.cache_resource(show_spinner="Loading sentiment model (first run only)...")
def load_model():
    """
    Cached so the model loads once per app session, not on every button click
    or page interaction — this is what keeps the app responsive after startup.
    """
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )


def get_sentence_aspects(sentence: str) -> list[str]:
    sentence_lower = sentence.lower()
    return [aspect for aspect, keywords in ASPECT_KEYWORDS.items()
            if any(kw in sentence_lower for kw in keywords)]


def analyze_review(review_text: str, sentiment_pipeline) -> pd.DataFrame:
    sentences = sent_tokenize(review_text)
    rows = []
    for sentence in sentences:
        aspects = get_sentence_aspects(sentence)
        if not aspects:
            continue
        sentiment = sentiment_pipeline(sentence, truncation=True)[0]
        for aspect in aspects:
            rows.append({
                "aspect": aspect,
                "sentence": sentence,
                "sentiment_label": sentiment["label"],
                "sentiment_score": round(sentiment["score"], 4),
            })
    return pd.DataFrame(rows)


st.set_page_config(page_title="Earbuds Review Intelligence", layout="wide")

st.title("🎧 Earbuds Review Intelligence")
st.markdown(
    "Aspect-based sentiment analysis on real wireless earbuds reviews — "
    "which product features drive negative feedback?"
)

sentiment_pipeline = load_model()

tab1, tab2 = st.tabs(["🔍 Try It Live", "📊 Dataset Insights"])

# ---------------------------------------------------------------------------
# TAB 1: Live prediction
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Paste a review to analyze")
    sample_review = (
        "Battery life is terrible, dies after 2 hours. But the sound quality "
        "is amazing and bass is really punchy. Comfortable fit too, wore them "
        "all day without any pain."
    )
    review_text = st.text_area("Review text", value=sample_review, height=120)

    if st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            results_df = analyze_review(review_text, sentiment_pipeline)

            if results_df.empty:
                st.warning("No known aspects (battery, sound, comfort, connectivity, "
                           "durability) detected in this review.")
            else:
                st.success(f"Found {len(results_df)} aspect mentions")
                for _, row in results_df.iterrows():
                    color = "🔴" if row["sentiment_label"] == "NEGATIVE" else "🟢"
                    st.markdown(
                        f"{color} **{row['aspect'].replace('_', ' ').title()}** "
                        f"({row['sentiment_label']}, confidence {row['sentiment_score']:.0%})  \n"
                        f"> {row['sentence']}"
                    )

# ---------------------------------------------------------------------------
# TAB 2: Aggregate insights from the full dataset (precomputed in the notebook)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Which aspects drive negative sentiment across all reviews?")
    if not os.path.exists(SUMMARY_PATH):
        st.warning("Summary file not found. Run the aspect sentiment notebook first "
                   "and make sure aspect_sentiment_summary.csv is committed to the repo.")
    else:
        summary_df = pd.read_csv(SUMMARY_PATH)
        summary_df["aspect"] = summary_df["aspect"].str.replace("_", " ").str.title()

        fig = px.bar(
            summary_df.sort_values("negative_rate", ascending=True),
            x="negative_rate",
            y="aspect",
            orientation="h",
            labels={"negative_rate": "Negative Sentiment Rate", "aspect": "Aspect"},
            title="Negative Sentiment Rate by Aspect",
            color="negative_rate",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(summary_df, width="stretch")

st.markdown("---")
st.caption("Built with DistilBERT and Streamlit | [GitHub Repo](https://github.com/harshaadire/earbud_review_intelligence)")