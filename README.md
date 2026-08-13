# 🎧 Earbuds Review Intelligence

Aspect-based sentiment analysis on real Amazon wireless earbuds reviews — identifying which product features (battery, sound quality, comfort, connectivity, durability) drive negative customer feedback, and how that varies across products.

🔗 [Live Demo](https://earbudreviewintelligence-21.streamlit.app/) · [GitHub](https://github.com/harshaadire)

## Problem

Amazon reviews are noisy — a single review often praises one feature while criticizing another ("great sound, but battery life is awful"). A single overall star rating hides which feature actually needs fixing. This project breaks reviews down at the aspect level to answer: which specific features drive negative sentiment, and for which products?

## Approach

**Data collection** — pulled real wireless earbuds/headphones reviews from the McAuley Lab Amazon Reviews 2023 dataset (Electronics category), filtered by product-title keyword matching.

**Cleaning** — deduplication, language filtering, empty/short-review removal, timestamp normalization. The final cleaned dataset contains **19,867 reviews**.
**Exploratory analysis** — rating distribution, review length patterns, volume trends over time. The dataset has a mean review length of **653 characters** and an average of **120 words** per review.

**Aspect extraction** — rule-based keyword tagging across 5 aspects (battery, sound quality, comfort/fit, connectivity, durability) at the sentence level.

**Sentiment classification** — DistilBERT (pretrained on SST-2) applied per aspect-tagged sentence, so a single review can register different sentiment for different features. The analysis sampled **2,000 reviews** and extracted **11,918 aspect-tagged sentences**.
**Serving** — a FastAPI microservice (`api/`) exposes the model for programmatic use; the deployed demo (`app/dashboard.py`) runs inference directly in-process for free, zero-dependency hosting on Streamlit Community Cloud.

## Key Findings

* **56.78%** of reviews are 5-star, showing the expected positive skew in the dataset.
* **Battery** has the highest negative sentiment rate at **60.25%**, followed by **connectivity at 59.34%**.
* **Comfort/fit** has the highest number of aspect mentions with **4,894**, followed by sound quality with **3,071**.
* **Durability** has the lowest negative sentiment rate at **38.76%**.
* The sentiment analysis produced **11,918 aspect-tagged sentences** across the 2,000-review sample.
* Review length varies considerably, with an average of **119.5 words** and a median of **72 words**.

## Tech Stack

Python · Pandas · HuggingFace Transformers (DistilBERT) · NLTK · FastAPI · Streamlit · Docker · Plotly

## Project Structure

```text
├── data/
│   ├── raw/              # scraped/collected data
│   └── processed/        # cleaned data + aspect sentiment results
├── notebooks/
│   ├── eda_earbuds_reviews.ipynb
│   └── aspect_sentiment_model.ipynb
├── src/
│   ├── data_collection.py
│   └── data_cleaning.py
├── api/
│   └── main.py           # FastAPI service (local/standalone use)
├── app/
│   └── dashboard.py      # Streamlit app (self-contained, deployed version)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone https://github.com/harshaadire/earbud-review-intelligence.git
cd earbud-review-intelligence
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Data pipeline
python src/data_collection.py
python src/data_cleaning.py

# Then run the notebooks in order (EDA, then aspect sentiment)

# Run the dashboard
streamlit run app/dashboard.py

# Or run the standalone API
uvicorn api.main:app --reload
```

Note: first run downloads the DistilBERT model (~260MB) and may take a minute or two.

## Future Improvements

Fine-tune DistilBERT on a manually labeled subset of aspect-sentences to improve accuracy over the generic pretrained baseline, with a measured before/after comparison

Replace rule-based aspect tagging with a trained aspect-extraction model

Add brand-level comparison view to the dashboard

CI/CD via GitHub Actions for automated testing on push

## Author

**Harshith Adire** — [LinkedIn](https://www.linkedin.com/in/adire-harshith-0232a6332/) · [GitHub](https://github.com/harshaadire)
