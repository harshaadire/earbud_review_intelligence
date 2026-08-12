#Importing the required Modules
import re
import pandas as pd
from datasets import load_dataset

# let's initialize keywords so that we can filter the data from the
#raw data
KEYWORDS = [
    "earbuds","earbud","airpods","ear buds","wireless headphones","wireless headphone",
    "in-ear headphones","in-ear headphone","bluetooth headset","bluetooth earphone",
    "bluetooth earphones"
]

#the "Electronics" data is huge so let decrease the size
CATEGORY = "raw_review_Electronics"
max_reviews = 20000 #stops when it reached this limit

def title_matches(title:str) -> bool:
    #checks the product we want filtered
    if not title:
        return False
    title_lower = title.lower()
    return any(kw in title_lower for kw in KEYWORDS)

def clean_text(text:str) -> str:
    #Helps to clean the strip the extra whitespaces and controls the characters
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+"," ",text).strip()
    return text

def main():
    print("Loading data...")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        CATEGORY,
        split="full",
        streaming=True,
        trust_remote_code=True
    )

    rows = []

    for i,review in enumerate(dataset):
        title = review.get("title","") or "" # review title
        text = review.get("text","") or "" # review body
        product_title = review.get("parent_asin","") or "" # product id

        #Here we filter using title/text

        combined = f"{title} {text}"

        if title_matches(combined):
            rows.append({
                "asin":review.get("asin",""),
                "parent_asin":product_title,
                "rating":review.get("rating",None),
                "review_title": clean_text(title),
                "review_text": clean_text(text),
                "timestamp":review.get("timestamp",""),
                "verified_purchase":review.get("verified_purchase",None),
                "helpful_vote":review.get("helpful_vote",None)
            })

        if len(rows) >= max_reviews:
            break

        if i % 50000 == 0 and i > 0:
            print(f"Scanned {i} reviews, matched {len(rows)} so far...")

    df = pd.DataFrame(rows)

    print(f"\nCollected {len(df)} earbud/headphone-related reviews.")

    output_path = "data/raw/earbuds_reviews.csv"

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()




