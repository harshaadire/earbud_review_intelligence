# importing required modules

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000" #have to change after the deployment

st.set_page_config(page_title="Earbud Reviews Intelligence",layout="wide")

st.title("🎧Earbud Reviews Intelligence")
st.markdown("Aspect-based sentiment analysis on real wireless reviews -"
            " Which product feature drive negative feedback?")

tab1,tab2 = st.tabs (["🔎Try It Live","📊Dataset Insights"])

#----------------------------------------------------------------------------------------------------
# Live prediction
#-----------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("Paste a review to analyze")
    sample_review = (
            "Battery life is terrible, dies after 2 hours. But the sound quality "
            "is amazing and bass is really punchy. Comfortable fit too, wore them "
            "all day without any pain."
    )

    review_text = st.text_area("Review Text",value= sample_review,height=120)

    if st.button("Analyze",type="primary"):
        with st.spinner('Analyzing...'):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"review_text": review_text},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                if not data['aspect_found']:
                    st.warning("No aspect found.")
                else:
                    result_df = pd.DataFrame(data['aspect_found'])
                    st.success(f"{len(result_df)} aspects found")

                    for _, row in result_df.iterrows():
                        color = "🔴" if row['sentiment_label'] == "NEGATIVE" else "🟢"
                        st.markdown(
                            f"{color}**{row['aspect'].replace('_', ' ').title()}**"
                            f"({row['sentiment_label']}, confidence {row['sentiment_score']:.0%}) \n"
                            f"> {row['sentence']}"
                        )
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to API:{e}")
                st.info("Make sure to FastAPI is running.")

#----------------------------------------------------------------------------------------------------
#Data insights from the precomputed dataset
#----------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("Which aspect drive negative feedback more across all reviews?")
    try:
        response = requests.get(f"{API_URL}/summary",timeout=30)
        response.raise_for_status()
        summary_data = response.json()

        if isinstance(summary_data,dict) and "error" in summary_data:
            st.warning(summary_data["error"])
        else:
            summary_df = pd.DataFrame(summary_data)
            summary_df["aspect"]  = summary_df["aspect"].str.replace('_',' ').str.title()

            fig = px.bar(
                summary_df.sort_values("negative_rate", ascending=True),
            x="negative_rate",
            y="aspect",
            orientation="h",
            labels={
                "negative_rate": "Negative Sentiment Rate",
                "aspect": "Aspect"
            },
            title="Negative Sentiment Rate by Aspect",
            color="negative_rate",
            color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(summary_df, use_container_width=True)
    except requests.exceptions.RequestException as e:
        st.error(f"could not reach API: {e}")

st.markdown("---")
st.caption("Built with FastAPI, DistilBERT, and Streamlit | [GitHub](https://github.com/harshaadire/earbud_review_intelligence)")

