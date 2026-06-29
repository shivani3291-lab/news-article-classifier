import re

import streamlit as st
from transformers import pipeline


st.set_page_config(page_title="News Classifier", layout="wide")

MODEL_NAME = "google/flan-t5-base"


@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the FLAN-T5 pipeline once per app session."""
    return pipeline("text2text-generation", model=MODEL_NAME)


def normalize_prediction(text: str) -> str:
    """Normalize model output to one of the supported classes."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^a-z/ ]", "", cleaned)

    if "sport" in cleaned:
        return "Sports"
    if "business" in cleaned or "finance" in cleaned or "economy" in cleaned:
        return "Business"
    if "sci" in cleaned or "tech" in cleaned or "science" in cleaned or "technology" in cleaned:
        return "Sci/Tech"
    if "world" in cleaned or "international" in cleaned or "global" in cleaned:
        return "World"
    return "Unknown"


def build_prompt(article_text: str) -> str:
    """Create a constrained prompt for reliable single-label classification."""
    return f"""Classify the following news article into exactly one label.

Allowed labels:
- World
- Sports
- Business
- Sci/Tech

Rules:
- Return only one label from the list.
- Do not add explanations.

Article:
{article_text[:1000]}

Label:"""


st.title("News Article Classifier")
st.caption("Standalone deployment app (no notebook/Colab required)")

with st.sidebar:
    st.header("About")
    st.write("Model: FLAN-T5 Base")
    st.write("Task: 4-class news categorization")
    st.write("Classes: World, Sports, Business, Sci/Tech")


article = st.text_area(
    "Enter news article text",
    height=220,
    placeholder="Paste a news article paragraph here...",
)

col1, col2 = st.columns([1, 4])
with col1:
    classify_clicked = st.button("Classify", type="primary", use_container_width=True)
with col2:
    if st.button("Use sample text", use_container_width=True):
        st.session_state["sample"] = (
            "The central bank announced a rate cut after inflation slowed for a third month, "
            "boosting banking and retail stocks in early trading."
        )

if st.session_state.get("sample") and not article:
    article = st.session_state["sample"]
    st.info("Sample text loaded. Click Classify.")


if classify_clicked:
    if not article.strip():
        st.warning("Please enter article text before classifying.")
    else:
        with st.spinner("Loading model and generating prediction..."):
            generator = load_model()
            prompt = build_prompt(article)
            output = generator(prompt, max_new_tokens=4, do_sample=False)
            raw_prediction = output[0]["generated_text"]
            prediction = normalize_prediction(raw_prediction)

        if prediction == "Unknown":
            st.error("Could not confidently map the model output to a supported class.")
            st.write("Raw model output:", raw_prediction)
        else:
            st.success(f"Predicted Category: {prediction}")
            st.write(f"Raw model output: {raw_prediction}")
            st.write(f"Article length: {len(article.split())} words")


st.markdown("---")
st.markdown(
    """
### Notes
- This app runs independently and does not require notebook upload.
- This deployment uses FLAN-T5 Base to keep the free hosting footprint manageable.
- For faster/cheaper inference in production, consider a tuned classical pipeline.
- Repository: https://github.com/shivani3291-lab/news-article-classifier
"""
)