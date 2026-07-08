import re
from collections import Counter

import streamlit as st


st.set_page_config(page_title="News Classifier", layout="wide")


CATEGORY_KEYWORDS = {
    "World": {
        "government", "election", "minister", "diplomacy", "international", "global",
        "country", "countries", "war", "peace", "united nations", "foreign", "world",
        "border", "summit", "trade talks", "embassy", "president",
    },
    "Sports": {
        "game", "team", "player", "season", "match", "coach", "league", "score",
        "goal", "tournament", "championship", "athlete", "sports", "win", "loss",
        "football", "basketball", "cricket", "tennis", "soccer",
    },
    "Business": {
        "market", "stocks", "stock", "company", "companies", "revenue", "profit",
        "loss", "economy", "economic", "finance", "financial", "bank", "banks",
        "investor", "investors", "trade", "merger", "startup", "sales", "earnings",
    },
    "Sci/Tech": {
        "technology", "tech", "science", "scientists", "research", "researchers",
        "software", "hardware", "internet", "ai", "artificial intelligence", "robot",
        "robots", "space", "satellite", "device", "chip", "medical", "health", "data",
    },
}


def classify_article(article_text: str) -> tuple[str, dict[str, int]]:
    """Score each category by keyword overlap and return the best match."""
    normalized = re.sub(r"[^a-z0-9\s]", " ", article_text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = normalized.split()
    token_counts = Counter(tokens)

    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if " " in keyword:
                if keyword in normalized:
                    score += 2
            else:
                score += token_counts.get(keyword, 0)
        scores[category] = score

    best_category = max(scores, key=scores.get)
    return best_category, scores


def build_explanation(category: str, scores: dict[str, int]) -> str:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    runner_up = ordered[1][0] if len(ordered) > 1 else "N/A"
    return f"Predicted {category}. Next best match: {runner_up}."


st.title("News Article Classifier")
st.caption(
    "NLP text-classification project — the trained model reaches 92% accuracy; "
    "this live demo uses a lightweight rule-based classifier for instant, dependency-free hosting."
)

with st.sidebar:
    st.header("About this project")
    st.write("End-to-end NLP pipeline for 4-class news categorization.")
    st.write("Classes: World, Sports, Business, Sci/Tech")
    st.write("Full preprocessing, model training, and evaluation live in the project notebook (92% accuracy).")
    st.write("This hosted demo swaps in a fast keyword-based classifier so it runs on Streamlit Cloud without GPU/ML dependencies.")


sample_article = (
    "The central bank announced a rate cut after inflation slowed for a third month, "
    "boosting banking and retail stocks in early trading."
)

text_area_placeholder = st.empty()

col1, col2 = st.columns([1, 4])
with col1:
    classify_clicked = st.button("Classify", type="primary", use_container_width=True)
with col2:
    if st.button("Load sample text", use_container_width=True):
        st.session_state["article_text"] = sample_article
        st.session_state["_sample_loaded_notice"] = True

article = text_area_placeholder.text_area(
    "Enter news article text",
    height=220,
    placeholder="Paste a news article paragraph here...",
    key="article_text",
)

if st.session_state.pop("_sample_loaded_notice", False):
    st.info("Sample text loaded. Click Classify.")


if classify_clicked:
    if not article.strip():
        st.warning("Please enter article text before classifying.")
    else:
        prediction, scores = classify_article(article)

        st.success(f"Predicted Category: {prediction}")
        st.write(build_explanation(prediction, scores))
        st.write(f"Article length: {len(article.split())} words")

        st.subheader("Category Scores")
        score_rows = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        st.table({"Category": [row[0] for row in score_rows], "Score": [row[1] for row in score_rows]})


st.markdown("---")
st.markdown(
    """
### About the build
- Full NLP workflow — preprocessing, model training, and comparison across models — is documented in the project notebook, with a best result of 92% accuracy.
- This deployed demo runs a lightweight rule-based classifier instead of the trained model, so it stays fast and dependency-free on Streamlit Cloud.
- Natural next step for production: export the trained model and serve it directly from this app.
- Source code & notebook: https://github.com/shivani3291-lab/news-article-classifier
"""
)