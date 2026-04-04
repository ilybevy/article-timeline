import requests
import numpy as np
from collections import defaultdict

from .config import (
    XAI_API_KEY,
    XAI_API_URL,
    GROK_MODEL,
    REQUEST_TIMEOUT,
    MAX_TOKENS_LABEL,
    TEMPERATURE_LABEL,
    TOP_K_KEYWORDS
)


# =========================================================
# 1. EXTRACT TOP KEYWORDS PER TOPIC
# =========================================================
def get_topic_keywords(model, k=10):
    topic_words = {}

    for topic_id in range(model.k):
        words = model.get_topic_words(topic_id, top_n=k)
        topic_words[topic_id] = [w for w, _ in words]

    return topic_words


# =========================================================
# 2. COMPUTE TOPIC WEIGHT IN A PERIOD
# =========================================================
def compute_topic_weight(df, start_year, end_year, topic_id):
    segment = df[
        (df["start_year"] == start_year) &
        (df["end_year"] == end_year)
    ]

    total = len(segment)
    if total == 0:
        return 0.0

    topic_count = (segment["topic"] == topic_id).sum()

    return topic_count / total


# =========================================================
# 3. EXTRACT KEYWORDS FROM TOPIC MAP
# =========================================================
def extract_period_keywords(period_row, topic_keywords_map):
    topic_id = int(period_row["topic"])
    return topic_keywords_map.get(topic_id, [])


# =========================================================
# 4. GROK LLM CALL
# =========================================================
def generate_period_label(keywords, start_year, end_year, topic_weight):
    prompt = f"""
You are an expert in scientific history analysis.

Time range: {start_year} - {end_year}
Topic dominance score: {topic_weight:.3f}

Dominant topic keywords:
{", ".join(keywords)}

Task:
Generate a concise label describing the research paradigm of the period.

Constraints:
- Maximum 12 words
- Must reflect scientific paradigm or methodology
- Must reflect dominance strength (high vs low dominance)
- No punctuation at the end
- No explanations
- Output only the label
""".strip()

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE_LABEL,
        "max_tokens": MAX_TOKENS_LABEL
    }

    response = requests.post(
        XAI_API_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"].strip()


# =========================================================
# 5. MAIN PIPELINE
# =========================================================
def build_period_labels(df, model):
    topic_keywords_map = get_topic_keywords(model, k=TOP_K_KEYWORDS)

    labels = []

    for _, row in df.iterrows():
        start_year = int(row["start_year"])
        end_year = int(row["end_year"])
        topic_id = int(row["topic"])

        keywords = extract_period_keywords(row, topic_keywords_map)

        topic_weight = compute_topic_weight(
            df,
            start_year,
            end_year,
            topic_id
        )

        label = generate_period_label(
            keywords=keywords,
            start_year=start_year,
            end_year=end_year,
            topic_weight=topic_weight
        )

        labels.append(label)

    df["period_label"] = labels

    return df