import requests

from .config import (
    XAI_API_KEY,
    XAI_API_URL,
    GROK_MODEL,
    REQUEST_TIMEOUT,
    MAX_TOKENS_LABEL,
    TEMPERATURE_LABEL
)


# =========================================================
# LLM LABEL GENERATION (ONLY KEYWORDS)
# =========================================================
def generate_period_label(keywords):

    keyword_str = ", ".join(keywords)

    prompt = f"""
You are an expert in scientific history analysis.

Dominant topic keywords:
{keyword_str}

Task:
Generate a concise label describing the research paradigm.

Constraints:
- Maximum 12 words
- Must reflect scientific methodology or paradigm
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