import json
from collections import defaultdict


def load_docs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    year_groups = defaultdict(list)

    for doc_id, doc in data.items():
        year = str(doc["pub_year"])

        title = doc.get("title", "")
        keywords = " ".join(doc.get("keywords", []))
        content = doc.get("content", "")

        full_text = f"{keywords} {content}".strip()

        year_groups[year].append({
            "id": doc_id,
            "text": full_text
        })

    return year_groups