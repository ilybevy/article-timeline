from collections import defaultdict
import json

def load_docs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    year_groups = defaultdict(list)

    for doc_id, doc in data.items():
        year = str(doc["pub_year"])

        keywords = doc.get("keywords", []) 
        content = doc.get("content", "")

        year_groups[year].append({
            "id": doc_id,
            "content": content,
            "keywords": keywords
        })

    return year_groups