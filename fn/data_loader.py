import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# load model 1 lần
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_docs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    year_groups = defaultdict(list)

    for doc_id, doc in data.items():
        year = doc["pub_year"]

        content = doc.get("content", "")
        embedding = model.encode(content) if content else None

        year_groups[year].append({
            "id": doc_id,
            "text": content,
            "keywords": doc.get("keywords", []),
            "citation": doc.get("cited_by_count", 0),
            "embedding": embedding
        })

    return year_groups