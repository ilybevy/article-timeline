import numpy as np


def select_core_topics(topic_vector, threshold=0.5):
    indexed = list(enumerate(topic_vector))
    indexed.sort(key=lambda x: x[1], reverse=True)

    selected = []
    total = 0.0

    for t, p in indexed:
        selected.append(t)
        total += p
        if total >= threshold:
            break

    return selected


def score_doc(doc_topic_vec, core_topics):
    return float(sum(doc_topic_vec[t] for t in core_topics))


def get_representative_papers(
    docs: list,
    doc_topic_map: dict,
    segment_topic_vector,
    top_k: int = 5,
    threshold: float = 0.5
):

    if not docs:
        return []

    core_topics = select_core_topics(segment_topic_vector, threshold)

    scored = []

    for d in docs:
        doc_id = d["id"]

        if doc_id not in doc_topic_map:
            continue

        doc_topic_vec = doc_topic_map[doc_id]

        score = score_doc(doc_topic_vec, core_topics)

        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_docs = []
    for score, d in scored[:top_k]:
        top_docs.append({
            "id": d["id"],
            "title": d["title"],
            "year": d["year"],
            "citation_count": d["citation_count"],
            "score": score,
            "content": d.get("content", "") 
        })

    return top_docs