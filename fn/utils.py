import numpy as np
from .metrics import jensen_shannon, cosine_distance


def build_doc_dist(doc):
    keywords = doc.get("keywords", [])
    if not keywords:
        return {}

    w = 1.0 / len(keywords)
    return {k: w for k in keywords}


def get_representative_docs(year_distributions, l, r, top_k=3):
    seg_dists = []
    seg_docs = []

    for i in range(l, r + 1):
        year_dist = year_distributions[i]["dist"]
        seg_dists.append(year_dist)

        for doc in year_distributions[i].get("docs", []):
            doc_copy = doc.copy()

            doc_dist = build_doc_dist(doc)
            if not doc_dist:
                doc_dist = year_dist

            doc_copy["_dist"] = doc_dist
            doc_copy["_emb"] = doc.get("embedding")

            seg_docs.append(doc_copy)

    # keyword barycenter
    bary = {}
    total = len(seg_dists)

    for dist in seg_dists:
        for k, v in dist.items():
            bary[k] = bary.get(k, 0) + v

    for k in bary:
        bary[k] /= total

    # embedding barycenter
    embs = [d["_emb"] for d in seg_docs if d["_emb"] is not None]
    emb_bary = np.mean(embs, axis=0) if embs else None

    # scoring
    scored = []
    for doc in seg_docs:
        js = jensen_shannon(doc["_dist"], bary)

        if emb_bary is not None and doc["_emb"] is not None:
            cos = cosine_distance(doc["_emb"], emb_bary)
        else:
            cos = 0.0

        score = 0.5 * js + 0.5 * cos
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0])

    return scored[:top_k]