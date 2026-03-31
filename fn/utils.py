import numpy as np
from .metrics import jensen_shannon
import pandas as pd
import matplotlib.pyplot as plt
from .pipeline import build_timeline_from_model
from .metrics import compute_total_distortion

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

def segments_to_dataframe(result):
    """
    Convert segmentation result → pandas DataFrame

    Columns:
        - start_year
        - end_year
        - period
        - num_docs
        - topic
        - topic_certainty
    """

    segments = result["segments"]
    year_distributions = result["year_distributions"]
    years = result["sorted_years"]

    rows = []

    for (l, r) in segments:
        seg_dists = []
        num_docs = 0

        for i in range(l, r + 1):
            seg_dists.append(year_distributions[i]["dist"])
            num_docs += len(year_distributions[i].get("docs", []))

        seg_matrix = np.array(seg_dists)

        # barycenter
        bary = np.mean(seg_matrix, axis=0)

        # topic + certainty
        topic = int(np.argmax(bary))
        certainty = float(np.max(bary))

        rows.append({
            "start_year": years[l],
            "end_year": years[r],
            "period": r - l + 1,
            "num_docs": num_docs,
            "topic": topic,
            "topic_certainty": certainty
        })

    df = pd.DataFrame(rows)

    return df


def sweep_lambda(dataset_path, model_path):
    lambdas = np.arange(0.0, 1.01, 0.05)

    all_lambdas = []
    all_costs = []
    all_segments = []

    for lam in lambdas:
        print(f"Running lambda = {lam:.2f}")

        result = build_timeline_from_model(
            dataset_path,
            model_path,
            lambda_penalty=float(lam)
        )

        cost, _ = compute_total_distortion(result['year_distributions'], result["segments"])
        num_segments = len(result["segments"])

        all_lambdas.append(lam)
        all_costs.append(cost)
        all_segments.append(num_segments)

    return all_lambdas, all_costs, all_segments


# =========================================
# Plot
# =========================================
def plot_elbow(lambdas, costs, segments):
    plt.figure(figsize=(10, 6))

    plt.plot(lambdas, costs, marker="o")

    # annotate số segment
    for x, y, s in zip(lambdas, costs, segments):
        plt.text(
            x, y,
            str(s),
            fontsize=9,
            ha="center",
            va="bottom"
        )

    plt.xlabel("Lambda")
    plt.ylabel("Total JS Cost")
    plt.title("Elbow Method (Lambda vs Cost)")
    plt.grid(True)

    plt.show()
