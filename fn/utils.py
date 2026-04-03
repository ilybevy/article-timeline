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


def segments_to_dataframe(result):
    """
    Convert segmentation result → pandas DataFrame

    Columns:
        - start_year
        - end_year
        - period
        - num_docs
        - topic
        - topic_vector
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

        # dominant topic
        topic = int(np.argmax(bary))

        rows.append({
            "start_year": years[l],
            "end_year": years[r],
            "period": r - l + 1,
            "num_docs": num_docs,
            "topic": topic,
            "topic_vector": bary.tolist()
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


import numpy as np
import pandas as pd


def merge_segments(df, min_docs=5):
    """
    Input:
        df: DataFrame có các cột:
            start_year, end_year, period, num_docs, topic, topic_vector

    Output:
        DataFrame đã:
            - filter num_docs >= min_docs
            - merge các segment liên tiếp cùng topic
    """

    # 1. filter
    df = df[df["num_docs"] >= min_docs].copy()
    df = df.sort_values("start_year").reset_index(drop=True)

    merged_rows = []

    current = None

    for _, row in df.iterrows():
        vec = np.array(row["topic_vector"], dtype=np.float32)

        if current is None:
            current = {
                "start_year": row["start_year"],
                "end_year": row["end_year"],
                "num_docs": row["num_docs"],
                "topic": row["topic"],
                "vector_sum": vec * row["num_docs"]
            }
            continue

        # nếu cùng topic → merge
        if row["topic"] == current["topic"]:
            current["end_year"] = row["end_year"]
            current["num_docs"] += row["num_docs"]
            current["vector_sum"] += vec * row["num_docs"]

        else:
            # finalize current
            bary = current["vector_sum"] / current["num_docs"]

            merged_rows.append({
                "start_year": current["start_year"],
                "end_year": current["end_year"],
                "period": current["end_year"] - current["start_year"] + 1,
                "num_docs": current["num_docs"],
                "topic": current["topic"],
                "topic_vector": bary.tolist()
            })

            # start new
            current = {
                "start_year": row["start_year"],
                "end_year": row["end_year"],
                "num_docs": row["num_docs"],
                "topic": row["topic"],
                "vector_sum": vec * row["num_docs"]
            }

    # finalize last
    if current is not None:
        bary = current["vector_sum"] / current["num_docs"]

        merged_rows.append({
            "start_year": current["start_year"],
            "end_year": current["end_year"],
            "period": current["end_year"] - current["start_year"] + 1,
            "num_docs": current["num_docs"],
            "topic": current["topic"],
            "topic_vector": bary.tolist()
        })

    return pd.DataFrame(merged_rows)