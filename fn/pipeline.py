import os
import pickle
import numpy as np
import tomotopy as tp

from .data_loader import load_docs
from .preprocessing import preprocess_docs
from .dmr import train_dmr, extract_year_topic_dist, build_doc_topic_mapping
from .segmentation import build_dp
from .metrics import compute_total_distortion


# =========================================================
# 1. PREPROCESS + SAVE DATASET
# =========================================================

def preprocess_and_save_dataset(
    raw_path,
    output_path="processed_dataset.pkl"
):
    """
    Input: raw json path
    Output: saved processed dataset (year_groups with tokens)
    """

    year_groups = load_docs(raw_path)

    print("Preprocessing dataset...")
    year_groups = preprocess_docs(year_groups)

    with open(output_path, "wb") as f:
        pickle.dump(year_groups, f)

    print(f"Saved processed dataset to {output_path}")
    return year_groups


def load_processed_dataset(path="processed_dataset.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError("Processed dataset not found.")

    with open(path, "rb") as f:
        return pickle.load(f)


# =========================================================
# 2. TRAIN DMR + SAVE MODEL
# =========================================================

def train_dmr_model(
    dataset_path,
    model_path="models/dmr_model.bin",
    mapping_filename="doc_topic_map.pkl",
    k_topics=20,
    iterations=1000,
    log_filename=None
):

    year_groups = load_processed_dataset(dataset_path)

    print("Training DMR model...")
    model, doc_id_list = train_dmr(
        year_groups,
        k=k_topics,
        iterations=iterations,
        log_filename=log_filename
    )

    os.makedirs("models", exist_ok=True)

    print("Saving DMR model...")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    print("Building doc-topic mapping...")
    doc_topic_map = build_doc_topic_mapping(model, doc_id_list)

    mapping_path = os.path.join("models", mapping_filename)

    import pickle
    with open(mapping_path, "wb") as f:
        pickle.dump(doc_topic_map, f)

    print(f"Mapping saved to {mapping_path}")

    return model, doc_topic_map

def load_dmr_model(model_path="dmr_model.bin"):
    if not os.path.exists(model_path):
        raise FileNotFoundError("DMR model not found.")

    return tp.DMRModel.load(model_path)


# =========================================================
# 3. BUILD SEGMENTS (TIMELINE)
# =========================================================

def build_timeline_from_model(
    dataset_path,
    model_path,
    lambda_penalty=0.1,
    k_topics=20
):
    """
    Input:
        processed dataset + trained DMR
    Output:
        segments + score + full topic vectors
    """

    year_groups = load_processed_dataset(dataset_path)
    model = load_dmr_model(model_path)

    sorted_years = sorted(year_groups.keys())

    print("Extracting topic distributions...")
    year_topic_dist = extract_year_topic_dist(model, year_groups)

    year_distributions = []

    for year in sorted_years:
        dist = year_topic_dist.get(year)

        if dist is None:
            dist = np.zeros(k_topics)

        year_distributions.append({
            "year": year,
            "dist": dist.tolist(),   # <-- convert sang list để serialize dễ hơn
            "docs": year_groups[year]
        })

    print("Running segmentation DP...")
    segments = build_dp(
        [np.array(y["dist"]) for y in year_distributions],  # convert lại khi dùng
        [None for _ in year_distributions],
        lambda_penalty
    )

    score, breakdown = compute_total_distortion(
        [
            {
                "year": y["year"],
                "dist": np.array(y["dist"])   # convert lại cho metrics
            }
            for y in year_distributions
        ],
        segments
    )

    return {
        "segments": segments,
        "year_distributions": year_distributions,
        "sorted_years": sorted_years,
        "total_score": score,
        "score_breakdown": breakdown
    }