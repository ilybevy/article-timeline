from tqdm import tqdm
import numpy as np

from .data_loader import load_docs
from .distribution import build_keyword_distribution
from .segmentation import build_dp
from .metrics import compute_total_distortion


def build_timeline(path, lambda_penalty=0.1):
    year_groups = load_docs(path)

    sorted_years = sorted(year_groups.keys())
    year_distributions = []

    for year in tqdm(sorted_years):
        docs = year_groups[year]

        dist = build_keyword_distribution(docs)

        embs = [
            doc["embedding"]
            for doc in docs
            if doc.get("embedding") is not None
        ]

        year_emb = np.mean(embs, axis=0) if embs else None

        year_distributions.append({
            "year": year,
            "dist": dist,
            "docs": docs,
            "embedding": year_emb
        })

    segments = build_dp(
        [y["dist"] for y in year_distributions],
        [y["embedding"] for y in year_distributions],
        lambda_penalty
    )

    score, breakdown = compute_total_distortion(
        year_distributions,
        segments
    )

    return {
        "segments": segments,
        "year_distributions": year_distributions,
        "sorted_years": sorted_years,
        "total_score": score,
        "score_breakdown": breakdown
    }