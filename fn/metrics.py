import numpy as np


def jensen_shannon(p, q):
    p = np.array(p)
    q = np.array(q)

    m = 0.5 * (p + q)

    def kl(a, b):
        mask = (a > 0)
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def segment_cost(seg_dists, seg_embs=None):
    """
    seg_dists: list of topic vectors
    """

    bary = np.mean(seg_dists, axis=0)

    cost = 0.0
    for d in seg_dists:
        cost += jensen_shannon(d, bary)

    return cost

def compute_total_distortion(year_distributions, segments):
    total_score = 0.0
    breakdown = []

    for (l, r) in segments:
        seg_dists = [
            year_distributions[i]["dist"]
            for i in range(l, r + 1)
        ]

        cost = segment_cost(seg_dists)

        total_score += cost

        breakdown.append({
            "range": (l, r),
            "start_year": year_distributions[l]["year"],
            "end_year": year_distributions[r]["year"],
            "length": r - l + 1,
            "cost": cost
        })

    return total_score, breakdown