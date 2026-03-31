import numpy as np


def jensen_shannon(p, q):
    keys = list(set(p.keys()) | set(q.keys()))

    p_vec = np.array([p.get(k, 0.0) for k in keys])
    q_vec = np.array([q.get(k, 0.0) for k in keys])

    p_vec /= (p_vec.sum() + 1e-12)
    q_vec /= (q_vec.sum() + 1e-12)

    m = 0.5 * (p_vec + q_vec)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))
    js = 0.5 * kl(p_vec, m) + 0.5 * kl(q_vec, m)
    return js / np.log(2)


def cosine_distance(a, b):
    if a is None or b is None:
        return 0.0

    a = np.array(a)
    b = np.array(b)

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    cos_sim = np.dot(a, b) / denom
    cos_dist = (1.0 - cos_sim) / 2.0
    return cos_dist

def segment_cost(segment_dists, segment_embs=None, weights=None):
    if not segment_dists:
        return 0.0

    if weights is None:
        weights = [1] * len(segment_dists)

    total_weight = sum(weights)

    # barycenter keyword
    bary = {}
    for dist, w in zip(segment_dists, weights):
        for k, v in dist.items():
            bary[k] = bary.get(k, 0) + w * v

    for k in bary:
        bary[k] /= total_weight

    # barycenter embedding
    if segment_embs:
        valid_embs = [e for e in segment_embs if e is not None]
        emb_bary = np.mean(valid_embs, axis=0) if valid_embs else None
    else:
        emb_bary = None

    # cost
    cost = 0.0

    for i, (dist, w) in enumerate(zip(segment_dists, weights)):
        js = jensen_shannon(dist, bary)

        if segment_embs and segment_embs[i] is not None and emb_bary is not None:
            cos = cosine_distance(segment_embs[i], emb_bary)
        else:
            cos = 0.0

        hybrid = 0.5 * js + 0.5 * cos
        cost += w * hybrid

    return cost


def compute_total_distortion(year_distributions, segments):
    total_distortion = 0.0
    breakdown = []

    for (l, r) in segments:
        seg_dists = [
            year_distributions[i]["dist"]
            for i in range(l, r + 1)
        ]

        seg_embs = [
            year_distributions[i]["embedding"]
            for i in range(l, r + 1)
        ]

        weights = [
            len(year_distributions[i].get("docs", [])) or 1
            for i in range(l, r + 1)
        ]

        dist_cost = segment_cost(seg_dists, seg_embs, weights)

        total_distortion += dist_cost

        breakdown.append({
            "range": (l, r),
            "distortion": dist_cost,
            "length": (r - l + 1),
            "weight_sum": sum(weights)
        })

    return total_distortion, breakdown