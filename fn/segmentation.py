import math
from .metrics import segment_cost


def build_dp(year_dists, year_embs, lambda_penalty=0.1):
    n = len(year_dists)

    cost_cache = {}

    def get_cost(i, j):
        if (i, j) in cost_cache:
            return cost_cache[(i, j)]

        seg_dists = year_dists[i:j+1]
        seg_embs = year_embs[i:j+1]

        cost_cache[(i, j)] = segment_cost(seg_dists, seg_embs)
        return cost_cache[(i, j)]

    dp = [math.inf] * (n + 1)
    prev = [-1] * (n + 1)

    dp[0] = 0

    for i in range(1, n + 1):
        for j in range(i):
            cost = dp[j] + get_cost(j, i - 1) + lambda_penalty
            if cost < dp[i]:
                dp[i] = cost
                prev[i] = j

    # reconstruct
    segments = []
    cur = n

    while cur > 0:
        j = prev[cur]
        segments.append((j, cur - 1))
        cur = j

    segments.reverse()
    return segments