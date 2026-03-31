# Timeline Generator

This project builds a semantic timeline from a corpus of documents grouped by publication year.
Each year is represented by:

1. A keyword probability distribution.
2. An embedding centroid (mean vector of document embeddings).

The timeline is then segmented into contiguous time intervals using dynamic programming (DP), minimizing a distortion objective with a complexity penalty.

## 1. Problem Formulation

Given an ordered sequence of years

$$
Y = (y_1, y_2, \dots, y_n),
$$

we want to partition it into contiguous segments

$$
\mathcal{S} = \{[l_1, r_1], [l_2, r_2], \dots, [l_K, r_K]\}
$$

such that the total within-segment distortion is minimized while controlling over-segmentation.

The optimization objective implemented in DP is:

$$
\min_{\mathcal{S}} \sum_{[l,r] \in \mathcal{S}} \left(C(l,r) + \lambda\right),
$$

where:

- $C(l,r)$ is the segment distortion cost for years $l..r$.
- $\lambda > 0$ is a regularization penalty (`lambda_penalty`) per segment.

## 2. Year Representation

### 2.1 Keyword Distribution

For a year $t$, let keyword count of term $w$ be $c_t(w)$. The normalized keyword distribution is:

$$
p_t(w) = \frac{c_t(w)}{\sum_{u} c_t(u)}.
$$

In code, keywords are lower-cased and trimmed before counting.

### 2.2 Embedding Centroid

If year $t$ has document embeddings $\{\mathbf{e}_{t,1}, \dots, \mathbf{e}_{t,m_t}\}$, then the yearly embedding is:

$$
\mathbf{\mu}_t = \frac{1}{m_t} \sum_{i=1}^{m_t} \mathbf{e}_{t,i}.
$$

If no embedding is available, the year embedding is treated as `None`.

## 3. Metrics and Distortion

### 3.1 Jensen-Shannon Divergence (Keyword Space)

For two distributions $p$ and $q$, define:

$$
m = \frac{1}{2}(p + q),
$$

$$
\mathrm{KL}(p\|m) = \sum_{w: p(w)>0} p(w)\log\frac{p(w)}{m(w)},
$$

$$
\mathrm{JS}(p,q) = \frac{1}{2}\mathrm{KL}(p\|m) + \frac{1}{2}\mathrm{KL}(q\|m).
$$

Implementation details:

- Union of keyword supports is used.
- Natural logarithm is used internally, then normalized by $\log 2$, yielding JS in bits:

$$
\mathrm{JS}_{\text{bits}}(p,q) = \frac{\mathrm{JS}(p,q)}{\log 2} \in [0,1].
$$

### 3.2 Cosine Distance (Embedding Space)

For vectors $\mathbf{a}, \mathbf{b}$:

$$
\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|_2\,\|\mathbf{b}\|_2},
$$

and the project uses a scaled cosine distance:

$$
d_{\cos}(\mathbf{a},\mathbf{b}) = \frac{1 - \cos(\mathbf{a},\mathbf{b})}{2} \in [0,1].
$$

If one side is missing (`None`), distance defaults to $0$ in the current implementation.

### 3.3 Segment Barycenter and Hybrid Cost

For a segment with yearly distributions $\{p_i\}$ and optional weights $\{w_i\}$, keyword barycenter is:

$$
\bar{p}(w) = \frac{\sum_i w_i p_i(w)}{\sum_i w_i}.
$$

Embedding barycenter is:

$$
\bar{\mathbf{\mu}} = \frac{1}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} \mathbf{\mu}_i,
$$

where $\mathcal{I}$ indexes years with non-null embeddings.

Per-year hybrid distortion inside the segment:

$$
h_i = \frac{1}{2}\,\mathrm{JS}(p_i, \bar{p}) + \frac{1}{2}\,d_{\cos}(\mathbf{\mu}_i, \bar{\mathbf{\mu}}).
$$

Segment cost:

$$
C(l,r) = \sum_{i=l}^{r} w_i h_i.
$$

Notes:

- During DP (`build_dp`), segment cost is computed with uniform weights.
- During final reporting (`compute_total_distortion`), weights are document counts per year (fallback $1$ if empty).

## 4. Dynamic Programming

Let $dp[i]$ be the minimum objective value for the first $i$ years.

Base condition:

$$
dp[0] = 0.
$$

Transition:

$$
dp[i] = \min_{0 \le j < i}\left\{dp[j] + C(j,i-1) + \lambda\right\}.
$$

Backpointer:

$$
\mathrm{prev}[i] = \arg\min_{0 \le j < i}\left\{dp[j] + C(j,i-1) + \lambda\right\}.
$$

Recovered segments are obtained by backtracking from $i=n$ via `prev`.

Complexity:

- States: $n+1$.
- Transitions: $O(n^2)$.
- Segment costs are memoized with cache key $(i,j)$.

## 5. Pipeline Summary

1. Load documents from JSON and group by publication year.
2. Compute per-document embeddings with `all-MiniLM-L6-v2`.
3. Build yearly keyword distributions and yearly embedding centroids.
4. Run DP segmentation with penalty $\lambda$.
5. Compute total distortion and per-segment breakdown.

## 6. Main Outputs

`build_timeline(...)` returns:

- `segments`: list of `(start_index, end_index)` over sorted years.
- `year_distributions`: per-year metadata (`year`, `dist`, `docs`, `embedding`).
- `sorted_years`: ascending year list.
- `total_score`: summed segment distortion.
- `score_breakdown`: per-segment distortion diagnostics.

## 7. Key Hyperparameter

- `lambda_penalty` ($\lambda$): controls the trade-off between fit and number of segments.
	- Larger $\lambda$ -> fewer, broader segments.
	- Smaller $\lambda$ -> more, finer segments.
