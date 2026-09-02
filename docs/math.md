# Mathematical Foundations

This document derives, rather than asserts, the scoring functions and
complexity claims used throughout `ragforge`. Each section names the
component it backs (`src/ragforge/...`) so the math can be checked against
the implementation directly.

---

## 1. Okapi BM25 (`src/ragforge/index.py::BM25Index`)

BM25 scores a document $d$ against a query $q = \{t_1, \dots, t_n\}$ as:

$$
\text{BM25}(d, q) = \sum_{i=1}^{n} \text{IDF}(t_i) \cdot \frac{f(t_i, d) \cdot (k_1 + 1)}{f(t_i, d) + k_1 \cdot \left(1 - b + b \cdot \dfrac{|d|}{\text{avgdl}}\right)}
$$

**Where this comes from.** BM25 is a pragmatic approximation to the
*Binary Independence Model* from probabilistic information retrieval, which
ranks documents by the log-odds that a document is relevant given the query
terms it contains, under a Naive-Bayes-style independence assumption between
terms:

$$
\text{score}(d, q) \propto \sum_{t \in q} \log \frac{P(t \mid R=1)(1 - P(t \mid R=0))}{P(t \mid R=0)(1 - P(t \mid R=1))}
$$

Robertson & Spärck Jones showed this reduces, under a two-Poisson model of
term frequency, to the IDF term used here:

$$
\text{IDF}(t) = \ln\left(\frac{N - n_t + 0.5}{n_t + 0.5} + 1\right)
$$

which is exactly `BM25Index._idf` — $N$ is the corpus size, $n_t$ the number
of documents containing $t$. The $+0.5$ (Laplace-style) smoothing keeps the
score finite and positive-tending even when $n_t \to N$ (a term in every
document still contributes a bounded, small value rather than
$\ln(0^+) \to -\infty$).

**Why saturate term frequency instead of using it linearly.** Raw
$f(t, d)$ implies "10 occurrences of *paris*" is 10x more relevant than one
occurrence — false in practice (diminishing returns from repetition, and
susceptible to keyword stuffing). The $k_1$ term implements a saturating
response: as $f(t,d) \to \infty$, the fraction $\frac{f \cdot (k_1+1)}{f + k_1 \cdot (\cdot)} \to k_1 + 1$,
a constant. $k_1$ controls *how fast* saturation kicks in — `ragforge`'s
default $k_1 = 1.5$ is the standard corpus-agnostic value from TREC-era
tuning.

**Why length-normalize.** A document twice as long has roughly twice the raw
term counts by chance alone, not because it's twice as relevant. The
normalization factor

$$
B = 1 - b + b \cdot \frac{|d|}{\text{avgdl}}, \qquad b \in [0, 1]
$$

interpolates between no length correction ($b=0$) and full length correction
($b=1$); `ragforge` defaults to $b=0.75$, the empirically robust middle
ground from the original Okapi experiments.

---

## 2. Weighted Reciprocal Rank Fusion (`HybridRetriever`)

BM25 scores are unbounded ($[0, \infty)$) while cosine similarity is bounded
($[-1, 1]$), and neither distribution is Gaussian or even unimodal in
general — so min-max or z-score normalization before summing the two is
fragile: a single outlier chunk with an extreme BM25 score can silently
dominate the fused ranking. RRF sidesteps calibration entirely by discarding
magnitude and fusing **ranks**:

$$
\text{RRF}(d) = \sum_{m \in \{\text{bm25}, \text{vector}\}} \frac{w_m}{k_{\text{rrf}} + \text{rank}_m(d)}
$$

**Why $\frac{1}{k + \text{rank}}$ specifically.** Three properties matter:

1. **Monotonically decreasing in rank** — rank 1 always outscores rank 2,
   regardless of the underlying score gap, so a system's own score scale is
   irrelevant.
2. **Diminishing marginal difference between adjacent ranks** — the gap
   between rank 1 and rank 2 is $\frac{1}{k+1} - \frac{1}{k+2}$, larger than
   the gap between rank 100 and 101. This encodes the belief that being
   first vs. second matters far more than being 100th vs. 101st — matching
   how users actually perceive ranked lists (position-biased attention).
3. **$k_{\text{rrf}}$ controls how "top-heavy" the fusion is.** As
   $k_{\text{rrf}} \to 0$, RRF degenerates toward "only rank 1 matters, all
   else is negligible"; as $k_{\text{rrf}} \to \infty$, RRF's per-rank gaps
   flatten toward a uniform vote (every retrieved document contributes
   $\approx 1/k_{\text{rrf}}$ regardless of exact position). $k_{\text{rrf}} = 60$
   is the standard default from Cormack et al. (2009), chosen because it
   performs well across TREC collections without per-corpus tuning.

**A document appearing in only one list is not penalized to zero** — it
still receives $\frac{w_m}{k_{\text{rrf}} + \text{rank}_m(d)}$ from whichever
list found it — which is precisely the desired hybrid-search behavior: an
exact SKU match found only by BM25 should still surface, even with zero
vector-similarity support.

---

## 3. Maximal Marginal Relevance (`MaxMarginalRelevanceReranker`)

$$
\text{MMR}(d_i) = \lambda \cdot \text{sim}(d_i, q) - (1-\lambda) \max_{d_j \in S} \text{sim}(d_i, d_j)
$$

MMR is a **greedy approximation to a submodular maximization problem**. Define
the *set utility* of a selected context set $S$ (relative to query $q$) as:

$$
U(S) = \lambda \sum_{d \in S} \text{sim}(d, q) - (1-\lambda) \sum_{\{d_i, d_j\} \subset S} \max(\text{sim}(d_i,d_j))
$$

Exactly maximizing $U(S)$ subject to $|S| = k$ is NP-hard (it subsumes
max-coverage). However, the *redundancy penalty* term is a special case of a
**facility-location-like function**, which is submodular (diminishing
returns: adding a chunk to a large, already-diverse set yields less marginal
novelty than adding it to a small set) and monotone under mild conditions.
For submodular set functions, the classical result of Nemhauser, Wolsey &
Fisher (1978) guarantees:

$$
U(S_{\text{greedy}}) \geq \left(1 - \frac{1}{e}\right) U(S_{\text{opt}}) \approx 0.632 \cdot U(S_{\text{opt}})
$$

i.e. the greedy algorithm MMR actually runs — pick the highest-marginal-value
item, add it, repeat — is never worse than 63.2% of the true combinatorial
optimum, for a $O(k \cdot n)$ algorithm instead of the $O(n^k)$ exhaustive
search. `MaxMarginalRelevanceReranker.rerank` is exactly this greedy loop.

**Reading the two limits.** $\lambda=1$ recovers pure top-k-by-relevance
(the redundancy term vanishes, MMR collapses to standard ranking). $\lambda=0$
maximizes diversity alone — the first pick is still the most relevant item
(there's nothing in $S$ yet to be redundant with), but every subsequent pick
actively avoids what's already selected, which can surface answers barely
related to the query. `ragforge`'s default $\lambda=0.7$ keeps relevance
dominant while still discounting near-duplicate chunks.

---

## 4. HNSW Approximate Nearest Neighbor Search (`src/ragforge/ann.py`)

### 4.1 Why brute force isn't enough

`VectorIndex.search` computes cosine similarity against every indexed chunk:
$O(n \cdot d)$ per query for $n$ chunks of dimension $d$. This is exact and
fine up to $n \sim 10^4$–$10^5$ depending on latency budget, but does not
scale to enterprise corpora ($10^6$–$10^9$ chunks) where sub-linear search is
required.

### 4.2 The skip-list intuition

A sorted array supports $O(\log n)$ search via binary search because
comparisons induce a *total order*. Vectors in $\mathbb{R}^d$ have no such
order — but a **skip list** achieves $O(\log n)$ search over a *linked list*
(no random access) by adding shortcut layers: layer $\ell$ contains roughly
$n / 2^\ell$ of the elements, so a search descends $O(\log n)$ layers, each
requiring $O(1)$ amortized hops.

HNSW applies the same idea to a *proximity graph* instead of a sorted
sequence: each vector is assigned a maximum layer

$$
\ell(v) = \lfloor -\ln(U) \cdot m_L \rfloor, \qquad U \sim \text{Uniform}(0,1), \quad m_L = \frac{1}{\ln(m)}
$$

(implemented in `HNSWIndex._random_level`), which is equivalent in
distribution to the skip list's geometric layer assignment: $P(\ell(v) \geq L) = m^{-L}$.
Layer 0 contains every node; each higher layer contains an exponentially
shrinking, random subset, forming long-range "express" edges. A query greedily
descends from the sparse top layer (few hops across large distances) into the
dense bottom layer (many hops across small distances) — this is exactly
`HNSWIndex.search`'s `for lc in range(max_level, 0, -1)` loop before the final
beam search at layer 0.

### 4.3 Expected complexity

Given bounded node degree $M$ (`HNSWIndex.m`), the expected number of layers
is $O(\log n)$, and greedy routing within a well-connected small-world graph
visits $O(\log n)$ nodes per layer under the navigable small-world
assumption (Kleinberg, 2000). This gives:

$$
\text{Expected query cost} = O(M \log n) \quad \text{vs. brute force } O(nd)
$$

For $n = 10^6$, $M=16$: HNSW visits on the order of $16 \times 20 \approx 320$
nodes vs. $10^6$ for brute force — a > 3000x reduction in distance
computations, at the cost of *exactness*: HNSW is a **approximate** nearest
neighbor structure, so `docs/benchmarks.md` reports measured recall@k against
brute force rather than assuming it.

### 4.4 Why greedy search can fail (and why it usually doesn't)

Greedy best-first search on a graph can get stuck in a local optimum: a node
whose neighbors are all farther from the query than itself, even though
better nodes exist elsewhere in the graph. HNSW mitigates this two ways,
both present in `HNSWIndex`:

1. **Beam width `ef` > 1** (`_search_layer`'s `ef` parameter): instead of
   keeping only the single best candidate, the search maintains the best
   `ef` candidates seen so far and only stops once none of the frontier can
   improve on the worst kept result — a bounded-width breadth expansion
   that escapes many local optima a strict greedy walk would not.
2. **Multi-layer entry**: the coarse top layers give the search a good
   starting region before the expensive, fine-grained layer-0 search begins,
   which is far less likely to start in a bad local neighborhood than
   picking a random layer-0 entry point.

### 4.5 Deletion is fundamentally hard, not an oversight

Removing a node's edges outright can partition the graph — if node $v$ was
the *only* bridge between two clusters, deleting it disconnects them
permanently, and no future search can recover reachability. This is why
`HNSWIndex.mark_deleted` and `ApproxVectorIndex.delete` use **lazy
tombstoning**: the node stays in the graph (preserving all paths through it)
but is filtered out of the *result set*. The same tradeoff appears in
production systems (hnswlib's `markDelete`, Qdrant's payload-based
soft-delete) — it is a structural property of graph-based ANN, not a gap
specific to this implementation.

---

## 5. Deterministic Hashing Embeddings (`hashing_embed`)

`hashing_embed` maps text to a fixed-`dims` vector via the *hashing trick*:
each token's CRC32 hash is reduced mod `dims`, and that bucket is
incremented. This is a bag-of-words embedding, not a semantic one — it will
score `"bank" (finance)` and `"bank" (river)` as identical, and it cannot
generalize across synonyms. Its purpose is purely to make the rest of the
pipeline (chunking, hybrid retrieval, reranking, evaluation, serialization)
exercisable **deterministically and without network calls** in tests, CI,
and examples. Two properties matter for that role:

**Determinism across processes.** Python's built-in `hash()` is
randomized per-process (via `PYTHONHASHSEED`) specifically to prevent
hash-flooding denial-of-service attacks on dict-based services — exactly the
opposite of what a reproducible embedding needs. CRC32 has no such salt: the
same string hashes identically on every run, every machine, every Python
version.

**Collision rate.** With `dims` buckets and $t$ distinct tokens hashed
uniformly, the expected number of colliding pairs follows the same
combinatorics as the birthday problem:

$$
E[\text{collisions}] \approx \binom{t}{2} \cdot \frac{1}{\text{dims}} = \frac{t(t-1)}{2 \cdot \text{dims}}
$$

At the library default `dims=128`, a chunk with $t=50$ distinct tokens has
$E[\text{collisions}] \approx \frac{50 \cdot 49}{256} \approx 9.6$ — non-trivial,
which is exactly why `hashing_embed` is documented as a CI/testing utility
and real deployments are expected to pass a trained embedding model (see
`src/ragforge/embeddings_providers.py`) as `embed_fn` instead.

---

## 6. Cosine Similarity as a Bounded, Scale-Invariant Metric

$$
\text{sim}(a, b) = \frac{a \cdot b}{\lVert a \rVert \lVert b \rVert} = \cos\theta_{ab} \in [-1, 1]
$$

Two properties make cosine the right choice for embedding comparison rather
than raw dot product or Euclidean distance:

- **Scale invariance**: $\text{sim}(a, b) = \text{sim}(\alpha a, \beta b)$ for
  any $\alpha, \beta > 0$. Embedding magnitude often reflects incidental
  factors (text length, model-specific norm drift) rather than semantic
  content, so discarding magnitude and comparing only direction is
  desirable.
- **Equivalence to Euclidean distance on the unit sphere**: if
  $\lVert a \rVert = \lVert b \rVert = 1$, then
  $\lVert a - b \rVert^2 = 2 - 2 \cos\theta_{ab} = 2(1 - \text{sim}(a,b))$ —
  so ranking by cosine similarity and ranking by Euclidean distance
  *coincide exactly* once vectors are L2-normalized (`normalize_vector`).
  This is why `hashing_embed` normalizes its output: it lets `HNSWIndex`
  (which internally computes cosine similarity, see `HNSWIndex._sim`) and
  any Euclidean-based ANN structure agree on the same nearest-neighbor
  ordering.
