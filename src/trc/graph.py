"""The read graph: windowed k-mers, TF-IDF vectors, cosine distances and
symmetrised nearest-neighbour edges."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse

from .util import log

# k-mers up to this length pack exactly into 64 bits; longer ones are hashed
EXACT_MAX_K = 32
MAX_K = 51

# byte -> 2-bit base code, either case; 255 for anything else
_LUT = np.full(256, 255, np.uint8)
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i

def _mix(x):
    """splitmix64 finaliser over a uint64 array."""
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64, copy=True)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def kmer_codes(seq, k):
    """Code and start of every k-mer in seq that has only ACGT."""
    if k > MAX_K:
        raise SystemExit(f"-k {k} exceeds the ceiling of {MAX_K}")
    if k < 2:
        raise SystemExit(f"-k {k} is too small to be a k-mer")
    if len(seq) < k:
        return np.empty(0, np.uint64), np.empty(0, np.int32)
    b = _LUT[np.frombuffer(seq.encode(), np.uint8)]
    w = sliding_window_view(b, k)
    ok = (w != 255).all(1)
    pos = np.flatnonzero(ok).astype(np.int32)
    if not pos.size:
        return np.empty(0, np.uint64), pos
    win = w[ok].astype(np.uint64)
    if k <= EXACT_MAX_K:
        powers = (np.uint64(1)
                  << (np.uint64(2) * np.arange(k - 1, -1, -1, dtype=np.uint64)))
        with np.errstate(over="ignore"):
            return (win * powers).sum(1), pos
    # hash each 32-bp chunk with its offset and fold it into the code
    code = np.zeros(win.shape[0], np.uint64)
    with np.errstate(over="ignore"):
        for start in range(0, k, EXACT_MAX_K):
            chunk = win[:, start:start + EXACT_MAX_K]
            n = chunk.shape[1]
            powers = (np.uint64(1) << (np.uint64(2)
                                       * np.arange(n - 1, -1, -1,
                                                   dtype=np.uint64)))
            code ^= _mix((chunk * powers).sum(1) + np.uint64(start + 1))
            code = _mix(code)
    return code, pos


def unpack(code, k):
    """Sequence of an exact (unhashed) k-mer code."""
    if k > EXACT_MAX_K:
        raise ValueError(f"k={k} codes are hashed and cannot be unpacked")
    return "".join("ACGT"[(int(code) >> (2 * (k - 1 - i))) & 3]
                   for i in range(k))


@dataclass
class KmerDB:
    """Every read's window k-mers; read i's are [off[i], off[i + 1])."""
    k: int
    hashed: bool
    code: np.ndarray
    # k-mer start in the read
    pos: np.ndarray
    off: np.ndarray
    # each read's window, [lo, hi) in the read
    lo: np.ndarray
    hi: np.ndarray

    @property
    def n(self):
        return len(self.off) - 1

    def read(self, i):
        a, b = int(self.off[i]), int(self.off[i + 1])
        return self.code[a:b], self.pos[a:b]

    def nbytes(self):
        return self.code.nbytes + self.pos.nbytes + self.off.nbytes


def build_kmerdb(seqs, b0, *, k, telo_bp, sub_bp):
    """K-mers of [b0 - telo_bp, b0 + sub_bp) per read, clipped to the read;
    telo_bp 0 starts at the read's tip."""
    codes, poss, off, lo, hi = [], [], [0], [], []
    for s, b in zip(seqs, np.asarray(b0)):
        b = int(b)
        beg = max(0, b - telo_bp) if telo_bp else 0
        end = min(len(s), b + sub_bp)
        beg = min(beg, len(s))
        end = max(end, beg)
        c, p = kmer_codes(s[beg:end], k)
        codes.append(c)
        poss.append(p + np.int32(beg))
        lo.append(beg)
        hi.append(end)
        off.append(off[-1] + c.size)
    db = KmerDB(
        k=k, hashed=k > EXACT_MAX_K,
        code=(np.concatenate(codes) if codes else np.empty(0, np.uint64)),
        pos=(np.concatenate(poss) if poss else np.empty(0, np.int32)),
        off=np.asarray(off, np.int64),
        lo=np.asarray(lo, np.int32), hi=np.asarray(hi, np.int32))
    win = "whole array" if not telo_bp else f"{telo_bp} bp array"
    log(f"graph: k={k}{' (HASHED)' if db.hashed else ''}, {win} + "
        f"{sub_bp} bp subtelomere, "
        f"{db.code.size / 1e6:.1f}M k-mers over {db.n:,} reads "
        f"({db.nbytes() / 1e6:.0f} MB)")
    return db


def kmerdb_arrays(db):
    return {"k": np.array(db.k), "hashed": np.array(int(db.hashed)),
            "code": db.code, "pos": db.pos, "off": db.off,
            "lo": db.lo, "hi": db.hi}


def kmerdb_from_arrays(z):
    return KmerDB(k=int(z["k"]), hashed=bool(int(z["hashed"])), code=z["code"],
                  pos=z["pos"], off=z["off"], lo=z["lo"], hi=z["hi"])


def kmer_weights(db, *, min_df, max_df):
    """Sorted codes of the informative k-mers and their IDF, log(n / df)."""
    n = db.n
    per = [np.unique(db.read(i)[0]) for i in range(n)]
    codes, df = np.unique(np.concatenate(per) if n else np.empty(0, np.uint64),
                          return_counts=True)
    w = np.maximum(np.log(np.maximum(n, 1) / np.maximum(df, 1)), 0.0)
    # w > 0 drops k-mers in every read
    keep = (df >= min_df) & (df <= max_df * n) & (w > 0)
    log(f"graph: {codes.size:,} distinct {db.k}-mers, {int(keep.sum()):,} "
        f"informative (in >= {min_df} reads and <= {max_df:.0%} of them, "
        f"and not in every one)")
    return codes[keep], w[keep].astype(np.float32)


def _transform(X):
    X = X.copy()
    X.data = np.log1p(X.data).astype(np.float32)
    return X


@dataclass
class Vectors:
    """Read x informative-k-mer CSR matrices sharing one sparsity pattern:
    X counts, L log1p, V TF-IDF, B presence."""
    indptr: np.ndarray
    indices: np.ndarray
    counts: np.ndarray
    log1p: np.ndarray
    tfidf: np.ndarray
    # column k-mer codes, sorted
    keep: np.ndarray
    idf: np.ndarray
    n: int

    @property
    def shape(self):
        return (int(self.n), int(self.keep.size))

    def _csr(self, data):
        return sparse.csr_matrix((data, self.indices, self.indptr),
                                 shape=self.shape)

    @property
    def X(self):
        return self._csr(self.counts)

    @property
    def L(self):
        return self._csr(self.log1p)

    @property
    def V(self):
        return self._csr(self.tfidf)

    @property
    def B(self):
        return self._csr(np.ones_like(self.counts))

    def nbytes(self):
        return (self.indptr.nbytes + self.indices.nbytes + self.counts.nbytes
                + self.log1p.nbytes + self.tfidf.nbytes + self.idf.nbytes
                + self.keep.nbytes)


def vectors_arrays(vec):
    return {"indptr": vec.indptr, "indices": vec.indices,
            "counts": vec.counts, "log1p": vec.log1p, "tfidf": vec.tfidf,
            "keep": vec.keep, "idf": vec.idf, "n": np.array(vec.n)}


def vectors_from_arrays(z):
    return Vectors(indptr=z["indptr"], indices=z["indices"],
                   counts=z["counts"], log1p=z["log1p"], tfidf=z["tfidf"],
                   keep=z["keep"], idf=z["idf"], n=int(z["n"]))


def vectors(db, keep, idf):
    """Count each read's informative k-mers; TF-IDF is log1p(count) x idf."""
    n, ncol = db.n, int(keep.size)
    indptr = np.zeros(n + 1, np.int64)
    cols, vals = [], []
    for i in range(n):
        code, _ = db.read(i)
        if code.size and ncol:
            # columns of the read's k-mers that are in keep
            j = np.searchsorted(keep, code)
            np.clip(j, 0, ncol - 1, out=j)
            j = j[keep[j] == code]
        else:
            j = np.empty(0, np.int64)
        if j.size:
            c, cnt = np.unique(j, return_counts=True)
            cols.append(c)
            vals.append(cnt)
            indptr[i + 1] = indptr[i] + c.size
        else:
            indptr[i + 1] = indptr[i]
    idx = np.concatenate(cols) if cols else np.empty(0, np.int64)
    dat = np.concatenate(vals) if vals else np.empty(0, np.int64)
    X = sparse.csr_matrix((dat.astype(np.float32), idx.astype(np.int32),
                           indptr), shape=(n, ncol))
    L = _transform(X)
    V = sparse.csr_matrix(L @ sparse.diags(idf.astype(np.float32)))
    V.sort_indices()
    if not (np.array_equal(V.indices, X.indices)
            and np.array_equal(V.indptr, X.indptr)):
        raise RuntimeError("graph: the TF-IDF rows did not keep the count "
                           "rows' sparsity structure")
    vec = Vectors(indptr=X.indptr, indices=X.indices, counts=X.data,
                  log1p=L.data, tfidf=V.data, keep=keep, idf=idf, n=n)
    med = int(np.median(np.diff(indptr))) if n else 0
    log(f"graph: {ncol:,} informative columns, {V.nnz / 1e6:.1f}M read/k-mer "
        f"incidences (median {med:,} distinct per read), "
        f"{vec.nbytes() / 1e6:.0f} MB of matrices")
    return vec


def distances(V):
    """Dense pairwise cosine distance between rows, on [0, 1]."""
    V = sparse.csr_matrix(V, dtype=np.float64)
    norm = np.sqrt(np.asarray(V.multiply(V).sum(axis=1)).ravel())
    inv = np.divide(1.0, norm, out=np.zeros_like(norm), where=norm > 0)
    U = sparse.diags(inv) @ V
    D = 1.0 - np.asarray((U @ U.T).todense(), np.float64)
    np.clip(D, 0.0, 1.0, out=D)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


# default --max-edge-distance
MAX_EDGE_DISTANCE = 0.1


def neighbours(D, ok, n_neighbors):
    """Each ok read's nearest ok reads, nearest first, as (positions among
    the ok reads, distances)."""
    m = np.flatnonzero(ok)
    sub = np.array(D[np.ix_(m, m)], np.float64, copy=True)
    np.fill_diagonal(sub, np.inf)
    k = int(min(max(n_neighbors, 1), max(m.size - 1, 1)))
    part = np.argpartition(sub, k - 1, axis=1)[:, :k]
    d = np.take_along_axis(sub, part, axis=1)
    order = np.argsort(d, axis=1, kind="stable")
    return (np.take_along_axis(part, order, axis=1),
            np.take_along_axis(d, order, axis=1))


def weights(D, ok, n_neighbors, max_dist=MAX_EDGE_DISTANCE):
    """Directed weights: W[i, j] = 1 - d for each of i's nearest j within
    max_dist, else 0."""
    n = D.shape[0]
    W = np.zeros((n, n), np.float64)
    m = np.flatnonzero(ok)
    if m.size < 2:
        return W
    idx, dist = neighbours(D, ok, n_neighbors)
    w = 1.0 - dist
    log(f"graph: step 3, 1 - d over each read's {idx.shape[1]} nearest  "
        f"(weight median {np.median(w):.4f}, p1 {np.percentile(w, 1):.4f}, "
        f"max {w.max():.4f})")
    far = dist > max_dist
    w[far] = 0.0
    log(f"graph: step 3, {int(far.sum()):,} of {far.size:,} neighbour claims "
        f"({far.mean():.1%}) are further than {max_dist:g} and cut; "
        f"{int(far.all(axis=1).sum()):,} reads lose every one")
    W[np.repeat(m, idx.shape[1]), m[idx.ravel()]] = w.ravel()
    return W


# r in refine: 1 is the fuzzy union, 0 the fuzzy intersection
MIX_RATIO = 0.5


def _mix_name(r):
    return ("fuzzy union" if r >= 1.0 else
            "fuzzy intersection" if r <= 0.0 else
            "the mean of the two" if r == 0.5 else "union/intersection mix")


def refine(W, ratio=MIX_RATIO):
    """Symmetric weights: r(a + b - ab) + (1 - r)ab of the two directions."""
    P = W * W.T
    out = ratio * (W + W.T - P) + (1.0 - ratio) * P
    np.fill_diagonal(out, 0.0)
    return out


def edge_list(W, ok, claim=None):
    """Edges among ok reads, heaviest first, and which end(s) chose each."""
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return (np.empty((0, 2), np.int64), np.empty(0, np.float64),
                np.empty((0, 2), bool))
    sub = np.array(W[np.ix_(idx, idx)], np.float64, copy=True)
    np.fill_diagonal(sub, 0.0)
    a, b = np.nonzero(np.triu(sub, 1) > 0.0)
    pairs = np.stack([idx[a], idx[b]], axis=1)
    w = sub[a, b]
    order = np.argsort(-w, kind="stable")
    pairs, w = pairs[order], w[order]
    if claim is None:
        claims = np.ones((pairs.shape[0], 2), bool)
    else:
        claim = np.asarray(claim, bool)
        claims = np.stack([claim[pairs[:, 0], pairs[:, 1]],
                           claim[pairs[:, 1], pairs[:, 0]]], axis=1)
    return pairs, w, claims


def shared_detail(B, idf, pairs):
    """Per edge, the informative k-mers both reads hold and their IDF sum."""
    ns = np.zeros(len(pairs), np.int64)
    gs = np.zeros(len(pairs), np.float64)
    idf = np.asarray(idf, np.float64)
    rows = [B.indices[B.indptr[i]:B.indptr[i + 1]] for i in range(B.shape[0])]
    for e, (i, j) in enumerate(pairs):
        common = np.intersect1d(rows[i], rows[j], assume_unique=True)
        ns[e] = common.size
        gs[e] = idf[common].sum()
    return ns, gs


def survey(W, ok):
    """Log and return weight percentiles over every pair of ok reads."""
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return {}
    w = W[np.ix_(idx, idx)][np.triu_indices(idx.size, 1)]
    qs = (1, 5, 10, 25, 50, 75, 90, 99)
    pct = {f"p{q}": float(np.percentile(w, q)) for q in qs}
    log("graph: pair weight  " + "  ".join(f"{k}={v:.3g}"
                                           for k, v in pct.items()))
    nz = w[w > 0.0]
    log(f"graph: {int((w <= 0.0).sum()):,} of {w.size:,} pairs weigh exactly "
        f"zero ({(w <= 0.0).mean():.2%})")
    npct = {}
    if nz.size:
        npct = {f"p{q}": float(np.percentile(nz, q)) for q in qs}
        log("graph: nonzero pair weight  "
            + "  ".join(f"{k}={v:.3g}" for k, v in npct.items()))
    return {"pair_weight_percentiles": pct, "n_pairs": int(w.size),
            "n_pairs_zero": int((w <= 0.0).sum()),
            "nonzero_pair_weight_percentiles": npct}


def weights_arrays(W):
    """Directed weights as CSR arrays for the cache."""
    S = sparse.csr_matrix(np.asarray(W, np.float64))
    S.eliminate_zeros()
    return {"data": S.data, "indices": S.indices, "indptr": S.indptr,
            "n": np.array(S.shape[0])}


def weights_from_arrays(z):
    n = int(z["n"])
    S = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    return np.asarray(S.todense(), np.float64)


@dataclass
class Graph:
    """The igraph graph plus per-read and per-edge detail; read arrays are
    indexed by read, not vertex."""
    g: object
    ids: list
    # read -> vertex, -1 for a read held out
    vertex_of: np.ndarray
    # read has an informative k-mer, so is a vertex
    ok: np.ndarray
    # edges as read index pairs
    pairs: np.ndarray
    weights: np.ndarray
    # (pair[0] chose pair[1], pair[1] chose pair[0])
    claims: np.ndarray
    n_shared: np.ndarray
    idf_shared: np.ndarray
    strength: np.ndarray
    degree: np.ndarray
    n_informative: np.ndarray
    ncol: int
    hashed: bool
    report: dict = field(default_factory=dict)


def to_igraph(pairs, w, ok, ids, claims=None):
    """igraph graph of the ok reads with weight and claim edge attributes,
    and the read -> vertex map."""
    import igraph as ig
    idx = np.flatnonzero(ok)
    vertex_of = np.full(ok.size, -1, np.int64)
    vertex_of[idx] = np.arange(idx.size)
    edges = [(int(vertex_of[int(i)]), int(vertex_of[int(j)]))
             for i, j in pairs]
    g = ig.Graph(n=int(idx.size), edges=edges)
    g.es["weight"] = [float(x) for x in w]
    if claims is None:
        claims = np.ones((len(edges), 2), bool)
    g.es["claim_i"] = [bool(x) for x in np.asarray(claims)[:, 0]]
    g.es["claim_j"] = [bool(x) for x in np.asarray(claims)[:, 1]]
    g.vs["name"] = [ids[int(i)] for i in idx]
    return g, vertex_of


def build(reads, *, k, telo_bp, sub_bp, min_df, max_df, n_neighbors,
          max_edge_distance=MAX_EDGE_DISTANCE, db=None, vec=None, W=None):
    """Build the Graph; db, vec and W skip their stage when given.
    Returns (Graph, db, vec, W) so the caller can cache them."""
    seqs = [r.seq for r in reads]
    b0 = np.asarray([r.b0 for r in reads], np.int64)
    if db is None:
        db = build_kmerdb(seqs, b0, k=k, telo_bp=telo_bp, sub_bp=sub_bp)
    if vec is None:
        keep, idf = kmer_weights(db, min_df=min_df, max_df=max_df)
        vec = vectors(db, keep, idf)
    else:
        log(f"graph: reusing the read/k-mer matrices "
            f"({vec.keep.size:,} informative columns, "
            f"{vec.tfidf.size / 1e6:.1f}M incidences)")
    keep, idf, V, B = vec.keep, vec.idf, vec.V, vec.B
    ok = np.asarray(B.getnnz(axis=1) > 0)
    if W is None:
        n = V.shape[0]
        log(f"graph: step 2, cosine distance over "
            f"{n * (n - 1) // 2:,} pairs of TF-IDF rows")
        D = distances(V)
        if n > 1:
            iu = np.triu_indices(D.shape[0], 1)
            log(f"graph: distance  p1={np.percentile(D[iu], 1):.4g}  "
                f"p50={np.median(D[iu]):.4g}  max={D[iu].max():.4g}")
            del iu
        W = weights(D, ok, n_neighbors, max_edge_distance)
        del D
    else:
        log("graph: reusing the directed weights")
    W = np.asarray(W, np.float64)
    # claim[i, j]: i chose j
    claim = W > 0.0
    log(f"graph: step 4, the two directions mixed at r={MIX_RATIO:g} "
        f"({_mix_name(MIX_RATIO)})")
    M = refine(W, MIX_RATIO)
    n_held = int((~ok).sum())
    if n_held:
        log(f"graph: {n_held:,} reads have no informative k-mer and are held "
            f"out of the graph")
    rep = survey(M, ok)

    pairs, w, claims = edge_list(M, ok, claim)
    ns, gs = shared_detail(B, idf, pairs)
    if len(pairs):
        log(f"graph: {len(pairs):,} edges at up to {n_neighbors} neighbours "
            f"per read within {max_edge_distance:g} (weight median "
            f"{np.median(w):.3f}, p10 {np.percentile(w, 10):.3f}, "
            f"max {w.max():.3f})")
    else:
        log("graph: no edges")

    m = np.zeros_like(M)
    m[np.ix_(ok, ok)] = M[np.ix_(ok, ok)]
    np.fill_diagonal(m, 0.0)
    strength = m.sum(axis=1)

    deg = np.zeros(len(reads), np.int64)
    if len(pairs):
        np.add.at(deg, pairs[:, 0], 1)
        np.add.at(deg, pairs[:, 1], 1)

    ids = [r.read_id for r in reads]
    n_one = int((claims.sum(axis=1) == 1).sum())
    if len(pairs):
        log(f"graph: {n_one:,} of {len(pairs):,} edges are one-way "
            f"({n_one / len(pairs):.1%}) -- one read chose them and the other "
            f"did not")
    g, vertex_of = to_igraph(pairs, w, ok, ids, claims)
    log(f"graph: {g.vcount():,} vertices, {g.ecount():,} weighted edges, "
        f"{len(g.connected_components()):,} connected components")
    rep.update({"n_measured": int(ok.sum()), "n_held_out": n_held,
                "n_informative_kmers": int(keep.size), "n_edges": len(pairs),
                "min_df": int(min_df), "max_df": float(max_df),
                "n_neighbors": int(n_neighbors),
                "max_edge_distance": float(max_edge_distance),
                "mix_ratio": float(MIX_RATIO), "n_vertices": g.vcount(),
                "n_components": len(g.connected_components())})
    return Graph(g=g, ids=ids, vertex_of=vertex_of, ok=ok, pairs=pairs,
                 weights=w, claims=claims,
                 n_shared=ns, idf_shared=gs, strength=strength,
                 degree=deg, n_informative=np.asarray(B.getnnz(axis=1),
                                                      np.int64),
                 ncol=int(keep.size), hashed=db.hashed, report=rep), \
        db, vec, W
