"""Module 2 of 3: reads become nodes, shared k-mers become weighted edges.

    G = build(reads, k=26, telo_bp=2500, sub_bp=1000, ...)

    v_i[m] = log1p(tf_i[m]) * idf[m]      idf[m] = log(n / df[m])
    w_ij   = <v_i, v_j> / (||v_i|| ||v_j||)              the edge weight
    edges  = the top `edge_k` weights at each node, symmetric union

**Order is discarded; what is left is composition.**  A k-mer is an edge
because both reads contain it, full stop -- no chaining, no positional
agreement, no requirement that the shared k-mers appear in the same arrangement
in both reads.  That is a choice and not a simplification: the composition
graph is what twenty-six independent community algorithms were measured to cut
identically, and an order-consistency term reinforces a structure those methods
already agree on.

**Cosine, not summed rarity.**  A raw sum of shared IDF is the more literal
reading of "weight by how rare the k-mer is", but it is unnormalised -- a 31 kb
read accumulates more of everything than a 4 kb one, so the longest reads
become hubs on length alone.  L2-normalising each read's vector first asks
about composition instead, and a read is then no better connected for being
long.  `n_shared` and `idf_shared` are exported on every edge anyway, so the
unnormalised quantity is one column away.

**Top-k per node, not a weight threshold.**  A single weight cut keeps hundreds
of edges at a dense node and none at a sparse one, which encodes the density
rather than the structure.  Top-k gives every read the same say, and the union
rather than the intersection keeps the edge a peripheral read casts to a core
that does not reciprocate -- which is exactly the case worth seeing.

`--edge-k` has a counting bound and it is the reason the default is small: a
read in a group of `m` reads has only `m - 1` possible in-group partners, so at
`edge_k >= m` it MUST link outside its own group.  The default of 10 was set
against curated groups as small as 11 reads.

**Reads with no informative k-mer are not vertices.**  A read scanned into a
window with nothing informative in it has an all-zero vector, so its cosine
against every other read is exactly 0.  A plateau of such reads is perfectly
uniform, and any clusterer returns them as a tight, confident group that means
the exact opposite.  They are held out and counted, never given a cluster.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse

from .util import log

# Two bits a base, so 2k bits.  At k <= 32 a k-mer packs into a uint64 exactly:
# there are no collisions at all, so a shared column is shared SEQUENCE and not
# a coincidence, and a column can be turned back into the bases it is made of.
# Above 32 that is impossible and the code becomes a 64-bit hash, which gives
# up both properties -- deliberately, and labelled as such wherever a build is
# reported.  The collision arithmetic, so the loss is a number and not a shrug:
# at ~5e6 distinct k-mers in a 2**64 space the expected number of colliding
# pairs is n**2 / 2**65 ~= 7e-7, and a collision adds one spurious shared k-mer
# to one pair out of millions whose weights are sums over hundreds of columns.
EXACT_MAX_K = 32
MAX_K = 51

# A=0 C=1 G=2 T=3, everything else 255 and therefore excluded.
_LUT = np.full(256, 255, np.uint8)
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i

def _mix(x):
    """splitmix64's finaliser.  Avalanches; wraps by construction in uint64."""
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64, copy=True)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def kmer_codes(seq, k):
    """`(codes, positions)` for every ACGT-only k-mer of `seq`, forward strand.

    `positions` indexes each k-mer's FIRST base in `seq`.  A k-mer containing
    any non-ACGT byte is dropped rather than substituted, so an N in a read
    removes the k k-mers that span it and nothing else.

    **Forward strand only.**  Intake has already oriented every read
    telomere-first, which puts them all on the C strand.  Canonicalising each
    k-mer to `min(forward, reverse-complement)` -- what a general-purpose
    sketch does -- would make `CCCTAA` and `TTAGGG` the same feature and throw
    away an orientation module 1 has already established.
    """
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
    # Above 32 bases: cut the window into 32-base chunks, pack each exactly,
    # fold them together with the mixer.  `unpack` no longer inverts this.
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
    """A k-mer code back to its sequence.  Exact builds only (`k <= 32`)."""
    if k > EXACT_MAX_K:
        raise ValueError(f"k={k} codes are hashed and cannot be unpacked")
    return "".join("ACGT"[(int(code) >> (2 * (k - 1 - i))) & 3]
                   for i in range(k))


# ------------------------------------------------------------------ k-mer db
@dataclass
class KmerDB:
    """Every read's k-mers, concatenated, with per-read offsets.

    `off[i]:off[i+1]` is read i's slice.  `lo`/`hi` record the read interval
    the k-mer starts were taken from, so a later stage can tell "this read has
    no k-mer here" from "this read was never read here".
    """
    k: int
    hashed: bool
    code: np.ndarray        # uint64, all reads concatenated
    pos: np.ndarray         # int32, read coordinate of each k-mer's first base
    off: np.ndarray         # int64, length n + 1
    lo: np.ndarray          # int32, first read index scanned
    hi: np.ndarray          # int32, one past the last read index scanned

    @property
    def n(self):
        return len(self.off) - 1

    def read(self, i):
        a, b = int(self.off[i]), int(self.off[i + 1])
        return self.code[a:b], self.pos[a:b]

    def nbytes(self):
        return self.code.nbytes + self.pos.nbytes + self.off.nbytes


def build_kmerdb(seqs, b0, *, k, telo_bp, sub_bp):
    """The k-mer database for one sample.

    ONE window, spanning both sides of the boundary:

        [b0 - telo_bp, b0 + sub_bp)

    with `telo_bp = 0` meaning "back to the read's own tip".

    It is deliberately not selectable.  Scanning either side alone is a subset
    of this window MINUS something: the k-mers straddling the boundary belong
    to neither one-sided window and are read only when the two are scanned
    together.  The junction is where the arms differ most, so those are the
    k-mers least worth giving up, and a one-sided run was never the better
    configuration on any curated sample.

    The array side is capped rather than truncated to a common depth.  A cap
    cannot lengthen a short read's array, so a read with 2 kb of array still
    has 2 kb under any cap; what the cap does is bound the DEEPEST pair, so a
    pooled window is a comparable depth of array against a comparable depth of
    subtelomere.  `pos` stays a READ coordinate throughout, so `lo`/`hi` say
    which interval was scanned.
    """
    codes, poss, off, lo, hi = [], [], [0], [], []
    for s, b in zip(seqs, np.asarray(b0)):
        b = int(b)
        beg = max(0, b - telo_bp) if telo_bp else 0
        end = min(len(s), b + sub_bp)
        beg = min(beg, len(s))
        end = max(end, beg)
        c, p = kmer_codes(s[beg:end], k)
        codes.append(c)
        poss.append(p + np.int32(beg))          # back to a read coordinate
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


# ------------------------------------------------------------------- weights
def kmer_weights(db, *, min_k_n, max_k_frac):
    """Document frequency over READS, and the IDF weight.  `(codes, idf)`.

    The counts are over THIS window's database and no other.  In the array the
    canonical repeat is in nearly every read and lands at weight ~0.01; a
    variant carried by twenty reads lands at ~11.  The separation is three
    orders of magnitude, so the canonical repeat costs almost nothing even when
    it survives: `log1p(tf)` and `log(n/df)` between them put it at ~0.3% of a
    read's vector before any gate sees it.

    The two gates cut either end of the df range.  `min_k_n` removes a k-mer in
    one read, which cannot form an edge at all.  `max_k_frac` removes what most
    of the pool shares -- at the default 0.5, a k-mer in more than half the
    reads, which is the canonical repeat and whatever else every arm carries.
    What is left is the middle, where a chromosome end's identity lives.

    THE CEILING IS A FRACTION OF THE POOL AND NOT OF A GROUP, so it has to stay
    above the largest group's share of the reads: at 0.5, a cluster holding
    more than half the sample loses the very k-mers that define it, and its
    reads are left to be grouped on whatever is under the ceiling.  That is not
    a concern at the ninety-odd ends of a whole-sample run and is one on a
    pool pre-filtered to a few arms.

    The IDF is plain `log(n/df)` with no exponent on it.  An exponent would
    re-weight the columns that survive either way without changing which ones
    do, since `w > 0` reduces to `df < n` for any positive exponent; the
    pipeline does not expose one.
    """
    n = db.n
    per = [np.unique(db.read(i)[0]) for i in range(n)]
    codes, df = np.unique(np.concatenate(per) if n else np.empty(0, np.uint64),
                          return_counts=True)
    w = np.maximum(np.log(np.maximum(n, 1) / np.maximum(df, 1)), 0.0)
    keep = (df >= min_k_n) & (df <= max_k_frac * n) & (w > 0)
    log(f"graph: {codes.size:,} distinct {db.k}-mers, {int(keep.sum()):,} "
        f"informative (in >= {min_k_n} reads and <= {max_k_frac:.0%} of them)")
    return codes[keep], w[keep].astype(np.float32)


def _transform(X):
    """Sublinear term frequency on the occurrence counts: `log1p`.

    A canonical 26-mer occurs hundreds of times in a 2.5 kb array and a variant
    one a handful.  IDF already flattens the canonical column, but among the
    columns that survive, a variant block repeated forty times would otherwise
    outweigh forty distinct one-off variants put together -- and it is the
    second that says which chromosome end this is.
    """
    X = X.copy()
    X.data = np.log1p(X.data).astype(np.float32)
    return X


@dataclass
class Vectors:
    """Every read x k-mer matrix of module 2, over ONE sparsity structure.

    `counts`, `log1p` and `tfidf` are three value vectors on the same
    `(indptr, indices)`, and presence is a fourth that is all ones.  They have
    to share it: multiplying by a positive diagonal cannot create or destroy an
    entry, `keep` admits only `w > 0` so every idf is positive, and
    `log1p(count) >= log 2` for a count that is there at all.  Storing the
    structure once instead of four times is what keeps the cache honest about
    holding every matrix rather than expensive about it.

        X = counts      raw occurrences of each informative k-mer in the window
        L = log1p(X)    sublinear term frequency, before the idf
        V = L * idf     L2-normalised per read -- the rows the cosine is over
        B = X > 0       presence only, which `n_shared` is counted in
    """
    indptr: np.ndarray          # int64, length n + 1
    indices: np.ndarray         # int32, the column of every entry
    counts: np.ndarray          # float32
    log1p: np.ndarray           # float32
    tfidf: np.ndarray           # float32
    keep: np.ndarray            # the informative k-mer codes, one per column
    idf: np.ndarray             # log(n/df) for each of them
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
    """The read x k-mer matrices for one set of informative columns.

    Returns a `Vectors` holding all four: the raw counts, their `log1p`, the
    L2-normalised TF-IDF rows the cosine is taken over, and presence.
    """
    n, ncol = db.n, int(keep.size)
    indptr = np.zeros(n + 1, np.int64)
    cols, vals = [], []
    for i in range(n):
        code, _ = db.read(i)
        if code.size and ncol:
            # `keep` is sorted (np.unique output), so membership is a search
            # rather than a hash of nineteen million keys.
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
    V = _transform(sparse.csr_matrix(
        X @ sparse.diags(np.exp(idf).astype(np.float32))))
    nrm = np.sqrt(np.asarray(V.multiply(V).sum(axis=1)).ravel())
    # A zero row stays a zero row rather than becoming a NaN one: those reads
    # are the held-out plateau the module docstring describes.
    inv = np.where(nrm > 0, 1.0 / np.maximum(nrm, 1e-12), 0.0)
    V = sparse.csr_matrix(sparse.diags(inv.astype(np.float32)) @ V)
    V.sort_indices()
    # Checked and not assumed, because one structure for four value vectors is
    # the whole basis of `Vectors` and of what the cache writes.
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


def similarity(V):
    """The dense read x read cosine, symmetrised and clipped.

    Dense on purpose: at a few thousand reads this is tens of MB, and every
    read needs its top-k against every other anyway.  `V @ V.T` is symmetric in
    exact arithmetic but not bitwise, so the average is taken rather than the
    asymmetry being argued with later.
    """
    S = np.asarray((V @ V.T).todense(), np.float64)
    S = 0.5 * (S + S.T)
    np.clip(S, 0.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    return S


def top_edges(S, ok, edge_k):
    """The `edge_k` strongest edges at each measured node, as a symmetric union.

    Returns `(pairs, weights)` sorted by descending weight, one row per
    undirected edge.
    """
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return np.empty((0, 2), np.int64), np.empty(0, np.float64)
    k = int(min(edge_k, idx.size - 1))
    if k < 1:
        return np.empty((0, 2), np.int64), np.empty(0, np.float64)
    sub = S[np.ix_(idx, idx)].copy()
    np.fill_diagonal(sub, -1.0)
    part = np.argpartition(-sub, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(idx.size), k)
    pairs = np.stack([idx[rows], idx[part.ravel()]], axis=1)
    pairs.sort(axis=1)                          # one row per undirected edge
    pairs = np.unique(pairs, axis=0)
    w = S[pairs[:, 0], pairs[:, 1]]
    order = np.argsort(-w, kind="stable")
    return pairs[order], w[order]


def shared_detail(B, idf, pairs):
    """`(n_shared, idf_shared)` for the exported edges only.

    Per pair rather than as two more n x n products: the products cost billions
    of multiply-adds each and every cell but the exported ones would be thrown
    away.  `B` rows are sorted column ids, so the intersection is a merge.
    """
    ns = np.zeros(len(pairs), np.int64)
    gs = np.zeros(len(pairs), np.float64)
    idf = np.asarray(idf, np.float64)
    rows = [B.indices[B.indptr[i]:B.indptr[i + 1]] for i in range(B.shape[0])]
    for e, (i, j) in enumerate(pairs):
        common = np.intersect1d(rows[i], rows[j], assume_unique=True)
        ns[e] = common.size
        gs[e] = idf[common].sum()
    return ns, gs


def survey(S, ok):
    """What the weights look like, printed before anything reads them.

    A threshold or a cut chosen off a distribution nobody printed is a
    threshold chosen off a hope.
    """
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return {}
    w = S[np.ix_(idx, idx)][np.triu_indices(idx.size, 1)]
    qs = (1, 5, 10, 25, 50, 75, 90, 99)
    pct = {f"p{q}": float(np.percentile(w, q)) for q in qs}
    log("graph: pair weight  " + "  ".join(f"{k}={v:.3f}"
                                           for k, v in pct.items()))
    return {"pair_weight_percentiles": pct}


# --------------------------------------------------------------------- graph
@dataclass
class Graph:
    """The built graph and everything a caller needs to describe it.

    `ok` is over the reads handed in; `g` and `vertex_of` cover only the
    measured ones, because a read with no informative k-mer has no edges by
    construction and would otherwise hand every clusterer a pile of singletons.
    """
    g: object                       # igraph.Graph over the measured reads
    ids: list                       # ids[v] is vertex v's read_id
    vertex_of: np.ndarray           # read index -> vertex, -1 if held out
    ok: np.ndarray                  # read index -> was it measurable
    pairs: np.ndarray               # (n_edges, 2) read indices
    weights: np.ndarray
    n_shared: np.ndarray
    idf_shared: np.ndarray
    strength: np.ndarray            # per read, summed weight to measured reads
    degree: np.ndarray              # per read, in the exported edge list
    n_informative: np.ndarray       # per read, distinct informative k-mers
    ncol: int
    hashed: bool
    report: dict = field(default_factory=dict)


def to_igraph(pairs, w, ok, ids):
    """An `igraph.Graph` over the measured reads.  `(g, vertex_of)`."""
    import igraph as ig
    idx = np.flatnonzero(ok)
    vertex_of = np.full(ok.size, -1, np.int64)
    vertex_of[idx] = np.arange(idx.size)
    edges = [(int(vertex_of[int(i)]), int(vertex_of[int(j)]))
             for i, j in pairs]
    g = ig.Graph(n=int(idx.size), edges=edges)
    g.es["weight"] = [float(x) for x in w]
    g.vs["name"] = [ids[int(i)] for i in idx]
    return g, vertex_of


def build(reads, *, k, telo_bp, sub_bp, min_k_n, max_k_frac,
          edge_k, db=None, vec=None, S=None):
    """Module 2 end to end.  `db`, `vec` and `S` may be supplied from a cache.

    Returns `(Graph, db, vec, S)` so a caller can cache what it built.  Nothing
    here decides group membership -- that is module 3 -- and nothing here has
    ever seen a label.
    """
    seqs = [r.seq for r in reads]
    b0 = np.asarray([r.b0 for r in reads], np.int64)
    if db is None:
        db = build_kmerdb(seqs, b0, k=k, telo_bp=telo_bp, sub_bp=sub_bp)
    if vec is None:
        keep, idf = kmer_weights(db, min_k_n=min_k_n,
                                 max_k_frac=max_k_frac)
        vec = vectors(db, keep, idf)
    else:
        log(f"graph: reusing the read/k-mer matrices "
            f"({vec.keep.size:,} informative columns, "
            f"{vec.tfidf.size / 1e6:.1f}M incidences)")
    keep, idf, V, B = vec.keep, vec.idf, vec.V, vec.B
    if S is None:
        n = V.shape[0]
        log(f"graph: cosine over {n * (n - 1) // 2:,} pairs")
        S = similarity(V)
    else:
        log("graph: reusing the cosine")
    S = np.asarray(S, np.float64)

    ok = np.asarray(B.getnnz(axis=1) > 0)
    n_held = int((~ok).sum())
    if n_held:
        log(f"graph: {n_held:,} reads have no informative k-mer and are held "
            f"out of the graph")
    rep = survey(S, ok)

    pairs, w = top_edges(S, ok, edge_k)
    ns, gs = shared_detail(B, idf, pairs)
    if len(pairs):
        log(f"graph: {len(pairs):,} edges at top-{edge_k} per node "
            f"(weight median {np.median(w):.3f}, p10 "
            f"{np.percentile(w, 10):.3f}, max {w.max():.3f})")
    else:
        log("graph: no edges")

    # Each read's summed weight to the other MEASURED reads.  Held-out columns
    # are structurally zero, so excluding them changes no number -- but leaving
    # them in would make the quantity depend on how many unscanned reads the
    # sample happened to contain.
    m = np.zeros_like(S)
    m[np.ix_(ok, ok)] = S[np.ix_(ok, ok)]
    np.fill_diagonal(m, 0.0)
    strength = m.sum(axis=1)

    deg = np.zeros(len(reads), np.int64)
    if len(pairs):
        np.add.at(deg, pairs[:, 0], 1)
        np.add.at(deg, pairs[:, 1], 1)

    ids = [r.read_id for r in reads]
    g, vertex_of = to_igraph(pairs, w, ok, ids)
    log(f"graph: {g.vcount():,} vertices, {g.ecount():,} weighted edges, "
        f"{len(g.connected_components()):,} connected components")
    rep.update({"n_measured": int(ok.sum()), "n_held_out": n_held,
                "n_informative_kmers": int(keep.size), "n_edges": len(pairs),
                "n_vertices": g.vcount(),
                "n_components": len(g.connected_components())})
    return Graph(g=g, ids=ids, vertex_of=vertex_of, ok=ok, pairs=pairs,
                 weights=w, n_shared=ns, idf_shared=gs, strength=strength,
                 degree=deg, n_informative=np.asarray(B.getnnz(axis=1),
                                                      np.int64),
                 ncol=int(keep.size), hashed=db.hashed, report=rep), \
        db, vec, S
