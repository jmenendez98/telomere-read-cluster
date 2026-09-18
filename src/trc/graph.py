"""Module 2 of 3: reads become nodes, shared k-mers become weighted edges.

    G = build(reads, k=48, telo_bp=3000, sub_bp=600, n_neighbors=10,
              mix_ratio=0.5)

FOUR STEPS, one function each, and nothing else stands between the k-mer
matrix and the graph module 3 cuts:

    1  normalise   v_i[m] = log1p(tf_i[m]) * idf[m]            `vectors`
                   idf[m] = log(n / df[m])
    2  distance    d(i,j) = || v_i - v_j ||                    `distances`
    3  weight      w(i->j) = exp(-(d(i,j) - rho_i) / sigma_i)  `weights`
                   over read i's `n_neighbors` nearest, and 0 everywhere else
    4  refine      w(i,j) = r*(a + b - a*b) + (1 - r)*a*b        `refine`
                   a = w(i->j), b = w(j->i), r = `MIX_RATIO` = 0.5

Each step reads only the step above it, so a change to one is answerable on
its own: a different vector space changes step 1, a different metric step 2,
and neither can reach into how a distance becomes a weight.

**Two different k's meet here, so they are named once.**  `-k` is the k-mer
LENGTH, which belongs to step 1 and is 48; steps 2 to 4 never read it.  The
`k` written in `log2(k)` below, and in every table in this file, is
`--n-neighbors`, the size of the neighbourhood steps 3 and 4 work over, which
is 10.  A third `k` turns up in the benchmark tables as the number of clusters
a run produced; that one is nobody's parameter.

**The exponential is the whole conversion.**  Step 3 is the only place a
distance becomes a weight; everything else in that line adjusts the distance
before it reaches the `exp` rather than adding a second rule about it.

`rho_i` is read i's distance to its single nearest neighbour, subtracted so
that neighbour sits at exactly zero in the numerator and therefore at weight
1.  Every read is then connected to something no matter how sparse its own
region is, which is the property a flat weight threshold cannot have and is
why one was never found for this data.  The shift is floored at zero, so no
weight exceeds 1.

`sigma_i` is read i's own scale -- small where reads are packed together,
large where they are spread out -- and dividing by it is what makes a weight
mean the same thing at both ends of a pool that holds both.  It is SOLVED FOR
rather than chosen: bisection finds the sigma_i at which read i's own weights
sum to log2(`--n-neighbors`), so every read spends the same total connection
and a read in
a dense arm cannot outvote one in a sparse arm merely by being close to more
things.  `smooth_knn` is that solve, and it is one function so that swapping
it for a closed form -- the mean neighbour distance, or the k-th one -- is a
change in one place.

**Step 4 symmetrises, and `MIX_RATIO` says how generously.**  `w(i->j)` and
`w(j->i)` are different numbers, because rho and sigma belong to the read the
arrow leaves, and the two are mixed the way UMAP mixes them.  At `r = 1` the
edge is the fuzzy UNION `a + b - ab`, the probability that AT LEAST ONE read
claims the other; at `r = 0` it is the fuzzy INTERSECTION `ab`, which keeps
mutual claims and deletes every one-way edge in the graph; at the shipped
`r = 0.5` the product cancels and the edge is the plain MEAN of the two
directions, `(a + b)/2`.  Above 0 it is continuous either way: no pair is
deleted for falling under a threshold, and a pair neither read has in its
`n_neighbors` was never given a weight to lose.

What the mix decides is what a ONE-WAY claim is worth beside a mutual one,
and that is the only place in these four steps where being CHOSEN differs
from choosing.  The union saturates -- two mutual claims of 1 give 1, not 2 --
so a read nobody chose keeps its outgoing weight in full while the pair that
chose each other is capped at what one of them could have claimed alone.  On
the curated sample fifteen such reads, each spending its whole budget on one
chromosome end and a trickle on a second, were enough to fuse two ends that
share no edge with one another at all.  The mean does not saturate: scaled
back up by two, a one-way claim is still `a` where a mutual pair is `a + b`.

**Order is discarded; what is left is composition.**  A k-mer is an edge
because both reads contain it, full stop -- no chaining, no positional
agreement, no requirement that the shared k-mers appear in the same
arrangement in both reads.  That is a choice and not a simplification: the
composition graph is what twenty-six independent community algorithms were
measured to cut identically, and an order-consistency term reinforces a
structure those methods already agree on.

**The rows are not L2-normalised, so step 2 can see read length.**  Two reads
off the same arm that were sequenced to different depths hold different
NUMBERS of the same k-mers, and an unnormalised Euclidean distance counts
that difference as distance.  Normalising the rows first would remove it --
and would make d = sqrt(2 - 2cos), a monotone restatement of the cosine this
module used to ship -- so leaving it in is what makes step 2 a different
question rather than the old one rewritten.  What it costs is that a long
read sits further from everything, which step 3 answers per read: a long
read's rho and sigma are ITS OWN, so its neighbours are still read off its
own profile and not off a scale set by the pool.

**IT COSTS SOMETHING ELSE TOO, AND NORMALISING IS STILL THE WRONG TRADE.**
Since `d^2 = ||a||^2 + ||b||^2 - 2<a,b>`, a pair sharing almost nothing has
`<a,b> ~= 0` and sits at `d ~= sqrt(||a||^2 + ||b||^2)` -- so among reads it
does not resemble, a read's NEAREST are simply the ones with the SMALLEST
NORMS.  On HG08434.LCL-ONT-UL that misplaces a near-duplicate pair of curated
cluster 32 (`a1a792f6`, `7c40d27d`, arrays of ~2 kb where their cluster-mates
run 6.6-9.5 kb).  Their 33 mates share 381-1,518 k-mers with them and sit at
distance ranks 417, 441, 452 ... median 1,137 OF 2,523; the nine reads that
take the rest of their log2(k) budget, and carry them into the wrong cluster,
share 6 to 13 k-mers whose summed IDF is 0.0 -- canonical repeat and nothing
else.

WHICH FORM IS BEST DEPENDS ON MODULE 3, which is why this table has two
halves.  Seven step-1 forms, everything else shipped, scored on the partition
against the curated labels of HG08434.LCL-ONT-UL.  `one cut` is module 3
cutting once and refusing afterwards; `settled` is `cut_until_settled`, which
deletes the refused reads and cuts again until a pass refuses nobody.

                                   ONE CUT                  SETTLED
    step 1                     k  unpl     ARI  caught   k  unpl     ARI  caught
    log1p(tf) * idf   raw     91    17  0.9862  12/12   92    17  0.9991  12/12  SHIPS
    log1p(tf e^idf)   raw     92    14  0.9972  12/12   92    14  0.9972  12/12
    log1p(tf)         raw      -     -       -      -   93    15  0.9954  12/12
    log1p(tf) * idf   unit    91     9  0.9856   5/12   91     9  0.9856   5/12
    log1p(tf e^idf)   unit    91     8  0.9852   5/12   91     8  0.9852   5/12
    presence          unit    90    13  0.9763   3/12   90    13  0.9763   3/12
    log1p(tf)         unit    89    10  0.9339   3/12   90    12  0.9700   3/12

**THE TWO CHANGES ONLY WIN TOGETHER.**  Under one cut the sum form leads by
0.011 and traditional TF-IDF looks like a regression that fuses curated 27, 47
and 62.  But every read joining 47 to 62 was ALREADY REFUSED -- all fourteen of
them -- so the fusion was held together entirely by reads module 3 throws away
a moment too late.  Cut again without them and it splits, the cluster count
lands on the curated 92, and ARI goes 0.9862 -> 0.9991, the best measured on
this sample.  Neither change would have survived being tested alone.

Every UNIT row fuses two curated clusters and the second cut does not save
them, because what fuses them is not a refused read: `a1a792f6` rejoins curated
32 there because curated 32 has absorbed curated 41.  Nothing shattered
afterwards because nothing was left apart to shatter, and the refusal goes with
it -- once two ends are one cluster a bridging read's weight is all internal,
it reads cohesion 1.0, and the rule that catches 12 of 12 catches 3 to 5.

`log1p(tf)` on raw rows is the one worth remembering: settled, it is the only
form with NO fused cluster at all and it still catches 12 of 12, but it cuts
93 clusters where the curators drew 92.  It trades 0.004 of ARI and one extra
split for never merging two ends.

MEASURED WITH `benchmark/row-normalisation/sweep.py` AND CONFIRMED BY A FULL
`python -m trc` RUN PER ROW, except the `log1p(tf)` raw row, which has the
harness only.  Seeding matters: igraph's Leiden draws from Python's `random`
and `cluster.run` does not seed it, so a harness that skips `main.seed_all`
does not reproduce a pipeline run -- and must reseed before EVERY variant, not
once, because each Leiden consumes the state.

A WARNING ABOUT HOW THAT WAS NEARLY MISSED.  Unit rows were chosen on `p@10`,
the share of a read's ten nearest carrying its curated label, which reads
0.9982 unnormalised and 0.9997-1.0000 normalised -- it saturates, and it can
only see a read's own neighbourhood where Leiden reads the whole graph.  A
step-1 change is not answerable on a neighbourhood measure; take it to a
partition.  `benchmark/row-normalisation/sweep.py` is that harness.

AND THE RARE K-MERS ARE NOT THE PROBLEM, which is worth writing down because
they are the obvious suspect: 67.7% of the columns here are in exactly ONE
read, and dropping them changes the unnormalised metric not at all (p@10
0.9982 -> 0.9979, the pair still misplaced) and makes every normalised variant
worse.  A private k-mer costs a row a little mass and nothing else.

What the pair actually needs is not in this module.  Their window is
`[b0-3000, b0+600)` over a ~2 kb array, so it covers their whole array plus 600
bp of subtelomere, while a mate with a 7.5 kb array covers the 3 kb of array
NEAREST the boundary plus the same 600 bp.  The 556 k-mers they do share are
that subtelomere -- the real signal, drowned by two non-overlapping stretches
of array.  That is the window, not the metric and not the kernel.

**Reads with no informative k-mer are not vertices.**  A read scanned into a
window with nothing informative in it has an all-zero vector, so it is at the
same distance from every other read.  A plateau of such reads is perfectly
uniform, and any clusterer returns them as a tight, confident group that
means the exact opposite.  They are held out and counted, never given a
cluster.
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


# ------------------------------------------------------------ 1. normalise
# The two document-frequency gates, both OFF.
#
#     MIN_K_N     a k-mer must be in at least this many reads
#     MAX_K_FRAC  a k-mer in more than this fraction of them is dropped
#
# At 1 and 1.0 neither can bind: the vocabulary is every k-mer the pool
# contains, less the ones every read carries, which `w > 0` drops anyway
# because their IDF is exactly 0 and a zero column cannot move a distance.
#
# What that costs, over the ten curated sets at `-k 48` on the uncapped cosine
# graph with module 3 frozen -- the whole 2x2, because the two gates pull
# opposite ways:
#
# EVERY NUMBER IN THE TABLE BELOW WAS MEASURED ON THE TF-IDF COSINE GRAPH
# THIS MODULE NO LONGER BUILDS.  The gates themselves are step 1 and are
# untouched by steps 2-4, but what they are WORTH is a statement about a graph
# that had no rho, no sigma and no per-read neighbour count, so the table says
# which direction each gate pulled and not what it is worth here.  Re-measuring
# it is a run of the curated sets away and has not been done on this branch.
#
#     MIN_K_N  MAX_K_FRAC      ARI    worst   mean k   columns
#           3         0.5    0.8610   0.7974     80.1     68.5k
#           3         1.0    0.8865   0.8121     80.8     68.7k
#           1         0.5    0.8312   0.7259     78.6    241.9k
#           1         1.0    0.8456   0.7334     79.7    242.1k  <- here
#
# Lifting the ceiling is worth +0.025, dropping the floor -0.030, and the two
# together -0.015 against the pair the parameterisation sweep chose.  The
# curators drew 92 clusters on every one of the ten.
#
# THE CEILING WAS NOT INERT, which has to be said because its flag's own
# documentation said it was: at k = 26 behind a per-node cap it gave byte
# identical tables at every value from 0.2 to 1.0, and here it is worth
# +0.025.  It deletes 150-200 columns out of ~68,500 -- a quarter of one
# percent of the vocabulary -- but they are the columns MOST reads carry, so
# it is ~10% of the incidences and ~120 of each read's ~1,420 of them
# (HG08434.PBMC: 3.9M -> 3.5M, median distinct per read 1,419 -> 1,300).  A
# 48-mer that more than half the pool shares is not the canonical repeat,
# which `w > 0` and `log(n/df)` handle without help; it is subtelomeric
# sequence many arms hold in common, and the cosine that measurement was
# taken on did better seeing it at a weight under 0.7 than not seeing it.
#
# What the ceiling was FOR still stands, and is now unguarded: it is a
# fraction of THE POOL and not of a group, so on a pool pre-filtered to a few
# arms one group can be half the reads and a ceiling of 0.5 deletes exactly
# the k-mers that name it.  At 1.0 nothing can do that.
#
# THE CEILING NEVER BOUND ON A WHOLE-SAMPLE RUN and could not: ~92 groups of
# ~30 reads put the largest at ~2% of the pool, so no k-mer that names a group
# comes near even 0.5.  It was a guard for a pool PRE-FILTERED to a few arms,
# where one group can be half the reads and a ceiling deletes exactly the
# k-mers that identify it.  That case is now unguarded, and it is the one
# place these two values are the wrong ones.
#
# THE FLOOR IS WHAT COSTS.  A k-mer in one read cannot enter any pair's inner
# product, but it IS in that read's L2 norm: admitting the singletons more
# than triples the vocabulary (68.5k -> 242k columns) and divides each of a
# read's similarities by how much private sequence -- at these depths very
# largely basecall error -- that read happens to carry.  At df = 2 the column
# is worse than inert, being one shared error and therefore an edge between
# two reads with nothing else in common.
MIN_K_N = 1
MAX_K_FRAC = 1.0


def kmer_weights(db):
    """Document frequency over READS, and the IDF weight.  `(codes, idf)`.

    The counts are over THIS window's database and no other.  In the array the
    canonical repeat is in nearly every read and lands at weight ~0.01; a
    variant carried by twenty reads lands at ~11.  The separation is three
    orders of magnitude, so the canonical repeat costs almost nothing even when
    it survives: `log1p(tf)` and `log(n/df)` between them put it at ~0.3% of a
    read's vector before any gate sees it.

    THE VOCABULARY IS EVERY K-MER THE POOL CONTAINS.  Both df gates are at
    their inert value (see `MIN_K_N` and `MAX_K_FRAC` above, and what turning
    them off is worth), so the whole df range survives and the only column
    dropped is one that every read carries.  Nothing here picks a middle any
    more; the weighting is left to say which columns matter.

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
    keep = (df >= MIN_K_N) & (df <= MAX_K_FRAC * n) & (w > 0)
    log(f"graph: {codes.size:,} distinct {db.k}-mers, {int(keep.sum()):,} "
        f"informative (in >= {MIN_K_N} reads and <= {MAX_K_FRAC:.0%} of them, "
        f"and not in every one)")
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

        X = counts          raw occurrences of each informative k-mer
        L = log1p(X)        sublinear term frequency, without the idf
        V = log1p(X) * idf  step 1's output -- the rows step 2 measures
                            between, NOT normalised to unit length
        B = X > 0           presence only, which `n_shared` is counted in

    **This is textbook TF-IDF: a PRODUCT of two compressed quantities.**  It
    was not always.  The module used to write `log1p(tf * e^idf)`, folding the
    raw ratio `e^idf = n/df` inside the log, which for any term that is present
    comes out as `log(tf) + idf` -- a SUM, an additive per-column offset.  That
    is much the weaker weighting: at n = 2,523 a canonical repeat column
    (df = 2000, tf = 300) reads 5.94 under the sum and 1.33 under the product,
    and private-to-canonical falls from 4.1x to 1.3x.  The module docstring
    carries what each is worth, and why the answer flipped once module 3
    learned to cut twice.
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
    TF-IDF rows step 2 measures distance between, and presence.

    THE ROWS ARE NOT NORMALISED TO UNIT LENGTH.  A row's norm carries how much
    informative sequence the read had, and step 2 is a Euclidean distance that
    reads it; see the module docstring for what that is for and what it costs.
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
    # `keep` admits only `w > 0`, so every idf here is strictly positive and
    # this diagonal cannot zero an entry -- which is what lets all four value
    # vectors share one sparsity structure.  The check below holds it to that.
    V = sparse.csr_matrix(L @ sparse.diags(idf.astype(np.float32)))
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


# ------------------------------------------------------------- 2. distances
def distances(V):
    """Every pair's Euclidean distance between TF-IDF rows.  Dense, `n x n`.

    Dense on purpose: at a few thousand reads this is tens of MB, and step 3
    needs each read's nearest `n_neighbors` out of every other read anyway.

    Taken from the Gram matrix, since

        ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>

    turns `n^2` distances into one sparse product.  The identity is exact in
    real arithmetic and can land a few ulp below zero for a pair that is very
    nearly identical, so the square is clamped before the root -- that clamp
    and the symmetrising average are the only two places a distance is touched
    after it is computed.
    """
    G = np.asarray((V @ V.T).todense(), np.float64)
    sq = np.diag(G).copy()
    D = sq[:, None] + sq[None, :] - 2.0 * G
    np.maximum(D, 0.0, out=D)
    np.sqrt(D, out=D)
    D = 0.5 * (D + D.T)          # symmetric in exact arithmetic, not bitwise
    np.fill_diagonal(D, 0.0)
    return D


# --------------------------------------------------------------- 3. weights
# The `sigma_i` solve: how many bisection steps, and the floor under what it
# returns.
#
# 64 halvings take the bracket to 2^-64 of its width, which is below the
# precision of the distances going in, so the count is a bound rather than a
# tuned number and the loop is run to the end instead of being given a
# tolerance to stop at.
#
# The floor is what a read gets when the budget is unreachable.  `psum` runs
# from the NUMBER OF NEIGHBOURS TIED AT RHO (at sigma -> 0) up to `k` (at
# sigma -> infinity), so a read with more than `log2(k)` neighbours at exactly
# its own nearest distance has no solution at all: the bisection drives sigma
# to zero, where `0/0` is a NaN rather than a weight.  Floored at a thousandth
# of the read's own mean neighbour distance it gets a very sharp kernel
# instead -- weight 1 on the tied neighbours and ~0 on the rest -- which is
# the right answer for a read whose cliff really is that steep.
#
# Measured on HG08434.LCL-ONT-UL, so the size of this is on the record rather
# than assumed.  At the shipped `--n-neighbors 10` the floor binds for 486 of
# 2,523 reads (19.3%): 293 of them have more than `log2(10) = 3.32` neighbours
# tied at rho and therefore no solution, and the other 193 have one below the
# floor.  At 15 it binds for 444 (17.6%), 305 of them without a solution --
# the budget rises with k faster than the ties do, so a wider neighbourhood
# leaves fewer reads with nothing to solve for.  210 reads have an exact
# duplicate somewhere in the pool -- `rho = 0` -- which is where the ties
# mostly come from, and that count does not depend on k at all.  For every one
# of these reads the far neighbours are at `exp(-large)` either way, so what
# the floor changes is the arithmetic and not the graph.
SMOOTH_ITERS = 64
MIN_SIGMA_FRAC = 1e-3


def neighbours(D, ok, n_neighbors):
    """Each measured read's `n_neighbors` nearest measured reads.

    `(idx, dist)`, both `(n_measured, k)`, each row sorted by increasing
    distance.  `idx` indexes the MEASURED reads -- `np.flatnonzero(ok)` --
    and not the input reads, so a caller maps it back through that array.

    A read is never its own neighbour, and the held-out reads are neither
    neighbours nor given neighbourhoods: they have no informative k-mer, so
    they sit at the same distance from everything and would otherwise be
    every sparse read's nearest.
    """
    m = np.flatnonzero(ok)
    sub = np.array(D[np.ix_(m, m)], np.float64, copy=True)
    np.fill_diagonal(sub, np.inf)
    k = int(min(max(n_neighbors, 1), max(m.size - 1, 1)))
    part = np.argpartition(sub, k - 1, axis=1)[:, :k]
    d = np.take_along_axis(sub, part, axis=1)
    order = np.argsort(d, axis=1, kind="stable")
    return (np.take_along_axis(part, order, axis=1),
            np.take_along_axis(d, order, axis=1))


def smooth_knn(dist, n_iter=SMOOTH_ITERS):
    """`(rho, sigma)` for every read: its nearest distance and its own scale.

    `rho_i` is the first column of `dist` -- read i's distance to its single
    nearest neighbour -- and step 3 subtracts it, which puts that neighbour at
    weight exactly 1 for every read in the pool.

    `sigma_i` is the number at which read i's own weights sum to `log2(k)`,
    where `k` is `--n-neighbors` -- the column count of `dist`, not the k-mer
    length:

        sum_j exp(-max(0, d(i,j) - rho_i) / sigma_i) = log2(k)

    The left side is continuous and strictly increasing in `sigma_i` between
    1 (a kernel so sharp only the nearest neighbour survives) and `k` (one so
    flat every neighbour weighs 1), so for any target in that range there is
    exactly one `sigma_i` and bisection finds it.  Nothing here is fitted or
    swept: the bracket is doubled until it contains the answer and then halved
    `n_iter` times.

    **`log2(k)` is a budget, not a threshold.**  It says every read spends the
    same total connection, so a read in a dense arm cannot outvote one in a
    sparse arm by being close to more things -- the same uniformity a per-node
    edge cap buys, reached by scaling rather than by cutting.  It is the
    target UMAP and Scanpy use, and it is the one constant in this step.

    Swapping the solve for a closed form -- `sigma_i = mean_j(d(i,j) - rho_i)`,
    or `d(i, k-th) - rho_i` -- is a change to this function and to nothing
    else; both are locally adaptive in the same direction and neither holds
    the per-read sum fixed.
    """
    n, k = dist.shape
    rho = np.array(dist[:, 0], np.float64, copy=True)
    shift = np.maximum(dist - rho[:, None], 0.0)
    target = float(np.log2(max(k, 2)))
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    sigma = np.ones(n)
    for _ in range(n_iter):
        psum = np.exp(-shift / sigma[:, None]).sum(axis=1)
        over = psum > target
        hi = np.where(over, sigma, hi)
        lo = np.where(over, lo, sigma)
        # `hi` is infinite only where the bracket's top has never been found,
        # which `over` has just ruled out wherever it is true.
        sigma = np.where(np.isinf(hi), sigma * 2.0, 0.5 * (lo + hi))
    scale = dist.mean(axis=1)
    if not np.all(scale > 0):
        whole = float(dist.mean()) if dist.size else 1.0
        scale = np.where(scale > 0, scale, whole if whole > 0 else 1.0)
    return rho, np.maximum(np.maximum(sigma, MIN_SIGMA_FRAC * scale), 1e-12)


def weights(D, ok, n_neighbors):
    """Step 3: `w(i->j) = exp(-(d(i,j) - rho_i) / sigma_i)`.  Dense `n x n`.

    **The matrix is asymmetric and is meant to be.**  `rho` and `sigma` belong
    to the read the arrow leaves, so `w(i->j)` is read i's claim on j measured
    on i's own scale and says nothing about what j thinks.  Step 4 is where
    the two become one number.

    Everything outside read i's `n_neighbors` nearest is a structural zero
    rather than a small weight.  That is the only cut in the module, and it is
    a cut on COUNT and not on weight: the same `n_neighbors` for every read,
    so a read in a dense region keeps no more edges than one in a sparse
    region, and no pair is ever deleted for the size of its weight.
    """
    n = D.shape[0]
    W = np.zeros((n, n), np.float64)
    m = np.flatnonzero(ok)
    if m.size < 2:
        return W
    idx, dist = neighbours(D, ok, n_neighbors)
    rho, sigma = smooth_knn(dist)
    w = np.exp(-np.maximum(dist - rho[:, None], 0.0) / sigma[:, None])
    W[np.repeat(m, idx.shape[1]), m[idx.ravel()]] = w.ravel()
    log(f"graph: step 3, exp(-(d - rho)/sigma) over each read's "
        f"{idx.shape[1]} nearest  "
        f"(rho median {np.median(rho):.4g}, sigma median {np.median(sigma):.4g}"
        f", weight median {np.median(w):.4f})")
    return W


# ---------------------------------------------------------------- 4. refine
# How far step 4 leans from the fuzzy union `a + b - ab` (1.0) towards the
# fuzzy intersection `a*b` (0.0).  UMAP calls it `set_op_mix_ratio` and ships
# 1.0; it is the one number in step 4.
#
# 0.5 is written here rather than a neighbouring value because the product
# term cancels there and the edge becomes the MEAN of the two directions,
# `(a + b)/2` -- the point at which step 4 stops being a set operation and
# becomes an average.  It is also the middle of the band that works.
#
# Measured on HG08434.LCL-ONT-UL against its curated labels, everything else
# at the shipped default, ARI over the reads the curators placed, at three
# neighbourhoods:
#
#      r      --n-neighbors 8    10 (shipped)      15
#     1.00        0.9952           0.9962        0.9812   <- the union merges
#     0.75        0.9952           0.9962        0.9962
#     0.50        0.9952           0.9962        0.9962
#     0.25        0.9952           0.9962        0.9962
#     0.00        0.2195           0.3229        0.5584   <- another graph
#
# The curators cut 92 clusters, the largest of them 53 reads, which is what
# every cell above 0.99 reproduces.
#
# **At the shipped `--n-neighbors 10` the mix changes nothing on this sample.**
# 0.5 is not buying the 0.9962 there -- 10 reaches it under the pure union too.
# What 0.5 buys is that the answer no longer depends on which neighbourhood it
# is asked at: under the union, 15, 25 and 30 merge a pair of chromosome ends
# that share no edge, and 8, 10, 12 and 20 do not.  It is insurance against a
# k the user picks, not a gain at the k this ships with.
#
# Every r above 0 leaves the edge list the same size and moves only the
# weights, so the band 0.25-0.75 is one graph reweighted three ways and not
# three graphs.  0.00 is a different graph: most edges here are one-way, so
# the intersection deletes three fifths of them at 15 and three quarters at
# 10, stranding 874 and 1,367 reads the curators placed -- mutual-kNN is far
# too strong a claim to ask of this data, and the sparser the neighbourhood
# the worse it gets.
#
# What the mix does NOT do is decide WHO is adjacent.  `--n-neighbors` is
# still the only cut in module 2 and still sets the edge count, the budget
# each read spends and the degree a hub can reach; r only prices the pairs
# that cut leaves behind.  The curve over `--n-neighbors` is the same shape at
# both, and differs at exactly the neighbourhoods where the union merged
# something (`-k` is 48 throughout, as are the two window flags):
#
#     --n-neighbors   ARI r=1.0   ARI r=0.5     edges   max degree
#           4           0.9911      0.9911      9,048        52
#           8           0.9952      0.9952     16,744        52
#          10           0.9962      0.9962     20,038        52
#          12           0.9962      0.9962     23,000        52
#          15           0.9812      0.9962     26,803        52
#          20           0.9863      0.9863     31,884       245
#          25           0.9645      0.9790     37,147       868
#          30           0.9464      0.9555     45,142     1,564
#
# So 0.5 does not widen the useful band from above -- what closes it is the
# hub that appears past 20, which is a property of the neighbourhood and not
# of the mix.  What it removes is the hole AT 15, which is what made 10, 12
# and 15 one flat plateau instead of two points either side of a dip.  The
# shipped `--n-neighbors` is 10: it scores what 12 and 15 score, on a third
# fewer edges and at half the median degree, and it is the end of the plateau
# furthest from that hub.
#
# The support is identical at every r above 0 but for pairs already at the
# bottom of double precision: one pair at k=6, one at k=30 and sixteen at
# k=100 weigh exactly 4.941e-324 under the union, and halving that is zero.
#
# ONE sample.  The ten-set benchmark has not been run on this branch, and
# 1.0's failure there is one merged pair of chromosome ends, not a trend.
MIX_RATIO = 0.5


def _mix_name(r):
    """What `r` is, for the one line that says it out loud."""
    return ("fuzzy union" if r >= 1.0 else
            "fuzzy intersection" if r <= 0.0 else
            "the mean of the two" if r == 0.5 else "union/intersection mix")


def refine(W, ratio=MIX_RATIO):
    """Step 4: the two directions become one edge.

    `w = r*(a + b - a*b) + (1 - r)*a*b`, with `a = w(i->j)`, `b = w(j->i)` and
    `r = ratio`: the fuzzy union at 1, the fuzzy intersection at 0, and at the
    shipped 0.5 their plain mean `(a + b)/2`, the product having cancelled.

    Three properties hold at every `r`:

      * it never sums past what a single claim can be worth, so a pair in the
        densest part of an arm cannot outweigh the arm's own structure;
      * it is continuous.  No pair is dropped for falling under a threshold,
        and a pair neither read listed was never given a weight to lose;
      * below 1 a one-way claim is worth less than a mutual one, which is the
        only thing in these four steps that tells a read something chose from
        a read that merely chose.

    Only at `r = 0` does the support move: a pair one read listed and the
    other did not goes to exactly zero and leaves the edge list.
    """
    P = W * W.T
    out = ratio * (W + W.T - P) + (1.0 - ratio) * P
    np.fill_diagonal(out, 0.0)
    return out


# ----------------------------------------------------------------- the edges
def edge_list(W, ok, claim=None):
    """Every positively weighted pair, `(pairs, weights, claims)`, strongest first.

    `claims[e]` is `(did pairs[e,0] choose this edge, did pairs[e,1])`.  Step 3
    gave a weight only to each read's own `n_neighbors` nearest, so an edge can
    reach a read that never asked for it -- and module 3 has to be able to tell
    the two apart, because a read cannot be held responsible for who chose IT.
    With `claim` left out every edge reads as chosen by both ends.

    One row per undirected edge.  There is no per-node cap and no weight
    threshold here: `W` is already sparse because step 3 gave a weight only to
    each read's `n_neighbors` nearest, and step 4 only ever mixes those.

    SOME OF THESE EDGES CARRY ESSENTIALLY NO WEIGHT, and that is worth knowing
    before reading a degree.  A read whose sigma is small and whose furthest
    listed neighbour is far gets `exp(-large)`: on the curated sample at the
    shipped defaults the weakest exported edge weighs 1.5e-222, and ~9% of the
    edge list weighs under 1e-6.  They cost modularity nothing -- Leiden sums
    weights, and these sum to nothing -- but they are rows in
    `<sample>.edges.tsv` and they count towards a read's degree.  They are
    kept because dropping them would be a threshold, and the one thing this
    module does not do is decide an edge by the size of its weight.
    """
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


def survey(W, ok):
    """What the weights look like, printed before anything reads them.

    A threshold or a cut chosen off a distribution nobody printed is a
    threshold chosen off a hope.
    """
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return {}
    w = W[np.ix_(idx, idx)][np.triu_indices(idx.size, 1)]
    qs = (1, 5, 10, 25, 50, 75, 90, 99)
    pct = {f"p{q}": float(np.percentile(w, q)) for q in qs}
    log("graph: pair weight  " + "  ".join(f"{k}={v:.3g}"
                                           for k, v in pct.items()))
    # Most pairs are at HARD zero -- they are outside both reads' neighbour
    # lists, not close to zero weight -- so the percentiles above are nearly
    # all "0" and say nothing.  The count of zeros and the shape of what is
    # left are the two numbers that do.
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


# ----------------------------------------------------------- the cache form
def weights_arrays(W):
    """The weight matrix on its way to a cache file: sparse, and float64.

    SPARSE, because step 3 gives a weight only to each read's `n_neighbors`
    nearest and the matrix is ~99% structural zero, so the dense form is fifty
    times the file for the same content -- and it is dense `n x n`, which is
    the one thing in this module that grows as the square of the sample.

    FLOAT64, because the kernel reaches genuinely tiny weights.  A read with a
    small sigma whose fifteenth neighbour is far gets `exp(-large)`, and on
    the curated sample the smallest nonzero weight is ~1e-124: in float32
    every weight under ~1e-38 becomes exactly zero, which does not round an
    edge, it DELETES one.  Measured, before this was stored properly: 26,803
    edges fresh and 25,179 restored from the same cache.  A run restored from
    a cache has to be the run that was cached.
    """
    S = sparse.csr_matrix(np.asarray(W, np.float64))
    S.eliminate_zeros()
    return {"data": S.data, "indices": S.indices, "indptr": S.indptr,
            "n": np.array(S.shape[0])}


def weights_from_arrays(z):
    """Back to the dense `n x n` the rest of the module indexes."""
    n = int(z["n"])
    S = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    return np.asarray(S.todense(), np.float64)


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
    claims: np.ndarray              # (n_edges, 2) bool, did that end choose it
    n_shared: np.ndarray
    idf_shared: np.ndarray
    strength: np.ndarray            # per read, summed weight to measured reads
    degree: np.ndarray              # per read, in the exported edge list
    n_informative: np.ndarray       # per read, distinct informative k-mers
    ncol: int
    hashed: bool
    report: dict = field(default_factory=dict)


def to_igraph(pairs, w, ok, ids, claims=None):
    """An `igraph.Graph` over the measured reads.  `(g, vertex_of)`.

    Carries `weight` and, per edge, the two `claim_i`/`claim_j` flags, in the
    order the edges were handed in -- which is the order `get_edgelist()`
    returns them, the same assumption `weight` has always relied on.
    """
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


def build(reads, *, k, telo_bp, sub_bp, n_neighbors, mix_ratio=MIX_RATIO,
          db=None, vec=None, W=None):
    """Module 2 end to end.  `db`, `vec` and `W` may be supplied from a cache.

    Returns `(Graph, db, vec, W)` so a caller can cache what it built.  Nothing
    here decides group membership -- that is module 3 -- and nothing here has
    ever seen a label.
    """
    seqs = [r.seq for r in reads]
    b0 = np.asarray([r.b0 for r in reads], np.int64)
    if db is None:
        db = build_kmerdb(seqs, b0, k=k, telo_bp=telo_bp, sub_bp=sub_bp)
    if vec is None:
        keep, idf = kmer_weights(db)
        vec = vectors(db, keep, idf)
    else:
        log(f"graph: reusing the read/k-mer matrices "
            f"({vec.keep.size:,} informative columns, "
            f"{vec.tfidf.size / 1e6:.1f}M incidences)")
    keep, idf, V, B = vec.keep, vec.idf, vec.V, vec.B
    ok = np.asarray(B.getnnz(axis=1) > 0)
    if W is None:
        n = V.shape[0]
        log(f"graph: step 2, Euclidean distance over "
            f"{n * (n - 1) // 2:,} pairs of TF-IDF rows")
        D = distances(V)
        if n > 1:
            # Guarded, because a pool can come down to a single measurable
            # read and `np.percentile` of no pairs is an IndexError, not a
            # summary line missing.
            iu = np.triu_indices(D.shape[0], 1)
            log(f"graph: distance  p1={np.percentile(D[iu], 1):.4g}  "
                f"p50={np.median(D[iu]):.4g}  max={D[iu].max():.4g}")
            del iu
        W = weights(D, ok, n_neighbors)
        del D
    else:
        log("graph: reusing the directed weights")
    # `W` is step 3's ASYMMETRIC matrix all the way to here, and it is what the
    # caller caches.  Its support is exactly "i chose j", which step 4 destroys
    # by construction and module 3 needs; keeping the directed form is also why
    # `--mix-ratio` no longer costs a rebuild of steps 2 and 3.
    W = np.asarray(W, np.float64)
    claim = W > 0.0
    log(f"graph: step 4, the two directions mixed at r={mix_ratio:g} "
        f"({_mix_name(mix_ratio)})")
    M = refine(W, mix_ratio)
    n_held = int((~ok).sum())
    if n_held:
        log(f"graph: {n_held:,} reads have no informative k-mer and are held "
            f"out of the graph")
    rep = survey(M, ok)

    pairs, w, claims = edge_list(M, ok, claim)
    ns, gs = shared_detail(B, idf, pairs)
    if len(pairs):
        log(f"graph: {len(pairs):,} edges at {n_neighbors} neighbours per read "
            f"(weight median {np.median(w):.3f}, p10 "
            f"{np.percentile(w, 10):.3f}, max {w.max():.3f})")
    else:
        log("graph: no edges")

    # Each read's summed weight to the other MEASURED reads.  Held-out columns
    # are structurally zero, so excluding them changes no number -- but leaving
    # them in would make the quantity depend on how many unscanned reads the
    # sample happened to contain.
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
                "n_neighbors": int(n_neighbors),
                "mix_ratio": float(mix_ratio), "n_vertices": g.vcount(),
                "n_components": len(g.connected_components())})
    return Graph(g=g, ids=ids, vertex_of=vertex_of, ok=ok, pairs=pairs,
                 weights=w, claims=claims,
                 n_shared=ns, idf_shared=gs, strength=strength,
                 degree=deg, n_informative=np.asarray(B.getnnz(axis=1),
                                                      np.int64),
                 ncol=int(keep.size), hashed=db.hashed, report=rep), \
        db, vec, W
