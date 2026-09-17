"""The read browser: one page, three views of one run's reads.

    trc reads.fq.gz -o out/HG08434 --browser
    python3 -m trc.browser out/HG08434 -o page.html

A page for LOOKING AT a decision `trc` already made.  It reads a run directory
and writes one self-contained HTML file -- no server, no network, no install --
with a tab strip over two drawings of the same read set:

    map     the read/k-mer graph laid out by t-SNE.  Reads are nodes, shared
            k-mers are the edges between them, colour is the cluster.
    reads   every read of the input as sequence, in cluster blocks, coloured
            by base, with the window the clustering actually saw marked off.
    stats   one statistic of the reads at a time, a box plot a cluster.

**The two tabs are one selection.**  Click a node on the map and the read is
selected on both; switch to `reads` and the page scrolls to its row.  That is
the whole point of putting them in one file: the map says *this read sits
between two blobs* and the sequence says *and here is why*.

**THE PAGE HOLDS EVERY RECORD OF THE INPUT FILE**, because
`<sample>.reads.tsv` does.  A read that no gate lost is drawn on all three
tabs; a read that one did is drawn on the reads tab and nowhere else, in a
block of its own named for the gate -- `unclustered:low_qs`,
`unclustered:no_array`, `unclustered:short_sub`.  That asymmetry is not a
compromise, it is the point: a dropped read has a SEQUENCE, which is the whole
of what a drop decision can be judged on, and it has no edges, no neighbours
and no position, so a dot on the map or a box in a plot would have to be
invented for it.  A read with no boundary is drawn anchored at its own first
base, which leaves the telomere half of its row empty -- the picture of "no
array was called here".

**THE EMBEDDING IS OF THE GRAPH, NOT OF THE DISTANCES.**  Affinities are built
from `<sample>.edges.tsv` -- the two directed kernels mixed at `--mix-ratio`,
which is what module 3 was handed -- and not from the dense distance matrix
the weights came from.  A read's neighbours here are the ones it was clustered
on, so a read drawn away from its cluster is a statement about the object that
produced the clusters.  It also means the
picture is only as complete as the graph: two reads with no edge between them
have no affinity at all, however similar their vectors were before each read's
`--n-neighbors` cut them apart.  The run's own `n_components` -- 89 on HG08434
at the shipped defaults -- is visible on the map as blobs that never touch.

**t-SNE IS IMPLEMENTED HERE, IN NUMPY.**  `environment.yml` has no
scikit-learn and this page is not worth adding one for: n is a few thousand, so
the O(n^2) gradient is a few seconds a run and the Barnes-Hut approximation
buys nothing.  `tsne()` below is the 2008 algorithm as written -- perplexity by
binary search over each read's own neighbours, early exaggeration, the
adaptive-gain update -- seeded, so the same run directory gives the same
picture every time.  It is not cached: a run builds the page once, and a
page built again from the same run directory is the same page.

**NOTHING HERE IS SCORED AND NOTHING HERE IS WRITTEN BACK.**  No stage of
the pipeline reads anything this module produces: `--browser` writes one file
beside the two result tables and nothing else, and the module is a view of a
decision rather than a step towards one.

**ONE SELECTION, THREE TABS, AND IT CAN BE A SET.**  Clicking a read selects
it everywhere.  Shift+drag on the map takes every read in the square, and
always ADDS -- two lobes of one chromosome end are two squares, and a gesture
that discarded the first when the second was drawn could not say so.  On the
reads tab shift+click takes the rows between the last click and this one and
ctrl+click adds or removes one read.  A plain click, or Escape, clears it.
A selection of one behaves exactly as the selection did when one was all there
could be.

**THE PAGE IS ALSO AN EDITOR, BEHIND A KEYSTROKE.**  Ctrl+Shift+E, confirmed
in a dialog, turns it into a curation tool.  A read dragged onto another read
takes that read's cluster, a read dragged onto a cluster's label joins that
cluster, and reads dropped in the white space BETWEEN two clusters make a new
cluster of their own; `N` makes a new cluster -- the selection's, or an empty
band to drop reads into; `U` sends the selection to `unclustered`; a
right-click offers
`move to...`, ctrl-Z undoes, and `Download CSV` writes
`<sample>.browser.manual.csv` -- one row a read, carrying the run's answer
beside the hand-made one.  A read that is part of a selection moves with the
whole selection, as one edit that one ctrl-Z undoes, and the ghost under the
cursor says how many reads are being carried.

**IN EDIT MODE THE READS TAB'S BLOCKS ARE THE CLUSTERS AS THEY NOW STAND.**  A
read moved to cluster 7 is drawn among the cluster 7 reads, so the tab answers
"what is in this cluster now" rather than "what was".  A read's row is a
function of its assignment and of nothing else -- block order the run's, reads
in the run's order inside it -- so there is no hand-made arrangement to lose
and the same CSV loaded back rebuilds the same page.  The read the gesture was
about keeps the screen line it was on while the rows move around it, and a
block emptied by an edit stays on the page, because putting the reads back is
what someone who emptied it by mistake wants to do next.  The CSV is:

    read_id,cluster,manual_cluster,changed

`Load CSV...` reads that back, matched by read id and never by row order, so a
file can be edited over several sittings or sorted in a spreadsheet between
them.  Nothing about edit mode is on the page until the keystroke, the edits
live in the tab and nowhere else, and the CSV is the only thing that leaves:
this module writes one HTML file and the browser writes one CSV, and no stage
of the pipeline reads either.

**Layout of the read tab** is the truth browser's, deliberately, so the two can
be read side by side: x is `t = b0 - pos`, subtelomere to the LEFT of zero,
telomere to the RIGHT, one shared coordinate down the page, bases coloured
A/C/G/T green/blue/amber/red.  The clustering window `[b0-telo_bp, b0+sub_bp)`
is therefore `t in [-sub_bp, +telo_bp]`, drawn between two dashed rules.
Sequence outside it is drawn faded: it is on the page because a read's context
is worth seeing, and it is faded because no k-mer in it reached the graph.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import os

# One thread per core is not what this wants.  The t-SNE gradient is a rank-2
# update producing an n x n matrix: the arithmetic is trivial and the cost is
# memory traffic, so the last cores bought are nearly free of any return.
# Measured on HG08434 -- 2,563 reads, 20,067 edges, 200 gradient steps -- the
# BLAS at its own default of 64 threads takes 33.2s of wall clock and 217s of
# CPU; at 8 it takes 28.8s and 205s.  Inside the run it was written for -- the
# same pipeline over the same cache, 1,000 steps, the KL trace identical to the
# digit -- it is 4m33s against 2m10s, the gap widening because the gradient is
# competing with the rest of the process for the same memory bandwidth.  Eight
# is both faster and eight, which on a shared machine is the whole argument.
#
# It has to be set before numpy is imported, which is the only time the
# environment is read, and with setdefault so an outer setting still wins.
# That makes this the `python -m trc.browser` half of the cap: inside a `trc
# --browser` run numpy has been up since the pipeline started, and the other
# half is `_blas_threads` below.
BLAS_THREADS = 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(BLAS_THREADS))

import numpy as np                                              # noqa: E402


from .cluster import (CLUSTERER_REASONS, REASON_TEXT,  # noqa: E402
                      UNPLACED_REASONS, unplaced_reason)
from .util import load_seqs, log                  # noqa: E402


# The truth browser's palette, so a read looks the same on both pages.
BASE_RGB = {"A": (61, 168, 83), "C": (66, 133, 244),
            "G": (249, 171, 0), "T": (234, 67, 53), "N": (208, 208, 208)}
GAP_RGB = (250, 250, 250)           # no sequence at this x for this read
GRID_RGB = (224, 224, 224)

# Sequence outside the clustering window, as a fraction of its full colour.
# Small enough to read as "not used", large enough to still see a satellite
# repeat sitting just past the window edge.
FLANK_ALPHA = 0.30

# How far past the window to draw, each side.  The window is 3,500 bp wide at
# the defaults; this puts roughly as much context around it again.
FLANK_BP = 3000

ROW_PX = 6                          # one read's height on the read tab
GAP_ROWS = 2                        # blank rows between two cluster blocks

# The page's own encoding of a cluster, because `cluster.UNCLUSTERED` is a
# sentinel object and `unclustered:<reason>` is a string, and neither goes into
# a typed array.  A real cluster is its own id; -1 is the bare `unclustered` a
# curator can assign in edit mode and that the run itself never writes; -2
# downwards are the reasons, in `UNPLACED_REASONS` order.
#
# ONE ORDERING, TAKEN FROM THE PIPELINE AND NOT FROM THE RUN: the code for
# `low_qs` is the same number on every page, so two pages can be compared and a
# CSV written against one can be loaded into another.
UNCLUSTERED = -1


def reason_code(reason):
    """The page's code for `unclustered:<reason>`."""
    return -2 - UNPLACED_REASONS.index(reason)



# ------------------------------------------------------------------- the run
def _first(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def find_run(outdir, sample=None):
    """The five files this page needs, wherever the run put them.

    `--emit-graph`/`--emit-json` used to write `edges.tsv` and `run.json` into
    `-o`; `--cache DIR` writes them into DIR.  Both layouts are in the tree, so
    both are looked in rather than one being declared the right one -- a page
    that refused to open a run from three weeks ago would be useless for
    exactly the comparison it exists to make.
    """
    outdir = os.path.abspath(outdir)
    if not os.path.isdir(outdir):
        raise SystemExit(f"{outdir} is not a directory -- the argument is a "
                         f"trc run directory, the one its `-o` wrote.")
    if sample is None:
        names = sorted(f[:-len(".reads.tsv")] for f in os.listdir(outdir)
                       if f.endswith(".reads.tsv"))
        if not names:
            raise SystemExit(f"no <sample>.reads.tsv in {outdir}")
        if len(names) > 1:
            raise SystemExit(f"{len(names)} samples in {outdir} "
                             f"({', '.join(names)}) -- name one with --sample")
        sample = names[0]

    cache = os.path.join(outdir, "cache")
    f = {
        "reads": _first(os.path.join(outdir, f"{sample}.reads.tsv")),
        "clusters": _first(os.path.join(outdir, f"{sample}.clusters.tsv")),
        "edges": _first(os.path.join(cache, f"{sample}.edges.tsv"),
                        os.path.join(outdir, f"{sample}.edges.tsv")),
        "json": _first(os.path.join(cache, f"{sample}.run.json"),
                       os.path.join(outdir, f"{sample}.run.json")),
        "intake": _first(os.path.join(cache, "intake.npz")),
        "seqs": _first(os.path.join(cache, "intake.seqs.txt.gz")),
    }
    missing = [k for k in ("reads", "edges", "intake", "seqs") if not f[k]]
    if missing:
        raise SystemExit(
            f"{outdir} is missing {', '.join(missing)}.\n"
            f"This page needs the read table, the edge list and the intake "
            f"cache: the run has to have been made with `--cache DIR` (for "
            f"the edges and the sequence) and DIR has to still be there.")
    return sample, f


def read_tsv(path):
    """A TSV with one header line, as a list of dicts."""
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(hdr, ln.rstrip("\n").split("\t"))) for ln in fh if ln]


def load_reads(path):
    """`<sample>.reads.tsv`: the read set this page draws, and its clusters.

    THE TABLE IS THE PAGE'S READ SET -- and that table is now every record of
    the input file: the reads that were clustered, the reads module 3
    unplaced, and the reads dropped before the graph was built, each under its
    own `unclustered:<reason>`.

    What separates them on the page is not whether they are on it but whether
    they HAVE A GRAPH.  A read dropped before module 2 has no edges, no
    neighbours and no position in the embedding; it is drawn on the reads tab,
    where a sequence needs nothing but itself, and it is absent from the map
    and from the box plots, where every number would have to be invented for
    it.  `ingraph` is that distinction, and it is read off the label rather
    than guessed at from a blank column.

    A table written before the drop reasons existed still opens: a bare
    `unclustered` is read as what it meant then, a read that was in the graph
    and was not placed.
    """
    return _reads_table(read_tsv(path))


def _cell(v, default):
    """One table cell as a number, with `NA` and an empty cell as `default`."""
    if v is None:
        return default
    v = v.strip() if isinstance(v, str) else v
    return default if v in ("", "NA") else v


def _reads_table(rows):
    """The read table as arrays, from the file or from the rows a run has just
    written.  Both ends go through here, so the page's read set is the table's
    by construction rather than by two functions agreeing.
    """
    ids = [r["read_id"] for r in rows]
    cl, why, unknown = [], [], set()
    for r in rows:
        lab = str(r["cluster"])
        w = unplaced_reason(lab)
        if w is None:
            cl.append(int(lab))
            why.append("")
        elif w in UNPLACED_REASONS:
            cl.append(reason_code(w))
            why.append(w)
        else:
            # A reason this page has no code for: drawn as the pool it is in
            # rather than refused, because a page that would not open a table
            # from a later version of the pipeline is a page that cannot be
            # used to compare two versions.
            if w:
                unknown.add(w)
            cl.append(UNCLUSTERED)
            why.append("")
    if unknown:
        log(f"browser: {', '.join(sorted(unknown))} is not a reason this "
            f"page knows; those reads are drawn as unclustered")
    cl = np.array(cl, np.int64)

    def col(name, dtype, default=0):
        return np.array([dtype(_cell(r.get(name), default)) for r in rows],
                        np.float64 if dtype is float else np.int64)

    return {
        "ids": ids, "cluster": cl, "reason": why,
        # A read is in the graph if it was clustered, or if it was in the
        # graph when module 3 declined to place it.
        "ingraph": np.array([c >= 0 or w in CLUSTERER_REASONS or w == ""
                             for c, w in zip(cl, why)], bool),
        "strength": col("strength", float),
        "degree": col("degree", int),
        "nkmer": col("n_informative_kmers", int),
        # 0 where there is no boundary: the read is then drawn from its own
        # first base leftwards, and the telomere half of the window stays
        # empty, which is the picture of a read no array was found in.  The
        # zero is a DRAWING coordinate and not a measurement, so `hasb0` keeps
        # the difference and the page reports NA rather than a boundary at
        # base 0.  It is read off the cell, not inferred from the reason.
        "b0": col("boundary_b0", int),
        "hasb0": np.array([_cell(r.get("boundary_b0"), None) is not None
                           for r in rows], bool),
        "sub_bp": col("sub_bp", int),
        "read_bp": col("read_bp", int),
        "qs": col("qs", float, -1.0),
        "orient": [r.get("orient", "?") for r in rows],
    }


def _edges(triples, pos):
    """(read_i, read_j, weight) -> arrays reindexed onto the page's reads.

    An edge touching a read that is not on the page is dropped and counted
    rather than silently ignored: from a file it means the table and the edge
    list came from different runs, and the count is the only warning of that
    there is.
    """
    i, j, w, dropped = [], [], [], 0
    for ri, rj, wt in triples:
        a, b = pos.get(ri, -1), pos.get(rj, -1)
        if a < 0 or b < 0:
            dropped += 1
            continue
        i.append(a)
        j.append(b)
        w.append(float(wt))
    return (np.array(i, np.int64), np.array(j, np.int64),
            np.array(w, np.float64), dropped)


def load_edges(path, pos):
    """`<sample>.edges.tsv`, reindexed onto the page's reads."""
    def rows():
        with open(path) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            ci, cj, cw = (hdr.index("read_i"), hdr.index("read_j"),
                          hdr.index("weight"))
            for ln in fh:
                f = ln.rstrip("\n").split("\t")
                yield f[ci], f[cj], f[cw]
    return _edges(rows(), pos)


def load_intake(npz_path, seq_path, ids):
    """The cached sequence, KEYED BY READ ID and never by position.

    `trc.intake`'s cache holds every read that survived intake; the read table
    holds the ones that reached the graph.  On HG08434 those two happen to be
    the same 2,592 reads, and pairing them by position would still be a bug
    waiting for the first run where they are not -- every row after the first
    missing read would draw one read's sequence under another read's name, and
    nothing on the page would look wrong.
    """
    z = np.load(npz_path, allow_pickle=True)
    cached = [str(s) for s in z["ids"]]
    seqs = load_seqs(seq_path)
    if len(seqs) != len(cached):
        raise SystemExit(f"{len(seqs)} cached sequences against "
                         f"{len(cached)} cached ids in {npz_path}")
    at = {r: k for k, r in enumerate(cached)}
    out, missing = [], []
    for r in ids:
        k = at.get(r)
        if k is None:
            missing.append(r)
            out.append("")
        else:
            out.append(seqs[k])
    return out, missing


# ----------------------------------------------------------------- the t-SNE
def _binary_search_sigma(d2, target, tol=1e-5, n_iter=60):
    """The precision beta = 1/2sigma^2 whose row entropy is `target`.

    `d2` is one read's squared distances to its own neighbours -- not to every
    read, because this graph only has the neighbours.  Entropy rises with
    sigma, so the search is the textbook bisection on beta with an expanding
    bracket.  A row whose distances are all equal cannot be made to hit any
    particular entropy; it converges to the uniform row, which is the right
    answer for it.
    """
    beta, lo, hi = 1.0, -np.inf, np.inf
    p = None
    for _ in range(n_iter):
        p = np.exp(-d2 * beta)
        s = p.sum()
        if s <= 0.0:
            # Everything underflowed: this beta is far too large.
            hi, beta = beta, beta / 2.0 if lo == -np.inf else (lo + beta) / 2.0
            continue
        h = np.log(s) + beta * float((d2 * p).sum()) / s
        p = p / s
        if abs(h - target) < tol:
            break
        if h > target:                  # too smooth -- sharpen
            lo = beta
            beta = beta * 2.0 if hi == np.inf else (beta + hi) / 2.0
        else:                           # too sharp -- smooth
            hi = beta
            beta = beta / 2.0 if lo == -np.inf else (beta + lo) / 2.0
    return p


def joint_p(n, i, j, w, perplexity):
    """The symmetric affinities P, from the graph's edges and nothing else.

    Each read's row is built over ITS OWN NEIGHBOURS at distance `1 - weight`,
    perplexity-matched the usual way, and the rows are then symmetrised as
    `(P + P')/2n`.  The edge weight is already in [0, 1], so `1 - w` is a
    distance without further scaling.  A pair with no edge gets exactly zero,
    which is the whole difference between this and a t-SNE of the dense
    distances: module 2's neighbour lists have already decided who is allowed
    to attract whom, and this picture is of that decision.
    """
    from scipy import sparse
    d = np.maximum(0.0, 1.0 - w)
    # Both directions, because the run writes each undirected edge once.
    A = sparse.coo_matrix((np.concatenate([d, d]),
                           (np.concatenate([i, j]), np.concatenate([j, i]))),
                          shape=(n, n)).tocsr()
    A.sum_duplicates()
    P = np.zeros((n, n), np.float32)
    target = np.log(perplexity)
    deg = np.diff(A.indptr)
    for r in range(n):
        a, b = A.indptr[r], A.indptr[r + 1]
        if a == b:
            continue
        nb, dd = A.indices[a:b], A.data[a:b]
        P[r, nb] = _binary_search_sigma(dd ** 2, target).astype(np.float32)
    P += P.T
    P /= max(P.sum(dtype=np.float64), 1e-12)
    np.maximum(P, 1e-12, out=P)
    return P, deg


def tsne(P, *, seed=0, n_iter=1000, exaggeration=12.0, exag_iter=250,
         lr=None, momentum=(0.5, 0.8), log_every=200):
    """van der Maaten & Hinton 2008, exactly, in about forty lines.

    The n^2 gradient is computed in full.  At a few thousand reads that is
    ~30 ms an iteration, so Barnes-Hut would save under a minute and cost a
    dependency and an approximation to explain.
    """
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    lr = float(lr or max(n / exaggeration, 50.0))
    Y = (rng.standard_normal((n, 2)) * 1e-4).astype(np.float32)
    upd = np.zeros_like(Y)
    gains = np.ones_like(Y)
    P = P * exaggeration

    # Every n^2 buffer is allocated once and written through `out=` from here
    # on.  Two of them at 2,592 reads is 54 MB; the same two RE-allocated a
    # thousand times is where most of the wall clock went before.
    num = np.empty((n, n), np.float32)
    PQ = np.empty((n, n), np.float32)
    sq = np.empty(n, np.float32)
    rsum = np.empty(n, np.float32)
    pqy = np.empty((n, 2), np.float32)

    for it in range(n_iter):
        if it == exag_iter:
            P /= exaggeration
        # num = 1 / (1 + ||y_i - y_j||^2), the Student-t kernel.
        np.dot(Y, Y.T, out=num)
        np.einsum("ij,ij->i", Y, Y, out=sq)
        num *= np.float32(-2.0)
        num += sq
        num += sq[:, None]
        num += np.float32(1.0)
        np.reciprocal(num, out=num)
        np.fill_diagonal(num, 0.0)
        z = num.sum(dtype=np.float64)

        # PQ = (p_ij - q_ij) * num_ij, all of it float32: at n^2 entries a
        # stray float64 temporary is 54 MB that buys nothing.
        np.multiply(num, np.float32(1.0 / z), out=PQ)
        np.subtract(P, PQ, out=PQ)
        PQ *= num
        np.sum(PQ, axis=1, out=rsum)
        np.dot(PQ, Y, out=pqy)
        grad = rsum[:, None] * Y
        grad -= pqy
        grad *= np.float32(4.0)

        mom = momentum[0] if it < 20 else momentum[1]
        same = np.sign(grad) == np.sign(upd)
        gains[same] *= 0.8
        gains[~same] += 0.2
        np.maximum(gains, 0.01, out=gains)
        upd *= mom
        upd -= lr * gains * grad
        Y += upd
        Y -= Y.mean(axis=0)

        if log_every and (it + 1) % log_every == 0:
            q = np.maximum(num * np.float32(1.0 / z), 1e-12)
            p = P / (exaggeration if it < exag_iter else 1.0)
            kl = float((p * np.log(np.maximum(p, 1e-12) / q)).sum())
            log(f"tsne: iteration {it + 1:>4}  KL {kl:7.4f}")
    return Y.astype(np.float64)


def _blas_threads(n):
    """Cap the BLAS at `n` threads, giving back what it was set to before.

    The environment variables at the top of this module are read when the BLAS
    is loaded, which is the process's first `import numpy` -- so they bite for
    `python -m trc.browser` and not inside a `trc --browser` run, where numpy
    has been up since the pipeline started.  OpenBLAS takes the same
    instruction at runtime, and this is the only way the cap reaches the run
    that needs it most -- worth 33.2s against 28.8s of wall clock and 64
    threads against 8 on the measurement at the top of this file.  A BLAS this
    cannot find still runs, just wide.
    """
    try:
        import ctypes
        path = next(ln.split()[-1] for ln in open("/proc/self/maps")
                    if "libopenblas" in ln)
        lib = ctypes.CDLL(path)
        was = int(lib.openblas_get_num_threads())
        lib.openblas_set_num_threads(int(n))
        return was
    except Exception:                                    # pragma: no cover
        return 0


def embed(reads, edges, *, perplexity, seed, n_iter):
    """The 2-D points, over THE READS THAT HAVE A GRAPH.  Seeded: one edge list
    gives one picture, every time.

    A read dropped before module 2 has no edge to anything, so it is left out
    of the embedding rather than embedded from an empty affinity row -- which
    would not place it nowhere, it would place it in the middle of the picture
    at coordinates that mean nothing and that a viewer would read as a
    position.  Those reads keep (0, 0) and the map never draws them.
    """
    i, j, w, _ = edges
    n = len(reads["ids"])
    keep = np.flatnonzero(reads["ingraph"])
    at = np.full(n, -1, np.int64)
    at[keep] = np.arange(keep.size)
    ok = (at[i] >= 0) & (at[j] >= 0) if len(i) else np.zeros(0, bool)
    log(f"tsne: {int(ok.sum()):,} edges over {keep.size:,} of {n:,} reads, "
        f"perplexity {perplexity:g}")
    xy, deg = np.zeros((n, 2)), np.zeros(n, np.int64)
    was = _blas_threads(BLAS_THREADS)
    try:
        P, d = joint_p(keep.size, at[i[ok]], at[j[ok]], w[ok], perplexity)
        xy[keep] = tsne(P, seed=seed, n_iter=n_iter)
        deg[keep] = d
        return xy, deg
    finally:
        if was:
            _blas_threads(was)


# ------------------------------------------------------------------ the order
def leaf_order(pts):
    """Average-linkage leaf order over 2-D points; identity under three points.

    Used twice: once over the cluster centroids to order the blocks, and once
    inside each block over its reads.  It is a 1-D flattening of the embedding
    and it is not drawn -- see `truth.py` at length on why a tree beside a
    partition is a hazard.  Here it is only deciding which row is above which.
    """
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return list(range(len(pts)))
    from scipy.cluster.hierarchy import leaves_list, linkage
    return [int(k) for k in leaves_list(linkage(pts, method="average"))]


def block_order(cluster, xy, ingraph):
    """Reads in blocks, each block in embedding order.

    Blocks themselves are ordered by the same flattening over their centroids,
    so two clusters the embedding puts beside each other are beside each other
    down the page -- which is where a seam between two ends that should be one,
    or the seam inside one that should be two, is visible without hunting.

    The unclustered blocks go last, one per reason, in `UNPLACED_REASONS`
    order: the reads module 3 looked at and refused, then the ones that never
    got that far, the reads that were never telomeric at all at the bottom of
    the page.  That order is the pipeline's and not the run's, so the blocks
    are in the same order on every page.

    A block whose reads have no embedding is left in TABLE ORDER -- there is
    no layout to flatten, and the table's order is the input file's.
    """
    cids = sorted({int(c) for c in cluster if c >= 0})
    cen = np.array([xy[cluster == c].mean(axis=0) for c in cids]) \
        if cids else np.zeros((0, 2))
    blocks = [(cids[k], np.flatnonzero(cluster == cids[k]))
              for k in leaf_order(cen)]
    # Descending, so -1 (the bare pool an old table carries) comes first and
    # the reasons follow in their own order.
    for c in sorted({int(c) for c in cluster if c < 0}, reverse=True):
        m = np.flatnonzero(cluster == c)
        if m.size:
            blocks.append((c, m))

    order, starts = [], []
    for cid, members in blocks:
        starts.append((cid, len(order), len(members)))
        if ingraph[members].all():
            order.extend(int(members[k]) for k in leaf_order(xy[members]))
        else:
            order.extend(int(m) for m in members)
    return np.array(order, np.int64), starts


# ----------------------------------------------------------------- the window
def pack_windows(seqs, b0, t_lo, t_hi):
    """Every read's drawn window, in ONE buffer, stored left to right in `t`.

    `t = b0 - pos`: the telomere is positive and to the right of zero, the
    subtelomere negative and to the left, which is the sequence pages' shared
    coordinate.  The bases are written in `t` order here, once, rather than the
    page reversing each read as it draws it -- the orientation arithmetic is
    the part of a sequence browser that is easy to get quietly wrong, and doing
    it in one place means a read cannot be drawn backwards on some rows only.

    Returns `(blob, tmin, off, length)`.
    """
    blob = bytearray()
    n = len(seqs)
    tmin = np.zeros(n, np.int64)
    off = np.zeros(n, np.int64)
    length = np.zeros(n, np.int64)
    for k, s in enumerate(seqs):
        if not s:
            off[k], tmin[k] = len(blob), t_lo
            continue
        lo = max(t_lo, b0[k] - (len(s) - 1))
        hi = min(t_hi, b0[k])
        off[k], tmin[k] = len(blob), lo
        if hi < lo:
            continue
        blob += s[b0[k] - hi:b0[k] - lo + 1][::-1].encode()
        length[k] = hi - lo + 1
    return bytes(blob), tmin, off, length


# ----------------------------------------------------------------- the payload
_DT = {"int8": "i1", "uint8": "u1", "int16": "i2", "int32": "i4",
       "float32": "f4"}


class Blob:
    """Named arrays into one buffer, with a manifest the page slices it by.

    Every array on the page goes through here rather than into JSON.  The edge
    list alone is 20,353 triples: as JSON that is half a megabyte of decimal
    text to parse at load, and as three typed arrays it is 244 kB that the
    browser maps in one go.
    """

    def __init__(self):
        self.buf = bytearray()
        self.man = []

    def add(self, name, arr, dtype=None):
        a = np.ascontiguousarray(arr, dtype or arr.dtype)
        if a.dtype.name not in _DT:
            raise ValueError(f"{name}: {a.dtype} is not a page dtype")
        self.man.append({"name": name, "dt": _DT[a.dtype.name],
                         "off": len(self.buf), "n": int(a.size)})
        self.buf += a.tobytes()
        return self

    def add_text(self, name, lines):
        b = "\n".join(lines).encode()
        self.man.append({"name": name, "dt": "txt",
                         "off": len(self.buf), "n": len(b)})
        self.buf += b
        return self


def gz_b64(data):
    """gzip, then base64, for a <script> literal.

    compresslevel 9 here and not `save_seqs`' 4: this runs once per page and
    the bytes are then copied around as an email attachment, where a third off
    the size is worth the seconds.

    `mtime=0` because gzip otherwise stamps the hour into its header, and two
    pages built from one run directory would differ in those four bytes and in
    every base64 character after them -- which is the difference between "the
    same page" being checkable and being asserted.
    """
    return base64.b64encode(
        gzip.compress(bytes(data), 9, mtime=0)).decode()


# -------------------------------------------------------------------- the page
CSS = """
:root{--fg:#1b1b1b;--mut:#666;--dim:#999;--line:#e0e0e0;--bg:#fff;
      --sel:#2b6cb0;--warn:#c0392b;--panel:#fbfbfb}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",
     Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);
     display:flex;flex-direction:column;overflow:hidden}
header{padding:9px 14px 0;border-bottom:1px solid var(--line);flex:0 0 auto}
h1{font-size:14px;margin:0 0 8px;font-weight:600}
h1 small{font-weight:400;color:var(--mut);margin-left:8px;
         font:12px ui-monospace,Menlo,Consolas,monospace}
#tabs{display:flex;gap:2px;align-items:flex-end}
.tab{padding:5px 15px;border:1px solid var(--line);border-bottom:none;
     border-radius:5px 5px 0 0;background:var(--panel);cursor:pointer;
     color:var(--mut);font-size:12.5px;position:relative;top:1px}
.tab.on{background:var(--bg);color:var(--fg);font-weight:600}
#tabs .grow{flex:1}
#find{border:1px solid #d8d8d8;border-radius:4px;padding:3px 7px;width:210px;
      font:11.5px ui-monospace,Menlo,Consolas,monospace;margin-bottom:4px}
#panels{flex:1 1 auto;position:relative;min-height:0}
.panel{position:absolute;inset:0;display:flex;flex-direction:column}
.panel[hidden]{display:none}
.ctl{flex:0 0 auto;display:flex;gap:14px;align-items:center;flex-wrap:wrap;
     padding:6px 14px;border-bottom:1px solid var(--line);
     font-size:11.5px;color:var(--mut);background:var(--panel)}
.ctl label{display:inline-flex;gap:5px;align-items:center;cursor:pointer}
.ctl input[type=range]{width:110px;vertical-align:middle}
.ctl b{font:11.5px ui-monospace,Menlo,Consolas,monospace;color:var(--fg);
       min-width:3.2em;display:inline-block}
button{font:inherit;font-size:11.5px;padding:2px 9px;border:1px solid #d0d0d0;
       border-radius:4px;background:#fff;cursor:pointer;color:var(--fg)}
button:hover{background:#f2f2f2}
.ctl select{font:inherit;font-size:11.5px;padding:1px 4px;border-radius:4px;
            border:1px solid #d0d0d0;background:#fff;color:var(--fg)}
.cv{flex:1 1 auto;min-height:0;position:relative}
#mapCv{display:block;width:100%;height:100%;cursor:grab}
#mapCv.drag{cursor:grabbing}
#readsScroll,#statsScroll{position:absolute;inset:0;overflow-y:auto;
     overflow-x:hidden}
#readsCv,#statsCv{display:block;position:sticky;top:0;z-index:1;width:100%}
#tip{position:fixed;pointer-events:none;background:#111;color:#fff;
     padding:6px 9px;border-radius:4px;font:11px ui-monospace,Menlo,Consolas,
     monospace;line-height:1.5;white-space:pre;z-index:9;display:none;
     max-width:46ch}
#boot{position:fixed;left:20px;bottom:20px;color:var(--mut);font-size:12px}
.key{display:inline-flex;gap:4px;align-items:center}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
/* edit mode: unmistakable, because the page now changes labels */
body.editing header{background:#fff8f0;box-shadow:inset 0 3px 0 #d9822b}
#editbar{display:flex;gap:10px;align-items:center;padding:5px 14px;
         background:#fff3e4;border-bottom:1px solid #f0d5b4;font-size:11.5px;
         color:#8a4b0d;flex:0 0 auto}
#editbar[hidden]{display:none}
#editbar b{font:11.5px ui-monospace,Menlo,Consolas,monospace}
#editbar .grow{flex:1}
#editbar .hint{color:#a97546}
#editmsg{color:#1f6f43;font-weight:600}
#newcl{border-color:#d9822b;color:#8a4b0d;background:#fff}
#newcl:hover{background:#ffeeda}
#dlcsv{border-color:#d9822b;color:#8a4b0d;background:#fff}
#dlcsv:hover{background:#ffeeda}
body.editing #mapCv,body.editing #readsCv{cursor:crosshair}
/* the read being dragged, and the read it would land on */
#ghost{position:fixed;pointer-events:none;z-index:10;
       background:#fff;border:1px solid #d9822b;border-radius:4px;
       padding:3px 7px;font:11px ui-monospace,Menlo,Consolas,monospace;
       box-shadow:0 2px 8px rgba(0,0,0,0.18);white-space:pre}
#ghost[hidden]{display:none}
#ghost .sw{margin-right:5px;vertical-align:-1px}
#ghost.ok{border-color:#2f855a;background:#f0fff4;color:#22543d}
/* the confirmation, and the file report */
#modal{position:fixed;inset:0;background:rgba(20,20,20,0.34);z-index:20;
       display:flex;align-items:center;justify-content:center}
#modal[hidden]{display:none}
#mcard{background:#fff;border-radius:8px;padding:18px 20px 15px;width:430px;
       box-shadow:0 10px 40px rgba(0,0,0,0.3);font-size:12.5px;line-height:1.5}
#mtitle{font-size:14px;font-weight:600;margin-bottom:8px}
#mbody{color:#444}
#mbtns{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
#myes{border-color:#2b6cb0;color:#1a4d80}
#myes.danger{border-color:#c0392b;color:#c0392b}
/* move to... */
#cmenu{position:fixed;z-index:30;width:214px;max-height:360px;overflow:auto;
       background:#fff;border:1px solid #d0d0d0;border-radius:6px;
       box-shadow:0 6px 24px rgba(0,0,0,0.22);font-size:11.5px;padding:4px 0}
#cmenu[hidden]{display:none}
#cmenu .mh{padding:6px 10px;border-bottom:1px solid #eee;
           font:11px ui-monospace,Menlo,Consolas,monospace;color:#333}
#cmenu .mh span{color:#999}
#cmenu .ms{padding:6px 10px 3px;color:#999;font-size:10.5px;
           text-transform:uppercase;letter-spacing:0.04em}
#cmenu .mi{display:flex;gap:7px;align-items:center;padding:3px 10px;
           cursor:pointer}
#cmenu .mi:hover{background:#eef4fb}
#cmenu .mi.on{background:#f3f3f3}
#cmenu .mi.on .mn{font-weight:600}
#cmenu .mn{flex:1}
#cmenu .mx{color:#aaa;font-size:10.5px}
#mfilt{width:calc(100% - 20px);margin:2px 10px 4px;padding:3px 6px;
       border:1px solid #ddd;border-radius:4px;
       font:11px ui-monospace,Menlo,Consolas,monospace}
"""

# The page's own JavaScript.  It is here rather than in a .js file because the
# page has to open from a Downloads folder with no server behind it.
JS = r"""
const DPR = Math.min(2, window.devicePixelRatio || 1);
const $ = s => document.querySelector(s);
let A = null, SEQ = null;            // the unpacked arrays and the sequence
let N = 0, M = 0;                    // reads, edges
let sel = -1, hov = -1;

// ---------------------------------------------------------------- unpacking
async function gunzip(b64){
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) u8[i] = bin.charCodeAt(i);
  const s = new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}
function unpack(bytes, man){
  const out = {}, ab = bytes.buffer, b0 = bytes.byteOffset;
  for(const m of man){
    const a = b0 + m.off;
    if(m.dt === 'txt')
      out[m.name] = new TextDecoder().decode(
        bytes.subarray(m.off, m.off + m.n)).split('\n');
    else if(m.dt === 'i4') out[m.name] = new Int32Array(ab.slice(a, a + 4*m.n));
    else if(m.dt === 'f4') out[m.name] = new Float32Array(ab.slice(a, a+4*m.n));
    else out[m.name] = new Uint8Array(ab.slice(a, a + m.n));
  }
  return out;
}

// ------------------------------------------------------------------- colours
// A cluster's colour is a function of its id and of nothing else, so the same
// cluster is the same colour on both tabs and stays that colour when a later
// run renumbers its neighbours.  The golden angle keeps 92 of them apart.
function ccol(c, l){ return c < 0 ? 'hsl(210,6%,72%)'
                     : 'hsl(' + ((c*137.508)%360).toFixed(1) + ',62%,' +
                       (l||48) + '%)'; }
const LR = new Uint8Array(256).fill(208), LG = new Uint8Array(256).fill(208),
      LB = new Uint8Array(256).fill(208);
for(const [b,c] of Object.entries(D.base))
  for(const ch of [b, b.toLowerCase()]){
    const k = ch.charCodeAt(0); LR[k]=c[0]; LG[k]=c[1]; LB[k]=c[2];
  }

// ----------------------------------------------------------------- the names
// `unclustered` is not one answer, it is ten, and a code says which: a real
// cluster is its own id, -1 is the bare pool (what a curator assigns in edit
// mode, and all an older run's table could say), and -2 downwards are
// D.reasons in the pipeline's own order.  One place turns a code into a
// string, so the gutter, the tooltip, the menu and the CSV cannot disagree
// about what a read is labelled -- and the CSV writes the same word the run's
// own reads.tsv does.
const UNCL = -1;
const RNAME = c => (c >= 0 || c === UNCL) ? '' : (D.reasons[-2 - c] || '');
const CNAME = c => c >= 0 ? String(c)
                 : (RNAME(c) ? 'unclustered:' + RNAME(c) : 'unclustered');
const CLABEL = c => c >= 0 ? 'cluster ' + c : CNAME(c);
const RWHY = c => D.reasonText[RNAME(c)] || '';
// 'unclustered:low_qs', 'unclustered' or '7', back to a code.  NaN is "this
// is not a label", which only a loaded CSV can produce.
function codeOf(raw){
  if(raw.slice(0, 11) === 'unclustered'){
    const w = raw.slice(11).replace(/^:/, '');
    const k = w ? D.reasons.indexOf(w) : -1;
    return k < 0 ? UNCL : -2 - k;
  }
  const c = parseInt(raw, 10);
  return Number.isFinite(c) ? c : NaN;
}

// ---------------------------------------------------------------- the tabs
let tab = 'map';
function show(t){
  tab = t;
  for(const el of document.querySelectorAll('.tab'))
    el.classList.toggle('on', el.dataset.tab === t);
  $('#mapPanel').hidden   = t !== 'map';
  $('#readsPanel').hidden = t !== 'reads';
  $('#statsPanel').hidden = t !== 'stats';
  if(t === 'map'){ sizeMap(); drawMap(); }
  else if(t === 'stats'){
    sizeStats(); if(sel >= 0) scrollToStats(sel); drawStats();
  }
  else { sizeReads(); if(sel >= 0) scrollToRead(sel); drawReads(); }
}

// ------------------------------------------------------------- the selection
// ONE SELECTION ACROSS THREE TABS, and a SET rather than a read.  `sel` is
// still the read the page reports on and scrolls to -- the last one added --
// and `selMask` is every read in the selection, `sel` among them.  A selection
// of one behaves exactly as the selection did when one was all there could be,
// which is why nothing that reads `sel` had to change.
//
// A read is in the set whether or not the tab in front of you can draw it: a
// shift+drag on the map and a shift+click down the reads tab put reads in the
// same set, and a read with no graph is simply not drawn on two of the three
// tabs.
let selMask = null, selN = 0;
const isSel = i => i >= 0 && selMask !== null && selMask[i] === 1;
function selected(){
  const out = [];
  if(selMask) for(let i = 0; i < N; i++) if(selMask[i]) out.push(i);
  return out;
}
// `add` unions with what is already selected; without it the set is replaced.
// `sel` follows the last read put in, so the page reports and scrolls to the
// read the gesture ended on.
function selPut(list, add){
  if(!selMask) selMask = new Uint8Array(N);
  if(!add){ selMask.fill(0); selN = 0; sel = -1; }
  for(const i of list){
    if(i < 0 || selMask[i]) continue;
    selMask[i] = 1; selN++; sel = i;
  }
  if(!selN) sel = -1;
}
function selDrop(i){
  if(!isSel(i)) return;
  selMask[i] = 0; selN--;
  if(sel === i){ sel = -1; for(let k = 0; k < N; k++) if(selMask[k]) sel = k; }
}
function redrawSel(from){
  const i = sel;
  if(i >= 0 && tab === 'reads' && from !== 'reads') scrollToRead(i);
  if(i >= 0 && tab === 'stats' && from !== 'stats') scrollToStats(i);
  drawMap(); drawReads(); drawStats(); status();
}
function select(i, from){
  selPut(i < 0 ? [] : [i], false);
  // WHERE A SHIFT+CLICK RANGE IS MEASURED FROM.  This function IS the plain
  // click -- from the reads tab, the map, a box plot or the find box -- so it
  // is the one place the anchor can be set without a way of selecting one read
  // being left out.  In edit mode a plain click on a row goes through the drag
  // and comes back here, which is how a click and then a shift+click came to
  // extend from whatever had been clicked before them.
  anchorRead = i;
  // A read with no graph exists on one tab only, so selecting it from the
  // find box while the map or the box plots are up moves to the tab where it
  // can be seen rather than selecting something invisible.
  if(i >= 0 && !A.ingraph[i] && tab !== 'reads'){
    show('reads'); status(); return;
  }
  redrawSel(from);
}
// Many reads at once: the map's rubber band, and the reads tab's shift+click
// range.  Neither ever moves the page to another tab -- the gesture was made
// on the tab you are looking at.
function selectMany(list, from, add){
  selPut(list, add);
  redrawSel(from);
}
function selectToggle(i, from){
  if(isSel(i)) selDrop(i); else selPut([i], true);
  redrawSel(from);
}
function status(){
  const el = $('#status');
  if(selN > 1){
    el.innerHTML = '<b>' + fmt(selN) + '</b> reads selected';
    return;
  }
  if(sel < 0){ el.textContent = 'no read selected'; return; }
  const c = cl(sel);
  el.innerHTML = '<b>' + A.ids[sel].slice(0,8) + '</b> → ' + CLABEL(c) +
    (A.ingraph[sel] ? '  strength ' + A.strength[sel].toFixed(1) +
                      '  degree ' + A.degree[sel]
                    : '  not in the graph') +
    '  array ' + bp(sel, 'b0') + '  sub ' + bp(sel, 'sub_bp');
}
const fmt = v => v.toLocaleString();
// A length the read never had measured is NA and not 0: `b0` carries a zero
// for those rows because the reads tab has to draw them somewhere, and a zero
// drawn as a number would read as a boundary at the first base.
const bp = (i, k) => A.hasb0[i] ? fmt(A[k][i]) : 'NA';
function detail(i){
  const c = cl(i);
  const why = RWHY(c);
  return A.ids[i] + '\n' + CLABEL(c) + '  (' + fmt(csize(c)) + ' reads)'
    + (why ? '\n' + why : '')
    + (A.ingraph[i]
       ? '\nstrength ' + A.strength[i].toFixed(2) + '   degree ' + A.degree[i]
         + '\ninformative k-mers ' + fmt(A.nkmer[i])
       : '\nnot in the graph: no edges, no embedding, no box plot')
    + '\ntelomere b0 ' + bp(i, 'b0') + ' bp' +
    '\nsub ' + bp(i, 'sub_bp') + ' bp   read ' + fmt(A.read_bp[i]) + ' bp' +
    '   ' + (A.hasb0[i] ? (A.orient[i] ? 'tail' : 'head') : 'unoriented') +
    (A.qs[i] >= 0 ? '\nqs ' + A.qs[i].toFixed(1) : '');
}
"""

JS += r"""
// ==================================================================== the map
const mcv = $('#mapCv'), mx = mcv.getContext('2d');
let MW = 0, MH = 0;                      // css pixels
let view = {cx:0, cy:0, s:1};            // world -> screen
let byCluster = null, centroid = null, ebuck = null;

function sizeMap(){
  const r = mcv.parentElement.getBoundingClientRect();
  MW = Math.max(1, Math.floor(r.width)); MH = Math.max(1, Math.floor(r.height));
  mcv.width = MW*DPR; mcv.height = MH*DPR;
}
function fitMap(){
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  for(let i=0;i<N;i++){
    if(!A.ingraph[i]) continue;
    const x=A.x[i], y=A.y[i];
    if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y;
  }
  view.cx = (x0+x1)/2; view.cy = (y0+y1)/2;
  view.s = 0.92*Math.min(MW/Math.max(x1-x0,1e-9), MH/Math.max(y1-y0,1e-9));
}
const SX = x => (x - view.cx)*view.s + MW/2;
const SY = y => MH/2 - (y - view.cy)*view.s;

function prep(){
  // Nodes grouped by cluster so the scatter is ~92 fills and not 2,592, and
  // edges bucketed by weight so the alpha ramp costs four strokes and not one
  // per edge.  The LAYOUT never changes, so this is built once -- except in
  // edit mode, where moving a read between two clusters moves it between two
  // of these groups and `afterEdit` runs it again.
  // Over the reads that HAVE a position.  A read dropped before the graph was
  // built was never embedded, and drawing it at the origin would put a dot in
  // the middle of the picture that no edge reaches and no cluster owns.
  byCluster = new Map();
  for(let i=0;i<N;i++){
    if(!A.ingraph[i]) continue;
    const c = cl(i);
    if(!byCluster.has(c)) byCluster.set(c, []);
    byCluster.get(c).push(i);
  }
  centroid = [];
  for(const [c, ix] of byCluster){
    if(c < 0) continue;
    let sx=0, sy=0;
    for(const i of ix){ sx += A.x[i]; sy += A.y[i]; }
    centroid.push([c, sx/ix.length, sy/ix.length, ix.length]);
  }
  const NB = 4;
  ebuck = Array.from({length:NB}, () => []);
  for(let e=0;e<M;e++){
    const w = A.ew[e];
    ebuck[Math.min(NB-1, Math.max(0, Math.floor(w*NB)))].push(e);
  }
}

function drawMap(){
  if(!A || tab !== 'map') return;
  mx.setTransform(DPR,0,0,DPR,0,0);
  mx.clearRect(0,0,MW,MH);
  const floor = +$('#wfloor').value, r = +$('#psize').value;

  if($('#edges').checked){
    mx.lineWidth = 1;
    for(let b=0;b<ebuck.length;b++){
      // Weight is the kernel's: the ramp is deliberately steep at the top, so a
      // 0.95 edge inside a cluster reads solid and a 0.3 edge between two of
      // them reads as the thread it is.
      const a = 0.035 + 0.20*Math.pow((b+0.5)/ebuck.length, 2);
      mx.strokeStyle = 'rgba(70,80,95,' + a.toFixed(3) + ')';
      mx.beginPath();
      let drew = false;
      for(const e of ebuck[b]){
        if(A.ew[e] < floor) continue;
        const i = A.ei[e], j = A.ej[e];
        mx.moveTo(SX(A.x[i]), SY(A.y[i]));
        mx.lineTo(SX(A.x[j]), SY(A.y[j]));
        drew = true;
      }
      if(drew) mx.stroke();
    }
  }
  for(const [c, ix] of byCluster){
    mx.fillStyle = ccol(c);
    mx.beginPath();
    for(const i of ix){
      const x = SX(A.x[i]), y = SY(A.y[i]);
      if(x < -8 || y < -8 || x > MW+8 || y > MH+8) continue;
      mx.moveTo(x+r, y); mx.arc(x, y, r, 0, 6.2832);
    }
    mx.fill();
  }
  if($('#labels').checked){
    mx.font = '600 10px ui-monospace,Menlo,Consolas,monospace';
    mx.textAlign = 'center'; mx.textBaseline = 'middle';
    for(const [c, x, y] of centroid){
      const sx = SX(x), sy = SY(y);
      if(sx < 0 || sy < 0 || sx > MW || sy > MH) continue;
      mx.lineWidth = 3; mx.strokeStyle = 'rgba(255,255,255,0.85)';
      mx.strokeText(String(c), sx, sy);
      mx.fillStyle = ccol(c, 30); mx.fillText(String(c), sx, sy);
    }
  }
  // Every read in the selection gets a ring, and the read the page is
  // reporting on gets its edges as well.  The edges are for ONE read on
  // purpose: "what is this read attached to" is a question about a read, and
  // forty reads' edges at once is the edge layer, which has its own switch.
  if(selN > 1){
    mx.strokeStyle = '#2b6cb0'; mx.lineWidth = 1; mx.globalAlpha = 0.9;
    mx.beginPath();
    for(let i = 0; i < N; i++){
      if(!selMask[i] || !A.ingraph[i]) continue;
      const x = SX(A.x[i]), y = SY(A.y[i]);
      if(x < -8 || y < -8 || x > MW+8 || y > MH+8) continue;
      mx.moveTo(x + r + 2.5, y); mx.arc(x, y, r + 2.5, 0, 6.2832);
    }
    mx.stroke(); mx.globalAlpha = 1;
  }
  const rings = [[hov,'#111',1.5],[sel,'#2b6cb0',2.5]];
  if(drag){
    rings.push([drag.i, '#d9822b', 2.5]);
    const over = drag.t ? drag.t.i : -1;
    if(over >= 0 && over !== drag.i) rings.push([over, '#2f855a', 3]);
  }
  for(const [i, col, w] of rings){
    if(i < 0 || !A.ingraph[i]) continue;
    // The selected read's own edges, over everything: this is the question
    // "what is this read actually attached to" and it should not need the
    // whole edge layer turned on to be answerable.
    mx.strokeStyle = col; mx.lineWidth = w === 2.5 ? 1.2 : 0.9;
    mx.globalAlpha = 0.55; mx.beginPath();
    for(let e=0;e<M;e++){
      const a = A.ei[e], b = A.ej[e];
      if(a !== i && b !== i) continue;
      const o = a === i ? b : a;
      mx.moveTo(SX(A.x[i]), SY(A.y[i]));
      mx.lineTo(SX(A.x[o]), SY(A.y[o]));
    }
    mx.stroke(); mx.globalAlpha = 1;
    mx.beginPath(); mx.arc(SX(A.x[i]), SY(A.y[i]), r+3.5, 0, 6.2832);
    mx.strokeStyle = col; mx.lineWidth = w; mx.stroke();
  }
  if(band){
    const x = Math.min(band.x0, band.x1), y = Math.min(band.y0, band.y1);
    const w = Math.abs(band.x1 - band.x0), h = Math.abs(band.y1 - band.y0);
    mx.fillStyle = 'rgba(43,108,176,0.08)';
    mx.fillRect(x, y, w, h);
    mx.strokeStyle = '#2b6cb0'; mx.lineWidth = 1;
    mx.setLineDash([4, 3]);
    mx.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, w, h);
    mx.setLineDash([]);
  }
}

function pick(px, py, preferSel){
  // Brute force over 2,592 points: a grid index would be faster and would also
  // be a second copy of the layout to keep in step with the first.
  //
  // `preferSel` is for picking a read UP while a selection is on the page.
  // Inside a blob the nearest dot to the cursor is often not the one being
  // aimed at, and grabbing a neighbour there does not move the selection -- it
  // moves one read out of it, quietly.  A selected dot within reach wins.
  let best = -1, bd = 144, bsel = -1, bsd = 144;
  for(let i=0;i<N;i++){
    if(!A.ingraph[i]) continue;
    const dx = SX(A.x[i]) - px, dy = SY(A.y[i]) - py, d = dx*dx + dy*dy;
    if(d < bd){ bd = d; best = i; }
    if(preferSel && isSel(i) && d < bsd){ bsd = d; bsel = i; }
  }
  return bsel >= 0 ? bsel : best;
}

// Shift is the map's select-a-square modifier, and it is checked before
// anything else a mousedown can start: without shift the same press pans, or
// in edit mode picks a read up, and a band that had to share the plain drag
// with either of those would be a band you could start by accident.
//
// THE BAND ALWAYS ADDS.  Two lobes of one chromosome end are two squares, and
// a gesture that threw the first square away when the second was drawn could
// not say so.  A plain click on empty space is what clears the selection.
let band = null;
let mdrag = null;
mcv.addEventListener('mousedown', e => {
  if(e.button === 0 && e.shiftKey){
    const r = mcv.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    band = {x0:px, y0:py, x1:px, y1:py};
    drawMap(); return;
  }
  if(edit && e.button === 0){
    // a dot under the cursor is a read to move; empty space still pans
    const r = mcv.getBoundingClientRect();
    const i = pick(e.clientX - r.left, e.clientY - r.top, selN > 1);
    if(i >= 0){ dragStart(i, e); return; }
  }
  mdrag = {x:e.clientX, y:e.clientY, cx:view.cx, cy:view.cy, moved:false};
  mcv.classList.add('drag');
});
// Every read whose dot falls inside the band.  Screen coordinates, because the
// square was drawn on the screen and the view may be panned or zoomed between
// one band and the next.
function inBand(b){
  const xl = Math.min(b.x0, b.x1), xh = Math.max(b.x0, b.x1);
  const yl = Math.min(b.y0, b.y1), yh = Math.max(b.y0, b.y1);
  const out = [];
  for(let i = 0; i < N; i++){
    if(!A.ingraph[i]) continue;
    const x = SX(A.x[i]), y = SY(A.y[i]);
    if(x >= xl && x <= xh && y >= yl && y <= yh) out.push(i);
  }
  return out;
}
window.addEventListener('mousemove', e => {
  if(drag && tab === 'map'){ dragMove(e); return; }
  const r = mcv.getBoundingClientRect();
  if(band){
    band.x1 = e.clientX - r.left; band.y1 = e.clientY - r.top;
    tipOff(); drawMap(); return;
  }
  if(mdrag){
    const dx = e.clientX - mdrag.x, dy = e.clientY - mdrag.y;
    if(Math.abs(dx) + Math.abs(dy) > 3) mdrag.moved = true;
    view.cx = mdrag.cx - dx/view.s; view.cy = mdrag.cy + dy/view.s;
    drawMap(); return;
  }
  if(tab !== 'map') return;
  const i = pick(e.clientX - r.left, e.clientY - r.top);
  if(i !== hov){ hov = i; drawMap(); }
  if(i >= 0) tipAt(e, detail(i)); else tipOff();
});
window.addEventListener('mouseup', e => {
  if(drag && tab === 'map'){ dragEnd(e); return; }
  if(band){
    const b = band; band = null;
    selectMany(inBand(b), 'map', true);
    return;
  }
  if(!mdrag) return;
  const moved = mdrag.moved;
  mdrag = null; mcv.classList.remove('drag');
  if(!moved && tab === 'map'){
    const r = mcv.getBoundingClientRect();
    select(pick(e.clientX - r.left, e.clientY - r.top), 'map');
  }
});
mcv.addEventListener('wheel', e => {
  e.preventDefault();
  const r = mcv.getBoundingClientRect();
  const px = e.clientX - r.left, py = e.clientY - r.top;
  const wx = (px - MW/2)/view.s + view.cx, wy = view.cy - (py - MH/2)/view.s;
  const f = Math.exp(-e.deltaY*0.0016);
  view.s *= f;
  view.cx = wx - (px - MW/2)/view.s; view.cy = wy + (py - MH/2)/view.s;
  drawMap();
}, {passive:false});
"""

JS += r"""
// ================================================================== the reads
const AXIS_H = 22, GID_W = 104, CHIP_W = 10, PADX = 6;
const READS_X = GID_W + CHIP_W + PADX;
const rcv = $('#readsCv'), rx = rcv.getContext('2d'),
      scroller = $('#readsScroll'), spacer = $('#readsSpacer');
let RW = 0, RH = 0, img = null;
let rowTop = null, blockOf = null, totalPx = 0;
// THE ROWS ARE THE CLUSTERS AS THEY NOW STAND.  Outside edit mode `BLOCKS` is
// the run's own `D.blocks` read for read and ORDER/RANK are `A.order`/`A.rank`
// unchanged -- a page nobody has edited is laid out exactly as it always was.
// Inside edit mode they are rebuilt from `manual` after every edit, so a read
// given cluster 7 is drawn among the cluster 7 reads.  Nothing outside
// `layout()` may read `A.order` or `A.rank` again: two answers to "which row
// is this read on" is how a gutter comes to disagree with the sequence beside
// it.
let BLOCKS = null, ORDER = null, RANK = null;
const EMPTY_ROWS = 3;              // an empty cluster's drop band, in row heights
let xLo = 0, xHi = 1;                    // the drawn t window

// t <-> x, ONCE.  The drawing and the hit test both go through these, so a
// change to the layout cannot move the sequence without moving the cursor with
// it -- a read reported under the wrong coordinate is a QC error, not a
// cosmetic one.
const tToX = t => READS_X + (t - xLo)/(xHi - xLo)*(RW - READS_X);
const xToT = x => xLo + (x - READS_X)/(RW - READS_X)*(xHi - xLo);

// One block a cluster, in the run's own block order with the clusters a
// curator has made since inserted after the run's and before the unclustered
// ones.  A block emptied by an edit STAYS: a label that vanished under the
// cursor would take its drop target with it, and putting the reads back is
// exactly what someone who emptied it by mistake wants to do next.
//
// Within a block the reads keep the run's own order, so a read's row is a
// function of its cluster and of nothing else -- there is no hand-made
// ordering to lose, and the same CSV loaded back rebuilds the same page.
function blockList(){
  if(!manual)
    return D.blocks.map(([cid, start, size]) => {
      const m = [];
      for(let r = start; r < start + size; r++) m.push(A.order[r]);
      return {cid, members:m};
    });
  const memb = new Map();
  for(let r = 0; r < N; r++){            // the run's row order, kept per block
    const i = A.order[r], c = manual[i];
    if(!memb.has(c)) memb.set(c, []);
    memb.get(c).push(i);
  }
  const out = [], seen = new Set();
  const put = c => { if(!seen.has(c)){ seen.add(c); out.push(c); } };
  for(const [c] of D.blocks) if(c >= 0) put(c);
  const made = [];
  for(const c of memb.keys()) if(c >= 0 && !seen.has(c)) made.push(c);
  for(const c of newClusters)
    if(c >= 0 && !seen.has(c) && made.indexOf(c) < 0) made.push(c);
  made.sort((a, b) => a - b).forEach(put);
  // the bare pool, then the run's own drop reasons last and in its order
  if(memb.has(UNCL) || newClusters.indexOf(UNCL) >= 0) put(UNCL);
  for(const [c] of D.blocks) if(c < 0) put(c);
  for(const c of memb.keys()) if(c < 0) put(c);
  return out.map(cid => ({cid, members: memb.get(cid) || []}));
}

function layout(){
  // Row tops once, with the gap between two cluster blocks folded in, so the
  // draw loop and the hit test read the same array and cannot disagree about
  // which row the cursor is on.  A block's `top`/`bot` is its whole span, the
  // empty band included, because that is what a drop onto a CLUSTER is hit
  // tested against.
  BLOCKS = blockList();
  ORDER = new Int32Array(N); RANK = new Int32Array(N);
  rowTop = new Float64Array(N + 1);
  blockOf = new Int32Array(N);
  let y = 0, r = 0;
  BLOCKS.forEach((blk, b) => {
    blk.b = b; blk.top = y; blk.start = r;
    for(const i of blk.members){
      ORDER[r] = i; RANK[i] = r; rowTop[r] = y; blockOf[r] = b;
      y += D.rowPx; r++;
    }
    if(!blk.members.length) y += EMPTY_ROWS*D.rowPx;
    blk.bot = y;
    y += D.gapRows*D.rowPx;
  });
  rowTop[N] = y;
  totalPx = y;
}
// The block a content y falls in -- its rows and, where it has none, its empty
// band -- or -1.  A hundred blocks scanned on a mousemove is nothing, and a
// second index of the layout would be a second thing to keep in step with it.
function blockAtY(y){
  if(!BLOCKS) return -1;
  for(const blk of BLOCKS) if(y >= blk.top && y < blk.bot) return blk.b;
  return -1;
}
// THE WHITE SPACE BETWEEN TWO BLOCKS IS A CLUSTER THAT DOES NOT EXIST YET.
// Dropping reads there makes one and puts them in it, which is the gesture for
// "these belong together and to nothing on this page" -- said in one motion,
// where making the cluster first and then filling it is two.  The gap is named
// by the block ABOVE it, the run of empty page below the last block included.
function gapAtY(y){
  if(!BLOCKS || y < 0) return -1;
  let g = -1;
  for(const blk of BLOCKS){
    if(y >= blk.bot) g = blk.b;
    else if(y >= blk.top) return -1;      // inside a block, not between two
    else break;
  }
  return g;
}
// The tab OPENS on the window the clustering saw, not on everything packed.
// At the full extent the array is ten bases to the pixel, and ten bases of a
// six-base repeat average to one flat colour whatever the repeat is doing --
// the variant blocks that are the whole reason to look at the array are only
// there under about four.  `all` is a button away.
function winView(){
  const pad = (D.telo_bp + D.sub_bp)*0.06;
  xLo = -D.sub_bp - pad; xHi = D.telo_bp + pad;
}
function rowFloor(y){        // the last row at or above content y
  let lo = 0, hi = N - 1, r = -1;
  while(lo <= hi){
    const m = (lo + hi) >> 1;
    if(rowTop[m] <= y){ r = m; lo = m + 1; } else hi = m - 1;
  }
  return r;
}
function rowAtY(y){          // content y -> row, or -1 in a block gap
  const r = rowFloor(y);
  return (r >= 0 && y < rowTop[r] + D.rowPx) ? r : -1;
}
function sizeReads(){
  RW = Math.max(1, scroller.clientWidth); RH = Math.max(1, scroller.clientHeight);
  rcv.style.height = RH + 'px';
  rcv.width = Math.floor(RW*DPR); rcv.height = Math.floor(RH*DPR);
  img = rx.createImageData(rcv.width, rcv.height);
  spacer.style.height = Math.max(0, totalPx - (RH - AXIS_H)) + 'px';
}
function scrollToRead(i){
  const r = RANK[i];
  scroller.scrollTop = Math.max(0, rowTop[r] - (RH - AXIS_H)/2);
}

function drawReads(){
  if(!A || !SEQ || tab !== 'reads') return;
  const cw = rcv.width, ch = rcv.height;
  const seqX = Math.round(READS_X*DPR), seqW = cw - seqX;
  const st = scroller.scrollTop;
  const axis = Math.round(AXIS_H*DPR);
  const bpp = (xHi - xLo)/seqW;                 // bases per device pixel
  const flankLo = -D.sub_bp, flankHi = D.telo_bp;
  const d = img.data;
  d.fill(255);

  const rowH = Math.max(1, Math.round(D.rowPx*DPR));
  const line = new Uint8ClampedArray(seqW*4);
  const r0 = Math.max(0, rowFloor(st) - 1);
  const rEnd = Math.min(N - 1, Math.max(r0, rowFloor(st + RH)) + 1);
  for(let r = r0; r <= rEnd; r++){
    const yTop = Math.round((rowTop[r] - st)*DPR) + axis;
    if(yTop + rowH <= axis || yTop >= ch) continue;
    const i = ORDER[r], off = A.off[i], len = A.slen[i], t0 = A.tmin[i];
    for(let px = 0; px < seqW; px++){
      const ta = xLo + px*bpp, tb = ta + bpp;
      let m0 = Math.floor(ta) - t0, m1 = Math.ceil(tb) - t0;
      if(m1 <= m0) m1 = m0 + 1;
      if(m0 < 0) m0 = 0;
      if(m1 > len) m1 = len;
      const o = px*4;
      let R, G, B;
      if(m1 <= m0){ R = D.gap[0]; G = D.gap[1]; B = D.gap[2]; }
      else{
        R = 0; G = 0; B = 0;
        for(let m = m0; m < m1; m++){
          const c = SEQ[off + m]; R += LR[c]; G += LG[c]; B += LB[c];
        }
        const k = m1 - m0;
        R /= k; G /= k; B /= k;
        // Outside the window the graph was built from, faded toward the page.
        // It is drawn because a read's context is worth seeing and it is faded
        // because not one k-mer in it reached the graph.
        const tc = ta + bpp/2;
        if(tc < flankLo || tc > flankHi){
          R = 255 - (255 - R)*D.flank;
          G = 255 - (255 - G)*D.flank;
          B = 255 - (255 - B)*D.flank;
        }
      }
      line[o] = R; line[o+1] = G; line[o+2] = B; line[o+3] = 255;
    }
    for(let s = 0; s < rowH; s++){
      const y = yTop + s;
      if(y < axis || y >= ch) continue;
      d.set(line, (y*cw + seqX)*4);
    }
  }
  rx.putImageData(img, 0, 0);

  // ------------------------------------------------------------ over the top
  rx.setTransform(DPR, 0, 0, DPR, 0, 0);
  const X = tToX;

  // the two dashed rules, and the boundary every read is aligned on
  rx.save();
  rx.beginPath(); rx.rect(READS_X, AXIS_H, RW - READS_X, RH - AXIS_H);
  rx.clip();
  rx.setLineDash([4, 3]); rx.lineWidth = 1;
  rx.strokeStyle = '#2b6cb0';
  for(const t of [flankLo, flankHi]){
    const x = Math.round(X(t)) + 0.5;
    rx.beginPath(); rx.moveTo(x, AXIS_H); rx.lineTo(x, RH); rx.stroke();
  }
  rx.setLineDash([]); rx.strokeStyle = 'rgba(27,27,27,0.35)';
  const xz = Math.round(X(0)) + 0.5;
  rx.beginPath(); rx.moveTo(xz, AXIS_H); rx.lineTo(xz, RH); rx.stroke();
  rx.restore();

  // the gutter: one chip a read, one label a block, both clipped to the rows
  rx.save();
  rx.beginPath(); rx.rect(0, AXIS_H, READS_X - PADX, RH - AXIS_H); rx.clip();
  rx.fillStyle = '#fff'; rx.fillRect(0, AXIS_H, READS_X - PADX, RH - AXIS_H);
  for(let r = r0; r <= rEnd; r++){
    const y = rowTop[r] - st + AXIS_H;
    if(y + D.rowPx <= AXIS_H || y >= RH) continue;
    const i = ORDER[r];
    rx.fillStyle = ccol(cl(i));
    rx.fillRect(GID_W, y, CHIP_W - 2, Math.max(1, D.rowPx - 0.5));
    // A read this session moved, so an edit can be found again by eye: the
    // row is in its new block among reads that were always there, and the
    // colour alone cannot tell the two apart.
    if(manual && manual[i] !== A.cluster[i]){
      rx.fillStyle = '#d9822b';
      rx.fillRect(GID_W - 3, y, 1.5, Math.max(1, D.rowPx - 0.5));
    }
    // every read being dragged, and the read it would land on
    const over = drag && drag.t ? drag.t.i : -1;
    if(drag && (isHeld(i) || i === over)){
      rx.strokeStyle = (i === over && over !== drag.i) ? '#2f855a' : '#d9822b';
      rx.lineWidth = 1.5;
      rx.strokeRect(GID_W - 2.5, y - 0.5, CHIP_W + 3, Math.max(1, D.rowPx));
    }
  }
  rx.font = '600 10.5px ui-monospace,Menlo,Consolas,monospace';
  rx.textBaseline = 'top';
  for(const blk of BLOCKS){
    const top = blk.top - st + AXIS_H, bot = blk.bot - st + AXIS_H;
    if(bot < AXIS_H || top > RH) continue;
    const cid = blk.cid, size = blk.members.length;
    // the block a drop would land in, drawn as the target it is
    const tgt = !!(drag && drag.t && drag.t.blk === blk.b);
    // Pinned to the top of its own block while any of it is on screen, so a
    // 53-read block is still labelled when you are in the middle of it.
    const y = Math.min(Math.max(top, AXIS_H + 2), Math.max(AXIS_H + 2, bot - 13));
    rx.fillStyle = ccol(cid, 34);
    rx.fillText(CNAME(cid) + '  ' + size, 6, y);
    rx.fillStyle = tgt ? '#2f855a' : ccol(cid, 70);
    rx.fillRect(GID_W - 6, Math.max(top, AXIS_H), tgt ? 4 : 2,
                Math.min(bot, RH) - Math.max(top, AXIS_H) - 1);
    // A cluster with nothing in it is a dashed band rather than nothing at
    // all: it is what a read is dropped onto to go there, so an empty cluster
    // has to be a place on the page before it can be filled.
    if(!size){
      rx.save();
      rx.setLineDash([3, 3]); rx.lineWidth = 1;
      rx.strokeStyle = tgt ? '#2f855a' : '#d9b892';
      rx.strokeRect(4.5, Math.max(top, AXIS_H) + 0.5, READS_X - PADX - 9,
                    Math.max(2, Math.min(bot, RH) - Math.max(top, AXIS_H) - 2));
      rx.restore();
    }
  }
  rx.restore();

  // The white space the drop would become a cluster in, drawn across the whole
  // width: the gap is twelve pixels of nothing between two blocks, and a
  // gesture that made a cluster out of it without saying so first would be a
  // cluster made by accident.
  if(drag && drag.t && drag.t.gap >= 0){
    const g = BLOCKS[drag.t.gap];
    const y = Math.round(g.bot + D.gapRows*D.rowPx/2 - st) + AXIS_H + 0.5;
    if(y > AXIS_H && y < RH){
      rx.save();
      rx.strokeStyle = '#2f855a'; rx.lineWidth = 2; rx.setLineDash([5, 3]);
      rx.beginPath(); rx.moveTo(0, y); rx.lineTo(RW, y); rx.stroke();
      rx.setLineDash([]);
      rx.font = '600 10.5px ui-monospace,Menlo,Consolas,monospace';
      rx.textBaseline = 'middle';
      const label = 'new cluster';
      const w = rx.measureText(label).width + 8;
      rx.fillStyle = '#2f855a';
      rx.fillRect(4, y - 7, w, 14);
      rx.fillStyle = '#fff';
      rx.fillText(label, 8, y);
      rx.restore();
    }
  }

  // Every selected row on screen is outlined, and only the visible ones are
  // looked at: a selection can be a thousand rows and the outline costs a
  // stroke a row.
  if(selN){
    rx.strokeStyle = '#2b6cb0'; rx.lineWidth = 1;
    for(let r = r0; r <= rEnd; r++){
      if(!selMask[ORDER[r]]) continue;
      const y = rowTop[r] - st + AXIS_H;
      if(y + D.rowPx <= AXIS_H || y >= RH) continue;
      rx.strokeRect(0.5, Math.round(y) - 0.5, RW - 1,
                    Math.max(2, D.rowPx) + 1);
    }
  }

  // ------------------------------------------------------------------- axis
  rx.fillStyle = '#fff'; rx.fillRect(0, 0, RW, AXIS_H);
  rx.fillStyle = 'rgba(43,108,176,0.09)';
  const bx0 = Math.max(READS_X, X(flankLo)), bx1 = Math.min(RW, X(flankHi));
  if(bx1 > bx0) rx.fillRect(bx0, 0, bx1 - bx0, AXIS_H);
  rx.strokeStyle = '#e0e0e0'; rx.lineWidth = 1;
  rx.beginPath(); rx.moveTo(0, AXIS_H - 0.5); rx.lineTo(RW, AXIS_H - 0.5);
  rx.stroke();
  rx.font = '10px ui-monospace,Menlo,Consolas,monospace';
  rx.textBaseline = 'middle'; rx.textAlign = 'center';
  const span = xHi - xLo, raw = span/((RW - READS_X)/110);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].find(m => m*mag >= raw)*mag;
  rx.fillStyle = '#888';
  for(let t = Math.ceil(xLo/step)*step; t <= xHi; t += step){
    const x = X(t);
    if(x < READS_X + 12) continue;
    rx.fillText(Math.abs(t) >= 1000 ? (t/1000) + ' kb' : t + '', x, AXIS_H/2);
    rx.strokeStyle = '#eee';
    rx.beginPath(); rx.moveTo(x, AXIS_H - 5); rx.lineTo(x, AXIS_H); rx.stroke();
  }
  rx.textAlign = 'left';
  rx.fillStyle = '#666';
  rx.fillText('subtelomere ←', READS_X + 4, AXIS_H/2);
  rx.textAlign = 'right';
  rx.fillText('→ telomere', RW - 4, AXIS_H/2);
  rx.textAlign = 'left';
}

// --------------------------------------------------------------- reads input
scroller.addEventListener('scroll', () => drawReads(), {passive:true});
function readsHit(e){
  const b = rcv.getBoundingClientRect();
  const px = e.clientX - b.left, py = e.clientY - b.top;
  // `cy` is the content coordinate the rows are laid out in, and -1 over the
  // axis -- which is not white space, and a drop there means nothing.
  if(py < AXIS_H) return {row:-1, blk:-1, cy:-1, px, py};
  const cy = py - AXIS_H + scroller.scrollTop;
  return {row:rowAtY(cy), blk:blockAtY(cy), cy, px, py, t:xToT(px)};
}
let rdrag = null;
// The last row a plain or shift click landed on: shift+click selects the rows
// BETWEEN that one and this one, which needs somewhere to measure from.  It is
// a row and not a read, because the range a person means is the one they can
// see -- the rows between two rows on the screen, whatever clusters those rows
// happen to belong to.
// The read a shift+click measures its range from: the last one CLICKED, set
// by `select` and by the modifier branch below.  A read and not a row number,
// because an edit re-lays the rows out under it.
let anchorRead = -1;
rcv.addEventListener('mousedown', e => {
  const h = readsHit(e);
  const i0 = h.row >= 0 ? ORDER[h.row] : -1;
  // Shift and ctrl are the selection modifiers, ahead of the drag and the pan:
  // shift+click takes the rows from the last click to this one, ctrl+click
  // adds or removes one row.  Anywhere on the row, not only in the gutter --
  // the row is what is being selected.
  if(e.button === 0 && h.row >= 0 && (e.shiftKey || e.ctrlKey || e.metaKey)){
    // The anchor a range is measured from is a READ and not a row number: an
    // edit re-lays the rows out under it, and a remembered row would then be
    // somebody else's.
    const ar = anchorRead >= 0 ? RANK[anchorRead] : -1;
    if(e.shiftKey && ar >= 0){
      const lo = Math.min(ar, h.row), hi = Math.max(ar, h.row);
      const run = [];
      for(let r = lo; r <= hi; r++) run.push(ORDER[r]);
      // The row clicked last is the one the page reports on, so a range picked
      // upwards reports the read at its top.
      if(ar > h.row) run.reverse();
      selectMany(run, 'reads', true);
    } else if(e.shiftKey){
      selectMany([i0], 'reads', true);
    } else {
      selectToggle(i0, 'reads');
    }
    anchorRead = i0;
    return;                       // no drag, no pan: this press was a select
  }
  // THE HANDLE IS THE GUTTER, AND A SELECTED ROW IS ITS OWN HANDLE.  A row is
  // six pixels tall: a drag that could only start inside the gutter picked up
  // the read one row off the one being aimed at often enough to matter, and
  // because that read was not in the selection it moved alone and left the
  // selection behind -- the edit looked like it had been made and had not.
  // Pressing anywhere on a row that is already selected now drags the whole
  // selection, and the sequence area of every other row still pans.
  if(edit && e.button === 0 && i0 >= 0 &&
     (h.px < READS_X || (selN > 1 && isSel(i0)))){
    dragStart(i0, e); return;
  }
  rdrag = {x:e.clientX, lo:xLo, hi:xHi, moved:false, row:h.row,
           pan:h.px >= READS_X};
});
rcv.addEventListener('mousemove', e => {
  if(drag && tab === 'reads'){ dragMove(e); return; }
  if(rdrag && rdrag.pan){
    const dx = e.clientX - rdrag.x;
    if(Math.abs(dx) > 3) rdrag.moved = true;
    const dt = dx/(RW - READS_X)*(rdrag.hi - rdrag.lo);
    xLo = rdrag.lo - dt; xHi = rdrag.hi - dt;
    drawReads(); return;
  }
  const h = readsHit(e);
  if(h.row < 0){ tipOff(); return; }
  const i = ORDER[h.row];
  let s = detail(i);
  if(h.px >= READS_X){
    const m = Math.round(h.t) - A.tmin[i];
    const base = (m >= 0 && m < A.slen[i])
      ? String.fromCharCode(SEQ[A.off[i] + m]) : '–';
    s += '\nt ' + fmt(Math.round(h.t)) + ' bp   ' + base +
         (h.t < -D.sub_bp || h.t > D.telo_bp ? '   (outside the window)' : '');
  }
  tipAt(e, s);
});
window.addEventListener('mouseup', e => {
  if(drag && tab === 'reads'){ dragEnd(e); return; }
  if(!rdrag) return;
  const {moved, row} = rdrag; rdrag = null;
  if(!moved && row >= 0 && tab === 'reads') select(ORDER[row], 'reads');
});
rcv.addEventListener('wheel', e => {
  if(!e.shiftKey) return;               // plain wheel scrolls the rows
  e.preventDefault();
  const b = rcv.getBoundingClientRect();
  const px = Math.max(READS_X, e.clientX - b.left);
  zoomX(Math.exp(e.deltaY*0.0015), xToT(px));
}, {passive:false});
function zoomX(f, about){
  const lo = about - (about - xLo)*f, hi = about + (xHi - about)*f;
  if(hi - lo < 40) return;              // 40 bp across the page is the floor
  xLo = lo; xHi = hi;
  drawReads();
}
"""

JS += r"""
// ================================================================== the stats
// One row a cluster, one statistic at a time across the full width.
//
// The telomere is `b0` -- teloBP's array/subtelomere boundary, the coordinate
// every window in the run was cut from and the one the reads tab draws its
// dashed rules at.  It is the array length as well: intake used to carry a
// second, independent `array_bp` from a canonical-hexamer density scan, but
// teloBP now calls the read's strand itself and that scan is gone, so there is
// one number here rather than two that disagreed.
//
// The rows stay in the reads tab's order whatever is being shown, so a row is
// in the same place on all three tabs and switching the statistic moves the
// boxes and not the clusters.
const S_AXIS = 34, S_GID = 138, S_ROW = 28, S_PAD = 9, S_RPAD = 12;
const SM = [{k:'read_bp', t:'read length'},
            {k:'b0',      t:'telomere length (b0)'}];
const scv = $('#statsCv'), sx = scv.getContext('2d'),
      sscroll = $('#statsScroll'), sspacer = $('#statsSpacer');
let SW = 0, SH = 0, sTotal = 0, smet = 0, shov = -1;
let SROWS = null, SDOM = null;

// ccol() collapses every lightness to one grey for the unclustered pool, which
// is right on the other two tabs and wrong here, where a box needs a fill and
// an outline that are not the same colour.
const bcol = (c, l) => c < 0 ? 'hsl(210,6%,' + l + '%)' : ccol(c, l);

// ------------------------------------------------------------- the five-number
// Linear interpolation between order statistics -- numpy's default, so a box
// drawn here and a quantile computed over the same column agree.
function quant(v, p){
  if(!v.length) return NaN;
  const h = (v.length - 1)*p, lo = Math.floor(h), hi = Math.ceil(h);
  return v[lo] + (v[hi] - v[lo])*(h - lo);
}
function boxOf(v){
  if(!v.length) return {n:0, min:0, max:0, q1:0, med:0, q3:0, wl:0, wh:0};
  const q1 = quant(v, 0.25), med = quant(v, 0.5), q3 = quant(v, 0.75);
  const iqr = q3 - q1, fLo = q1 - 1.5*iqr, fHi = q3 + 1.5*iqr;
  let wl = v[0], wh = v[v.length - 1];
  for(let i = 0; i < v.length; i++) if(v[i] >= fLo){ wl = v[i]; break; }
  for(let i = v.length - 1; i >= 0; i--) if(v[i] <= fHi){ wh = v[i]; break; }
  return {n:v.length, min:v[0], max:v[v.length - 1], q1, med, q3, wl, wh};
}
// Both statistics are summarised for every row whichever one is on screen:
// the row tooltip reports both, and switching the dropdown is then a redraw
// and not a recomputation.
// A box plot needs the numbers the graph produced, so this tab is over the
// reads that HAVE them: a block of reads dropped before module 2 gets no row,
// and a read dragged out of one in edit mode does not acquire one.  Their
// lengths are on the reads tab, drawn rather than summarised, which is the
// only honest thing a page can do with one measurement out of five.
//
// SROWI maps a block to its row here, or to -1, because the rows are no longer
// the blocks one for one and `rowOf` is what puts the selection on the right
// line.
let SROWI = null;
function statsPrep(){
  // One row a block, over the blocks as they NOW STAND: a read dragged out of
  // a cluster leaves that box plot and joins another, and a cluster a curator
  // made gets a row of its own as soon as a read with a graph is in it -- or
  // the tab would go on describing a labelling that no longer exists.  The row
  // order is the reads tab's, so a cluster is in the same place on both.
  //
  // A block with no read in the graph gets no row: five numbers over nothing
  // is not a distribution.  That is why the block-to-row map is rebuilt here
  // rather than frozen at boot.
  SROWI = new Int32Array(BLOCKS.length).fill(-1);
  SROWS = [];
  BLOCKS.forEach((blk, b) => {
    const idx = Int32Array.from(blk.members.filter(i => A.ingraph[i]));
    if(!idx.length) return;
    SROWI[b] = SROWS.length;
    const bx = SM.map(m => {
      const v = Array.from(idx, i => A[m.k][i]).sort((p, q) => p - q);
      return boxOf(v);
    });
    SROWS.push({cid: blk.cid, size: idx.length, idx, b: bx});
  });
  SDOM = SM.map(m => {
    const v = [];
    for(let i = 0; i < N; i++) if(A.ingraph[i]) v.push(A[m.k][i]);
    v.sort((p, q) => p - q);
    return {lo: v[0], hi: v[v.length - 1], med: quant(v, 0.5)};
  });
  sTotal = SROWS.length*S_ROW;
  sspacer.style.height = Math.max(0, sTotal - (SH - S_AXIS)) + 'px';
  sstat();
}
function sstat(){
  smet = +($('#smetric').value || 0);
  $('#sstatus').innerHTML =
    '<b>' + SM[smet].t + '</b>  ·  sample median ' +
    fmt(Math.round(SDOM[smet].med)) + ' bp  ·  ' +
    fmt(SDOM[smet].lo) + '–' + fmt(SDOM[smet].hi) + ' bp over ' + D.nGraph +
    ' reads in the graph' +
    (N > D.nGraph ? '  ·  ' + fmt(N - D.nGraph) + ' dropped before it, on '
                    + 'the reads tab only' : '');
}

// ------------------------------------------------------------------ the scale
function spanel(){ return {x0: S_GID, w: SW - S_GID - S_RPAD}; }
function sX(v){
  const d = SDOM[smet], g = spanel();
  const f = (v - d.lo)/(d.hi - d.lo);
  return g.x0 + S_PAD + Math.max(0, Math.min(1, f))*(g.w - 2*S_PAD);
}
function sticks(){
  const d = SDOM[smet], g = spanel(), out = [];
  // One label per 70 px, not the reads tab's 110: the nice-number step is
  // rounded UP, so a target that lands just above a power of ten doubles the
  // step and halves the labels.
  const raw = (d.hi - d.lo)/Math.max(1, (g.w - 2*S_PAD)/70);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = ([1, 2, 5, 10].find(m => m*mag >= raw) || 10)*mag;
  for(let v = Math.ceil(d.lo/step)*step; v <= d.hi; v += step) out.push(v);
  return out;
}
function kb(v){
  if(v < 1000) return String(Math.round(v));
  const k = v/1000;
  return (k < 10 ? k.toFixed(1) : Math.round(k)) + ' kb';
}
// A read's jitter is a function of its index and of nothing else, so a dot
// does not walk around its row between redraws or between statistics.
function jit(i){
  let h = Math.imul(i + 1, 2654435761) >>> 0;
  h ^= h >>> 15; h = Math.imul(h, 2246822519) >>> 0; h ^= h >>> 13;
  return (h >>> 8)/8388608 - 1;
}

function sizeStats(){
  SW = Math.max(1, sscroll.clientWidth); SH = Math.max(1, sscroll.clientHeight);
  scv.style.height = SH + 'px';
  scv.width = Math.floor(SW*DPR); scv.height = Math.floor(SH*DPR);
  sspacer.style.height = Math.max(0, sTotal - (SH - S_AXIS)) + 'px';
}
function rowOf(i){
  if(!blockOf || !SROWI) return -1;
  return SROWI[blockOf[RANK[i]]];
}
function scrollToStats(i){
  if(!SROWS || !blockOf) return;
  const q = rowOf(i);
  if(q < 0) return;              // a read with no box plot to scroll to
  sscroll.scrollTop = Math.max(0, q*S_ROW - (SH - S_AXIS)/2);
}

function drawStats(){
  if(!A || !SROWS || tab !== 'stats') return;
  sx.setTransform(DPR, 0, 0, DPR, 0, 0);
  sx.clearRect(0, 0, SW, SH);
  sx.fillStyle = '#fff'; sx.fillRect(0, 0, SW, SH);
  const st = sscroll.scrollTop, dots = $('#sdots').checked;
  const col = A[SM[smet].k];
  const q0 = Math.max(0, Math.floor(st/S_ROW) - 1);
  const q1 = Math.min(SROWS.length - 1, Math.ceil((st + SH)/S_ROW));
  const half = S_ROW*0.5 - 5, jhalf = S_ROW*0.5 - 9;
  const selRow = sel >= 0 ? rowOf(sel) : -1;

  sx.save();
  sx.beginPath(); sx.rect(0, S_AXIS, SW, SH - S_AXIS); sx.clip();

  // the whole sample's median, behind everything, as the reference a row is
  // long or short against
  sx.strokeStyle = 'rgba(27,27,27,0.16)'; sx.lineWidth = 1;
  sx.setLineDash([3, 3]);
  const xmed = Math.round(sX(SDOM[smet].med)) + 0.5;
  sx.beginPath(); sx.moveTo(xmed, S_AXIS); sx.lineTo(xmed, SH); sx.stroke();
  sx.setLineDash([]);

  for(let q = q0; q <= q1; q++){
    const row = SROWS[q];
    const y = q*S_ROW - st + S_AXIS, mid = y + S_ROW/2;
    if(y + S_ROW <= S_AXIS || y >= SH) continue;
    if(q & 1){
      sx.fillStyle = 'rgba(27,27,27,0.028)';
      sx.fillRect(S_GID - 8, y, SW - S_GID + 8, S_ROW);
    }
    if(q === selRow){
      sx.fillStyle = 'rgba(43,108,176,0.07)';
      sx.fillRect(S_GID - 8, y, SW - S_GID + 8, S_ROW);
    }
    const b = row.b[smet];
    const xl = sX(b.wl), xh = sX(b.wh), x1 = sX(b.q1), x3 = sX(b.q3);
    // whisker, then box, then median, then the reads over the top
    sx.strokeStyle = bcol(row.cid, 66); sx.lineWidth = 1;
    sx.beginPath(); sx.moveTo(xl, mid); sx.lineTo(xh, mid); sx.stroke();
    for(const x of [xl, xh]){
      const xr = Math.round(x) + 0.5;
      sx.beginPath(); sx.moveTo(xr, mid - 4.5); sx.lineTo(xr, mid + 4.5);
      sx.stroke();
    }
    sx.fillStyle = bcol(row.cid, 89);
    sx.fillRect(x1, mid - half*0.66, Math.max(1, x3 - x1), half*1.32);
    sx.strokeStyle = bcol(row.cid, 50);
    sx.strokeRect(Math.round(x1) + 0.5, Math.round(mid - half*0.66) + 0.5,
                  Math.max(1, Math.round(x3 - x1)), Math.round(half*1.32));
    sx.strokeStyle = bcol(row.cid, 30); sx.lineWidth = 2;
    const xm = Math.round(sX(b.med)) + 0.5;
    sx.beginPath(); sx.moveTo(xm, mid - half*0.72); sx.lineTo(xm, mid + half*0.72);
    sx.stroke();
    sx.lineWidth = 1;
    if(dots){
      sx.fillStyle = bcol(row.cid, 40); sx.globalAlpha = 0.5;
      for(let z = 0; z < row.idx.length; z++){
        const i = row.idx[z];
        sx.beginPath();
        sx.arc(sX(col[i]), mid + jit(i)*jhalf, 1.7, 0, 6.2831853);
        sx.fill();
      }
      sx.globalAlpha = 1;
      // The selection, then the hovered and reported reads over the top of it,
      // so the one read the page is talking about is still findable inside a
      // selection of forty.
      if(selN > 1){
        sx.fillStyle = '#2b6cb0';
        for(let z = 0; z < row.idx.length; z++){
          const i = row.idx[z];
          if(!selMask[i]) continue;
          sx.beginPath();
          sx.arc(sX(col[i]), mid + jit(i)*jhalf, 2.2, 0, 6.2831853);
          sx.fill();
        }
      }
      for(const i of [shov, sel]){
        if(i < 0 || rowOf(i) !== q) continue;
        const cx = sX(col[i]), cy = mid + jit(i)*jhalf;
        sx.fillStyle = i === sel ? '#2b6cb0' : '#1b1b1b';
        sx.beginPath(); sx.arc(cx, cy, 2.6, 0, 6.2831853); sx.fill();
        sx.strokeStyle = '#fff'; sx.lineWidth = 1;
        sx.beginPath(); sx.arc(cx, cy, 4.2, 0, 6.2831853); sx.stroke();
      }
    }
  }
  sx.restore();

  // ---------------------------------------------------------------- the gutter
  sx.save();
  sx.beginPath(); sx.rect(0, S_AXIS, S_GID - 10, SH - S_AXIS); sx.clip();
  sx.fillStyle = '#fff'; sx.fillRect(0, S_AXIS, S_GID - 10, SH - S_AXIS);
  sx.font = '600 10.5px ui-monospace,Menlo,Consolas,monospace';
  sx.textBaseline = 'middle';
  for(let q = q0; q <= q1; q++){
    const row = SROWS[q], mid = q*S_ROW - st + S_AXIS + S_ROW/2;
    if(mid < S_AXIS - S_ROW || mid > SH + S_ROW) continue;
    sx.fillStyle = bcol(row.cid, 62);
    sx.fillRect(4, mid - 5, 8, 10);
    sx.textAlign = 'left';
    sx.fillStyle = bcol(row.cid, 32);
    sx.fillText(CNAME(row.cid), 17, mid);
    sx.textAlign = 'right';
    sx.fillStyle = '#aaa';
    sx.fillText(row.size + '', S_GID - 14, mid);
  }
  sx.textAlign = 'left';
  sx.restore();

  // ------------------------------------------------------------------ the axis
  sx.fillStyle = '#fff'; sx.fillRect(0, 0, SW, S_AXIS);
  sx.strokeStyle = '#e0e0e0'; sx.lineWidth = 1;
  sx.beginPath(); sx.moveTo(0, S_AXIS - 0.5); sx.lineTo(SW, S_AXIS - 0.5);
  sx.stroke();
  const g = spanel();
  sx.font = '600 11px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,' +
            'Helvetica,Arial,sans-serif';
  sx.fillStyle = '#444'; sx.textAlign = 'left'; sx.textBaseline = 'top';
  sx.fillText(SM[smet].t + '  ·  bp', g.x0 + S_PAD, 3);
  sx.font = '10px ui-monospace,Menlo,Consolas,monospace';
  sx.textAlign = 'center'; sx.textBaseline = 'middle';
  for(const v of sticks()){
    const x = sX(v);
    sx.fillStyle = '#888'; sx.fillText(kb(v), x, S_AXIS - 10);
    sx.strokeStyle = '#eee';
    sx.beginPath(); sx.moveTo(x, S_AXIS - 4); sx.lineTo(x, S_AXIS); sx.stroke();
  }
  sx.textAlign = 'left'; sx.textBaseline = 'alphabetic';
}

// --------------------------------------------------------------- stats input
// A dot is hit in SCREEN space, not in value space, so the test that a click
// lands on the read it looks like it landed on is a test of the same scale the
// drawing used.
function statsHit(e){
  if(!SROWS) return null;
  const b = scv.getBoundingClientRect();
  const px = e.clientX - b.left, py = e.clientY - b.top;
  if(py < S_AXIS) return null;
  const q = Math.floor((py - S_AXIS + sscroll.scrollTop)/S_ROW);
  if(q < 0 || q >= SROWS.length) return null;
  const row = SROWS[q], col = A[SM[smet].k];
  const mid = q*S_ROW - sscroll.scrollTop + S_AXIS + S_ROW/2;
  const jhalf = S_ROW*0.5 - 9, g = spanel();
  let best = -1, bd = 30;
  if(px >= g.x0 - 2)
    for(let z = 0; z < row.idx.length; z++){
      const i = row.idx[z];
      const dx = sX(col[i]) - px, dy = mid + jit(i)*jhalf - py;
      const d = dx*dx + dy*dy;
      if(d < bd){ bd = d; best = i; }
    }
  return {row, q, i:best, px, py};
}
// Both statistics, whichever one is drawn: the row is a cluster and not a
// column, and the comparison between the two is the reason to look at it.
function sdetail(row){
  let s = CLABEL(row.cid) + '   ' + row.size + ' reads';
  for(let p = 0; p < 2; p++){
    const b = row.b[p], r = v => fmt(Math.round(v));
    s += '\n' + SM[p].t + (p === smet ? '   (drawn)' : '') +
         '\n  median ' + r(b.med) + '   IQR ' + r(b.q1) + '–' + r(b.q3) +
         '\n  range ' + r(b.min) + '–' + r(b.max);
  }
  return s;
}
sscroll.addEventListener('scroll', () => drawStats(), {passive:true});
scv.addEventListener('mousemove', e => {
  const h = statsHit(e);
  if(!h){ tipOff(); if(shov >= 0){ shov = -1; drawStats(); } return; }
  tipAt(e, h.i >= 0 ? detail(h.i) : sdetail(h.row));
  if(h.i !== shov){ shov = h.i; drawStats(); }
});
scv.addEventListener('mouseleave', () => {
  tipOff(); if(shov >= 0){ shov = -1; drawStats(); }
});
scv.addEventListener('click', e => {
  const h = statsHit(e);
  if(h && h.i >= 0) select(h.i, 'stats');
});
"""
JS += r"""
// ===================================================================== chrome
const tip = $('#tip');
function tipAt(e, text){
  tip.textContent = text;
  tip.style.display = 'block';
  tip.style.left = Math.min(window.innerWidth - 380, e.clientX + 14) + 'px';
  tip.style.top  = Math.min(window.innerHeight - 130, e.clientY + 16) + 'px';
}
function tipOff(){ tip.style.display = 'none'; }
window.addEventListener('mouseleave', tipOff);

for(const el of document.querySelectorAll('.tab'))
  el.addEventListener('click', () => show(el.dataset.tab));
for(const id of ['edges','labels','wfloor','psize'])
  $('#' + id).addEventListener('input', () => {
    $('#wfloorv').textContent = (+$('#wfloor').value).toFixed(2);
    drawMap();
  });
$('#fitmap').addEventListener('click', () => { fitMap(); drawMap(); });
$('#zin').addEventListener('click', () => zoomX(0.6, (xLo + xHi)/2));
$('#zout').addEventListener('click', () => zoomX(1/0.6, (xLo + xHi)/2));
$('#zfit').addEventListener('click',
  () => { xLo = D.tLo; xHi = D.tHi; drawReads(); });
$('#zwin').addEventListener('click', () => { winView(); drawReads(); });
$('#smetric').addEventListener('change', () => { sstat(); drawStats(); });
$('#sdots').addEventListener('change', () => drawStats());
$('#find').addEventListener('keydown', e => {
  if(e.key !== 'Enter') return;
  const q = $('#find').value.trim().toLowerCase();
  if(!q) return;
  let hit = -1;
  for(let i=0;i<N;i++) if(A.ids[i].toLowerCase().startsWith(q)){ hit = i; break; }
  if(hit < 0)
    for(let i=0;i<N;i++) if(A.ids[i].toLowerCase().includes(q)){ hit = i; break; }
  if(hit >= 0) select(hit, 'find');
  else $('#status').textContent = 'no read id matches ' + q;
});
window.addEventListener('keydown', e => {
  if(e.ctrlKey && e.shiftKey && (e.key === 'E' || e.key === 'e')){
    e.preventDefault(); toggleEdit(); return;
  }
  if(e.key === 'Escape'){
    if(!$('#cmenu').hidden){ closeMenu(); return; }
    if(!$('#modal').hidden){ askClose(); return; }
  }
  if(edit && (e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')){
    e.preventDefault(); undo(); return;
  }
  if(e.target.tagName === 'INPUT') return;
  // Edit mode's two one-key moves, and both are about the selection: U is the
  // way out of every cluster at once, N the way into a new one.  Bare letters
  // because they are the two things done a hundred times in a sitting, and
  // they do nothing whatever until edit mode is on -- nor behind a dialog,
  // where a keystroke that quietly moved reads would be read as an answer to
  // the question on screen.
  if(edit && $('#modal').hidden && !e.ctrlKey && !e.metaKey && !e.altKey){
    if(e.key === 'u' || e.key === 'U'){
      e.preventDefault();
      if(selN) assignMany(selected(), UNCL, sel);
      else editNote('nothing selected -- U moves the selection to unclustered');
      return;
    }
    if(e.key === 'n' || e.key === 'N'){
      e.preventDefault(); newCluster(); return;
    }
  }
  if(e.key === '1') show('map');
  if(e.key === '2') show('reads');
  if(e.key === '3') show('stats');
  if(e.key === 'Escape'){
    select(-1, tab);
  }
});
let rt = null;
window.addEventListener('resize', () => {
  clearTimeout(rt);
  rt = setTimeout(() => {
    sizeMap(); sizeReads(); sizeStats();
    if(tab === 'map') drawMap();
    else if(tab === 'stats') drawStats();
    else drawReads();
  }, 80);
});

// ======================================================================= boot
(async function(){
  const el = $('#boot');
  if(typeof DecompressionStream === 'undefined'){
    el.textContent = 'This browser has no DecompressionStream, so the page '
      + 'cannot unpack itself. Chrome 80+, Firefox 113+ or Safari 16.4+ will.';
    return;
  }
  try{
    A = unpack(await gunzip(ARR_B64), D.man);
    A.ids = A.idtext;
    SEQ = await gunzip(SEQ_B64);
  }catch(err){ el.textContent = 'Could not unpack the page: ' + err; return; }
  N = A.x.length; M = A.ew.length;
  selMask = new Uint8Array(N);
  el.remove();
  $('#panels').style.visibility = 'visible';
  prep(); layout(); winView(); statsPrep();
  sizeMap(); sizeReads(); sizeStats(); fitMap();
  $('#wfloorv').textContent = (+$('#wfloor').value).toFixed(2);
  status();
  show('map');
})();
"""



JS += r"""
// =================================================================== the edits
// Ctrl+Shift+E, behind a confirmation, makes the page a curation tool: a read
// dragged onto another read takes that read's cluster.
//
// ONE RULE, BOTH TABS.  The drop target is a READ, not a cluster picked from a
// list, so the same gesture works on the map (drop on a dot) and on the reads
// tab (drop on a row) with no per-tab vocabulary -- and `unclustered` needs no
// special target, because the unclustered reads are reads.
//
// REASSIGNING A READ MOVES ITS ROW.  The reads tab's blocks are the clusters
// as they now stand, so a read given cluster 7 leaves its old block and is
// drawn in cluster 7's -- see `blockList`.  The BLOCK order is still the
// t-SNE's and the three tabs still share it, and the read the gesture was
// about holds its screen line while the rows move around it, so the page does
// not jump out from under the cursor.  An edited read takes its new cluster's
// colour and is marked in the gutter.
//
// The edits live in the page and nowhere else.  Leaving edit mode discards
// them, and the CSV is the record -- so the exit asks first when there are any.
//
// A DROPPED READ CAN BE MOVED INTO A CLUSTER like any other: it is on the page
// because the question "did a gate lose a read it should not have" is a
// question about this run, and answering it yes has to be expressible.  What
// cannot be expressed is the other direction -- dropping a read onto a dropped
// read makes it `unclustered`, bare, because `low_qs` is the target read's
// history and not a verdict a curator can hand to a different read.
//
// RUN_CLIDS is every cluster the run made, in id order.
const RUN_CLIDS = D.blocks.map(b => b[0]).filter(c => c >= 0)
                          .sort((a, b) => a - b);
let edit = false, manual = null, nEdits = 0, undoStack = [];
let drag = null;                 // {i, list, set, x, y, t} while reads are held
// The clusters made in this session, the empty ones included: an empty cluster
// is a block you drop reads into, so it has to outlive the moment it was made
// with nothing in it.  Cleared with the edits, because it is one of them.
let newClusters = [];

// Every id a read could be given now: the run's, the ones made here, and any a
// loaded CSV brought with it.
function clusterIds(){
  const s = new Set(RUN_CLIDS);
  for(const c of newClusters) if(c >= 0) s.add(c);
  if(manual) for(let i = 0; i < N; i++) if(manual[i] >= 0) s.add(manual[i]);
  return [...s].sort((a, b) => a - b);
}

// The cluster a read is in NOW: its manual assignment if it has one, the run's
// otherwise.  Everything that draws a read's colour goes through this, so one
// function is the difference between edited and not.
function cl(i){ return manual ? manual[i] : A.cluster[i]; }

let sizeCache = null;
// A cluster's size is a count of the reads in it NOW, so the tooltip and the
// menu do not go on quoting the run's numbers at someone who has just changed
// them.  Rebuilt lazily, thrown away by every edit.
function csize(c){
  if(!manual) return D.size[c] || 0;
  if(!sizeCache){
    sizeCache = new Map();
    for(let i = 0; i < N; i++)
      sizeCache.set(manual[i], (sizeCache.get(manual[i]) || 0) + 1);
  }
  return sizeCache.get(c) || 0;
}
// Everything an edit invalidates, in one place: `prep` groups the map's dots
// by cluster and `statsPrep` groups the box plots' reads, and both of them
// read `cl`, so both have to be redone when `cl` changes for one read.
// `anchor` is the read the gesture was about.  The reads tab is re-laid out on
// every edit, so the page has to hold something still: that read keeps the
// screen line it was on and the rows move around it.  Without it a drop near
// the bottom of cluster 3 leaves you somewhere in the middle of cluster 40.
function afterEdit(anchor){
  sizeCache = null;
  nEdits = countEdits();
  const held = (anchor >= 0 && RANK) ? rowTop[RANK[anchor]] - scroller.scrollTop
                                     : null;
  prep(); layout(); statsPrep(); sizeReads();
  if(held !== null)
    scroller.scrollTop = Math.max(0, Math.min(
      Math.max(0, totalPx - (RH - AXIS_H)), rowTop[RANK[anchor]] - held));
  editStatus();
  redraw();
}
function editOn(){
  edit = true;
  manual = Int32Array.from(A.cluster);
  undoStack = []; newClusters = [];
  document.body.classList.add('editing');
  $('#editbar').hidden = false;
  afterEdit();
}
function editOff(){
  edit = false; manual = null; undoStack = []; drag = null; newClusters = [];
  editNote('');
  document.body.classList.remove('editing');
  $('#editbar').hidden = true;
  $('#ghost').hidden = true;
  closeMenu();
  afterEdit();
}
// The cluster a drop onto this read means: its own, unless it has none, in
// which case it means the bare pool and not that read's drop reason.
const dropCluster = c => c < 0 ? UNCL : c;
// ONE GESTURE IS ONE EDIT, whether it moved one read or forty: the undo stack
// holds a batch a step, so ctrl-Z after dropping a selection puts the whole
// selection back where it was rather than releasing it a read at a time.
function assignMany(list, c, anchor){
  if(!manual) return;
  const batch = [];
  for(const i of list)
    if(i >= 0 && manual[i] !== c){ batch.push([i, manual[i]]); manual[i] = c; }
  if(!batch.length) return;
  undoStack.push(batch);
  afterEdit(anchor === undefined ? batch[0][0] : anchor);
}
// The reads a gesture on read `i` acts on: the selection when `i` is part of
// it, and `i` alone otherwise -- so dragging a read that is not in the
// selection never moves the selection by surprise.
const actOn = i => (isSel(i) && selN > 1) ? selected() : [i];
function undo(){
  const u = undoStack.pop();
  if(!u) return;
  for(const [i, c] of u) manual[i] = c;
  afterEdit(u[0][0]);
}
function countEdits(){
  if(!manual) return 0;
  let n = 0;
  for(let i = 0; i < N; i++) if(manual[i] !== A.cluster[i]) n++;
  return n;
}
// A CLUSTER A CURATOR MAKES.  The next free id, and empty unless something is
// selected: an empty cluster is a dashed band on the reads tab and reads are
// dropped onto it, so "make it, then fill it" needs no reads to exist first.
// A new cluster is a cluster in every other respect -- it has a colour, a
// block, a box plot and a line in the CSV, and `trc` will never have heard of
// it, which is the point.
// The next id no cluster on this page has used, remembered so that a cluster
// stays on the page while it is still empty.
function newClusterId(){
  const ids = clusterIds();
  const c = (ids.length ? ids[ids.length - 1] : -1) + 1;
  newClusters.push(c);
  return c;
}
// Reads into a cluster that did not exist a moment ago: the white space drop,
// and `N` with a selection.  One edit, so one ctrl-Z takes the reads back --
// and leaves the empty cluster, which is the block they were dropped into.
function clusterFrom(list, anchor){
  if(!list.length) return -1;
  const c = newClusterId();
  assignMany(list, c, anchor);
  const n = csize(c);
  editNote((n === 1 ? '1 read' : fmt(n) + ' reads') + ' moved into cluster '
           + c + ', new');
  return c;
}
function newCluster(){
  if(!edit) return;
  const last = newClusters.length ? newClusters[newClusters.length - 1] : -1;
  // One empty cluster at a time: a second N while the first is still empty is
  // someone looking for the band, not asking for another one.
  if(!selN && last >= 0 && csize(last) === 0){
    gotoCluster(last);
    editNote('cluster ' + last + ' is still empty');
    return;
  }
  if(selN){ clusterFrom(selected(), sel); return; }
  const c = newClusterId();
  afterEdit();
  gotoCluster(c);
  editNote('cluster ' + c + ' is empty -- drop reads on its band');
}
// The reads tab is where a cluster is a place, so that is where this goes.
function gotoCluster(c){
  if(tab !== 'reads') show('reads');
  const blk = BLOCKS.find(b => b.cid === c);
  if(blk) scroller.scrollTop = Math.max(0, Math.min(
    Math.max(0, totalPx - (RH - AXIS_H)), blk.top - (RH - AXIS_H)/3));
  drawReads();
}
// The edit bar's own line, for what a keystroke did.  It clears itself: a note
// that stayed would be read as a state the page is in.
let noteT = null;
function editNote(s){
  $('#editmsg').textContent = s;
  clearTimeout(noteT);
  if(s) noteT = setTimeout(() => { $('#editmsg').textContent = ''; }, 5000);
}
function editStatus(){
  $('#editn').textContent = nEdits === 1 ? '1 read moved'
                                         : fmt(nEdits) + ' reads moved';
}
function redraw(){ drawMap(); drawReads(); drawStats(); status(); }

// ------------------------------------------------------------------- the CSV
// Every read, not only the moved ones, and every read now means every record
// of the input file: the file is a labelling that stands on its own and diffs
// against reads.tsv line for line, rather than a list of corrections that
// means nothing without the run beside it.  `CNAME` writes the run's own
// composite labels, so `unclustered:low_qs` round-trips.
function csv(){
  const out = ['read_id,cluster,manual_cluster,changed'];
  for(let i = 0; i < N; i++){
    const a = A.cluster[i], b = manual ? manual[i] : a;
    out.push(A.ids[i] + ',' + CNAME(a) + ',' + CNAME(b) + ',' + (a === b ? 0 : 1));
  }
  return out.join('\n') + '\n';
}
function download(){
  const blob = new Blob([csv()], {type:'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = D.sample + '.browser.manual.csv';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

// ------------------------------------------------------------- the confirmation
// A modal rather than confirm(): it has to say what edit mode does, and the
// keystroke is deliberately obscure enough that someone who hits it by accident
// needs telling.
let modalYes = null;
function ask(title, body, yes, onYes, danger){
  modalYes = onYes;
  $('#mno').hidden = false;
  $('#mtitle').textContent = title;
  $('#mbody').innerHTML = body;
  $('#myes').textContent = yes;
  $('#myes').className = danger ? 'danger' : '';
  $('#modal').hidden = false;
  $('#myes').focus();
}
// A report, not a question: the upload has already happened by the time it is
// shown, so there is nothing to cancel and no second button to offer.
function tell(title, body){
  ask(title, body, 'OK', null);
  $('#mno').hidden = true;
}
function askClose(){
  $('#modal').hidden = true; modalYes = null; $('#mno').hidden = false;
}
$('#myes').addEventListener('click', () => {
  const f = modalYes; askClose(); if(f) f();
});
$('#mno').addEventListener('click', askClose);
$('#modal').addEventListener('mousedown', e => {
  if(e.target === $('#modal')) askClose();
});

function toggleEdit(){
  if(!edit){
    ask('Enter edit mode?',
        'Drag a read onto another read to take that read&rsquo;s cluster, or '
        + 'onto a cluster&rsquo;s label in the left gutter to join that '
        + 'cluster. Drop reads in the white space BETWEEN two clusters and a '
        + 'new cluster is made to hold them. Drag a read that is part of a '
        + 'selection and the whole selection moves with it, as one edit that '
        + 'one ctrl-Z undoes.<br><br>'
        + '<b>N</b> makes a new cluster &mdash; the selection&rsquo;s, or an '
        + 'empty band on the reads tab to drop reads into. <b>U</b> moves the '
        + 'selection to <b>unclustered</b>. Right-click a read for '
        + '<b>move to&hellip;</b>.<br><br>'
        + 'On the reads tab the blocks are the clusters as they now stand, so '
        + 'a read you move is drawn in its new cluster from then on.<br><br>'
        + 'Reads dropped before the graph was built are on the reads tab '
        + 'and can be moved into a cluster like any other &mdash; that is '
        + 'how this page says a gate lost a read it should not have.<br><br>'
        + 'Nothing is written to the run. The edits live in this page until you '
        + '<b>Download CSV</b>, and leaving edit mode discards them.',
        'Enter edit mode', editOn);
    return;
  }
  if(nEdits){
    ask('Leave edit mode?',
        '<b>' + nEdits + '</b> edit' + (nEdits === 1 ? '' : 's') + ' '
        + (nEdits === 1 ? 'has' : 'have') + ' not been downloaded. Leaving '
        + 'discards ' + (nEdits === 1 ? 'it' : 'them') + ' &mdash; the CSV is '
        + 'the only record.',
        'Discard and leave', editOff, true);
    return;
  }
  editOff();
}
$('#newcl').addEventListener('click', newCluster);
$('#dlcsv').addEventListener('click', download);
$('#editx').addEventListener('click', toggleEdit);

// ------------------------------------------------------------------ dragging
// The map drags a dot; the reads tab drags a row BY ITS GUTTER -- the cluster
// id and colour chip left of the sequence.  The sequence area keeps panning,
// because losing the horizontal pan in edit mode would cost more than the drag
// is worth, and the gutter is where a row's cluster is written anyway.
//
// The drop target is whatever read is under the cursor when it is released.
// THE READS A DRAG CARRIES ARE FIXED WHEN IT STARTS.  `drag.list` is settled
// by the mousedown and the ghost says how many there are, so what will be
// moved is decided and visible before the cursor goes anywhere -- a set read
// off the selection at the drop instead would be a set nothing on the page had
// shown you.
function dragStart(i, e){
  const list = actOn(i);
  drag = {i, list, set:new Set(list), x:e.clientX, y:e.clientY, t:null,
          moved:false};
  const g = $('#ghost');
  g.innerHTML = ghostHtml();
  g.hidden = false;
  dragMove(e);
}
// What is being carried, and -- over the white space -- what it would become.
function ghostHtml(){
  const i = drag.i, t = drag.t;
  return '<span class="sw" style="background:' + ccol(cl(i)) + '"></span>'
       + (drag.list.length > 1 ? fmt(drag.list.length) + ' reads'
                               : A.ids[i].slice(0, 16))
       + (t && t.gap >= 0 ? '  \u2192 new cluster' : '');
}
const isHeld = i => !!(drag && drag.set.has(i));
function dragMove(e){
  if(!drag) return;
  if(Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3)
    drag.moved = true;
  const g = $('#ghost');
  g.style.left = (e.clientX + 12) + 'px';
  g.style.top  = (e.clientY + 12) + 'px';
  const key = t => t ? t.i + '/' + t.blk + '/' + t.gap : '-';
  const was = key(drag.t);
  drag.t = dropTarget(e);
  const t = drag.t;
  g.className = (t && t.i !== drag.i) ? 'ok' : '';
  if(key(t) !== was){
    g.innerHTML = ghostHtml();
    if(tab === 'map') drawMap(); else drawReads();
  }
}
// WHAT A DROP MEANS, in one place, and there are three things it can mean.
//
// A READ is the target it always was -- drop on a read, take that read's
// cluster -- and the same gesture works on both tabs with no per-tab
// vocabulary.  A BLOCK is a target too: its label column on the reads tab, and
// the dashed band of a cluster with nothing in it yet.  A block is the only
// way to reach an empty cluster, and on rows six pixels tall it is a target
// that can be hit.  And the WHITE SPACE between two blocks is a cluster that
// does not exist yet: dropping there makes one and puts the reads in it.
//
// The three read in one line off the page: onto reads, into a labelled block,
// or into the gap where no cluster is -- so what a drop will do is a question
// about where the cursor is and nothing else.
//
// A drop onto a block the run named for a gate means the bare pool and not
// that gate, by the same rule as a drop onto a read it dropped: `low_qs` is
// the run's account of those reads and not a verdict a curator can hand out.
function dropTarget(e){
  if(tab === 'map'){
    const r = mcv.getBoundingClientRect();
    const i = pick(e.clientX - r.left, e.clientY - r.top);
    return i >= 0 ? {i, blk:-1, gap:-1, c:dropCluster(cl(i))} : null;
  }
  const h = readsHit(e);
  if(h.row >= 0 && h.px >= GID_W){
    const i = ORDER[h.row];
    return {i, blk:-1, gap:-1, c:dropCluster(cl(i))};
  }
  if(h.blk >= 0)
    return {i:-1, blk:h.blk, gap:-1, c:dropCluster(BLOCKS[h.blk].cid)};
  const g = gapAtY(h.cy);
  if(g >= 0) return {i:-1, blk:-1, gap:g, c:null};
  return null;
}
function dragEnd(e){
  if(!drag) return;
  const {i, list, t, moved} = drag;
  drag = null;
  $('#ghost').hidden = true;
  if(moved && t && t.gap >= 0) clusterFrom(list, i);
  else if(moved && t && t.i !== i) assignMany(list, t.c, i);
  else if(!moved) select(i, tab);
  redraw();
}

// ------------------------------------------------------------- move to...
// Right-click a read for the same assignment the drag makes, when the cluster
// you want is nowhere near the cursor.  Its own edges come first: the question
// "what is this read actually attached to" is the one the map answers by
// lighting them up, and it is the question a curator is asking here too.
let menuRead = -1;
function neighbourClusters(i){
  const w = new Map();
  for(let e = 0; e < M; e++){
    let o = -1;
    if(A.ei[e] === i) o = A.ej[e];
    else if(A.ej[e] === i) o = A.ei[e];
    else continue;
    const c = cl(o);
    w.set(c, (w.get(c) || 0) + A.ew[e]);
  }
  return [...w.entries()].sort((a, b) => b[1] - a[1]);
}
function menuRow(c, i, note){
  const here = cl(i) === c;
  return '<div class="mi' + (here ? ' on' : '') + '" data-c="' + c + '">'
       + '<span class="sw" style="background:' + ccol(c) + '"></span>'
       + '<span class="mn">' + CLABEL(c)
       + '</span><span class="mx">' + (note || (c < 0 ? ''
       : fmt(csize(c)) + ' reads')) + '</span></div>';
}
function openMenu(i, cx, cy){
  if(!edit || i < 0) return;
  menuRead = i;
  const near = neighbourClusters(i);
  const many = actOn(i).length;
  let h = '<div class="mh">'
        + (many > 1 ? fmt(many) + ' reads selected' : A.ids[i].slice(0, 18))
        + '<br><span>now ' + CLABEL(cl(i)) + '</span></div>';
  if(near.length){
    h += '<div class="ms">attached to</div>';
    for(const [c, wsum] of near.slice(0, 6))
      h += menuRow(c, i, wsum.toFixed(2) + ' weight');
  }
  h += '<div class="ms">move to</div>'
     + '<input id="mfilt" placeholder="cluster id" spellcheck="false">'
     + '<div id="mlist">' + menuRow(UNCL, i) + allRows(i, '') + '</div>';
  const m = $('#cmenu');
  m.innerHTML = h;
  m.hidden = false;
  // kept on screen: a read near the bottom right must not open a menu that
  // runs off it
  const w = 214, hh = Math.min(360, m.scrollHeight || 360);
  m.style.left = Math.min(cx, window.innerWidth - w - 8) + 'px';
  m.style.top  = Math.min(cy, window.innerHeight - hh - 8) + 'px';
  const f = $('#mfilt');
  if(f){
    f.addEventListener('input', () => {
      $('#mlist').innerHTML = menuRow(UNCL, menuRead) +
                              allRows(menuRead, f.value.trim());
    });
    f.focus();
  }
}
function allRows(i, q){
  let h = '';
  for(const c of clusterIds()){
    if(q && !String(c).startsWith(q)) continue;
    h += menuRow(c, i);
  }
  return h;
}
function closeMenu(){ $('#cmenu').hidden = true; menuRead = -1; }
$('#cmenu').addEventListener('mousedown', e => {
  const row = e.target.closest ? e.target.closest('.mi') : null;
  if(!row) return;
  e.preventDefault();
  assignMany(actOn(menuRead), +row.dataset.c, menuRead);
  closeMenu();
});
window.addEventListener('mousedown', e => {
  if($('#cmenu').hidden) return;
  if(!$('#cmenu').contains(e.target)) closeMenu();
}, true);

// ------------------------------------------------------------ back in again
// A downloaded CSV is the only record of a session's edits, so it has to be
// able to become one again.  Matched BY READ ID and never by row order: the
// file may have been sorted, filtered or round-tripped through a spreadsheet
// since it was written, and a positional read would then relabel the wrong
// reads with nothing on the page looking wrong.
function applyCsv(text){
  const lines = text.split(/\r?\n/).filter(s => s.length);
  if(!lines.length) return {err:'the file is empty'};
  const hdr = lines[0].split(',').map(s => s.trim().toLowerCase());
  const ci = hdr.indexOf('read_id');
  let cc = hdr.indexOf('manual_cluster');
  if(cc < 0) cc = hdr.indexOf('cluster');
  if(ci < 0 || cc < 0)
    return {err:'no read_id and manual_cluster columns in the header'};
  const pos = new Map();
  for(let i = 0; i < N; i++) pos.set(A.ids[i], i);
  const next = Int32Array.from(A.cluster);
  let applied = 0, unknown = 0, bad = 0, seen = 0;
  for(let r = 1; r < lines.length; r++){
    const f = lines[r].split(',');
    if(f.length <= Math.max(ci, cc)) continue;
    seen++;
    const i = pos.get(f[ci].trim());
    if(i === undefined){ unknown++; continue; }
    const c = codeOf(f[cc].trim());
    if(!Number.isFinite(c)){ bad++; continue; }
    next[i] = c;
    applied++;
  }
  manual = next;
  afterEdit();
  return {rows:seen, applied, unknown, bad, edits:nEdits};
}
$('#upcsv').addEventListener('change', e => {
  const f = e.target.files && e.target.files[0];
  if(!f) return;
  const fr = new FileReader();
  fr.onload = () => {
    const r = applyCsv(String(fr.result));
    tell(r.err ? 'That file could not be read' : 'Loaded ' + f.name,
        r.err ? r.err
              : '<b>' + fmt(r.applied) + '</b> of ' + fmt(r.rows)
                + ' rows matched a read on this page, leaving <b>' + r.edits
                + '</b> read' + (r.edits === 1 ? '' : 's')
                + ' assigned away from the run.'
                + (r.unknown ? '<br>' + fmt(r.unknown) + ' read ids are not on '
                   + 'this page and were skipped.' : '')
                + (r.bad ? '<br>' + fmt(r.bad) + ' rows had no readable '
                   + 'cluster.' : ''));
  };
  fr.readAsText(f);
  e.target.value = '';
});
$('#upbtn').addEventListener('click', () => $('#upcsv').click());

// The Ctrl+Shift+E keystroke itself lives in the page's own keydown handler,
// so that Escape and the find box are not fought over by two listeners.

// Right-click is the same assignment as the drag, for when the cluster you want
// is not on screen next to the read you are moving.
for(const cv of [mcv, rcv])
  cv.addEventListener('contextmenu', e => {
    if(!edit) return;                   // outside edit mode the browser's own
    e.preventDefault();                 // menu is still what a right-click is for
    const r = cv === mcv ? mcv.getBoundingClientRect() : null;
    const i = cv === mcv
      ? pick(e.clientX - r.left, e.clientY - r.top, selN > 1)
      : (() => { const h = readsHit(e);
                 return h.row >= 0 ? ORDER[h.row] : -1; })();
    if(i >= 0) openMenu(i, e.clientX, e.clientY);
  });
"""

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>__CSS__</style>
<header>
  <h1>__SAMPLE__ <small>__HEAD__</small></h1>
  <div id="tabs">
    <div class="tab on" data-tab="map">map</div>
    <div class="tab" data-tab="reads">reads</div>
    <div class="tab" data-tab="stats">cluster stats</div>
    <span class="grow"></span>
    <input id="find" placeholder="read id, then Enter" spellcheck="false">
  </div>
</header>
  <div id="editbar" hidden>
    <b>EDIT MODE</b>
    <span id="editn">0 edits</span>
    <button id="newcl">+ cluster</button>
    <span class="hint">drag reads onto a read, onto a cluster's label, or
      into the space between two clusters for a new one · <b>N</b> new cluster
      · <b>U</b> unclustered · right-click for <b>move to…</b> · ctrl-Z
      undoes</span>
    <span id="editmsg"></span>
    <span class="grow"></span>
    <button id="upbtn">Load CSV…</button>
    <button id="dlcsv">Download CSV</button>
    <button id="editx">Exit</button>
    <input type="file" id="upcsv" accept=".csv,text/csv" hidden>
  </div>
<div id="panels" style="visibility:hidden">
  <div class="panel" id="mapPanel">
    <div class="ctl">
      <label><input type="checkbox" id="edges" checked> shared-k-mer edges</label>
      <label>weight ≥ <input type="range" id="wfloor" min="0" max="0.99"
             step="0.01" value="0"> <b id="wfloorv">0.00</b></label>
      <label>point <input type="range" id="psize" min="1" max="6" step="0.5"
             value="2.5"></label>
      <label><input type="checkbox" id="labels" checked> cluster ids</label>
      <button id="fitmap">fit</button>
      <span style="color:#999">· shift-drag selects a square</span>
      <span id="status"></span>
    </div>
    <div class="cv"><canvas id="mapCv"></canvas></div>
  </div>
  <div class="panel" id="readsPanel" hidden>
    <div class="ctl">
      <span>x: <button id="zout">−</button> <button id="zin">+</button>
        <button id="zwin">window</button> <button id="zfit">all</button>
        <span style="color:#999">· shift-wheel zooms, drag pans, wheel
        scrolls · shift-click selects a range, ctrl-click one
        read</span></span>
      <span class="key">__LEGEND__</span>
      <span style="color:#999">dashed: the window the graph was built from</span>
    </div>
    <div class="cv"><div id="readsScroll"><canvas id="readsCv"></canvas>
      <div id="readsSpacer"></div></div></div>
  </div>
  <div class="panel" id="statsPanel" hidden>
    <div class="ctl">
      <label>stat <select id="smetric">
        <option value="0">read length</option>
        <option value="1">telomere length (b0)</option></select></label>
      <label><input type="checkbox" id="sdots" checked> reads</label>
      <span style="color:#999">box: IQR, whiskers 1.5×IQR, dashed: the
        sample median</span>
      <span id="sstatus"></span>
    </div>
    <div class="cv"><div id="statsScroll"><canvas id="statsCv"></canvas>
      <div id="statsSpacer"></div></div></div>
  </div>
</div>
<div id="modal" hidden><div id="mcard">
  <div id="mtitle"></div>
  <div id="mbody"></div>
  <div id="mbtns"><button id="mno">Cancel</button><button id="myes"></button></div>
</div></div>
<div id="cmenu" hidden></div>
<div id="ghost" hidden></div>
<div id="tip"></div>
<div id="boot">unpacking __NREADS__ reads…</div>
<script>
const D = __D__;
const ARR_B64 = "__ARR__";
const SEQ_B64 = "__SEQ__";
__JS__
</script>
"""


def _swatch(rgb, label):
    return (f'<span class="sw" style="background:rgb{rgb}"></span>'
            f'<span style="margin-right:7px">{label}</span>')


# The t-SNE's own two numbers.  They are not command-line flags on `trc`:
# `--browser` is one switch over a view of a run, and a view that needed
# tuning to be read would not be worth writing.  `python -m trc.browser` does
# expose them, because a page built on its own is usually being compared.
PERPLEXITY = 5.0
N_ITER = 1000


def _page(sample, reads, edges, seqs, *, telo_bp, sub_bp, out_path,
          perplexity, seed, n_iter, row_px, flank):
    """The page itself, over arrays that came off disk or out of a live run."""
    ids = reads["ids"]
    n = len(ids)
    if edges[3]:
        log(f"browser: {edges[3]:,} of {edges[3] + len(edges[2]):,} edges "
            f"touch a read that is not in the read table and were dropped -- "
            f"the edge list and the table may be from different runs")
    blank = sum(1 for q in seqs if not q)
    if blank:
        log(f"browser: {blank:,} reads have no sequence and draw a blank row")
    xy, _ = embed(reads, edges, perplexity=perplexity, seed=seed,
                  n_iter=n_iter)

    order, starts = block_order(reads["cluster"], xy, reads["ingraph"])
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    t_lo, t_hi = -(sub_bp + flank), telo_bp + flank
    # A read with no boundary is anchored at its own first base: t = 0 is the
    # boundary, so the whole of what is drawn falls on the subtelomere side and
    # the telomere half of its row stays empty.  That IS the finding -- no
    # array was called in this read -- and it is drawn rather than asserted.
    blob, tmin, off, length = pack_windows(seqs, np.maximum(reads["b0"], 0),
                                           t_lo, t_hi)
    log(f"browser: {len(blob) / 1e6:.1f} Mbp packed over "
        f"t {t_lo:,} to {t_hi:,}")

    B = Blob()
    B.add("x", xy[:, 0], "float32").add("y", xy[:, 1], "float32")
    B.add("cluster", reads["cluster"], "int32")
    B.add("strength", reads["strength"], "float32")
    B.add("qs", reads["qs"], "float32")
    for c in ("degree", "nkmer", "b0", "sub_bp", "read_bp"):
        B.add(c, reads[c], "int32")
    B.add("orient", np.array([1 if o == "tail" else 0
                              for o in reads["orient"]]), "uint8")
    # The one flag the page branches on: a read with no graph is drawn as
    # sequence and as nothing else.
    B.add("ingraph", reads["ingraph"].astype(np.uint8), "uint8")
    B.add("hasb0", reads["hasb0"].astype(np.uint8), "uint8")
    B.add("order", order, "int32").add("rank", rank, "int32")
    B.add("tmin", tmin, "int32").add("off", off, "int32")
    B.add("slen", length, "int32")
    B.add("ei", edges[0], "int32").add("ej", edges[1], "int32")
    B.add("ew", edges[2], "float32")
    B.add_text("idtext", ids)

    # Every block's size, unclustered ones included, because the page reports
    # "of 214 unclustered:low_qs" in the same breath as "of 31 in cluster 3".
    sizes = {int(c): int(s) for c, _, s in starts}
    n_clusters = sum(1 for c, _, _ in starts if c >= 0)
    n_graph = int(reads["ingraph"].sum())
    data = {
        "sample": sample, "man": B.man,
        # The reason vocabulary, in the pipeline's order and indexed by the
        # page's codes, plus a sentence a person can read for each.
        "reasons": list(UNPLACED_REASONS),
        "reasonText": {k: REASON_TEXT[k] for k in UNPLACED_REASONS},
        "nGraph": n_graph,
        "tLo": int(t_lo), "tHi": int(t_hi),
        "rowPx": int(row_px), "gapRows": GAP_ROWS,
        "base": {b: list(c) for b, c in BASE_RGB.items()},
        "gap": list(GAP_RGB), "flank": FLANK_ALPHA,
        "sub_bp": sub_bp, "telo_bp": telo_bp,
        "blocks": [[int(c), int(s), int(m)] for c, s, m in starts],
        "size": sizes,
    }
    head = (f"{n:,} reads · {n_graph:,} in the graph · "
            f"{n_clusters:,} clusters")
    legend = "".join(_swatch(BASE_RGB[b], b) for b in "ACGT")

    page = (PAGE
            .replace("__TITLE__", f"{sample} t-SNE read browser")
            .replace("__CSS__", CSS)
            .replace("__SAMPLE__", html.escape(sample))
            .replace("__HEAD__", head)
            .replace("__LEGEND__", legend)
            .replace("__NREADS__", f"{n:,}")
            .replace("__D__", json.dumps(data, separators=(",", ":")))
            .replace("__ARR__", gz_b64(B.buf))
            .replace("__SEQ__", gz_b64(blob))
            .replace("__JS__", JS))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = f"{out_path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(page)
    os.replace(tmp, out_path)
    log(f"browser: {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


def build(outdir, *, sample=None, out_path=None, perplexity=PERPLEXITY,
          seed=0, n_iter=N_ITER, row_px=ROW_PX, flank=FLANK_BP):
    """One run directory in, one self-contained HTML file out."""
    sample, f = find_run(outdir, sample)
    reads = load_reads(f["reads"])
    ids = reads["ids"]
    pos = {r: k for k, r in enumerate(ids)}
    if len(pos) != len(ids):
        raise SystemExit(f"{f['reads']} has duplicate read ids")
    log(f"browser: {sample}, {len(ids):,} reads from "
        f"{os.path.basename(f['reads'])}")
    cfg = {}
    if f["json"]:
        with open(f["json"]) as fh:
            cfg = json.load(fh).get("params", {})
    seqs, _ = load_intake(f["intake"], f["seqs"], ids)
    return _page(sample, reads, load_edges(f["edges"], pos), seqs,
                 telo_bp=int(cfg.get("telo_bp", 2500)),
                 sub_bp=int(cfg.get("sub_bp", 1000)),
                 out_path=out_path or os.path.join(
                     os.path.abspath(outdir), f"{sample}.browser.html"),
                 perplexity=perplexity, seed=seed, n_iter=n_iter,
                 row_px=row_px, flank=flank)


def build_from_run(out_path, sample, rows, reads, dropped, G, *, telo_bp,
                   sub_bp, seed=0, perplexity=PERPLEXITY, n_iter=N_ITER,
                   row_px=ROW_PX, flank=FLANK_BP):
    """`trc --browser`: the page off a run that is still in memory.

    Nothing is read back from disk, so the page does not need the run to have
    been made with `--cache`.  `rows` is `<sample>.reads.tsv` as it was just
    written and the page draws exactly that table; `reads` and `dropped` are
    module 1's two lists, which between them are where the sequence is; and
    `G` carries the edge list the clusterer was handed.
    """
    table = _reads_table(rows)
    ids = table["ids"]
    pos = {r: k for k, r in enumerate(ids)}
    if len(pos) != len(ids):
        raise SystemExit("the read table has duplicate read ids")
    log(f"browser: {sample}, {len(ids):,} reads")
    seq = {r.read_id: r.seq for r in list(reads) + list(dropped)}
    edges = _edges(((reads[int(i)].read_id, reads[int(j)].read_id, float(w))
                    for (i, j), w in zip(G.pairs, G.weights)), pos)
    return _page(sample, table, edges, [seq.get(r, "") for r in ids],
                 telo_bp=telo_bp, sub_bp=sub_bp, out_path=out_path,
                 perplexity=perplexity, seed=seed, n_iter=n_iter,
                 row_px=row_px, flank=flank)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="A three-tab HTML browser over one trc run: the "
                    "read/k-mer graph as t-SNE, the same reads as sequence, "
                    "and the clusters as distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("outdir", help="a trc run directory (needs the run's "
                                   "--cache DIR beside its tables)")
    ap.add_argument("--sample", default=None,
                    help="which sample, if the directory holds more than one")
    ap.add_argument("-o", "--out", default=None, metavar="HTML",
                    help="the page; default is <sample>.browser.html in the "
                         "run directory")
    ap.add_argument("--perplexity", type=float, default=PERPLEXITY,
                    help="t-SNE perplexity.  Keep it well under the graph's "
                         "degree -- a read has at least --n-neighbors "
                         "neighbours and there is no more neighbourhood "
                         "than that to match")
    ap.add_argument("--iterations", type=int, default=N_ITER, dest="n_iter",
                    help="t-SNE gradient steps")
    ap.add_argument("--seed", type=int, default=0,
                    help="the embedding's seed; the same seed gives the same "
                         "picture")
    ap.add_argument("--row-px", type=int, default=ROW_PX, dest="row_px",
                    help="height of one read on the read tab")
    ap.add_argument("--flank", type=int, default=FLANK_BP,
                    help="bp drawn past each edge of the clustering window, "
                         "faded")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="no progress on stderr")
    a = ap.parse_args(argv)
    if a.quiet:
        from .util import verbose
        verbose(False)
    build(a.outdir, sample=a.sample, out_path=a.out, perplexity=a.perplexity,
          seed=a.seed, n_iter=a.n_iter, row_px=a.row_px, flank=a.flank)


if __name__ == "__main__":
    main()
