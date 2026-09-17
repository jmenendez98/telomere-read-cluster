"""The command line.  Almost every parameter of every stage is a flag here.

    trc reads.fq.gz -o out/HG08434
    trc reads.bam   -o out/HG08434 -k 30 --telo-bp 2500
    trc reads.fq.gz -o out/HG08434 --cache out/HG08434/cache
    trc reads.fq.gz -o out/HG08434 --browser

One invocation runs all three modules end to end:

    1  intake   FASTQ/BAM -> reads oriented telomere-first, boundary called
    2  graph    reads as nodes, shared k-mers as weighted edges
    3  cluster  the graph cut into clusters, weakly-attached reads unplaced

`--browser` adds one more file and no stage: `trc.browser` draws the finished
run as a self-contained HTML page, built from what is still in memory when the
tables are written, so it needs nothing on disk that `-o` does not already
hold.

**THE TWO TABLES HOLD EVERY RECORD OF THE INPUT FILE.**  A read that no gate
lost carries its cluster id; every other read carries the gate that lost it,
as `unclustered:<reason>` -- `unclustered:low_qs`, `unclustered:no_kmers`,
`unclustered:weak_edges`.  So `wc -l` on `<sample>.reads.tsv` is the record
count of the input, `<sample>.clusters.tsv` has a row per kind of unclustered
beside its rows per cluster, and a read that is not in a cluster is in the
table saying why rather than being absent from it.  A column belonging to a
stage the read never reached is `NA`.

**No configuration files.**  Every number the pipeline uses is declared either
below with its default or as a named constant in the module that reads it, so
`trc --help` plus those constants is the complete specification of a run and
`run.json` is a complete record of one.  Nothing reads an environment variable,
a config file, or a hardcoded path.

Module 2 holds two of them -- `graph.MIN_K_N` and `graph.MAX_K_FRAC`, the two
document-frequency gates of step 1 -- and module 3 one more,
`cluster.REJECT_FRAC`.  Each is a value that was a flag until a sweep found it
had one setting worth shipping, so it is written where the measurement that
chose it can be read beside it.  All three are now at the setting that does
nothing, which is the only setting a constant can hold without being a tuned
number in hiding.  Module 2's kernel adds two more that are not of that kind:
`graph.SMOOTH_ITERS` and `graph.MIN_SIGMA_FRAC` are a bisection's step count
and a guard against dividing by zero, neither swept nor sweepable.  Intake
used to carry two more, a lead-in and an orientation ratio, for a canonical-
hexamer density scan that decided which end of a read was the telomere.
teloBP answers that itself, and with its own strand call in place the scan,
its three flags and both constants are gone.

**The defaults are a measured configuration, not a guess.**  `-k 48
--telo-bp 3000 --sub-bp 600 --reject-min-size 5` is the point selected over
eight hand-curated samples in `benchmark/parameterization/`, where it scores
ARI 0.995 against the curators' labels where the previous configuration scored
0.989, and recovers the cluster splits the curators had to make by hand.  It
is the CENTRE of a plateau rather than its argmax: 144 of 1,033 configurations
scored within 0.001 of the best, and with eight samples the standard error on
that mean is ~0.0009, so the points inside the plateau are not distinguishable
from one another.  That sweep was run against the TF-IDF cosine graph module 2
used to build, so what it selected is `-k`, `--telo-bp` and `--sub-bp` -- step
1's window, which steps 2 to 4 did not change.

`--n-neighbors` has not been swept over those samples either; its 10 is the
low end of the 10-15 plateau on the ONE sample where it has been swept, chosen
there over 15 -- UMAP's own default -- because the two score the same and 10
sits further from the hub that appears past 20.  It is still the first number
to put on a curve over all ten.  `--mix-ratio` has been swept over that same
one sample, where 0.25-0.75 are indistinguishable and 1.0 -- UMAP's own
default -- merges a pair of chromosome ends that share no edge; 0.5 is the
middle of that band and the point where the mix becomes a plain average.  The
weight-rejection threshold is `cluster.REJECT_FRAC` and is 0: a read is now
unplaced because the partition left it alone, not because its strength fell
under a swept fraction -- see `cluster.py`.

The two length gates are set above what those samples required.
`--min-subtelo-bp` is held at or above `--sub-bp` rather than at the 150 bp
that was lossless there, so a read reaching the graph can fill the subtelomere
window instead of merely clearing it; `--min-telo-bp` is likewise above the
252 bp floor the density scan can report, so it trims short arrays rather than
only arrayless reads.  Both trade reads for profiles built entirely of
sequence that is present.

**Caching is opt-in and stamped.**  `--cache DIR` stores everything a run
produces that is not one of the two result tables: intake (teloBP is ~280 ms a
read, a quarter of an hour a sample), the k-mer database, every read x k-mer
matrix -- raw counts, `log1p` of them, TF-IDF, presence -- the dense weight
matrix,
and the exported edge list and `run.json` alongside them.  A stamp covers the
input file, the parameters each stage reads and the source of the modules that
produce it, so a re-run that changes only the clusterer reuses all of it and
a re-run that changes `-k` reuses only intake.  Without the flag nothing
is written or read, and `<sample>.edges.tsv` and `<sample>.run.json` are not
produced at all.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import __version__
from . import cluster as _cluster_mod
from . import graph as _graph_mod
from . import intake as _intake_mod
from . import teloboundary as _telobp_mod
from .cluster import (UNCLUSTERED, UNPLACED_REASONS, unplaced_label,
                      unplaced_reason)
from .intake import UNKNOWN, Dropped, Read
from .util import (cache_path, code_sig, input_sig, load_npz, load_seqs, log,
                   n_threads, param_sig, save_npz, save_seqs, verbose,
                   write_json, write_tsv)

__all__ = ["main", "build_parser"]


def build_parser():
    p = argparse.ArgumentParser(
        prog="trc",
        description="Cluster telomere reads into chromosome ends: reads are "
                    "nodes, shared k-mers are weighted edges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ------------------------------------------------------------ input/output
    io = p.add_argument_group("input and output")
    io.add_argument("input", metavar="READS",
                    help="one FASTQ (.fq/.fastq, optionally .gz) or BAM "
                         "(.bam/.cram/.sam).  A BAM is read as a container of "
                         "sequences; its alignments are never consulted")
    io.add_argument("-o", "--out", default=".",
                    help="directory the result files are written to")
    io.add_argument("-s", "--sample", default=None,
                    help="name the output files are prefixed with "
                         "(default: the input file's basename)")

    # ------------------------------------------------------------ 1. intake
    a = p.add_argument_group("intake")
    a.add_argument("--min-qs", type=float, default=20.0,
                   help="drop reads whose basecaller mean qscore (the `qs` "
                        "tag) is below this.  A read with no tag is kept")
    a.add_argument("--min-telo-bp", type=int, default=400,
                   help="a read matching --telo-regex over less than this "
                        "many bases is not a telomere read.  Measured on the "
                        "oriented read before the boundary is called, so it "
                        "is a composition test rather than the length of "
                        "anything contiguous.  0 turns it off")
    a.add_argument("--telo-regex", default="CTTCTT|CCTGG|CCC(?!CCC)[ATCG]{3}",
                   help="telomeric sequence, as a pattern.  Before any "
                        "boundary is called, a read must match it over at "
                        "least --min-telo-bp and fail to match it over at "
                        "least --min-subtelo-bp.  The default is teloBP's "
                        "own C-strand nanopore pattern, which is the strand "
                        "teloBP has just put the read on.  Empty turns it off")
    a.add_argument("--min-subtelo-bp", type=int, default=1000,
                   help="drop reads with less than this much subtelomere past "
                        "the boundary, before the graph is built.  Such a "
                        "read's profile is mostly sequence that is not there, "
                        "and it contributes edges to everything it touches.  "
                        "At the default it is at or above --sub-bp, so every "
                        "read in the graph can fill the subtelomere window; "
                        "keep it there if you raise --sub-bp")

    # ------------------------------ 2. telomere/subtelomere boundary detection
    t = p.add_argument_group("subtelomere/telomere boundary detection")
    t.add_argument("--bound-margin", type=int, default=5000,
                   help="teloBP: how far past the array the boundary scan may "
                        "look.  Bounding it stops a distant interstitial "
                        "telomeric block from capturing the call")
    t.add_argument("--snap-bp", type=int, default=500,
                   help="teloBP: snap the boundary onto the end of the last "
                        "run of --telo-regex within this distance, so reads "
                        "from one chromosome end are anchored on a landmark "
                        "they share instead of on the smoothing width of the "
                        "boundary call.  A read with no run end that close "
                        "keeps the boundary teloBP gave it.  0 takes every "
                        "boundary as it comes and skips the scan the snap "
                        "needs; so does an empty --telo-regex, which leaves "
                        "nothing to snap onto")

    # ------------------------------------------------------------- 3. graph
    b = p.add_argument_group("read/k-mer graph")
    b.add_argument("-k", type=int, default=48,
                   help=f"k-mer length.  Exact and collision-free up to "
                        f"{_graph_mod.EXACT_MAX_K}; {_graph_mod.EXACT_MAX_K + 1}"
                        f"-{_graph_mod.MAX_K} are 64-bit hashes and are "
                        f"labelled as such.  48 over 32 is worth 0.003 of ARI "
                        f"on the curated sets and is the single largest "
                        f"module-2 effect measured: every weighting that beat "
                        f"the cosine this module used to ship did so by "
                        f"reading MORE SEQUENCE PER FEATURE, and raising k "
                        f"is the cheapest way to do it")
    b.add_argument("--telo-bp", type=int, default=3000,
                   help="how far back into the array to scan; 0 means back to "
                        "the read's own tip")
    b.add_argument("--sub-bp", type=int, default=600,
                   help="how far into the subtelomere to scan")
    b.add_argument("--n-neighbors", type=int, default=10, metavar="K",
                   help="how many nearest reads each read is given a weight "
                        "to.  It sets the neighbourhood the kernel's rho and "
                        "sigma are read off, and it is the only cut in module "
                        "2: a pair outside both reads' K is never given a "
                        "weight, and no pair is ever dropped for the SIZE of "
                        "its weight.  Every read gets the same K, so a read "
                        "in a dense arm has no more say than one in a sparse "
                        "arm")
    b.add_argument("--mix-ratio", type=float, default=_graph_mod.MIX_RATIO,
                   metavar="R",
                   help="how step 4 joins the two directions of a pair: the "
                        "fuzzy union a+b-ab at 1, the fuzzy intersection a*b "
                        "at 0, and their mean (a+b)/2 at the default 0.5, "
                        "where the product cancels.  It is what a ONE-WAY "
                        "claim is worth beside a mutual one, and the only "
                        "place in module 2 where being CHOSEN differs from "
                        "choosing.  Above 0 it reweights the same edges; at "
                        "0 it deletes every one-way pair, which on the "
                        "curated sample is three fifths of them")
    # ----------------------------------------------------------- 4. cluster
    c = p.add_argument_group("cluster assignment")
    c.add_argument("--reject-min-size", type=int, default=5,
                   help="report a read unclustered when what is left of its "
                        "cluster is smaller than this.  Nothing is ever moved "
                        "into a neighbouring cluster.  1 rejects nothing")

    # ------------------------------------------------------------- 5. other
    d = p.add_argument_group("other")
    d.add_argument("--threads", type=int, default=4,
                   help="processes for the teloBP boundary scan; 0 means "
                        "every core this job can see")
    d.add_argument("--seed", type=int, default=0,
                   help="seeds igraph's own RNG, which Leiden draws from "
                        "without taking a seed argument")
    d.add_argument("--cache", default=None, metavar="DIR",
                   help="write everything but the two result tables to this "
                        "directory, each file stamped on the input and the "
                        "parameters it depends on: intake, the k-mer "
                        "database, every read/k-mer matrix, the weights, and "
                        "<sample>.edges.tsv and <sample>.run.json.  Without "
                        "it those last two are not produced")
    d.add_argument("--browser", action="store_true",
                   help="also write <sample>.browser.html beside the tables: "
                        "one self-contained page over this run -- the "
                        "read/k-mer graph as t-SNE, the same reads as "
                        "sequence, the clusters as distributions.  It is "
                        "built from the run in memory, so it needs no "
                        "--cache, and it costs a t-SNE over the edge list "
                        "(seconds at a few thousand reads, O(n^2) in them).  "
                        "Nothing reads it back")
    d.add_argument("-v", "--verbose", action="store_true",
                   help="progress on stderr")
    d.add_argument("--version", action="version",
                   version=f"trc {__version__}")
    return p


# ------------------------------------------------------------------- staging
def _intake_stage(args, cachedir):
    """Module 1, cached on the input file and the fields intake reads.

    EVERY RECORD OF THE INPUT IS CACHED, dropped ones included, in one set of
    parallel arrays with `reason` empty for the reads that survived.  The
    dropped reads are rows in `<sample>.reads.tsv` and rows on the browser's
    reads tab, so their sequence is part of what a run has to be able to
    reproduce without going back to the FASTQ -- and keeping them in the same
    arrays as the survivors is what makes the cached read set the input file's
    read set rather than a subset of it.  It costs the rejected sequence on
    disk, which is the only thing `--cache` did not already hold.
    """
    kw = dict(min_qs=args.min_qs, min_telo_bp=args.min_telo_bp,
              telo_regex=args.telo_regex, bound_margin=args.bound_margin,
              snap_bp=args.snap_bp, min_subtelo_bp=args.min_subtelo_bp)
    sig = (f"{input_sig(args.input)}|{param_sig(**kw)}"
           f"|{code_sig(_intake_mod, _telobp_mod)}")
    path = cache_path(cachedir, "intake.npz") if cachedir else None
    seqpath = cache_path(cachedir, "intake.seqs.txt.gz") if cachedir else None

    z = load_npz(path, sig)
    if z is not None and seqpath and os.path.exists(seqpath):
        log("intake: reusing cache")
        seqs = load_seqs(seqpath)
        reads, dropped = [], []
        for i, s, o, b, sb, q, why in zip(z["ids"], seqs, z["orient"],
                                          z["b0"], z["sub_bp"], z["qs"],
                                          z["reason"]):
            if str(why):
                dropped.append(Dropped(read_id=str(i), reason=str(why), seq=s,
                                       orient=str(o), b0=int(b),
                                       sub_bp=int(sb), qs=float(q)))
            else:
                reads.append(Read(read_id=str(i), seq=s, orient=str(o),
                                  b0=int(b), sub_bp=int(sb), qs=float(q)))
        return reads, dropped, sig

    reads, dropped = _intake_mod.intake(args.input, threads=n_threads(
        args.threads), **kw)
    if path:
        every = list(reads) + list(dropped)
        save_seqs(seqpath, [r.seq for r in every])
        save_npz(path, sig,
                 ids=np.array([r.read_id for r in every]),
                 orient=np.array([r.orient for r in every]),
                 b0=np.array([r.b0 for r in every]),
                 sub_bp=np.array([r.sub_bp for r in every]),
                 qs=np.array([float(r.qs) for r in every]),
                 reason=np.array([getattr(r, "reason", "") for r in every]))
    return reads, dropped, sig


def _window_name(args):
    """Every field that moves the window is in the FILENAME, not only in the
    stamp.  Two databases that share a file take turns overwriting it, each
    finding the other's stamp wrong and rebuilding, so a sweep alternating
    between them pays for every build twice and never hits the cache."""
    cap = f"telo{args.telo_bp}" if args.telo_bp else "teloall"
    return f"k{args.k}.{cap}.sub{args.sub_bp}"


def _graph_stage(args, reads, cachedir, intake_sig):
    """Module 2, with the database, the matrices and the weights cached apart.

    Three caches and not one, because they cost different things to rebuild
    and to store: the weights are the expensive half of the module and the
    smallest file, and the matrices are wanted whole by anything that wants a
    column.  The first two carry the SAME stamp -- the df gates that used to
    move the matrices while leaving the database alone are `graph.MIN_K_N` and
    `graph.MAX_K_FRAC`, constants, and a change to either is a change to the
    module's code signature, which every one of the three stamps holds.

    The weights carry that stamp plus `--n-neighbors` and `--mix-ratio`, the
    two parameters that move steps 2-4 without moving the k-mer matrices above
    them: sweeping either rebuilds the weights and reuses the database and the
    matrices, which is where the time is.  Both are in the file name, so the
    settings of a sweep sit side by side in one cache directory.
    """
    name = _window_name(args)
    db_kw = dict(k=args.k, telo_bp=args.telo_bp, sub_bp=args.sub_bp)
    csig = code_sig(_graph_mod)
    db_sig = f"{intake_sig}|{param_sig(**db_kw)}|{csig}|n={len(reads)}"
    db_path = cache_path(cachedir, f"kmerdb.{name}.npz") if cachedir else None

    db = None
    z = load_npz(db_path, db_sig)
    if z is not None:
        log(f"graph: reusing k-mer cache ({z['code'].size / 1e6:.1f}M k-mers)")
        db = _graph_mod.kmerdb_from_arrays(z)
    else:
        db = _graph_mod.build_kmerdb([r.seq for r in reads],
                                     np.array([r.b0 for r in reads]), **db_kw)
        save_npz(db_path, db_sig, **_graph_mod.kmerdb_arrays(db))

    vec_path = (cache_path(cachedir, f"vectors.{name}.npz")
                if cachedir else None)
    vec = None
    zv = load_npz(vec_path, db_sig)
    if zv is not None:
        vec = _graph_mod.vectors_from_arrays(zv)

    w_par = param_sig(n_neighbors=args.n_neighbors,
                      mix_ratio=args.mix_ratio)
    w_sig = f"{db_sig}|{w_par}"
    w_path = (cache_path(cachedir, f"weights.{name}.nn{args.n_neighbors}"
                         f".mix{args.mix_ratio:g}.npz")
              if cachedir else None)
    W = None
    zc = load_npz(w_path, w_sig)
    if zc is not None:
        W = _graph_mod.weights_from_arrays(zc)

    G, db, vec, W = _graph_mod.build(reads, db=db, vec=vec, W=W,
                                     n_neighbors=args.n_neighbors,
                                     mix_ratio=args.mix_ratio, **db_kw)
    if vec_path and zv is None:
        # Raw counts, log1p of them, TF-IDF, and the columns they are over.
        # Presence is all ones and is rebuilt from the same structure.
        save_npz(vec_path, db_sig, **_graph_mod.vectors_arrays(vec))
    if w_path and zc is None:
        save_npz(w_path, w_sig, **_graph_mod.weights_arrays(W))
    return G


# -------------------------------------------------------------------- output
def _num(v, missing=UNKNOWN):
    """A measurement, or `None` where the sentinel says there is not one.

    `None` is what the tables write as `NA`.  It is kept distinct from 0 all
    the way to the file: a read dropped at `--min-qs` has no boundary, which
    is not the same statement as a boundary at base 0.
    """
    return None if v is None or v == missing else v


def _write_reads(path, reads, dropped, G, a):
    """ONE ROW PER RECORD OF THE INPUT FILE.  Nothing is left out.

    `cluster` says which of three things happened to the read, and the two
    that are not "it was clustered" are told apart by what follows the colon:

        <id>                  Leiden put the read in that cluster
        unclustered:<rule>    it reached module 3, which unplaced it --
                              `weak_edges` or `small_cluster`
        unclustered:no_kmers  it reached module 2 with no informative k-mer
                              and was held out of the graph
        unclustered:<gate>    module 1 dropped it: `low_qs`, `both_ends`,
                              `no_array`, `thin_telo`, `thin_sub`,
                              `no_boundary`, `short_sub`

    The earlier a read was lost the fewer of its columns are filled, because
    the columns ARE the stages: `strength` and `degree` are module 2's, `b0`
    and `sub_bp` are module 1's, and a read that never reached a stage carries
    `NA` for it rather than a zero that reads like a measurement.  A read held
    out of the graph is the one exception -- its strength and degree are real
    zeros, since a read with no informative k-mer shares none with anything.

    Survivors are written first, in the order module 1 returned them, then the
    dropped reads in the order they were dropped.
    """
    cols = ["read_id", "cluster", "strength", "degree", "n_informative_kmers",
            "boundary_b0", "sub_bp", "read_bp", "orient", "qs"]
    labels = a.labels()
    rows = []
    for i, r in enumerate(reads):
        v = int(G.vertex_of[i])
        rows.append({"read_id": r.read_id,
                     "cluster": (labels[v] if v >= 0
                                 else unplaced_label("no_kmers")),
                     "strength": float(G.strength[i]),
                     "degree": int(G.degree[i]),
                     "n_informative_kmers": int(G.n_informative[i]),
                     "boundary_b0": r.b0, "sub_bp": r.sub_bp,
                     "read_bp": len(r.seq), "orient": r.orient,
                     "qs": _num(r.qs)})
    for d in dropped:
        rows.append({"read_id": d.read_id,
                     "cluster": unplaced_label(d.reason),
                     "strength": None, "degree": None,
                     "n_informative_kmers": None,
                     "boundary_b0": _num(d.b0), "sub_bp": _num(d.sub_bp),
                     "read_bp": len(d.seq), "orient": d.orient or None,
                     "qs": _num(d.qs)})
    write_tsv(path, rows, cols)
    return rows


def _write_clusters(path, rows, G, a):
    """One row per cluster, and one per KIND of unclustered.

    Built from the rows `_write_reads` just wrote and not from the reads
    again, so the two tables cannot disagree about which read is in which
    group: this table is that one, grouped.

    Clusters come first, largest first, then the unclustered groups in
    `UNPLACED_REASONS` order -- the reads that nearly made it next to the
    clusters, the ones that were never telomeric at the bottom.  A column a
    group has no measurement for is `NA`, and for a group whose reads never
    entered the graph that is every column derived from an edge.
    """
    import collections
    # Edge weights within each cluster, so a row says how tight it is rather
    # than only how big.
    inside = collections.defaultdict(list)
    for (i, j), w in zip(G.pairs, G.weights):
        vi, vj = int(G.vertex_of[i]), int(G.vertex_of[j])
        if vi >= 0 and vj >= 0 and a.membership[vi] == a.membership[vj] \
                and a.membership[vi] is not UNCLUSTERED:
            inside[str(a.membership[vi])].append(float(w))

    members = collections.defaultdict(list)
    for r in rows:
        members[r["cluster"]].append(r)

    def order(label):
        why = unplaced_reason(label)
        if why is None:
            return (0, -len(members[label]), label)
        return (1, UNPLACED_REASONS.index(why)
                if why in UNPLACED_REASONS else len(UNPLACED_REASONS), label)

    def med(rs, col):
        v = [r[col] for r in rs if r[col] is not None]
        return float(np.median(v)) if v else None

    out = []
    for label in sorted(members, key=order):
        rs = members[label]
        w = inside.get(label, [])
        st = [r["strength"] for r in rs if r["strength"] is not None]
        out.append({
            "cluster": label,
            "n_reads": len(rs),
            "n_internal_edges": len(w) if st else None,
            "median_internal_weight": float(np.median(w)) if w else None,
            "min_internal_weight": float(np.min(w)) if w else None,
            "mean_strength": float(np.mean(st)) if st else None,
            "median_b0": med(rs, "boundary_b0"),
            "median_sub_bp": med(rs, "sub_bp"),
            # The best-connected read of the group, as a handle for pulling
            # its sequence back out of the input file.  Where there is no
            # strength to rank by -- a group that never entered the graph --
            # it is the longest read, which is the only measurement those
            # reads all have.
            "representative": max(
                rs, key=lambda r: (r["strength"] if r["strength"] is not None
                                   else -1.0, r["read_bp"]))["read_id"],
        })
    write_tsv(path, out, ["cluster", "n_reads", "n_internal_edges",
                          "median_internal_weight", "min_internal_weight",
                          "mean_strength", "median_b0",
                          "median_sub_bp", "representative"])
    return len(out)


def _unplaced_tally(rows):
    """`{unclustered:<reason>: n}` over the rows, for `run.json`.

    Read off the table rather than accumulated alongside it, so the record of
    a run and the run's own output cannot drift apart.
    """
    import collections
    t = collections.Counter(r["cluster"] for r in rows
                            if unplaced_reason(r["cluster"]) is not None)
    return dict(sorted(t.items()))


def _write_edges(path, reads, G):
    rows = [{"read_i": reads[int(i)].read_id, "read_j": reads[int(j)].read_id,
             "weight": float(w), "n_shared": int(n), "idf_shared": float(g)}
            for (i, j), w, n, g in zip(G.pairs, G.weights, G.n_shared,
                                       G.idf_shared)]
    write_tsv(path, rows, ["read_i", "read_j", "weight", "n_shared",
                           "idf_shared"])


# ----------------------------------------------------------------------- main
def main(argv=None):
    p = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    args = p.parse_args(argv)
    verbose(args.verbose)

    # Before the cache stamp, which stats the file: a mistyped path should be
    # a sentence and not a traceback from os.stat.
    _intake_mod.check_input(args.input)
    if args.k > _graph_mod.MAX_K:
        raise SystemExit(f"-k {args.k} exceeds the ceiling of "
                         f"{_graph_mod.MAX_K}")
    if not 0.0 <= args.mix_ratio <= 1.0:
        raise SystemExit(f"--mix-ratio {args.mix_ratio} is outside [0, 1]; "
                         f"it is a mix of two set operations, not a scale")
    sample = args.sample or os.path.basename(args.input).split(".")[0]
    os.makedirs(args.out, exist_ok=True)
    cachedir = args.cache
    if cachedir:
        os.makedirs(cachedir, exist_ok=True)
        log(f"cache: {cachedir}")

    reads, dropped, isig = _intake_stage(args, cachedir)
    G = _graph_stage(args, reads, cachedir, isig)
    _cluster_mod.seed_all(args.seed)
    chosen = _cluster_mod.run(
        G, seed=args.seed, reject_min_size=args.reject_min_size)

    base = os.path.join(args.out, sample)
    rows = _write_reads(f"{base}.reads.tsv", reads, dropped, G, chosen)
    n_rows = len(rows)
    n_clu = _write_clusters(f"{base}.clusters.tsv", rows, G, chosen)
    if cachedir:
        # The edge list and the run record go where the matrices go.  They are
        # the account of how one run reached its answer, which is what --cache
        # holds; `-o` holds the answer itself and nothing else.
        _write_edges(os.path.join(cachedir, f"{sample}.edges.tsv"), reads, G)
        write_json(os.path.join(cachedir, f"{sample}.run.json"), {
            "sample": sample,
            "input": os.path.abspath(args.input),
            "params": {k: v for k, v in sorted(vars(args).items())
                       if k != "verbose"},
            "kmer_coding": "hashed" if G.hashed else "exact",
            "n_input_records": len(rows),
            "n_input_records_kept": len(reads),
            "intake_rejects": _intake_mod.reject_counts(dropped),
            "unplaced": _unplaced_tally(rows),
            "graph": G.report,
            "cluster_method": chosen.name,
            "n_clusters": chosen.n_clusters,
            "n_unclustered": chosen.n_unclustered,
            "largest_cluster": chosen.largest,
            "modularity": chosen.modularity,
            "cluster_seconds": chosen.seconds,
        })

    if args.browser:
        # Imported here and not at the top: without the flag nothing in this
        # module needs it, and the module sets BLAS thread limits at import.
        from . import browser as _browser_mod
        _browser_mod.build_from_run(f"{base}.browser.html", sample, rows,
                                    reads, dropped, G, telo_bp=args.telo_bp,
                                    sub_bp=args.sub_bp, seed=args.seed)

    log(f"done: {n_rows:,} reads, {G.g.vcount():,} of them in the graph, "
        f"{chosen.n_clusters:,} clusters ({n_clu:,} rows), "
        f"{n_rows - G.g.vcount() + chosen.n_unclustered:,} unclustered, "
        f"method {chosen.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
