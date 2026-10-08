"""The `trc` command: intake, graph, clustering, then the output tables."""

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

    io = p.add_argument_group("input and output")
    io.add_argument("input", metavar="READS",
                    help="reads as FASTQ (.fq/.fastq, optionally .gz) or "
                         "BAM/CRAM/SAM")
    io.add_argument("-o", "--out", default=".",
                    help="output directory")
    io.add_argument("-s", "--sample", default=None,
                    help="prefix for the output files (the input file's "
                         "basename if unset)")

    a = p.add_argument_group("intake")
    a.add_argument("--min-qs", type=float, default=0.0,
                   help="drop reads whose mean qscore (`qs` tag) is below "
                        "this; reads without the tag are kept")
    a.add_argument("--telo-regex", default="CTTCTT|CCTGG|CCC(?!CCC)[ATCG]{3}",
                   help="pattern for telomeric sequence; empty disables")
    a.add_argument("--min-telo-bp", type=int, default=400,
                   help="drop reads with less telomeric array than this "
                        "before the boundary; 0 disables")
    a.add_argument("--min-subtelo-bp", type=int, default=1000,
                   help="drop reads with less subtelomere than this past the "
                        "boundary")
    a.add_argument("--bound-margin", type=int, default=5000,
                   help="how far past the telomeric array to search for the "
                        "boundary")
    a.add_argument("--snap-bp", type=int, default=500,
                   help="snap the boundary to the end of the nearest "
                        "--telo-regex run within this distance; 0 disables")

    b = p.add_argument_group("read/k-mer graph")
    b.add_argument("-k", type=int, default=48,
                   help="k-mer length")
    b.add_argument("--telo-bp", type=int, default=2500,
                   help="bp of telomeric array before the boundary to take "
                        "k-mers from; 0 reads back to the read's tip")
    b.add_argument("--sub-bp", type=int, default=1000,
                   help="bp of subtelomere past the boundary to take k-mers "
                        "from")
    b.add_argument("--min-df", type=int, default=20, metavar="N",
                   help="drop k-mers found in fewer than this many reads")
    b.add_argument("--max-df", type=float, default=1.0, metavar="F",
                   help="drop k-mers found in more than this fraction of "
                        "reads")
    b.add_argument("--n-neighbors", type=int, default=10, metavar="K",
                   help="number of nearest reads each read gets an edge to")
    b.add_argument("--max-edge-distance", type=float,
                   default=_graph_mod.MAX_EDGE_DISTANCE, metavar="D",
                   help="never connect two reads further apart than this "
                        "cosine distance; 1 disables")
    c = p.add_argument_group("cluster assignment")
    c.add_argument("--reject-min-size", type=int, default=5,
                   help="leave reads unclustered when their cluster has fewer "
                        "reads than this; 1 disables")
    c.add_argument("--min-cohesion", type=float,
                   default=_cluster_mod.MIN_COHESION, metavar="F",
                   help="leave reads unclustered when less than this fraction "
                        "of the edge weight they chose lies inside their own "
                        "cluster; 0 disables")
    c.add_argument("--max-iterations", type=int,
                   default=_cluster_mod.PASSES, metavar="N",
                   help="cut the graph again without the reads the last cut "
                        "left unclustered, at most this many times in all; "
                        "stops early once a cut leaves no read unclustered")
    c.add_argument("--seed", type=int, default=0,
                   help="random seed for Leiden")

    d = p.add_argument_group("other")
    d.add_argument("--threads", type=int, default=4,
                   help="processes for the boundary scan; 0 uses every "
                        "available core")
    d.add_argument("--cache", default=None, metavar="DIR",
                   help="cache intermediate results in this directory, and "
                        "write <sample>.edges.tsv and <sample>.run.json there")
    d.add_argument("--browser", action="store_true",
                   help="also write <sample>.browser.html, an interactive "
                        "view of the run")
    d.add_argument("-v", "--verbose", action="store_true",
                   help="print progress to stderr")
    d.add_argument("--version", action="version",
                   version=f"trc {__version__}")
    return p


def _intake_stage(args, cachedir):
    """Intake, from the cache when input, parameters and code all match.
    Returns (reads, dropped, sig)."""
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
    """Cache-file tag for k and the window."""
    cap = f"telo{args.telo_bp}" if args.telo_bp else "teloall"
    return f"k{args.k}.{cap}.sub{args.sub_bp}"


def _graph_stage(args, reads, cachedir, intake_sig):
    """Build the graph, reusing each cached stage (k-mers, vectors,
    weights) whose signature matches; each extends the one before."""
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

    name_df = f"{name}.df{args.min_df}-{args.max_df:g}"
    vec_path = (cache_path(cachedir, f"vectors.{name_df}.npz")
                if cachedir else None)
    vec_sig = (f"{db_sig}|"
               f"{param_sig(min_df=args.min_df, max_df=args.max_df)}")
    vec = None
    zv = load_npz(vec_path, vec_sig)
    if zv is not None:
        vec = _graph_mod.vectors_from_arrays(zv)

    w_par = param_sig(n_neighbors=args.n_neighbors,
                      max_edge_distance=args.max_edge_distance)
    w_sig = f"{vec_sig}|{w_par}"
    w_path = (cache_path(cachedir, f"weights.{name_df}.nn{args.n_neighbors}"
                         f".d{args.max_edge_distance:g}.npz")
              if cachedir else None)
    W = None
    zc = load_npz(w_path, w_sig)
    if zc is not None:
        W = _graph_mod.weights_from_arrays(zc)

    G, db, vec, W = _graph_mod.build(reads, db=db, vec=vec, W=W,
                                     min_df=args.min_df, max_df=args.max_df,
                                     n_neighbors=args.n_neighbors,
                                     max_edge_distance=args.max_edge_distance,
                                     **db_kw)
    if vec_path and zv is None:
        save_npz(vec_path, vec_sig, **_graph_mod.vectors_arrays(vec))
    if w_path and zc is None:
        save_npz(w_path, w_sig, **_graph_mod.weights_arrays(W))
    return G


def _num(v, missing=UNKNOWN):
    """None for a missing value, so it is written as NA."""
    return None if v is None or v == missing else v


def _write_reads(path, reads, dropped, G, a):
    """Write reads.tsv, kept reads then dropped ones; returns the rows."""
    cols = ["read_id", "cluster", "strength", "degree", "n_informative_kmers",
            "boundary_b0", "sub_bp", "read_bp", "orient", "qs"]
    score = getattr(a, "score_col", "") or ""
    if score:
        cols.append(score)
    labels = a.labels()
    rows = []
    for i, r in enumerate(reads):
        v = int(G.vertex_of[i])
        rows.append({"read_id": r.read_id,
                     "cluster": (labels[v] if v >= 0
                                 else unplaced_label("no_kmers")),
                     "strength": float(G.strength[i]) if v >= 0 else None,
                     "degree": int(G.degree[i]) if v >= 0 else None,
                     "n_informative_kmers": int(G.n_informative[i]),
                     "boundary_b0": r.b0, "sub_bp": r.sub_bp,
                     "read_bp": len(r.seq), "orient": r.orient,
                     "qs": _num(r.qs)})
        if score:
            rows[-1][score] = (float(a.scores[v])
                               if 0 <= v < len(a.scores) else None)
    for d in dropped:
        rows.append({"read_id": d.read_id,
                     "cluster": unplaced_label(d.reason),
                     "strength": None, "degree": None,
                     "n_informative_kmers": None,
                     "boundary_b0": _num(d.b0), "sub_bp": _num(d.sub_bp),
                     "read_bp": len(d.seq), "orient": d.orient or None,
                     "qs": _num(d.qs)})
        if score:
            rows[-1][score] = None
    write_tsv(path, rows, cols)
    return rows


def _write_clusters(path, rows, G, a):
    """Write clusters.tsv from the read rows; returns the row count."""
    import collections
    # weights of the edges inside each cluster
    inside = collections.defaultdict(list)
    for (i, j), w in zip(G.pairs, G.weights):
        vi, vj = int(G.vertex_of[i]), int(G.vertex_of[j])
        if vi >= 0 and vj >= 0 and a.membership[vi] == a.membership[vj] \
                and a.membership[vi] is not UNCLUSTERED:
            inside[str(a.membership[vi])].append(float(w))

    members = collections.defaultdict(list)
    for r in rows:
        members[r["cluster"]].append(r)

    # clusters largest first, then unclustered groups in reason order
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
    """Read count per unclustered label."""
    import collections
    t = collections.Counter(r["cluster"] for r in rows
                            if unplaced_reason(r["cluster"]) is not None)
    return dict(sorted(t.items()))


def _write_edges(path, reads, G):
    """Write edges.tsv, with which end(s) chose each edge."""
    rows = [{"read_i": reads[int(i)].read_id, "read_j": reads[int(j)].read_id,
             "weight": float(w), "n_shared": int(n), "idf_shared": float(g),
             "chose": "both" if (ci and cj) else ("i" if ci else "j")}
            for (i, j), w, n, g, (ci, cj) in zip(G.pairs, G.weights, G.n_shared,
                                                 G.idf_shared, G.claims)]
    write_tsv(path, rows, ["read_i", "read_j", "weight", "n_shared",
                           "idf_shared", "chose"])


def main(argv=None):
    """Run trc; returns the exit status."""
    p = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    args = p.parse_args(argv)
    verbose(args.verbose)

    _intake_mod.check_input(args.input)
    if args.k > _graph_mod.MAX_K:
        raise SystemExit(f"-k {args.k} exceeds the ceiling of "
                         f"{_graph_mod.MAX_K}")
    if args.min_df < 1:
        raise SystemExit(f"--min-df {args.min_df} is below 1")
    if not 0.0 < args.max_df <= 1.0:
        raise SystemExit(f"--max-df {args.max_df} is outside (0, 1]")
    if not 0.0 <= args.min_cohesion <= 1.0:
        raise SystemExit(f"--min-cohesion {args.min_cohesion} is outside "
                         f"[0, 1]")
    if args.max_iterations < 1:
        raise SystemExit(f"--max-iterations {args.max_iterations} is below 1")
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
        G, seed=args.seed, reject_min_size=args.reject_min_size,
        min_cohesion=args.min_cohesion, max_iterations=args.max_iterations)

    base = os.path.join(args.out, sample)
    rows = _write_reads(f"{base}.reads.tsv", reads, dropped, G, chosen)
    n_rows = len(rows)
    n_clu = _write_clusters(f"{base}.clusters.tsv", rows, G, chosen)
    if cachedir:
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
            "cluster_detail": getattr(chosen, "detail", {}),
            "n_clusters": chosen.n_clusters,
            "n_unclustered": chosen.n_unclustered,
            "largest_cluster": chosen.largest,
            "modularity": chosen.modularity,
            "cluster_seconds": chosen.seconds,
        })

    if args.browser:
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
