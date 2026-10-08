"""Cut the graph with Leiden/CPM and refuse the reads that do not belong;
repeat on what is left."""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field

import numpy as np

from .util import log

# edge attribute holding the weight
W = "weight"
# membership of a refused read
UNCLUSTERED = None
UNCLUSTERED_LABEL = "unclustered"

# reasons a read is unclustered, latest stage first; this module gives the
# first two
CLUSTERER_REASONS = ("disputed", "small_cluster")
UNPLACED_REASONS = CLUSTERER_REASONS + (
    "no_kmers",
    "short_sub", "short_telo", "no_boundary", "thin_sub", "thin_telo",
    "no_array", "both_ends", "low_qs")

REASON_TEXT = {
    "disputed": "under --min-cohesion of its edge weight lies inside "
                "the cluster it was placed in",
    "small_cluster": "what was left of its cluster was under "
                     "--reject-min-size",
    "no_kmers": "no informative k-mer, so it was held out of the graph",
    "short_sub": "under --min-subtelo-bp of subtelomere past the boundary",
    "short_telo": "under --min-telo-bp of array before the boundary",
    "no_boundary": "teloBP found no array/subtelomere transition",
    "thin_sub": "under --min-subtelo-bp of sequence not matching "
                "--telo-regex",
    "thin_telo": "under --min-telo-bp of sequence matching --telo-regex",
    "no_array": "teloBP could not call the read's strand",
    "both_ends": "telomeric repeat at both ends of the read",
    "low_qs": "basecaller mean qscore under --min-qs",
}


def unplaced_label(reason):
    """"unclustered:<reason>", or "unclustered" with no reason."""
    return f"{UNCLUSTERED_LABEL}:{reason}" if reason else UNCLUSTERED_LABEL


def unplaced_reason(label):
    """Reason in an unclustered label, "" if it has none; None for a
    cluster."""
    if label == UNCLUSTERED_LABEL:
        return ""
    if label.startswith(UNCLUSTERED_LABEL + ":"):
        return label.split(":", 1)[1]
    return None

# CPM resolution, in multiples of the graph's weight density
GAMMA_MULT = 2.0

METHOD = f"igraph/leiden CPM at {GAMMA_MULT:g}x the graph's weight density"

# reads.tsv column for the per-read score
SCORE_COL = "cohesion"

# default --min-cohesion
MIN_COHESION = 0.99


def seed_all(seed):
    """Seed Python's and igraph's RNGs."""
    import igraph as ig
    random.seed(seed)
    ig.set_random_number_generator(random.Random(seed))


def gamma(g):
    """CPM resolution: GAMMA_MULT x total edge weight / (n choose 2)."""
    n = g.vcount()
    pairs = n * (n - 1) / 2.0
    if pairs <= 0:
        return 1.0
    total = float(np.sum(g.es[W])) if g.ecount() else 0.0
    return GAMMA_MULT * total / pairs


def cluster(g):
    """Leiden/CPM membership at gamma(g), iterated until it is stable."""
    return list(g.community_leiden(objective_function="CPM", weights=W,
                                   resolution=gamma(g),
                                   n_iterations=-1).membership)


def cohesion(g, mem):
    """Per vertex, the share of the weight of the edges it chose that stays
    in its cluster; 0 if it chose none."""
    n = g.vcount()
    if g.ecount() == 0:
        return np.zeros(n)
    edges = np.asarray(g.get_edgelist(), np.int64).reshape(-1, 2)
    w = np.asarray(g.es[W], float)
    attrs = g.es.attributes()
    ci = (np.asarray(g.es["claim_i"], bool) if "claim_i" in attrs
          else np.ones(w.size, bool))
    cj = (np.asarray(g.es["claim_j"], bool) if "claim_j" in attrs
          else np.ones(w.size, bool))
    mem = np.asarray(mem)
    num, den = np.zeros(n), np.zeros(n)
    same = (mem[edges[:, 0]] == mem[edges[:, 1]]).astype(float)
    # an edge counts toward each end that chose it
    np.add.at(num, edges[:, 0], w * same * ci)
    np.add.at(num, edges[:, 1], w * same * cj)
    np.add.at(den, edges[:, 0], w * ci)
    np.add.at(den, edges[:, 1], w * cj)
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def reject(mem, coh, min_size, floor=MIN_COHESION):
    """Unplace reads under floor cohesion, then reads in clusters left
    under min_size; returns (membership, reasons)."""
    mem = list(mem)
    why = ["" for _ in mem]
    for i, m in enumerate(mem):
        if m is not UNCLUSTERED and coh[i] < floor:
            mem[i], why[i] = UNCLUSTERED, "disputed"
    if min_size > 1:
        size = collections.Counter(m for m in mem if m is not UNCLUSTERED)
        for i, m in enumerate(mem):
            if m is not UNCLUSTERED and size[m] < min_size:
                mem[i], why[i] = UNCLUSTERED, "small_cluster"
    return mem, why


@dataclass
class Assignment:
    """Per-vertex membership, reasons and scores, plus a summary."""
    name: str
    membership: list
    reasons: list
    n_clusters: int
    n_unclustered: int
    largest: int
    modularity: float
    seconds: float
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    score_col: str = ""
    detail: dict = field(default_factory=dict)

    def labels(self):
        """reads.tsv cluster label per vertex."""
        return [unplaced_label(w) if m is UNCLUSTERED else str(m)
                for m, w in zip(self.membership, self.reasons)]


def _summarise(g, name, mem, why, dt, scores, detail):
    """Assignment for mem; modularity counts each unclustered vertex as
    its own cluster."""
    placed = [m for m in mem if m is not UNCLUSTERED]
    sizes = collections.Counter(placed)
    try:
        nxt = (max(placed) + 1) if placed else 0
        full, i = [], 0
        for m in mem:
            if m is UNCLUSTERED:
                full.append(nxt + i)
                i += 1
            else:
                full.append(m)
        q = float(g.modularity(full, weights=W))
    except Exception:
        q = float("nan")
    return Assignment(name=name, membership=mem, reasons=why,
                      n_clusters=len(sizes),
                      n_unclustered=int(sum(1 for m in mem
                                            if m is UNCLUSTERED)),
                      largest=max(sizes.values()) if sizes else 0,
                      modularity=q, seconds=dt,
                      scores=scores, score_col=SCORE_COL, detail=detail)


# default --max-iterations
PASSES = 16


def iterated_cut(g, reject_min_size, min_cohesion=MIN_COHESION,
                 passes=PASSES):
    """Cut, refuse, and cut the survivors again until a pass refuses no
    read or `passes` run out. A refused read keeps the reason and score of
    the pass that refused it. Returns (mem, why, scores, passes run)."""
    n = g.vcount()
    mem = [UNCLUSTERED] * n
    why = ["" for _ in range(n)]
    scores = np.zeros(n)
    alive = list(range(n))
    done = 0
    for p in range(1, passes + 1):
        if not alive:
            break
        sub = g if len(alive) == n else g.subgraph(alive)
        m = cluster(sub)
        c = cohesion(sub, m)
        m, w = reject(m, c, reject_min_size, min_cohesion)
        for s, v in enumerate(alive):
            mem[v], why[v], scores[v] = m[s], w[s], c[s]
        still = [alive[s] for s, x in enumerate(m) if x is not UNCLUSTERED]
        done = p
        if len(still) == len(alive):
            break
        log(f"cluster: pass {p} refused {len(alive) - len(still):,}; "
            f"{len(still):,} remain")
        alive = still
    return mem, why, scores, done


def run(G, *, seed=0, reject_min_size=5, min_cohesion=MIN_COHESION,
        max_iterations=PASSES):
    """Cluster G and return its Assignment."""
    import time
    g = G.g
    if g.vcount() == 0:
        raise SystemExit("cluster: the graph has no vertices")
    log(f"cluster: {METHOD} on {g.vcount():,} vertices, "
        f"gamma {gamma(g):.3g}, seed {seed}")

    t0 = time.time()
    mem, why, coh, passes = iterated_cut(g, reject_min_size, min_cohesion,
                                         max_iterations)
    kept = np.array([m is not UNCLUSTERED for m in mem], bool)
    detail = {"min_cohesion": float(min_cohesion), "passes": passes,
              "max_iterations": int(max_iterations),
              "gamma_mult": GAMMA_MULT, "gamma": gamma(g),
              "cohesion_placed_p50": float(np.median(coh[kept]))
              if kept.any() else float("nan"),
              "cohesion_unplaced_p50": float(np.median(coh[~kept]))
              if (~kept).any() else float("nan")}
    a = _summarise(g, METHOD, mem, why, time.time() - t0, coh, detail)
    log(f"cluster: cohesion  placed p50 {detail['cohesion_placed_p50']:.4f}"
        f"  unplaced p50 {detail['cohesion_unplaced_p50']:.4f}"
        f"  ({passes} pass{'es' if passes != 1 else ''})")
    tally = collections.Counter(w for w in why if w)
    log(f"cluster: {a.n_clusters:>5,} clusters  {a.n_unclustered:>4,} "
        f"unclustered  Q={a.modularity:6.3f}  {a.seconds:5.1f}s"
        + (f"  ({', '.join(f'{v:,} {k}' for k, v in sorted(tally.items()))})"
           if tally else ""))
    return a
