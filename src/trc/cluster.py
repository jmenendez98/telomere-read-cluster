"""Module 3 of 3: the graph is cut into clusters, and some reads are not placed.

    result = run(G, seed=0, reject_min_size=5)

**One method, pinned: igraph's Leiden on modularity, run to convergence.**
`community_leiden(objective_function="modularity", n_iterations=-1)` -- the
negative count is igraph's "keep iterating until the partition stops moving",
so the cut is the converged one and not whatever two passes happened to reach.
There is no method flag and no panel: the choice is fixed here, in the source,
where it can be read.

It takes no resolution parameter.  It cannot be aimed at an expected number of
chromosome ends; it returns whatever the graph's structure says.  That is the
point of choosing it over a CPM-style objective, whose resolution would have to
come from somewhere -- and the only honest place for that number to come from
is the answer, which is not available at run time.

Nothing in this module reads a label of any kind.

**"Nowhere" is an answer this module is allowed to give.**  A read is reported
`unclustered:<rule>` rather than pulled into the nearest cluster, and one rule
puts it there:

    --reject-min-size  what is left of its cluster is smaller than this.

That rule reads the PARTITION, not the graph.  A read is unplaced because
Leiden left it by itself or in a group too small to be a chromosome end --
which is a thing the clustering said, not a threshold carried in from a sweep.

There was a second rule, `--reject-frac`, which unplaced a read whose total
edge weight fell under a fraction of the graph's median, and on its own terms
it worked: total edge weight separates the reads a curator refused to group
from the rest at AUC 0.973, where cluster size does so at 0.25-0.69, at or
below a coin flip.  It is gone anyway, because what it needed was a NUMBER and
the number had no source in the reads.  0.45 was the peak of a flat 0.42-0.46
band swept against a graph that capped every node at eight edges, and module
2 no longer builds that graph at all, so the strength distribution that cut is
a different distribution, and
a fraction of the median lands at a different depth on every sample in any
case -- the median of total edge weight varies 0.4% across the curated sets
while the width of the low tail varies 13%.  `REJECT_FRAC` below is 0 and
records what setting it there costs.

Nothing is ever moved.  A pass that dissolves a small cluster and gives its
reads to whichever neighbour holds most of their edge weight is the right pass
when every read belongs somewhere; it is the wrong one when the answer for some
reads is "nowhere", because it guarantees that answer can never be produced.
"""
from __future__ import annotations

import collections
import random
from dataclasses import dataclass

import numpy as np

from .util import log

W = "weight"
UNCLUSTERED = None
UNCLUSTERED_LABEL = "unclustered"

# `unclustered` is not one answer, it is ten.  A read can be outside every
# cluster because this module put it there, because module 2 could not measure
# it, or because module 1 never let it past a gate -- and those are different
# enough that reporting them under one label loses the only thing that
# distinguishes "attached to nothing" from "never offered to anything".  Each
# one is written as `unclustered:<reason>` and is a sub-cluster of its own in
# the tables and in the browser.
#
# IN THE ORDER A READ REACHES THEM, LATEST FIRST: this module's own two rules,
# then the reads module 2 held out, then module 1's gates from the last to the
# first.  Every table that lists them lists them in this order, so the reads
# that nearly made it sit next to the clusters and the reads that were never
# telomeric at all sink to the bottom.
CLUSTERER_REASONS = ("weak_edges", "small_cluster")
UNPLACED_REASONS = CLUSTERER_REASONS + (
    "no_kmers",                                         # trc.graph
    "short_sub", "no_boundary", "thin_sub", "thin_telo",
    "no_array", "both_ends", "low_qs")                  # trc.intake

# `CLUSTERER_REASONS` is the line between the two kinds of unplaced read, and
# it is drawn here because it is this module's rules that define it: a read
# unplaced for one of those two WAS a vertex, with edges and neighbours and a
# position in the browser's embedding, and every other reason is a read that
# never became one.  Anything that needs to know whether an unclustered read
# has a graph to be looked at in asks this tuple rather than listing the
# reasons again.

# For a person reading a label, not for anything that branches on one.
REASON_TEXT = {
    "weak_edges": "total edge weight under cluster.REJECT_FRAC x the "
                  "graph's median",
    "small_cluster": "what was left of its cluster was under "
                     "--reject-min-size",
    "no_kmers": "no informative k-mer, so it was held out of the graph",
    "short_sub": "under --min-subtelo-bp of subtelomere past the boundary",
    "no_boundary": "teloBP found no array/subtelomere transition",
    "thin_sub": "under --min-subtelo-bp of sequence not matching "
                "--telo-regex",
    "thin_telo": "under --min-telo-bp of sequence matching --telo-regex",
    "no_array": "teloBP could not call the read's strand",
    "both_ends": "telomeric repeat at both ends of the read",
    "low_qs": "basecaller mean qscore under --min-qs",
}


def unplaced_label(reason):
    """`unclustered:<reason>`, the one place the composite label is spelled."""
    return f"{UNCLUSTERED_LABEL}:{reason}" if reason else UNCLUSTERED_LABEL


def unplaced_reason(label):
    """The reason out of a composite label, or `None` if it is not one."""
    if label == UNCLUSTERED_LABEL:
        return ""
    if label.startswith(UNCLUSTERED_LABEL + ":"):
        return label.split(":", 1)[1]
    return None

# The pinned method, named for the log line and for `run.json`.
METHOD = "igraph/leiden modularity"

# The weight rule's threshold, in units of the graph's median total edge
# weight, or 0 for "no weight rule -- the partition decides".
#
# **Shipped at 0**, and it costs nothing to give up.  Swept over the ten
# curated sets on the uncapped TF-IDF cosine graph module 2 used to build, at
# `--reject-min-size 5`, with the mean unplaced count per sample -- so the
# table says which direction the rule pulled and not what it is worth against
# today's kernel, which has not been measured:
#
#     REJECT_FRAC = 0.00   ARI 0.8527   worst 0.7446   mean k 80.9    2 unplaced
#     REJECT_FRAC = 0.25   ARI 0.8544   worst 0.7466   mean k 80.2   12 unplaced
#     REJECT_FRAC = 0.45   ARI 0.8456   worst 0.7334   mean k 79.7   38 unplaced
#     REJECT_FRAC = 0.50   ARI 0.8342   worst 0.7178   mean k 79.2   58 unplaced
#
# 0.45 was the peak of a flat 0.42-0.46 band when it was swept, and that sweep
# ran against a graph that capped every node at eight edges.  No such cap has
# existed since, so the strength distribution this rule cuts is a different
# distribution and 0.45 has fallen off the far side of its own peak: it is
# worth -0.009 of ARI against the curve's best and -0.007 against not running
# the rule at all.  Turning it off is not a concession -- 0 is the second-best
# point on the curve, and it is the only point on the curve that is not a
# number somebody chose.
#
# What is given up is real all the same.  Total edge weight separates the reads
# a curator refused to group at AUC 0.973, where cluster size does so at
# 0.25-0.69.  At 0.45 the rule unplaced 38 reads a sample; what is left
# unplaces 2.  So "nowhere" is still an answer this module can give and it is
# now the partition that gives it, but it is given about twenty times less
# often, and `--reject-min-size` is the flag that would have to carry it --
# over 1 to 10 that flag spans 0.0002 of ARI and cannot.
REJECT_FRAC = 0.0


def seed_all(seed):
    """igraph's own methods draw from Python's `random`, NOT from an argument.

    Leiden moves between runs without this, and a pipeline whose output moves
    between runs is not a result.
    """
    import igraph as ig
    random.seed(seed)
    ig.set_random_number_generator(random.Random(seed))


def cluster(g):
    """The cut: Leiden on modularity, iterated to convergence.

    `n_iterations=-1` is igraph's "until the partition stops changing".  The
    objective takes no resolution parameter, so there is nothing here to aim.
    """
    return list(g.community_leiden(objective_function="modularity",
                                   weights=W, n_iterations=-1).membership)


# ------------------------------------------------------------------- rejects
def total_weight(g):
    """Each vertex's summed incident edge weight."""
    tot = np.zeros(g.vcount())
    for e, w in zip(g.es, g.es[W]):
        a, b = e.tuple
        tot[a] += w
        tot[b] += w
    return tot


def weak_mask(g, frac):
    """Vertices whose total edge weight is below `frac` x the graph's median.

    RELATIVE TO THE MEDIAN, not absolute.  An absolute cut in kernel-weight
    units is a different cut on every graph -- the per-node edge count alone
    moves the median by a factor of three -- and would not survive being
    carried to a new
    sample, which is the one place a reject rule has to work unsupervised.
    `frac = 0` rejects nothing.
    """
    if frac <= 0:
        return np.zeros(g.vcount(), bool)
    tot = total_weight(g)
    med = float(np.median(tot)) if g.vcount() else 0.0
    return tot < frac * med


def reject(g, mem, frac, min_size):
    """The weight rule, then the size floor on what survives it.

    Returns `(mem, why)`: the membership with the rejected reads set to
    `UNCLUSTERED`, and per vertex the rule that did it -- `weak_edges`,
    `small_cluster`, or `""` for a read still in a cluster.

    In that order on purpose: the floor is computed on the clusters that remain
    after the weak reads leave, so a real group that was only small because it
    had two weak reads hanging off it is not then ejected whole.  A read that
    trips both rules is therefore `weak_edges`, which is the rule that actually
    decided it -- by the time the floor is applied the read has already gone.

    Cluster ids are NOT relabelled, so a row can be compared against the same
    row computed without this pass.
    """
    mem = list(mem)
    why = ["" for _ in mem]
    if frac > 0:
        bad = weak_mask(g, frac)
        for i, m in enumerate(mem):
            if bad[i] and m is not UNCLUSTERED:
                mem[i], why[i] = UNCLUSTERED, "weak_edges"
    if min_size > 1:
        size = collections.Counter(m for m in mem if m is not UNCLUSTERED)
        for i, m in enumerate(mem):
            if m is not UNCLUSTERED and size[m] < min_size:
                mem[i], why[i] = UNCLUSTERED, "small_cluster"
    return mem, why


# -------------------------------------------------------------------- assign
@dataclass
class Assignment:
    """The answer, plus what it looked like."""
    name: str
    membership: list        # per vertex; None where the read is unplaced
    reasons: list           # per vertex; which rule unplaced it, else ""
    n_clusters: int
    n_unclustered: int
    largest: int
    modularity: float
    seconds: float

    def labels(self):
        """One label a vertex: the cluster id, or `unclustered:<reason>`."""
        return [unplaced_label(w) if m is UNCLUSTERED else str(m)
                for m, w in zip(self.membership, self.reasons)]


def _summarise(g, name, mem, why, dt):
    placed = [m for m in mem if m is not UNCLUSTERED]
    sizes = collections.Counter(placed)
    try:
        # Modularity needs a label for every vertex; the unplaced reads are
        # each given one of their own so they are neither silently merged into
        # a cluster nor dropped from the denominator.
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
                      modularity=q, seconds=dt)


def run(G, *, seed=0, reject_min_size=5):
    """Module 3 end to end.  Returns one `Assignment`."""
    import time
    g = G.g
    if g.vcount() == 0:
        raise SystemExit("cluster: the graph has no vertices")
    log(f"cluster: {METHOD} on {g.vcount():,} vertices, seed {seed}")
    if REJECT_FRAC > 0:
        weak = int(weak_mask(g, REJECT_FRAC).sum())
        log(f"cluster: {weak:,} reads are under {REJECT_FRAC:g} x the median "
            f"total edge weight and are unplaced")

    t0 = time.time()
    mem = cluster(g)
    mem, why = reject(g, mem, REJECT_FRAC, reject_min_size)
    a = _summarise(g, METHOD, mem, why, time.time() - t0)
    tally = collections.Counter(w for w in why if w)
    log(f"cluster: {a.n_clusters:>5,} clusters  {a.n_unclustered:>4,} "
        f"unclustered  Q={a.modularity:6.3f}  {a.seconds:5.1f}s"
        + (f"  ({', '.join(f'{v:,} {k}' for k, v in sorted(tally.items()))})"
           if tally else ""))
    return a
