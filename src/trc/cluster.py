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
`unclustered:<rule>` rather than pulled into the nearest cluster, and two rules
put it there:

    disputed           under `MIN_COHESION` of its edge weight lies inside the
                       cluster it was placed in
    --reject-min-size  what is left of its cluster is smaller than this

The second reads the PARTITION.  The first reads the GRAPH against the
partition, and it is the rule this module was rebuilt around, because a
partition is a partition: Leiden labels every vertex and has no way to say
"nowhere", so every unplaced read this module produces is produced after it.

**A read is unplaced when most of what module 2 measured about it points
somewhere other than where it was put.**  `cohesion` is that share -- the
fraction of a read's incident edge weight that stays inside its own cluster --
and it is WEIGHTED, which is the whole of the rule.  A read at the boundary
between two real clusters has edges leaving it, but module 2 has already priced
those edges, and at a real boundary they are worth almost nothing.  Counting
those edges instead of weighing them is a rule that throws out reads whose
placement nothing ever doubted; weighing them, the same reads read 0.9999.

**It is a RATIO, and the reason is the rule it replaces.**  `REJECT_FRAC`
unplaced a read whose total edge weight fell under a fraction of the graph's
median, and on its own terms it worked -- total edge weight separates the reads
a curator refused from the rest at AUC 0.973, where cluster size does so at
0.25-0.69, at or below a coin flip.  It is gone, and not only because the
fraction was a swept number with no source in the reads: module 2's kernel
solves each read's sigma so its outgoing weight sums to exactly `log2(k)`, which
puts a hard FLOOR under every read's strength.  A level-based rule has nothing
left to cut.  Nor would one transfer if it had: a fraction of the median lands
at a different depth on every sample -- across the ten curated sets the median
of total edge weight varies 0.4% while the width of the low tail varies 13%.  A
ratio is unaffected by both -- it asks where a read's budget went, not how large
the budget was -- which is why this is the shape the rule had to take.

**`MIN_COHESION` is a constant, and it sits in a band the data leaves empty.**
It is the one number in this module, and what makes it not a tuned number is
below, at the constant itself, with the distribution that opened the band.

**What it does.**  On the curated sample, at `--reject-min-size 5`, over the
2,523 reads that reached the graph and that the curators sorted into 92
clusters while refusing 12:

    rule               k  unplaced   ARI  shattered  merged  caught
    no refusal        92         0  0.9962      3       2      0/12
    cohesion < 0.99   92        19  0.9972      2       2     12/12

`shattered` is curated clusters split across more than one found cluster,
`merged` is found clusters holding more than one curated cluster, `caught` is
how many of the 12 refused reads this module also refused, and ARI is over the
reads both the module and the curators placed.  Every one of the 12 is caught,
at a precision of 0.63, and cohesion ranks the refused reads against the rest
at AUC 0.998 -- the best refusal signal measured on this graph, where total
edge weight reaches 0.973 and cluster size 0.25-0.69.

**Pulling the disputed reads out makes the partition better, not just
smaller.**  One of the three curated clusters the partition used to shatter was
shattered BY a read that bridged it, and with that read unplaced the cluster
comes back whole; the other two are unchanged.  That is the argument for
refusing rather than reassigning: the read was not in the wrong cluster, it was
in no cluster, and moving it anywhere would have kept the bridge.

**The ensemble this rule was first measured inside is NOT here, on purpose.**
It was developed against ECG -- sixteen Leidens on bootstraps of the edge list,
every edge scored by how often its ends stayed together, the graph reweighted by
that score and cut once more.  Cohesion never read that score: it reads module
2's own weights, and on the curated sample it separates the refused reads at the
same AUC against a single Leiden's partition as against the consensus, because
the two partitions are identical read for read.  What is here was checked
against that branch's output directly: all 3,333 rows of `<sample>.reads.tsv`
match it exactly, cluster ids and refusals alike, at a sixteenth of the
clustering cost.  So what is kept is the rule, without the machinery that
happened to surround it when it was found.

Nothing is ever moved.  A pass that dissolves a small cluster and gives its
reads to whichever neighbour holds most of their edge weight is the right pass
when every read belongs somewhere; it is the wrong one when the answer for some
reads is "nowhere", because it guarantees that answer can never be produced.
"""
from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field

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
CLUSTERER_REASONS = ("disputed", "small_cluster")
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
    "disputed": "under cluster.MIN_COHESION of its edge weight lies inside "
                "the cluster it was placed in",
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

# The name this module's per-read diagnostic is written under, in
# `<sample>.reads.tsv` and on the browser's read tab.  The table's other ten
# columns are fixed; this one is module 3's, and the page discovers it rather
# than being told it.
SCORE_COL = "cohesion"

# The share of a read's edge weight that must lie INSIDE its own cluster for
# the read to be kept.  The one constant in this module.
#
# **It sits in a band the data leaves empty.**  On the curated sample
# (HG08434.LCL-ONT-UL, 2,523 graph reads) the whole distribution of cohesion
# is four groups and two gaps:
#
#     19 reads    0.5367 to 0.9655
#     ---------   nothing at all across 0.0343 of the range
#      8 reads    0.999875 to 0.999999
#     ---------
#  2,496 reads    exactly 1
#
# Every cut in (0.9655, 0.9998] therefore returns the same 19 reads, and all
# 12 of the reads a curator refused to group are among them at every one:
#
#     cut     0.950  0.960  0.970  0.980  0.990  0.999
#     rejects    18     18     19     19     19     19
#     caught     12     12     12     12     12     12   (of 12)
#
# So this is not the peak of a swept curve -- which is what `REJECT_FRAC` was,
# and why it is gone -- it is a point inside a gap the distribution itself
# opened, with nothing within 0.024 below it or 0.0098 above it, and nothing
# was fitted to choose it.
#
# WHAT IT MUST NOT BE IS 1.  The 8 reads just under 1 are there because a
# handful of their edges weigh about 1e-20 and land outside their cluster,
# which is module 2's kernel underflowing rather than a read being badly
# placed; `cohesion < 1` would reject those 8 as well, and on this sample they
# are reads of two adjacent clusters whose shared boundary carries almost no
# weight -- exactly the reads weighing the edges rather than counting them was
# meant to keep.
MIN_COHESION = 0.99


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
def cohesion(g, mem):
    """Per READ, the share of its edge weight that stays inside its cluster.

        sum of w(i,j) over j in i's own cluster / sum of w(i,j) over all j

    Read off the edge list rather than an `n x n` co-association: the quantity
    is only ever wanted on the edges module 2 actually built, and a dense
    matrix would be quadratic in the reads for the same answer.

    **A read with no edges reads 0.**  Nothing holds it anywhere, so nothing
    holds it where it was put -- the vacuous alternative, calling an empty sum
    perfectly cohesive, would let the one read module 2 could measure least
    through the one rule meant to catch it.

    Computed against whatever partition it is handed, before any read has been
    unplaced, so every read is scored against the cluster Leiden chose for it.
    """
    n = g.vcount()
    if g.ecount() == 0:
        return np.zeros(n)
    edges = np.asarray(g.get_edgelist(), np.int64).reshape(-1, 2)
    w = np.asarray(g.es[W], float)
    mem = np.asarray(mem)
    num, den = np.zeros(n), np.zeros(n)
    same = (mem[edges[:, 0]] == mem[edges[:, 1]]).astype(float)
    np.add.at(num, edges[:, 0], w * same)
    np.add.at(num, edges[:, 1], w * same)
    np.add.at(den, edges[:, 0], w)
    np.add.at(den, edges[:, 1], w)
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def reject(mem, coh, min_size, floor=MIN_COHESION):
    """The cohesion rule, then the size floor on what survives it.

    Returns `(mem, why)`: the membership with the rejected reads set to
    `UNCLUSTERED`, and per vertex the rule that did it -- `disputed`,
    `small_cluster`, or `""` for a read still in a cluster.

    In that order on purpose: the floor is computed on the clusters that remain
    after the disputed reads leave, so a real group that was only small because
    it had two disputed reads hanging off it is not then ejected whole.  A read
    that trips both rules is therefore `disputed`, which is the rule that
    actually decided it -- by the time the floor is applied the read has
    already gone.

    Cluster ids are NOT relabelled, so a row can be compared against the same
    row computed without this pass.
    """
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
    # This module's own per-read number and the name it is written under.
    # NOTHING BRANCHES ON THE COLUMN -- it is reported so the tables and the
    # page can be read, and the one rule that does branch on the number is
    # `reject` above, here in the module that defines it.
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    score_col: str = ""
    detail: dict = field(default_factory=dict)

    def labels(self):
        """One label a vertex: the cluster id, or `unclustered:<reason>`."""
        return [unplaced_label(w) if m is UNCLUSTERED else str(m)
                for m, w in zip(self.membership, self.reasons)]


def _summarise(g, name, mem, why, dt, scores, detail):
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
                      modularity=q, seconds=dt,
                      scores=scores, score_col=SCORE_COL, detail=detail)


def run(G, *, seed=0, reject_min_size=5):
    """Module 3 end to end.  Returns one `Assignment`."""
    import time
    g = G.g
    if g.vcount() == 0:
        raise SystemExit("cluster: the graph has no vertices")
    log(f"cluster: {METHOD} on {g.vcount():,} vertices, seed {seed}")

    t0 = time.time()
    mem = cluster(g)
    coh = cohesion(g, mem)
    mem, why = reject(mem, coh, reject_min_size)
    kept = np.array([m is not UNCLUSTERED for m in mem], bool)
    detail = {"min_cohesion": MIN_COHESION,
              "cohesion_placed_p50": float(np.median(coh[kept]))
              if kept.any() else float("nan"),
              "cohesion_unplaced_p50": float(np.median(coh[~kept]))
              if (~kept).any() else float("nan")}
    a = _summarise(g, METHOD, mem, why, time.time() - t0, coh, detail)
    log(f"cluster: cohesion  placed p50 {detail['cohesion_placed_p50']:.4f}"
        f"  unplaced p50 {detail['cohesion_unplaced_p50']:.4f}")
    tally = collections.Counter(w for w in why if w)
    log(f"cluster: {a.n_clusters:>5,} clusters  {a.n_unclustered:>4,} "
        f"unclustered  Q={a.modularity:6.3f}  {a.seconds:5.1f}s"
        + (f"  ({', '.join(f'{v:,} {k}' for k, v in sorted(tally.items()))})"
           if tally else ""))
    return a
