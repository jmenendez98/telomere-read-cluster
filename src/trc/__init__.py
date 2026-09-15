"""telomere-read-cluster: telomere reads grouped into chromosome ends.

Three modules, run end to end by one command (`trc`, or `python -m trc`):

    1  trc.intake    FASTQ/BAM -> reads put on the C strand by teloBP's own
                     strand call, with its array/subtelomere boundary `b0`
    2  trc.graph     reads become nodes and shared k-mers become weighted
                     edges: a TF-IDF cosine over the k-mers of one window
                     across the array/subtelomere junction, kept at the top
                     `edge_k` per node
    3  trc.cluster   the graph is cut by igraph's Leiden on modularity,
                     iterated to convergence, and reads attached to nothing in
                     particular are reported unclustered rather than moved
                     into a neighbour

`trc.teloboundary` is TeloBP (MIT, (c) 2024 Ramin Kahidi), vendored and
lightly adapted so module 1 has no import to satisfy outside this package.  It
is upstream's code, not this project's; see its module docstring.

`--browser` adds a fourth module, `trc.browser`, which writes one
self-contained HTML page over a finished run: the graph as t-SNE, the reads as
sequence, the clusters as distributions.  It is a view of a decision and never
a step towards one -- no stage reads anything it produces.

Every parameter of every stage is a flag on the command line; see
`trc.main.build_parser`.  Nothing in these modules ever reads a label.
"""
__version__ = "0.1"

__all__ = ["intake", "graph", "cluster", "browser", "main", "util",
           "teloboundary"]
