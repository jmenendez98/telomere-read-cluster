# telomere-read-cluster

Group telomere reads by the chromosome end they came from. Reads are nodes,
shared k-mers are weighted edges, and the graph is cut by community detection.

```
trc reads.fq.gz -o out/HG08434
```

Three modules and one command:

| | module | what it does |
|---|---|---|
| 1 | `trc/intake.py` | FASTQ or BAM → reads put on the C strand by teloBP's own strand call, with the array/subtelomere boundary called by teloBP |
| 2 | `trc/graph.py` | reads → an `igraph.Graph`: a TF-IDF cosine over the k-mers of one window across the boundary, kept at the top `--edge-k` per node |
| 3 | `trc/cluster.py` | the graph → clusters, with weakly-attached reads reported `unclustered:<rule>` rather than moved into a neighbour |

`trc/teloboundary.py` is [TeloBP](https://github.com/GEN-DBIO/TeloBP) (MIT, © 2024 Ramin Kahidi), vendored and lightly adapted, so module 1 has nothing to install.

`trc/main.py` is the CLI. **Every parameter of every stage is a flag**, so
`trc --help` is the complete specification of a run and `--cache` writes a
complete record of one. There are no configuration files and no hardcoded
paths.

## Install

Two steps. There is nothing to install outside this repository.

```bash
# 1. the environment, in-tree at ./.env  (~1.5 GB; .gitignore'd)
micromamba create -y -p ./.env -f environment.yml     # or conda/mamba

# 2. this package
./.env/bin/pip install -e .
```

Then `./.env/bin/trc --help`, or activate the env and just use `trc`.

`environment.yml` pins the versions the pipeline was measured against:

```
python 3.10   numpy 1.26.4   scipy 1.12.0
python-igraph 1.0.0   pysam 0.24.1
```

That is the complete list. No scikit-learn — nothing in `src/` embeds or
scores. No matplotlib — nothing in `trc` plots; it was only ever here because
upstream TeloBP imports `matplotlib.pyplot` at module scope, and the vendored
copy defers that import into the one plotting function nothing calls. No
biopython, no pandas: the vendored copy carries only the code module 1
reaches.

### The vendored boundary caller

`trc/teloboundary.py` is TeloBP, carried in-tree. It used to be a third
install step that could not go in `environment.yml`, because the fork the
pipeline needs lives in a checkout whose path differs per machine — so it got
missed, and an environment built from the file came out silently without its
boundary caller.

The fork matters. `getTeloNPArrayBoundary` bounds the scan by the array's own
extent and snaps the call onto the last run of `--telo-regex`; unpatched
TeloBP cannot
bound the scan, which is what lets a distant interstitial telomeric block — an
ITS at 20q, Xp's C-rich non-telomeric stretch — capture the boundary and give
the read a fingerprint belonging to the wrong locus.

Stock TeloBP supplies the other half of module 1: `getIsGStrandFromSeq` decides
which end of a read is the telomere, so intake carries no orientation code of
its own.

The copy was cut from the checkout by line range rather than retyped, so it
diffs cleanly against upstream. Its module docstring lists exactly what was
dropped and what was changed. `--cache` stamps its source alongside
`trc/intake.py`, so editing the algorithm invalidates a cached intake.

## Usage

```bash
# the measured default configuration
trc reads.fq.gz -o out/HG08434

# a BAM, a different k
trc reads.bam -o out/HG08434 -k 30 --edge-k 14

# every intermediate, the edge list and run.json, reused by the next run
trc reads.fq.gz -o out/HG08434 --cache out/HG08434/cache

# progress on stderr; silent otherwise
trc reads.fq.gz -o out/HG08434 -v
```

A BAM is read as a container of sequences: primary records only, unmapped
records included, and the alignments themselves are never consulted.

## Output

Always written:

**`<sample>.reads.tsv`** — **one row per record of the input file.** Nothing
is left out.

```
read_id  cluster  strength  degree  n_informative_kmers
         boundary_b0  sub_bp  read_bp  orient  qs
```

`cluster` is an integer, or `unclustered:<reason>` naming the gate that lost
the read. The reasons are, in the order a read reaches them:

| `cluster` | from | the read |
|---|---|---|
| `0`, `1`, … | `cluster.py` | was placed in that cluster |
| `unclustered:weak_edges` | `cluster.py` | had total edge weight under `--reject-frac` × the graph's median |
| `unclustered:small_cluster` | `cluster.py` | was in a cluster left under `--reject-min-size` |
| `unclustered:no_kmers` | `graph.py` | had no informative k-mer, so it was held out of the graph |
| `unclustered:short_sub` | `intake.py` | had under `--min-subtelo-bp` of subtelomere past the boundary |
| `unclustered:no_boundary` | `intake.py` | had no array/subtelomere transition teloBP could find |
| `unclustered:thin_sub` | `intake.py` | had under `--min-subtelo-bp` not matching `--telo-regex` |
| `unclustered:thin_telo` | `intake.py` | had under `--min-telo-bp` matching `--telo-regex` |
| `unclustered:no_array` | `intake.py` | had no strand teloBP could call |
| `unclustered:both_ends` | `intake.py` | was telomeric at both ends |
| `unclustered:low_qs` | `intake.py` | had a mean qscore under `--min-qs` |

**The earlier a read was lost the fewer of its columns are filled**, because
the columns are the stages: `strength` and `degree` are module 2's, `b0` and
`sub_bp` are module 1's, and a column the read never reached is `NA` rather
than a zero that would read as a measurement. A read held out of the graph is
the one exception — its strength and degree are real zeros, since a read with
no informative k-mer shares none with anything.

**`<sample>.clusters.tsv`** — one row per cluster, then one per kind of
unclustered, in the same order as the table above.

```
cluster  n_reads  n_internal_edges  median_internal_weight
         min_internal_weight  mean_strength  median_b0
         median_sub_bp  representative
```

Same rule: a column a group has no measurement for is `NA`, which for a group
that never entered the graph is every column derived from an edge.
`representative` is the best-connected read of the group, or its longest read
where there is no strength to rank by.

`-o` holds the answer and nothing else. Everything describing *how* a run
reached it goes to `--cache DIR`, and without that flag is not written at all:

- **`<DIR>/<sample>.edges.tsv`**: `read_i read_j weight n_shared idf_shared`
- **`<DIR>/<sample>.run.json`**: every parameter, the intake reject tally, the
  pair-weight distribution, and the cluster summary

`--cache` also holds the sequence of every read, dropped ones included, so a
page can be redrawn from a run directory without going back to the FASTQ.

**`--browser`** writes `<sample>.browser.html` beside the tables: one
self-contained page over the run — the graph as t-SNE, every read as sequence,
the clusters as distributions. It draws `reads.tsv`, so it draws every record
of the input. A read that no gate lost is on all three tabs; a read that one
did is on the reads tab only, in a block of its own named for the gate, since
it has a sequence but no edges, no neighbours and no position. A read with no
boundary is anchored at its own first base, which leaves the telomere half of
its row empty — the picture of "no array was called here".

## The algorithm

### 1. Intake

Which end of a read is the telomere is **teloBP's own call** — the one decision
here that could have used an aligner and does not.
`getIsGStrandFromSeq` counts C-strand telomere runs from the read's start and
G-strand runs from its end and answers with the *strand*: C means the array
already reads CCCTAA at the front, G means it reads TTAGGG at the back and the
read is reverse-complemented. A read telomeric at both ends comes back
`fusedRead` and one telomeric at neither comes back `strandType`; both are
dropped rather than guessed at. Reads are **not** trimmed, so `b0` counts
whatever preceded the array along with the array, and `b0` is the array length
this pipeline reports.

This replaced a canonical-hexamer density scan with three flags
(`--telo-win`, `--telo-step`, `--telo-min-dens`) and two tuned constants. The
scan picked an *end* rather than a strand, so a G-strand all-telomere fragment
could be oriented head-first on its TTAGGG array and then scanned by teloBP as
if it were C-strand — against a pattern its array cannot match. Over six files
the two rules disagree on 1 read in 2,509, 5 in 2,710, 1 in 2,153, 8 in 618, 3
in 573 and 3 in 782, and against the hand-curated sample they score identically
(ARI 0.9988 now, 0.9989 before).

**Nothing is masked, and no hexamer is named.** Masking the canonical repeat
before extracting features also destroys the k-mers that *straddle* a variant
repeat — the ones that say what context it sits in. The array is removed
downstream by the IDF, because everything that reached it has one.

### 2. The graph

```
v_i[m] = log1p(tf_i[m]) * idf[m]       idf[m] = log(n / df[m])
w_ij   = <v_i, v_j> / (||v_i|| ||v_j||)
edges  = the top `edge_k` weights at each node, symmetric union
```

**Order is discarded.** A k-mer is an edge because both reads contain it, full
stop — no chaining, no positional agreement. That is a choice, not a
simplification: the composition graph is what twenty-six independent community
algorithms were measured to cut identically.

**Cosine, not summed rarity.** A raw sum of shared IDF is unnormalised, so a
31 kb read accumulates more of everything than a 4 kb one and the longest reads
become hubs on length alone. L2-normalising first asks about composition
instead.

**Top-k per node, not a weight threshold.** A single weight cut keeps hundreds
of edges at a dense node and none at a sparse one, encoding the density rather
than the structure.

**The two df gates cut either end of the range.** `--min-k-n` removes a k-mer
in one read, which cannot form an edge at all. `--max-k-frac` removes what most
of the pool shares — at the default 0.5, a k-mer in more than half the reads.
The canonical repeat was already cheap without it: `log1p(tf)` and `log(n/df)`
between them put it at ~0.3% of a read's vector before any gate sees it. What
is left is the middle, where a chromosome end's identity lives.

The ceiling is a fraction of **the pool, not of a group**, so it has to stay
above the largest group's share of the reads. At 0.5, a cluster holding more
than half the sample loses the k-mers that define it. That is not a concern at
the ninety-odd ends of a whole-sample run; it is one on a pool pre-filtered to
a few arms, where the ceiling should be raised toward 1.0.

### 3. Cluster assignment

The cut is **igraph's Leiden on modularity, iterated to convergence** —
`community_leiden(objective_function="modularity", n_iterations=-1)`, where the
negative count is igraph's "keep going until the partition stops moving". It is
pinned in `cluster.py` and is not a flag.

Modularity **takes no resolution parameter**, which is the reason to prefer it
over a CPM-style objective here: it cannot be aimed at an expected number of
chromosome ends, and returns whatever the graph's structure says. A resolution
would have to come from somewhere, and the only honest source for that number
is the answer, which is not available at run time. Nothing in these three
modules reads a label of any kind.

**"Nowhere" is an answer this pipeline is allowed to give.** A read is reported
`unclustered:weak_edges` when its total edge weight is under `--reject-frac` ×
the graph's median, and `unclustered:small_cluster` when what is left of its
cluster is under `--reject-min-size`. A read that trips both is
`weak_edges` — the weight rule runs first, and by the time the size floor is
applied the read has already gone. Nothing is ever moved into a neighbouring
cluster.

The weight rule is the one that does the work. Total edge weight separates the
reads a curator refused to group at AUC 0.973; cluster size does so at
0.25–0.69, at or below a coin flip, because most such reads sit *inside* real
thirty-read groups. Degree cannot work and could not: top-k gives every node
`edge_k` neighbours by construction, so degree is floored and carries almost
nothing. It is the *weight* on those edges that says a read is attached to
nothing in particular. The threshold is computed on the graph before the
clustering runs, so it is a statement about the read and not about the
partition.

## Defaults

The defaults are a measured configuration, not a guess: the point selected over
**eight** hand-curated samples in `benchmark/parameterization/`, which scores
ARI 0.995 against the curators' labels where the previous configuration scored
0.989, and finds the cluster splits the curators had to make by hand — 92
clusters where it used to find 90, 91 where it used to find 88.

It is the **centre of a plateau, not its argmax**. 144 of 1,033 configurations
scored within 0.001 of the best, and with eight samples the standard error on
that mean is ~0.0009, so nothing inside the plateau is distinguishable from
anything else inside it. Leiden's cut is seed-deterministic here — twelve seeds
give identical scores to five decimals — so the uncertainty is entirely
cross-sample. Seven of the eight samples improve; HG08435.PBMC-ONT-LSK does
not, splitting into 93 clusters where the curator drew 92.

Two of these parameters were measured to do **nothing** on a whole-sample pool
and are kept only for the pre-filtered case: see `--max-k-frac` and
`--reject-min-size` below.

| flag | default | why |
|---|---|---|
| `-k` | 32 | the largest exact, collision-free length; 33–51 are 64-bit hashes and are labelled as such in `run.json`. Scores flat from 28 to 36 |
| `--telo-bp` | 3000 | bounds the deepest pair, so a pooled window is a comparable depth of array against a comparable depth of subtelomere. The window always spans both sides of the boundary, `[b0-telo_bp, b0+sub_bp)`, which is the only way to see the k-mers straddling it |
| `--sub-bp` | 600 | the selected value sits in a flat band from 450 to 600; below 400 the profile loses the subtelomere that identifies the arm |
| `--edge-k` | 8 | **a counting bound, not a tuning knob**: a read in a group of *m* reads has only *m*−1 possible in-group partners, so `edge_k ≥ m` forces it to link outside its own group. Set against curated groups as small as 11 reads. At 20 the score collapses to 0.92, which is the bound asserting itself |
| `--min-qs` | 20 | the basecaller's own mean qscore, from the `qs` tag |
| `--min-telo-bp` | 400 | bases of the oriented read matching `--telo-regex`. A composition test, not the length of anything contiguous |
| `--min-subtelo-bp` | 1000 | pinned to `--sub-bp`, so a read reaching the graph can fill the subtelomere window instead of merely clearing it. 150 bp was the lossless point across five curated samples, the shortest subtelomere on any *grouped* read there being 151 bp |
| `--telo-regex` | teloBP's C-strand pattern | a first pass at the two length gates, from composition alone and before the boundary is called, so a hopeless read does not cost teloBP's ~280 ms. Empty turns it off |
| `--bound-margin` | 5000 | how far past the array teloBP's boundary scan may look. Bounding it is the whole reason for the `itsfix` fork; widening it gives a nearby ITS more room to capture the call |
| `--snap-bp` | 500 | the boundary is the origin every read is measured from, and teloBP calls it from a 750 bp smoothed window, so it carries that smoothing as slop. Snapping moves it onto the end of the last run of `--telo-regex` — a landmark the reads *share*, since it is the pattern the whole pool is judged by. Measured on HG08434 LCL-ONT-UL, per-column agreement across the 300 bp of array behind `b0` with the clusters held fixed: **0.859** unsnapped, 0.959 onto the last tandem run end (what this used to do), **0.964** onto the last regex run end. It fires on 81% of reads and moves `b0` by a median of 6 bp (p90 105); 76 of 92 clusters improve, the worst loses 0.027. The score saturates by 250 — every value from there to 600 gives the same tables on this sample, since a run end further than that from the call is rare — so 500 is headroom rather than a fitted number. 0 is off, and so is an empty `--telo-regex` |
| `--reject-frac` | 0.45 | the peak of a flat 0.42–0.46 band, and the one swept parameter that peaks rather than plateaus. It sits deep in the tail because the tail is narrow: total edge weight has a median that varies 0.4% across samples but a low-tail width that varies 13%, so a cut in units of the median lands at a different depth on every sample. See `benchmark/parameterization/README.md` |
| `--reject-min-size` | 5 | **near-inert**: 1 through 10 span 0.00006 ARI. Cluster size separates the reads a curator refused to group at AUC 0.25–0.69, at or below a coin flip, because most such reads sit inside real thirty-read groups. Left at 5 rather than moved on a difference this small |
| `--min-k-n` | 3 | below 2 a k-mer cannot form an edge; 3 drops the singleton-pair k-mers that link two reads by a shared error. 2–4 score alike |
| `--max-k-frac` | 0.5 | drops what more than half the pool shares. **Inert on a whole-sample run**: every value from 0.2 to 1.0 gives byte-identical tables, because ~92 groups of ~30 reads put the largest at ~2% of the pool and the ceiling never binds. It is the right guard only on a pool pre-filtered to a few arms, where a group can be half the reads — raise it toward 1.0 there. Below 0.1 it deletes real signal |

## Caching

`--cache DIR` is opt-in. It stores every intermediate, plus the two files
above:

```
<DIR>/intake.npz + intake.seqs.txt.gz    teloBP is ~280 ms a read
<DIR>/kmerdb.<window>.npz                the k-mer database
<DIR>/vectors.<window>.<gates>.npz       every read × k-mer matrix
<DIR>/cosine.<window>.<gates>.npz        the dense read × read matrix
<DIR>/<sample>.edges.tsv  + .run.json
```

`vectors` holds all four matrices of the weighting — raw counts, `log1p` of
them, the L2-normalised TF-IDF rows the cosine is taken over, and presence —
over **one** sparsity structure. They have to share it: a positive diagonal
cannot create or destroy an entry, and `keep` admits only `w > 0`. Storing the
structure once rather than four times is what makes "all of them" affordable.
The informative k-mer codes and their IDF are in the same file, so a re-run
skips the df pass as well.

Each is stamped with the input file's size and mtime, the parameters that stage
actually reads, **and a hash of the source of the modules that produce it** — so
editing a function without touching any constant still invalidates it. There is
no cache to clear by hand.

They are stamped separately because different flags move them: `--min-k-n` and
`--max-k-frac` move the matrices and the cosine and leave the database alone,
and `--edge-k` and everything in module 3 move none of them. So a re-run that
only changes the clusterer reuses everything.

Every field that moves a file is in the *filename* as well as the stamp — the
window for all of them, and the two df gates for those the gates move. Two
builds sharing a file take turns overwriting it, each finding the other's stamp
wrong and rebuilding, so a sweep alternating between them pays for every build
twice and never hits the cache.

## Reproducibility

`--seed` seeds igraph's own RNG. igraph's community methods, Leiden included,
draw from Python's `random` and take no seed argument at all; without seeding
it, the cut moves between runs.

## Repository

```
src/trc/        the four files above, plus util.py (logging, TSV/JSON, cache)
benchmark/      the measurement history the defaults come from.  Reads the
                arm track and curated truth sets; nothing in src/ ever does
```
