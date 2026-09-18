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
| 2 | `trc/graph.py` | reads → an `igraph.Graph`: a TF-IDF cosine over the k-mers of one window across the boundary, kept where each read's own neighbourhood gate leaves them |
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
trc reads.bam -o out/HG08434 -k 30 --telo-bp 2500

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
         boundary_b0  sub_bp  read_bp  orient  qs  cohesion
```

The first ten columns are fixed. The eleventh is module 3's own per-read
number, under the name that module gives it — `cohesion`, the share of the
edge weight the read *chose* that stays inside its cluster — and the browser discovers
it as whatever column is not one of the ten rather than being told which
module 3 wrote the table.

`cluster` is an integer, or `unclustered:<reason>` naming the gate that lost
the read. The reasons are, in the order a read reaches them:

| `cluster` | from | the read |
|---|---|---|
| `0`, `1`, … | `cluster.py` | was placed in that cluster |
| `unclustered:disputed` | `cluster.py` | had under `cluster.MIN_COHESION` of its edge weight inside the cluster it was placed in |
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

- **`<DIR>/<sample>.edges.tsv`**: `read_i read_j weight n_shared idf_shared
  chose`, where `chose` is `both`, `i` or `j` — which end asked for the edge.
  Step 3 weights each read's own `--n-neighbors` nearest and nothing else, so
  an edge is one-way whenever the read at the other end had ten better
  options. On the curated sample **75.3% of the 20,038 edges are one-way**, and
  module 3's refusal reads the same distinction.
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
1 normalise  v_i[m] = log1p(tf_i[m]) * idf[m]     idf[m] = log(n / df[m])
2 distance   d(i,j) = || v_i - v_j ||             rows NOT unit length
3 weight     w(i->j) = exp(-(d(i,j) - rho_i) / sigma_i)   over i's 10 nearest
4 refine     w(i,j) = r(a + b - ab) + (1 - r)ab   a = w(i->j), b = w(j->i)
```

Step 1 is textbook TF-IDF, a **product** of two compressed quantities. It was
`log1p(tf * e^idf)` until the measurement in `graph.py` — which folds the raw
ratio `n/df` inside the log and comes out as `log(tf) + idf`, a **sum**, and a
much weaker weighting: a canonical repeat column reads 5.94 that way against
1.33 as a product. The sum form wins if module 3 cuts once; the product wins
once it cuts repeatedly, and neither change survives being tested alone.

`rho_i` is read i's nearest distance, subtracted so that neighbour sits at
weight 1; `sigma_i` is solved so each read's outgoing weight sums to
`log2(--n-neighbors)`, which is what stops a read in a dense arm outvoting one
in a sparse arm. `--n-neighbors` is the only cut in module 2 and it is a cut on
**count**, never on weight.

**Order is discarded.** A k-mer is an edge because both reads contain it, full
stop — no chaining, no positional agreement. That is a choice, not a
simplification: the composition graph is what twenty-six independent community
algorithms were measured to cut identically.

**Cosine, not summed rarity.** A raw sum of shared IDF is unnormalised, so a
31 kb read accumulates more of everything than a 4 kb one and the longest reads
become hubs on length alone. L2-normalising first asks about composition
instead.

**The edge rule is a gate, not a threshold and no longer a count.** A single
weight cut keeps hundreds of edges at a dense node and none at a sparse one,
encoding the density rather than the structure, so each read's neighbourhood is
read off the largest drop in its *own* sorted similarity profile and a pair
survives when either read calls the other a neighbour. The per-node cap that
used to follow it is `graph.EDGE_K`, a constant, and it is 0 — there is no cap.
What that costs is measured in `graph.py`, beside the constant: it is the
largest single effect in module 2.

**The vocabulary is every k-mer the pool contains.** The two df gates that
used to cut either end of the range are `graph.MIN_K_N` and `graph.MAX_K_FRAC`,
constants, and both are at their inert value — so the only column dropped is one
every read carries, whose `log(n/df)` is exactly 0 and which could not move a
cosine anyway. The canonical repeat was always cheap without a gate: `log1p(tf)`
and `log(n/df)` between them put it at ~0.3% of a read's vector.

`graph.py` carries the 2×2 that measures them. The floor is what the choice
costs — admitting the singletons triples the vocabulary and puts each read's
private sequence, which at these depths is very largely basecall error, into its
own L2 norm. The ceiling turns out to have been the opposite: it was documented
as inert and is not, because it was deleting ~10% of every read's incidences
(the columns most reads share) and keeping those is worth +0.025 ARI.

What the ceiling was *for* is now unguarded. It is a fraction of **the pool, not
of a group**, so on a pool pre-filtered to a few arms one group can be half the
reads and a ceiling of 0.5 deletes exactly the k-mers that name it. At 1.0
nothing can.

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

**"Nowhere" is an answer this pipeline is allowed to give.** Leiden cannot give
it — a partition labels every vertex — so both rules that produce an unplaced
read run after it, and nothing is ever moved into a neighbouring cluster:

| reason | what it reads |
|---|---|
| `unclustered:disputed` | the **graph**, against the partition: under `cluster.MIN_COHESION` of the edge weight the read *chose* lies inside the cluster Leiden put it in |
| `unclustered:small_cluster` | the **partition**: what is left of the read's cluster is under `--reject-min-size` |

In that order, so the size floor is applied to the clusters that remain after
the disputed reads leave.

**And the cut is repeated.** A refusal that arrives after the cut arrives too
late: Leiden has already used a bridging read to merge two chromosome ends by
the time `cohesion` says it belongs to neither, so the rule catches the read
and keeps the damage. `cluster.PASSES = 16` times, the refused reads are
*deleted* — the vertex, and its edges with it — and what remains is cut again.
On the curated sample all fourteen reads joining curated 47 to curated 62 were
already refused; cutting again without them splits the pair, lands the cluster
count on the curated 92 and takes ARI from 0.9862 to **0.9991**.

Pass 1 refuses 17 and pass 2 refuses nobody, and passes 3 to 32 return the same
partition every time from a different RNG start — the same stability the ECG
work ran into, where sixteen bootstrapped Leidens matched a single one. So the
spare passes cost ~2s and buy nothing measurable *here*; they are run because
the graph that needs them is the one whose structure is less settled, and that
has not been ruled out on the other nine samples.

`cohesion` is that share, and it is **weighted**, which is the whole of the
rule. A read on the boundary between two real clusters has edges leaving it,
but module 2 has already priced those edges and at a real boundary they are
worth almost nothing; counting them instead of weighing them throws out reads
whose placement nothing ever doubted, and weighing them the same reads read
0.9999. It is also a **ratio rather than a level**, and that is forced: module
2 solves each read's sigma so its outgoing weight sums to exactly `log2(k)`,
which floors every read's strength however empty its neighbourhood is, so a
level-based rule has nothing left to cut. A share is unaffected by that floor.

**The denominator is what the read chose, not what chose it.** Three quarters
of the edges here are one-way, so a read can collect a large share of its
incident weight from reads that picked *it* — and under the first version of
this rule it was refused for their sake. `09480b4c` is the case that named it:
47.8% of its incident weight came from reads that chose it, among them one the
curator marked `weak_edges` and one marked `no_kmers`, dragging it to 0.679 and
out of a cluster Leiden had placed it in correctly. Over its own ten choices it
reads 1.0000. A read answers for where it points; it does not answer for who
points at it.

On **one** sample — HG08434.LCL-ONT-UL, 2,523 graph reads the curators sorted
into 92 clusters while refusing 12, at `--reject-min-size 5`:

| rule | k | unplaced | ARI | shattered | merged | refused caught | precision |
|---|---|---|---|---|---|---|---|
| no refusal | 92 | 0 | 0.9962 | 3 | 2 | 0/12 | — |
| `cohesion < 0.99`, all incident edges | 92 | 19 | **0.9972** | 2 | 2 | **12/12** | 0.63 |
| `cohesion < 0.99`, edges the read chose | 92 | 14 | **0.9972** | 2 | 2 | **12/12** | **0.86** |

All twelve under either denominator; over a read's own choices the five false
refusals go away and cohesion ranks the refused reads against the rest at
**AUC 0.999** — where total edge weight reaches 0.973 and
cluster size 0.25–0.69, at or below a coin flip, because most refused reads sit
*inside* real thirty-read groups. Degree has never been a usable signal either,
though for a reason that has since gone away: the per-node edge cap floored it
by construction. With `graph.EDGE_K = 0` degree varies again and has not been
re-measured. The partition also gets **better** rather
than just smaller: one of the three curated clusters it used to shatter was
shattered by a single read bridging it, and with that read unplaced the cluster
comes back whole. Moving the read anywhere would have kept the bridge. This is
one sample; the ten-set numbers below predate the rule.

**The rule it replaces was `--reject-frac`**, which unplaced a read whose total
edge weight fell under a fraction of the graph's median. It is gone, constant
and all. What it needed was a number with no source in the reads: 0.45 was the
peak of a flat 0.42–0.46 band swept against a graph that capped every node at
eight edges, and no such cap has existed since. Over the ten curated sets at
`--reject-min-size 5` it had already fallen off the far side of its own peak —
0.45 scored 0.846 (worst 0.733, 80 clusters, 38 reads unplaced a sample) where
turning it off scored **0.853** (worst 0.745, 81 clusters, 2 unplaced), and the
curve peaked at 0.25 (0.854). `cluster.MIN_COHESION` is a number too, and the
difference is where it comes from: it is not the peak of a swept curve but a
point inside a gap the distribution opens by itself — 19 reads at 0.9655 and
below, then nothing at all until 0.9999, then 8 reads and 2,496 at exactly 1 —
so every cut in (0.9655, 0.9998] returns the same 19 reads.

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

One of these parameters was measured to do **nothing** on a whole-sample pool
and is kept for what it guards against rather than for its effect: see
`--reject-min-size` below.

**Modules 2 and 3 have since moved off the point that sweep selected**, and
these numbers are the sweep's. `-k` is 48 rather than 32; the per-node edge cap
is gone (`graph.EDGE_K = 0`); both df gates are off (`graph.MIN_K_N = 1`,
`graph.MAX_K_FRAC = 1.0`); the weight-rejection rule is gone entirely, replaced
by the cohesion refusal. End to end over the ten curated sets in
`out/manually-curated-truth-sets`, that configuration scores ARI **0.853**
(worst 0.745) and finds 81 clusters where the curators drew 92. Each constant
records its own cost beside itself, in `graph.py` and `cluster.py`.

| flag | default | why |
|---|---|---|
| `-k` | 48 | 33–51 are 64-bit hashes, not exact codes, and are labelled as such in `run.json`; 32 is the largest exact, collision-free length. 48 over 32 is worth 0.003 ARI on the curated sets and is the largest weighting effect measured in module 2 — everything that beat the plain cosine did so by reading more sequence per feature |
| `--telo-bp` | 3000 | bounds the deepest pair, so a pooled window is a comparable depth of array against a comparable depth of subtelomere. The window always spans both sides of the boundary, `[b0-telo_bp, b0+sub_bp)`, which is the only way to see the k-mers straddling it |
| `--sub-bp` | 600 | the selected value sits in a flat band from 450 to 600; below 400 the profile loses the subtelomere that identifies the arm |
| `--min-qs` | 20 | the basecaller's own mean qscore, from the `qs` tag |
| `--min-telo-bp` | 400 | bases of the oriented read matching `--telo-regex`. A composition test, not the length of anything contiguous |
| `--min-subtelo-bp` | 1000 | pinned to `--sub-bp`, so a read reaching the graph can fill the subtelomere window instead of merely clearing it. 150 bp was the lossless point across five curated samples, the shortest subtelomere on any *grouped* read there being 151 bp |
| `--telo-regex` | teloBP's C-strand pattern | a first pass at the two length gates, from composition alone and before the boundary is called, so a hopeless read does not cost teloBP's ~280 ms. Empty turns it off |
| `--bound-margin` | 5000 | how far past the array teloBP's boundary scan may look. Bounding it is the whole reason for the `itsfix` fork; widening it gives a nearby ITS more room to capture the call |
| `--snap-bp` | 500 | the boundary is the origin every read is measured from, and teloBP calls it from a 750 bp smoothed window, so it carries that smoothing as slop. Snapping moves it onto the end of the last run of `--telo-regex` — a landmark the reads *share*, since it is the pattern the whole pool is judged by. Measured on HG08434 LCL-ONT-UL, per-column agreement across the 300 bp of array behind `b0` with the clusters held fixed: **0.859** unsnapped, 0.959 onto the last tandem run end (what this used to do), **0.964** onto the last regex run end. It fires on 81% of reads and moves `b0` by a median of 6 bp (p90 105); 76 of 92 clusters improve, the worst loses 0.027. The score saturates by 250 — every value from there to 600 gives the same tables on this sample, since a run end further than that from the call is rare — so 500 is headroom rather than a fitted number. 0 is off, and so is an empty `--telo-regex` |
| `--reject-min-size` | 5 | **near-inert**: over the ten curated sets, 1 through 10 span 0.0002 ARI, and all it moves is the cluster count — 81.6 at 1, 80.2 at 10, against a truth of 92, so raising it walks away from the answer. Cluster size separates the reads a curator refused to group at AUC 0.25–0.69, at or below a coin flip, because most such reads sit inside real thirty-read groups. Left at 5 rather than moved on a difference this small. It is no longer the only rejection rule — `cluster.MIN_COHESION` is the one that does the work — and it runs second, on the clusters left after the disputed reads go, so a real group that was only small because two disputed reads hung off it is not then ejected whole |

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

They are stamped alike and stored apart, because they cost different things
to rebuild and to store: the cosine is the expensive half of the module and the
smallest file. Nothing between the database and the cosine takes a flag any
more — the df gates are constants, and a change to either is a change to the
module's source hash, which every one of the three stamps holds. Everything in
module 3 moves none of them, so a re-run that only changes the clusterer reuses
everything.

Every field that moves a file is in the *filename* as well as the stamp: the
window, for all of them. Two
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
