# telomere-read-cluster

Designed to quickly group telomere reads together by the locus they came from. 
`trc` counts k-mers to build a between read distance matrix. Matrix distances
are used to build a graph which is cut into loci clusters by community detection.

## Install

```bash
# 0. git clone repo
git clone https://github.com/jmenendez98/telomere-read-cluster.git
cd telomere-read-cluster

# 1. conda/mamba package install
conda env create -n trc -f environment.yml
conda activate trc

# 2. pip install
pip install .
```

Then `trc --help` inside the activated env.

`environment.yml` pins the versions the pipeline was measured against:

```
python 3.10   numpy 1.26.4   scipy 1.12.0
python-igraph 1.0.0   pysam 0.24.1
```

## Usage

```bash
# trc default configuration
trc {sample}.fq.gz -o out/{sample}

# a BAM, a different k
trc {sample}.bam -o out/{sample} -k 30

# keep intermediates for the next run to reuse, plus the edge list and run.json
trc {sample}.fq.gz -o out/{sample} --cache out/{sample}/cache

# progress on stderr; silent otherwise
trc {sample}.fq.gz -o out/{sample} -v
```

## Output

`-o DIR` gets two tables:

| file | one row per |
|---|---|
| `<sample>.reads.tsv` | read in the input, dropped reads included (from a BAM, primary records with a sequence) |
| `<sample>.clusters.tsv` | cluster, then one per `unclustered:<reason>` |

`<sample>` is `-s`, or the input's file name up to its first `.`.

### `reads.tsv`

| column | |
|---|---|
| `read_id` | |
| `cluster` | cluster number, or `unclustered:<reason>` (below) |
| `strength`, `degree` | summed edge weight and edge count in the graph |
| `n_informative_kmers` | the read's distinct k-mers left after `--min-df` / `--max-df` |
| `boundary_b0` | telomere/subtelomere boundary position in the oriented read |
| `sub_bp` | bp of subtelomere past the boundary |
| `read_bp`, `orient`, `qs` | length, telomeric end (`head`/`tail`), mean qscore |
| `cohesion` | fraction of the weight of the edges the read chose that lies inside its own cluster |

A column from a stage the read never reached is `NA`, not `0`.

Reasons a read is left unclustered, latest stage first:

| reason | the read |
|---|---|
| `small_cluster` | was in a cluster smaller than `--reject-min-size` |
| `disputed` | had under `--min-cohesion` of its edge weight in its cluster |
| `no_kmers` | had no informative k-mers |
| `short_sub` | had under `--min-subtelo-bp` of subtelomere past the boundary |
| `short_telo` | had under `--min-telo-bp` of telomere before the boundary |
| `no_boundary` | had no telomere/subtelomere boundary |
| `thin_sub` | had under `--min-subtelo-bp` not matching `--telo-regex` |
| `thin_telo` | had under `--min-telo-bp` matching `--telo-regex` |
| `no_array` | had no telomeric strand teloBP could call |
| `both_ends` | was telomeric at both ends |
| `low_qs` | had a mean qscore under `--min-qs` |

### `clusters.tsv`

`cluster  n_reads  n_internal_edges  median_internal_weight  min_internal_weight  mean_strength  median_b0  median_sub_bp  representative`

`representative` is the group's read with the highest `strength`, or its
longest read when the group has no edges.

### Optional

- **`--cache DIR`** writes intermediates that later runs reuse, plus:
   - `<sample>.edges.tsv`: `read_i  read_j  weight  n_shared  idf_shared  chose`,
   where `chose` is `both`, `i` or `j`, the end(s) that picked the edge
   - `<sample>.run.json`: parameters, intake tallies and a summary of the
   weights and clusters
- **`--browser`** writes `<sample>.browser.html`, a self-contained page that
   shows the graph (t-SNE), the reads as sequence, and the clusters as
   distributions.

------

## The algorithm

### 1. Intake

  1. **Quality.** Drop reads whose basecaller mean qscore (`qs` tag) is under
     `--min-qs` (`low_qs`). Reads with no tag are kept.
  2. **Orient.** [teloBP](https://github.com/GreiderLab/TeloBP) works out which strand the telomere is on. C-strand reads
     are kept as they are and G-strand reads are reverse-complemented, so every
     read starts on its CCCTAA array. Reads that are telomeric at both ends
     (`both_ends`) or at neither (`no_array`) are dropped.
  3. **Pre-screen.** Before the boundary call, count the bases that match
     `--telo-regex`. Drop the read if fewer than `--min-telo-bp` match
     (`thin_telo`) or fewer than `--min-subtelo-bp` don't (`thin_sub`).
  4. **Boundary.** teloBP calls `b0`, where the array ends and the subtelomere
     begins. It looks no further than `--bound-margin` past the array. A read
     with no call is dropped (`no_boundary`).
  5. **Snap.** If the end of a `--telo-regex` run is within `--snap-bp` of `b0`,
     `b0` moves onto the nearest one. That puts every read's origin on a
     landmark the reads share.
  6. **Length gates.** Drop reads with less than `--min-telo-bp` of array before
     `b0` (`short_telo`) or less than `--min-subtelo-bp` of subtelomere after it
     (`short_sub`).

### 2. The graph

  1. **Window.** From each read, take `[b0 - --telo-bp, b0 + --sub-bp)`,
     clipped to the read (`--telo-bp 0` runs back to its tip). The window
     spans the boundary, so it includes the k-mers that cross it.
  2. **k-mers.** Count every k-mer of length `-k` in the window. Their order and
     position are not used.
  3. **Vocabulary.** Keep k-mers found in at least `--min-df` reads and in no
     more than a `--max-df` fraction of them, minus any found in every read.
     Reads with no k-mers left are held out of the graph (`no_kmers`).
  4. **Normalise.** TF-IDF: `v[m] = log1p(count) × log(n / df)`.
  5. **Distance.** Cosine distance between every pair of reads, on [0, 1].
  6. **Edges.** Each read points at its `--n-neighbors` nearest reads. A
     neighbour further away than `--max-edge-distance` gets no edge. Each edge
     that survives weighs `1 - d`.
  7. **Symmetrise.** For each pair, combine the two directed weights:
     `w = r(a + b - ab) + (1 - r)ab`, with `r = 0.5`. A mutual edge keeps its
     weight, and an edge only one read chose counts for half.

### 3. Cluster assignment

  1. **Resolution.** `γ = 2 × total edge weight / (n choose 2)`, which is twice
     the graph's own weight density.
  2. **Cut.** Run Leiden on CPM at `γ` until the partition stops changing.
  3. **Cohesion.** For each read, take the edges *it* chose and find the share
     of their weight that lands inside its own cluster. A read that chose no
     edges scores 0.
  4. **Refuse.** Unplace reads below `--min-cohesion` (`disputed`). Then unplace
     reads whose cluster has fallen below `--reject-min-size` (`small_cluster`).
  5. **Repeat.** Delete the refused reads from the graph and run steps 1–4 again
     on what is left. Steps 1–4 run at most `--max-iterations` times in all,
     and stop early once a pass refuses no read.

