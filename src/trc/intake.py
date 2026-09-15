"""Module 1 of 3: reads in, oriented telomere-first with a boundary called.

    reads, dropped = intake(path, ...)

FASTQ or BAM in; out comes one `Read` per surviving record, its sequence
rewritten so index 0 is the first base of the telomere array and `b0` marking
where that array ends.  Everything downstream measures from `b0`, so this is
the module that establishes the coordinate system.

**Which end is the telomere is teloBP's own call.**  `getIsGStrandFromSeq`
counts C-strand telomere runs from the read's START and G-strand runs from its
END, and answers with the STRAND: `False` means the array reads CCCTAA at the
front, `True` that it reads TTAGGG at the back, and a read matching at both
ends or at neither comes back as an error code rather than a guess.  A G-strand
read is reverse-complemented, so every read this module returns is C-strand
with its array at the front -- the convention `_boundary` then asserts to
teloBP.  No aligner is consulted: a BAM is read as a container of sequences,
and where its records happen to be placed is never used.

This replaced a canonical-hexamer density scan that carried three flags of its
own and two tuned constants.  The scan picked an END rather than a strand, so a
G-strand all-telomere fragment could be oriented head-first on its TTAGGG array
and then handed to teloBP with `isGStrand=False` -- scanned against a pattern
its array cannot match.  Asking teloBP for the strand fixes that at the source.
Over six files the two rules disagree on 1 read in 2,509, 5 in 2,710, 1 in
2,153, 8 in 618, 3 in 573 and 3 in 782, and on the one hand-curated sample they
score identically against it (ARI 0.9989 either way).

**Nothing is masked, and no hexamer is named here.**  An earlier revision of
this pipeline replaced every canonical repeat with N before extracting
features, which also destroyed the k-mers that STRADDLE a variant repeat -- the
ones that say what context it sits in.  The array is removed downstream by the
IDF, because everything that reached it has one.

**Six gates.**  In the order they are applied:

    --min-qs          the basecaller's own mean qscore, read from the header
    both_ends         teloBP finds telomeric repeat at BOTH ends of the read
    no_array          teloBP cannot call the read's strand at all
    --telo-regex      too little sequence matching it, or too little not
                      matching it, to be worth calling a boundary on
    no_boundary       teloBP finds no array/subtelomere transition
    --min-subtelo-bp  less than this much subtelomere past the boundary

Three are flags; the other three are teloBP's own verdicts.  There is no tuned
constant left in this module for a caller to have to know about.

**Nothing is thrown away.**  Every record the file held comes back: the ones
that passed all six gates as `Read`, the rest as `Dropped` carrying the name
of the gate that stopped them.  Each gate's name is the sub-cluster its reads
appear under in `<sample>.reads.tsv` -- `unclustered:low_qs`,
`unclustered:no_array` -- so a read that is not in a cluster is in the table
saying why, and the row count of that table is the record count of the input.

`--telo-regex` puts the same two questions to the read before teloBP is paid
for.  It counts the bases of the oriented read the pattern covers and the
bases it does not, and holds those two counts to `--min-telo-bp` and
`--min-subtelo-bp`.  Neither is the length of anything contiguous -- the
default pattern is teloBP's own C-strand one, permissive enough to fire on
C-rich subtelomere as readily as on the array -- so both counts run well above
what they are named after and the gate is loose by construction.  It is meant
to be: a first pass that is wrong in the direction of keeping reads.

It is loose and it still finds something.  Measured against the density scan
it replaced, over 40 files it rejected 1,702 of 66,631 reads, 1,574 of which
teloBP or `--min-subtelo-bp` was going to lose anyway.  Of the 128 that were
new, 99 were almost entirely array and teloBP had still put a boundary halfway
down them, which is a boundary in the middle of a telomere.  The other 29
carried a TTAGGG array where a CCCTAA one should be -- the strand bug teloBP's
own call now prevents, so reads of that kind should no longer reach this gate.

`--telo-regex ''` turns the gate off.

`--min-subtelo-bp` is the one that is not obvious.  A read with 138 bp of
subtelomere cannot fill a 1,000 bp `--sub-bp` window, so its k-mer profile is
built mostly out of sequence that is not there -- and it does not merely fail
to be placed, it contributes edges to everything it touches.  That is a build
problem rather than a scoring one, which is why the gate is here and not in the
clusterer.

Its default is pinned to `--sub-bp`, so every read that reaches the graph can
fill the subtelomere window completely.  That is stricter than it needs to be
to avoid the failure above: 150 bp was the lossless point across five
hand-curated samples, the shortest subtelomere on any grouped read there being
151 bp.  The two are not linked in code, so raising `--sub-bp` without raising
this leaves partly-empty windows again.
"""
from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np

from .teloboundary import (errorReturns, getIsGStrandFromSeq,
                           getTeloNPArrayBoundary,
                           teloNPTeloCompositionCStrand,
                           teloNPTeloCompositionGStrand)
from .util import log

COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")
# Out of band for every field that carries one: a boundary and a subtelomere
# length are counts of bases and a qscore is not negative, so -1 cannot be
# confused with a measurement.  The tables write "NA" for it.
NO_BOUNDARY = -1
UNKNOWN = -1

# The basecaller's own mean qscore, as dorado writes it into a FASTQ header
# (`@<id>\tqs:f:25.7299\tdu:f:...`) or a BAM tag.  Read rather than recomputed
# from the quality string, because it is the number the basecaller stands
# behind and the one a person means by "read quality".
QS_TAG = re.compile(r"\bqs:f:(-?[0-9.]+)")


@dataclass
class Read:
    """One surviving read, in the pipeline's coordinate system.

    `seq` is the read as teloBP saw it, reverse-complemented if the array was
    at its far end.  It is NOT trimmed, so `b0` counts whatever preceded the
    array along with the array itself; `seq[:b0]` is array (plus any lead-in)
    and `seq[b0:]` is subtelomere.  `b0` is the array length this module
    reports -- there is no second, independent measurement of it.
    """
    read_id: str
    seq: str
    orient: str             # head (C strand) | tail (G strand, flipped)
    b0: int                 # array/subtelomere boundary, teloBP
    sub_bp: int             # len(seq) - b0, the subtelomere actually present
    qs: float = UNKNOWN     # the basecaller's own mean qscore, or UNKNOWN


@dataclass
class Dropped:
    """One read a gate rejected, and HOW FAR IT GOT before it was rejected.

    A dropped read is kept whole rather than tallied, because the tally cannot
    answer the question a reject list is usually opened for -- *which* read,
    and does it look like one of the ones that survived.  The fields are
    filled as far as the read reached and are `UNKNOWN` past that point: a
    read that failed `--min-qs` was never given to teloBP and so has no strand
    and no boundary, one that failed `--telo-regex` has a strand but no
    boundary, and one that failed `--min-subtelo-bp` has both.

    `seq` is oriented where the strand was called and raw where it was not, so
    it is always the most that is known about the read and never a guess: the
    two reasons that ARE a failure to call the strand (`both_ends`,
    `no_array`) are exactly the ones that cannot be oriented.
    """
    read_id: str
    reason: str
    seq: str
    orient: str             # head | tail, or "" where no strand was called
    b0: int                 # or NO_BOUNDARY where none was called
    sub_bp: int             # or UNKNOWN
    qs: float = UNKNOWN


def rc(s):
    return s.translate(COMP)[::-1]


# ------------------------------------------------------------------- reading
def _read_fastq(path):
    """`(ids, seqs, qs)` from a FASTQ, gzipped or not.

    Quality strings are not kept: nothing downstream reads them, and `qs` above
    is the score this pipeline judges a read by.
    """
    ids, seqs, qs = [], [], []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for i, line in enumerate(fh):
            m = i & 3
            if m == 0:
                ids.append(line[1:].split()[0])
                t = QS_TAG.search(line)
                qs.append(float(t.group(1)) if t else None)
            elif m == 1:
                seqs.append(line.strip().upper())
    if len(ids) != len(seqs):
        raise SystemExit(f"{path}: truncated FASTQ "
                         f"({len(ids)} ids, {len(seqs)} seqs)")
    return ids, seqs, qs


def _read_bam(path):
    """`(ids, seqs, qs)` from a BAM/CRAM/SAM, primary records only.

    Secondary (0x100) and supplementary (0x800) records are skipped: they carry
    a fragment of a sequence this file already holds in full, and admitting
    them would enter one read into the graph several times over.  Unmapped
    records are KEPT, because an unaligned dorado BAM is a normal input here
    and every record in it is unmapped.

    A minus-strand record's SEQ has been reverse-complemented by the aligner,
    so it is put back before the strand is called.  That is not because the
    original orientation matters -- teloBP decides it from composition a moment
    later -- but so that the answer describes the READ rather than describing
    which way an aligner happened to place it.
    """
    import pysam
    ids, seqs, qs = [], [], []
    mode = "rc" if path.endswith(".cram") else ("r" if path.endswith(".sam")
                                                else "rb")
    # check_sq=False so a header with no @SQ lines -- what dorado emits for
    # unaligned output -- is read instead of refused.
    with pysam.AlignmentFile(path, mode, check_sq=False) as fh:
        for rec in fh.fetch(until_eof=True):
            if rec.is_secondary or rec.is_supplementary:
                continue
            s = rec.query_sequence
            if not s:
                continue
            ids.append(rec.query_name)
            seqs.append(rc(s.upper()) if rec.is_reverse else s.upper())
            qs.append(float(rec.get_tag("qs"))
                      if rec.has_tag("qs") else None)
    return ids, seqs, qs


BAM_EXT = (".bam", ".cram", ".sam")
FASTQ_EXT = (".fq", ".fastq", ".fq.gz", ".fastq.gz")


def check_input(path):
    """`"BAM"` or `"FASTQ"`, or exit saying why this file is neither.

    Called before anything reads or stats the file -- including the cache
    stamp, which would otherwise turn a mistyped path into a traceback from
    `os.stat` rather than a sentence the caller can act on.  Dispatch is on the
    extension and not on file magic, so an input this cannot name is an error
    rather than a guess it has to trust.
    """
    low = path.lower()
    if low.endswith(BAM_EXT):
        kind = "BAM"
    elif low.endswith(FASTQ_EXT):
        kind = "FASTQ"
    else:
        raise SystemExit(
            f"intake: cannot tell what {os.path.basename(path)} is from its "
            f"name.  Expected one of {', '.join(FASTQ_EXT + BAM_EXT)}")
    if not os.path.exists(path):
        raise SystemExit(f"intake: no such file: {path}")
    if not os.path.isfile(path):
        raise SystemExit(f"intake: not a file: {path}")
    return kind


def read_input(path):
    """`(ids, seqs, qs)` from whichever of the two formats this is."""
    kind = check_input(path)
    ids, seqs, qs = (_read_bam(path) if kind == "BAM" else _read_fastq(path))
    if not ids:
        raise SystemExit(f"intake: {os.path.basename(path)} holds no reads")
    log(f"intake: {len(ids):,} {kind} records in {os.path.basename(path)}")
    return ids, seqs, qs


# ------------------------------------------------------------- composition
def composition(seq, rgx):
    """`(matching, non-matching)` bases of `seq` under `rgx`.

    Matches are the non-overlapping ones `finditer` walks left to right, so
    every base is counted once and the two numbers sum to the read's length.
    """
    m = sum(x.end() - x.start() for x in rgx.finditer(seq))
    return m, len(seq) - m


# ------------------------------------------------- strand, gate and boundary
# Set once per worker by the pool initialiser, so `_classify` can stay a
# single-argument module-level function that Pool.map can carry.
_MARGIN, _SNAP, _RGX, _MIN_TELO, _MIN_SUB = 5_000, 0, None, 0, 0


def _init(margin, snap, pattern, min_telo, min_sub):
    global _MARGIN, _SNAP, _RGX, _MIN_TELO, _MIN_SUB
    _MARGIN, _SNAP = margin, snap
    _RGX = re.compile(pattern) if pattern else None
    _MIN_TELO, _MIN_SUB = min_telo, min_sub


# `getTeloBoundary` hands `composition[targetPatternIndex]` -- the inner
# `[pattern, ratio, unit]` triple -- to `getIsGStrandFromSeq`, not the outer
# list, so the same is done here.
_GPAT = teloNPTeloCompositionGStrand[-1]
_CPAT = teloNPTeloCompositionCStrand[-1]


def _classify(seq):
    """`(reason, flipped, b0)` for one raw read; `reason` is "" if it survived.

    The strand is reported even when the read is then rejected, and that is
    the point of returning a triple rather than the pair this used to: a read
    dropped at `--telo-regex` or at the boundary call has had its strand
    called, so the caller can store it the right way round and a reject can be
    looked at in the same coordinate system as the reads that survived.
    `flipped` is `None` only for the two rejections that ARE a failure to call
    the strand.

    Strand, the composition gate and the boundary in ONE worker call.  The gate
    exists to avoid paying teloBP's ~280 ms on a hopeless read, which only
    works if it runs on the same side of the process boundary as teloBP does;
    and the strand call is what decides which sequence the gate should see.

    `getIsGStrandFromSeq` counts C-strand runs from the read's start and
    G-strand runs from its end, so `False` means the array is already at the
    front and `True` means the read has to be flipped.  It returns an error
    code instead when the read is telomeric at both ends (`fusedRead`) or at
    neither (`strandType`), and those are the two rejections a density scan had
    to invent a ratio and a lead-in to make.

    `isGStrand=False` on the boundary call is now always true by construction:
    the read has just been put on the C strand, so its array reads CCCTAA.

    The scan is bounded because teloBP's nanopore preset takes the *last*
    window still above its change threshold anywhere in the read, so a distant
    stretch whose composition matches -- an interstitial telomeric block at
    20q, Xp's C-rich non-telomeric stretch -- recaptures the search and the
    read acquires a fingerprint belonging to the wrong locus.

    The snap is handed `_RGX`, the SAME pattern the composition gate above
    just ran.  The boundary is the origin every read in the pool is measured
    from, so the landmark it is moved onto has to be one they all share, and
    `--telo-regex` is the only such pattern this pipeline has.  It is skipped
    outright at `_SNAP == 0`, or when there is no pattern to snap onto, rather
    than asked for and ignored -- which saves the scan it would need.
    """
    st = getIsGStrandFromSeq(seq, _GPAT, _CPAT)
    if st == errorReturns["fusedRead"]:
        return "both_ends", None, NO_BOUNDARY
    if not isinstance(st, bool) and not isinstance(st, np.bool_):
        return "no_array", None, NO_BOUNDARY
    flip = bool(st)
    o = rc(seq) if flip else seq
    if _RGX is not None:
        telo, sub = composition(o, _RGX)
        if telo < _MIN_TELO:
            return "thin_telo", flip, NO_BOUNDARY
        if sub < _MIN_SUB:
            return "thin_sub", flip, NO_BOUNDARY
    b = getTeloNPArrayBoundary(o, isGStrand=False, margin=_MARGIN,
                               snap=_SNAP > 0 and _RGX is not None,
                               snapBP=_SNAP, snapRgx=_RGX)
    if b is None or b < 0:
        return "no_boundary", flip, NO_BOUNDARY
    return "", flip, int(b)


def classify(seqs, threads, margin, snap, pattern, min_telo, min_sub):
    """`_classify` over many reads, in parallel.

    teloBP is ~280 ms a read, which is a quarter of an hour a sample
    single-threaded and the reason `--cache` exists.  `Pool.map` preserves
    order and teloBP carries no JIT, so this adds none of the CPU-generation
    sensitivity a numba stage would.
    """
    if not seqs:
        return []
    log(f"intake: teloBP over {len(seqs):,} reads on {threads} processes")
    args = (margin, snap, pattern, min_telo, min_sub)
    if threads <= 1:
        _init(*args)
        return [_classify(x) for x in seqs]
    with get_context("fork").Pool(threads, initializer=_init,
                                 initargs=args) as p:
        return p.map(_classify, seqs, chunksize=8)


# ----------------------------------------------------------------------- run
def reject_counts(dropped):
    """The reject tally, `{reason: n}`, over a list of `Dropped`.

    The tally is DERIVED from the list and never kept beside it: two counters
    for one set of reads is two things that can disagree, and the one that
    would then be believed is whichever a given log line happened to read.
    """
    c = {}
    for d in dropped:
        c[d.reason] = c.get(d.reason, 0) + 1
    return c


def intake(path, *, min_qs, min_telo_bp, telo_regex, bound_margin, snap_bp,
           min_subtelo_bp, threads):
    """Module 1 end to end.  Returns `(reads, dropped)`.

    The quality gate is applied FIRST, before anything else: a read under
    `min_qs` is not a read this pipeline has an opinion about, and classifying
    it only to discard it would put its verdict in the reject tally.

    Reads with no teloBP boundary do not become `Read`s.  There is no landmark
    to measure from, so every position such a read could offer is
    uninterpretable and no window can be cut from it.  They are RETURNED, as
    `Dropped`, with the reason they failed and as much of the coordinate
    system as was established before they did -- so the row a read gets in
    `<sample>.reads.tsv` says which gate lost it rather than the read simply
    being absent from the file.
    """
    try:
        re.compile(telo_regex) if telo_regex else None
    except re.error as e:
        raise SystemExit(f"intake: --telo-regex is not a valid pattern: {e}")
    if snap_bp > 0 and not telo_regex:
        log(f"intake: --snap-bp {snap_bp} has no --telo-regex to snap onto; "
            f"boundaries are taken as teloBP calls them")

    ids, seqs, qs = read_input(path)
    dropped, no_tag = [], 0

    kept_ids, kept_seq, kept_qs = [], [], []
    for rid, seq, q in zip(ids, seqs, qs):
        if q is None:
            no_tag += 1
        elif q < min_qs:
            dropped.append(Dropped(read_id=rid, reason="low_qs", seq=seq,
                                   orient="", b0=NO_BOUNDARY, sub_bp=UNKNOWN,
                                   qs=float(q)))
            continue
        kept_ids.append(rid)
        kept_seq.append(seq)
        kept_qs.append(UNKNOWN if q is None else float(q))
    if min_qs > 0:
        log(f"intake: {len(dropped):,} reads under qs {min_qs:g} "
            f"dropped before anything else is asked of them" +
            (f"; {no_tag:,} kept with no qs tag to judge them by"
             if no_tag else ""))

    verdicts = classify(kept_seq, threads, bound_margin, snap_bp, telo_regex,
                        min_telo_bp, min_subtelo_bp)
    reads = []
    for rid, seq, q, (reason, flip, b) in zip(kept_ids, kept_seq, kept_qs,
                                              verdicts):
        # `flip is None` is the strand call itself having failed, and the read
        # is then stored as it came off the sequencer -- the one case where a
        # sequence here is not in the pipeline's coordinate system.
        o = rc(seq) if flip else seq
        orient = "" if flip is None else ("tail" if flip else "head")
        if reason:
            dropped.append(Dropped(read_id=rid, reason=reason, seq=o,
                                   orient=orient, b0=NO_BOUNDARY,
                                   sub_bp=UNKNOWN, qs=q))
            continue
        b = int(b)
        sub = len(o) - b
        if sub < min_subtelo_bp:
            dropped.append(Dropped(read_id=rid, reason="short_sub", seq=o,
                                   orient=orient, b0=b, sub_bp=int(sub),
                                   qs=q))
            continue
        reads.append(Read(read_id=rid, seq=o, orient=orient, b0=b,
                          sub_bp=int(sub), qs=q))

    rejects = reject_counts(dropped)
    n_flip = sum(1 for r in reads if r.orient == "tail")
    n_placed = len(reads) + rejects.get("short_sub", 0)
    log(f"intake: {n_placed:,} reads placed on the C strand "
        f"({n_flip:,} reverse-complemented), "
        f"{rejects.get('both_ends', 0):,} telomeric at both ends and "
        f"{rejects.get('no_array', 0):,} on neither strand")
    if rejects.get("thin_telo") or rejects.get("thin_sub"):
        log(f"intake: {rejects.get('thin_telo', 0):,} reads matching "
            f"--telo-regex over under {min_telo_bp} bp and "
            f"{rejects.get('thin_sub', 0):,} with under {min_subtelo_bp} bp "
            f"not matching it dropped before the boundary call")
    if rejects.get("no_boundary"):
        log(f"intake: {rejects['no_boundary']:,} reads with no boundary "
            f"(all-array) dropped")
    if reads:
        med = int(np.median([r.b0 for r in reads]))
        log(f"intake: median boundary {med:,} bp")
    if min_subtelo_bp > 0:
        log(f"intake: {rejects.get('short_sub', 0):,} reads with under "
            f"{min_subtelo_bp} bp of subtelomere past the boundary dropped")
    log(f"intake: {len(reads):,} reads of {len(ids):,} survive "
        f"({', '.join(f'{v:,} {k}' for k, v in sorted(rejects.items())) or 'none dropped'})")
    if not reads:
        raise SystemExit("intake: no reads survived; nothing to build a graph "
                         "from.  Loosen --min-qs, --min-telo-bp or "
                         "--min-subtelo-bp, or clear --telo-regex.")
    return reads, dropped
