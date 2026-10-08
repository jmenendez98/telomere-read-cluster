"""Intake: read the input, filter it, orient each read telomere-first and
call its telomere/subtelomere boundary."""

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
# b0 of a read with no boundary call
NO_BOUNDARY = -1
# a value that was never measured
UNKNOWN = -1

# basecaller mean qscore in a FASTQ header
QS_TAG = re.compile(r"\bqs:f:(-?[0-9.]+)")


@dataclass
class Read:
    """A read that passed intake."""
    read_id: str
    # oriented: the telomeric array comes first
    seq: str
    # the end the array was on as sequenced: "head" or "tail"
    orient: str
    # boundary position in seq
    b0: int
    # bp past b0
    sub_bp: int
    qs: float = UNKNOWN


@dataclass
class Dropped:
    """A read intake dropped, with why and what was measured before it."""
    read_id: str
    reason: str
    seq: str
    orient: str
    b0: int
    sub_bp: int
    qs: float = UNKNOWN


def rc(s):
    return s.translate(COMP)[::-1]


def _read_fastq(path):
    """Ids, upper-case sequences and qs tags (None if absent)."""
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
    """Same from primary records, reverse-strand ones back as sequenced."""
    import pysam
    ids, seqs, qs = [], [], []
    mode = "rc" if path.endswith(".cram") else ("r" if path.endswith(".sam")
                                                else "rb")
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
    """"FASTQ" or "BAM" from the extension; exits if unknown or missing."""
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
    kind = check_input(path)
    ids, seqs, qs = (_read_bam(path) if kind == "BAM" else _read_fastq(path))
    if not ids:
        raise SystemExit(f"intake: {os.path.basename(path)} holds no reads")
    log(f"intake: {len(ids):,} {kind} records in {os.path.basename(path)}")
    return ids, seqs, qs


def composition(seq, rgx):
    """bp of seq matching rgx, and bp not matching it."""
    m = sum(x.end() - x.start() for x in rgx.finditer(seq))
    return m, len(seq) - m


# _classify's settings, set per process by _init
_MARGIN, _SNAP, _RGX, _MIN_TELO, _MIN_SUB = 5_000, 0, None, 0, 0


def _init(margin, snap, pattern, min_telo, min_sub):
    global _MARGIN, _SNAP, _RGX, _MIN_TELO, _MIN_SUB
    _MARGIN, _SNAP = margin, snap
    _RGX = re.compile(pattern) if pattern else None
    _MIN_TELO, _MIN_SUB = min_telo, min_sub


_GPAT = teloNPTeloCompositionGStrand[-1]
_CPAT = teloNPTeloCompositionCStrand[-1]


def _classify(seq):
    """(reason, flipped, b0) for one read; reason is "" if it is kept."""
    st = getIsGStrandFromSeq(seq, _GPAT, _CPAT)
    if st == errorReturns["fusedRead"]:
        return "both_ends", None, NO_BOUNDARY
    if not isinstance(st, bool) and not isinstance(st, np.bool_):
        return "no_array", None, NO_BOUNDARY
    # G-strand reads are reverse-complemented onto the C strand
    flip = bool(st)
    o = rc(seq) if flip else seq
    # pre-screen before the boundary call
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
    """_classify over every read, in a fork pool when threads > 1."""
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


def reject_counts(dropped):
    c = {}
    for d in dropped:
        c[d.reason] = c.get(d.reason, 0) + 1
    return c


def intake(path, *, min_qs, min_telo_bp, telo_regex, bound_margin, snap_bp,
           min_subtelo_bp, threads):
    """The reads that pass every intake gate, and the ones dropped."""
    try:
        re.compile(telo_regex) if telo_regex else None
    except re.error as e:
        raise SystemExit(f"intake: --telo-regex is not a valid pattern: {e}")
    if snap_bp > 0 and not telo_regex:
        log(f"intake: --snap-bp {snap_bp} has no --telo-regex to snap onto; "
            f"boundaries are taken as teloBP calls them")

    ids, seqs, qs = read_input(path)
    dropped, no_tag = [], 0

    # quality gate; reads without a qs tag pass
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
        o = rc(seq) if flip else seq
        orient = "" if flip is None else ("tail" if flip else "head")
        if reason:
            dropped.append(Dropped(read_id=rid, reason=reason, seq=o,
                                   orient=orient, b0=NO_BOUNDARY,
                                   sub_bp=UNKNOWN, qs=q))
            continue
        # length gates either side of the boundary
        b = int(b)
        sub = len(o) - b
        if b < min_telo_bp:
            dropped.append(Dropped(read_id=rid, reason="short_telo", seq=o,
                                   orient=orient, b0=b, sub_bp=int(sub),
                                   qs=q))
            continue
        if sub < min_subtelo_bp:
            dropped.append(Dropped(read_id=rid, reason="short_sub", seq=o,
                                   orient=orient, b0=b, sub_bp=int(sub),
                                   qs=q))
            continue
        reads.append(Read(read_id=rid, seq=o, orient=orient, b0=b,
                          sub_bp=int(sub), qs=q))

    rejects = reject_counts(dropped)
    n_flip = sum(1 for r in reads if r.orient == "tail")
    n_placed = (len(reads) + rejects.get("short_telo", 0)
                + rejects.get("short_sub", 0))
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
    if min_telo_bp > 0:
        log(f"intake: {rejects.get('short_telo', 0):,} reads with under "
            f"{min_telo_bp} bp of array before the boundary dropped")
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
