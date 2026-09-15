"""Logging, thread count, TSV/JSON writing, and a cache that detects staleness.

Not a fourth module in the pipeline sense -- nothing here decides anything
about a read.  It is the plumbing the three modules and the CLI share.

**The caching discipline.**  A stage's file is stamped with the parameters that
stage actually reads, with the size and mtime of the upstream input, and with a
hash of the SOURCE of the modules that produce it.  It is reused only when
every one of them is unchanged, so editing a constant -- or editing a function
without touching any constant -- invalidates exactly the stages downstream of
it and nothing above.  There is no cache to clear by hand.

Writes go through a temporary name and are renamed into place, so a job killed
partway leaves either the old cache or the new one, never a truncated file a
later run would trust.
"""
from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np

T0 = time.time()
# On for anything that imports these modules directly.  The CLI turns it off
# unless -v/--verbose was given, so a `trc` run is silent by default.
_QUIET = False


def verbose(on=True):
    global _QUIET
    _QUIET = not on


def log(msg):
    if not _QUIET:
        print(f"[{time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def n_threads(requested=0):
    """`--threads`, or every core this job can actually see."""
    if requested:
        return max(int(requested), 1)
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = int(os.environ.get(var, 0) or 0)
        if v:
            return v
    return os.cpu_count() or 1


# --------------------------------------------------------------------- stamps
def input_sig(path):
    """Size and mtime of an input file, as the upstream half of a cache stamp."""
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


def code_sig(*modules):
    """A hash of the source of the modules a cached stage is produced by.

    Stamping parameter VALUES alone leaves a standing hazard: change a function
    without changing any constant and the cache is reused and is wrong.  This
    closes that by construction, with nothing to remember.
    """
    h = hashlib.sha1()
    for m in modules:
        try:
            h.update(inspect.getsource(m).encode())
        except (OSError, TypeError):
            # Source unavailable (frozen, or defined interactively); invalidate
            # nothing rather than crash a run.
            h.update(repr(m).encode())
    return h.hexdigest()[:12]


def param_sig(**kw):
    """The parameters a stage reads, as one order-independent string."""
    return "|".join(f"{k}={kw[k]}" for k in sorted(kw))


# ---------------------------------------------------------------------- cache
def cache_path(cachedir, name):
    os.makedirs(cachedir, exist_ok=True)
    return os.path.join(cachedir, name)


def load_npz(path, sig):
    """The cached arrays if the stamp matches, else None."""
    if not path or not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
        if str(z["__sig__"]) != sig:
            return None
        return z
    except Exception:
        # A corrupt or half-written cache is a cache miss, not a crash.
        return None


def save_npz(path, sig, **arrays):
    if not path:
        return
    # Through a file handle, not a name: np.savez appends ".npz" to a path that
    # lacks it, which would rename a file that is not the one just written.
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, __sig__=np.array(sig), **arrays)
    os.replace(tmp, path)


def save_seqs(path, seqs):
    """Sequences as gzipped text, one per line, not as a numpy array.

    A numpy unicode array is fixed-width: it pads every row out to the longest
    read.  On a sample whose longest read is ~1 Mb that turns a 220 Mbp cache
    into a 13.6 GB file.  compresslevel 4 because DNA reaches most of its ratio
    in the first few levels and 9 costs minutes to save a few percent.
    """
    tmp = f"{path}.tmp{os.getpid()}"
    with gzip.open(tmp, "wt", compresslevel=4) as fh:
        for s in seqs:
            fh.write(s + "\n")
    os.replace(tmp, path)


def load_seqs(path):
    with gzip.open(path, "rt") as fh:
        return [line.rstrip("\n") for line in fh]


# --------------------------------------------------------------------- output
def write_tsv(path, rows, columns):
    """One header, one row per record, LF line endings.

    LF and not the CRLF `csv` defaults to, so `awk -F'\\t'` works on the result.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", newline="\n") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join(_fmt(r.get(c, "")) for c in columns) + "\n")
    os.replace(tmp, path)
    log(f"wrote {path} ({len(rows):,} rows)")


def _fmt(v):
    # None is the tables' "this read never reached the stage that measures
    # this" -- a row for a read dropped at intake has no strength and no
    # degree, and writing 0 for them would be a measurement rather than a gap.
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_jsonable)
        fh.write("\n")
    os.replace(tmp, path)
    log(f"wrote {path}")


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))
