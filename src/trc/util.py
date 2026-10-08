"""Logging, thread counts, cache signatures and atomic file writes."""

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
_QUIET = False


def verbose(on=True):
    """Turn progress logging on or off."""
    global _QUIET
    _QUIET = not on


def log(msg):
    """Print msg to stderr, stamped with seconds since import."""
    if not _QUIET:
        print(f"[{time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def n_threads(requested=0):
    """The requested count, else SLURM's CPUs, else every core."""
    if requested:
        return max(int(requested), 1)
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = int(os.environ.get(var, 0) or 0)
        if v:
            return v
    return os.cpu_count() or 1


def input_sig(path):
    """Cache signature of a file: its size and mtime."""
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


def code_sig(*modules):
    """Hash of the modules' source, so any edit invalidates the cache."""
    h = hashlib.sha1()
    for m in modules:
        try:
            h.update(inspect.getsource(m).encode())
        except (OSError, TypeError):
            h.update(repr(m).encode())
    return h.hexdigest()[:12]


def param_sig(**kw):
    """Cache signature of keyword parameters, in sorted order."""
    return "|".join(f"{k}={kw[k]}" for k in sorted(kw))


def cache_path(cachedir, name):
    os.makedirs(cachedir, exist_ok=True)
    return os.path.join(cachedir, name)


def load_npz(path, sig):
    """Cached arrays, or None if absent, unreadable or of another sig."""
    if not path or not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
        if str(z["__sig__"]) != sig:
            return None
        return z
    except Exception:
        return None


def save_npz(path, sig, **arrays):
    """Write arrays and their sig atomically; no-op without a path."""
    if not path:
        return
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, __sig__=np.array(sig), **arrays)
    os.replace(tmp, path)


def save_seqs(path, seqs):
    """Write one sequence per line, gzipped, atomically."""
    tmp = f"{path}.tmp{os.getpid()}"
    with gzip.open(tmp, "wt", compresslevel=4) as fh:
        for s in seqs:
            fh.write(s + "\n")
    os.replace(tmp, path)


def load_seqs(path):
    with gzip.open(path, "rt") as fh:
        return [line.rstrip("\n") for line in fh]


def write_tsv(path, rows, columns):
    """Write dict rows as a TSV atomically; None is written as NA."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", newline="\n") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join(_fmt(r.get(c, "")) for c in columns) + "\n")
    os.replace(tmp, path)
    log(f"wrote {path} ({len(rows):,} rows)")


def _fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_json(path, obj):
    """Write obj as sorted, indented JSON atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_jsonable)
        fh.write("\n")
    os.replace(tmp, path)
    log(f"wrote {path}")


def _jsonable(o):
    """json.dump fallback for numpy values and tuples."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))
