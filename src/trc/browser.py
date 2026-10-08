"""A self-contained HTML page over one trc run: the graph as t-SNE, the
reads as sequence, the clusters as box plots, and a manual edit mode."""

from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import os

# cap BLAS threads before numpy is imported
BLAS_THREADS = 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(BLAS_THREADS))

import numpy as np


from .cluster import (CLUSTERER_REASONS, REASON_TEXT,
                      UNPLACED_REASONS, unplaced_reason)
from .util import load_seqs, log


BASE_RGB = {"A": (61, 168, 83), "C": (66, 133, 244),
            "G": (249, 171, 0), "T": (234, 67, 53), "N": (208, 208, 208)}
GAP_RGB = (250, 250, 250)
GRID_RGB = (224, 224, 224)

# strength of bases drawn outside the window
FLANK_ALPHA = 0.30

# default --flank
FLANK_BP = 3000

# px per read row; blank rows between clusters
ROW_PX = 6
GAP_ROWS = 2

# page cluster codes: >= 0 cluster, -1 unclustered with no known reason,
# -2 - k unclustered for UNPLACED_REASONS[k]
UNCLUSTERED = -1


def reason_code(reason):
    return -2 - UNPLACED_REASONS.index(reason)



def _first(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def find_run(outdir, sample=None):
    """Sample name and the paths of its tables and cache under outdir."""
    outdir = os.path.abspath(outdir)
    if not os.path.isdir(outdir):
        raise SystemExit(f"{outdir} is not a directory -- the argument is a "
                         f"trc run directory, the one its `-o` wrote.")
    if sample is None:
        names = sorted(f[:-len(".reads.tsv")] for f in os.listdir(outdir)
                       if f.endswith(".reads.tsv"))
        if not names:
            raise SystemExit(f"no <sample>.reads.tsv in {outdir}")
        if len(names) > 1:
            raise SystemExit(f"{len(names)} samples in {outdir} "
                             f"({', '.join(names)}) -- name one with --sample")
        sample = names[0]

    cache = os.path.join(outdir, "cache")
    f = {
        "reads": _first(os.path.join(outdir, f"{sample}.reads.tsv")),
        "clusters": _first(os.path.join(outdir, f"{sample}.clusters.tsv")),
        "edges": _first(os.path.join(cache, f"{sample}.edges.tsv"),
                        os.path.join(outdir, f"{sample}.edges.tsv")),
        "json": _first(os.path.join(cache, f"{sample}.run.json"),
                       os.path.join(outdir, f"{sample}.run.json")),
        "intake": _first(os.path.join(cache, "intake.npz")),
        "seqs": _first(os.path.join(cache, "intake.seqs.txt.gz")),
    }
    missing = [k for k in ("reads", "edges", "intake", "seqs") if not f[k]]
    if missing:
        raise SystemExit(
            f"{outdir} is missing {', '.join(missing)}.\n"
            f"This page needs the read table, the edge list and the intake "
            f"cache: the run has to have been made with `--cache DIR` (for "
            f"the edges and the sequence) and DIR has to still be there.")
    return sample, f


def read_tsv(path):
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(hdr, ln.rstrip("\n").split("\t"))) for ln in fh if ln]


def load_reads(path):
    """reads.tsv as page arrays."""
    return _reads_table(read_tsv(path))


def _cell(v, default):
    """v, or default if it is missing, blank or NA."""
    if v is None:
        return default
    v = v.strip() if isinstance(v, str) else v
    return default if v in ("", "NA") else v


# reads.tsv columns other than the score
BASE_COLUMNS = ("read_id", "cluster", "strength", "degree",
                "n_informative_kmers", "boundary_b0", "sub_bp", "read_bp",
                "orient", "qs")

NO_SCORE = -1.0


def score_column(rows):
    """The one reads.tsv column beyond BASE_COLUMNS, or ""."""
    if not rows:
        return ""
    extra = [k for k in rows[0] if k not in BASE_COLUMNS]
    return extra[0] if len(extra) == 1 else ""


def _reads_table(rows):
    """Read rows as page arrays, cluster labels as page codes."""
    ids = [r["read_id"] for r in rows]
    cl, why, unknown = [], [], set()
    for r in rows:
        lab = str(r["cluster"])
        w = unplaced_reason(lab)
        if w is None:
            cl.append(int(lab))
            why.append("")
        elif w in UNPLACED_REASONS:
            cl.append(reason_code(w))
            why.append(w)
        else:
            if w:
                unknown.add(w)
            cl.append(UNCLUSTERED)
            why.append("")
    if unknown:
        log(f"browser: {', '.join(sorted(unknown))} is not a reason this "
            f"page knows; those reads are drawn as unclustered")
    cl = np.array(cl, np.int64)

    def col(name, dtype, default=0):
        return np.array([dtype(_cell(r.get(name), default)) for r in rows],
                        np.float64 if dtype is float else np.int64)

    name = score_column(rows)
    return {
        "ids": ids, "cluster": cl, "reason": why,
        "score": (col(name, float, NO_SCORE) if name
                  else np.full(len(ids), NO_SCORE)),
        "score_name": name,
        # a vertex: placed, refused by the clusterer, or reason unknown
        "ingraph": np.array([c >= 0 or w in CLUSTERER_REASONS or w == ""
                             for c, w in zip(cl, why)], bool),
        "strength": col("strength", float),
        "degree": col("degree", int),
        "nkmer": col("n_informative_kmers", int),
        "b0": col("boundary_b0", int),
        "hasb0": np.array([_cell(r.get("boundary_b0"), None) is not None
                           for r in rows], bool),
        "sub_bp": col("sub_bp", int),
        "read_bp": col("read_bp", int),
        "qs": col("qs", float, -1.0),
        "orient": [r.get("orient", "?") for r in rows],
    }


def _edges(triples, pos):
    """(read_i, read_j, weight) by id as index arrays, plus the count
    dropped for an id not in pos."""
    i, j, w, dropped = [], [], [], 0
    for ri, rj, wt in triples:
        a, b = pos.get(ri, -1), pos.get(rj, -1)
        if a < 0 or b < 0:
            dropped += 1
            continue
        i.append(a)
        j.append(b)
        w.append(float(wt))
    return (np.array(i, np.int64), np.array(j, np.int64),
            np.array(w, np.float64), dropped)


def load_edges(path, pos):
    """edges.tsv as _edges arrays."""
    def rows():
        with open(path) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            ci, cj, cw = (hdr.index("read_i"), hdr.index("read_j"),
                          hdr.index("weight"))
            for ln in fh:
                f = ln.rstrip("\n").split("\t")
                yield f[ci], f[cj], f[cw]
    return _edges(rows(), pos)


def load_intake(npz_path, seq_path, ids):
    """Cached oriented sequence per id ("" if absent), and the absent ids."""
    z = np.load(npz_path, allow_pickle=True)
    cached = [str(s) for s in z["ids"]]
    seqs = load_seqs(seq_path)
    if len(seqs) != len(cached):
        raise SystemExit(f"{len(seqs)} cached sequences against "
                         f"{len(cached)} cached ids in {npz_path}")
    at = {r: k for k, r in enumerate(cached)}
    out, missing = [], []
    for r in ids:
        k = at.get(r)
        if k is None:
            missing.append(r)
            out.append("")
        else:
            out.append(seqs[k])
    return out, missing


def _binary_search_sigma(d2, target, tol=1e-5, n_iter=60):
    """Row of exp(-beta d2), normalised, with beta bisected to entropy
    target."""
    beta, lo, hi = 1.0, -np.inf, np.inf
    p = None
    for _ in range(n_iter):
        p = np.exp(-d2 * beta)
        s = p.sum()
        if s <= 0.0:
            hi, beta = beta, beta / 2.0 if lo == -np.inf else (lo + beta) / 2.0
            continue
        h = np.log(s) + beta * float((d2 * p).sum()) / s
        p = p / s
        if abs(h - target) < tol:
            break
        if h > target:
            lo = beta
            beta = beta * 2.0 if hi == np.inf else (beta + hi) / 2.0
        else:
            hi = beta
            beta = beta / 2.0 if lo == -np.inf else (beta + lo) / 2.0
    return p


def joint_p(n, i, j, w, perplexity):
    """Symmetric t-SNE affinities over the graph's edges only, at distance
    1 - w; and each vertex's degree."""
    from scipy import sparse
    d = np.maximum(0.0, 1.0 - w)
    A = sparse.coo_matrix((np.concatenate([d, d]),
                           (np.concatenate([i, j]), np.concatenate([j, i]))),
                          shape=(n, n)).tocsr()
    A.sum_duplicates()
    P = np.zeros((n, n), np.float32)
    target = np.log(perplexity)
    deg = np.diff(A.indptr)
    for r in range(n):
        a, b = A.indptr[r], A.indptr[r + 1]
        if a == b:
            continue
        nb, dd = A.indices[a:b], A.data[a:b]
        P[r, nb] = _binary_search_sigma(dd ** 2, target).astype(np.float32)
    P += P.T
    P /= max(P.sum(dtype=np.float64), 1e-12)
    np.maximum(P, 1e-12, out=P)
    return P, deg


def tsne(P, *, seed=0, n_iter=1000, exaggeration=12.0, exag_iter=250,
         lr=None, momentum=(0.5, 0.8), log_every=200):
    """Exact t-SNE on a dense P, with early exaggeration, momentum and
    gains; returns n x 2 coordinates."""
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    lr = float(lr or max(n / exaggeration, 50.0))
    Y = (rng.standard_normal((n, 2)) * 1e-4).astype(np.float32)
    upd = np.zeros_like(Y)
    gains = np.ones_like(Y)
    P = P * exaggeration

    num = np.empty((n, n), np.float32)
    PQ = np.empty((n, n), np.float32)
    sq = np.empty(n, np.float32)
    rsum = np.empty(n, np.float32)
    pqy = np.empty((n, 2), np.float32)

    for it in range(n_iter):
        if it == exag_iter:
            P /= exaggeration
        np.dot(Y, Y.T, out=num)
        np.einsum("ij,ij->i", Y, Y, out=sq)
        num *= np.float32(-2.0)
        num += sq
        num += sq[:, None]
        num += np.float32(1.0)
        np.reciprocal(num, out=num)
        np.fill_diagonal(num, 0.0)
        z = num.sum(dtype=np.float64)

        np.multiply(num, np.float32(1.0 / z), out=PQ)
        np.subtract(P, PQ, out=PQ)
        PQ *= num
        np.sum(PQ, axis=1, out=rsum)
        np.dot(PQ, Y, out=pqy)
        grad = rsum[:, None] * Y
        grad -= pqy
        grad *= np.float32(4.0)

        mom = momentum[0] if it < 20 else momentum[1]
        same = np.sign(grad) == np.sign(upd)
        gains[same] *= 0.8
        gains[~same] += 0.2
        np.maximum(gains, 0.01, out=gains)
        upd *= mom
        upd -= lr * gains * grad
        Y += upd
        Y -= Y.mean(axis=0)

        if log_every and (it + 1) % log_every == 0:
            q = np.maximum(num * np.float32(1.0 / z), 1e-12)
            p = P / (exaggeration if it < exag_iter else 1.0)
            kl = float((p * np.log(np.maximum(p, 1e-12) / q)).sum())
            log(f"tsne: iteration {it + 1:>4}  KL {kl:7.4f}")
    return Y.astype(np.float64)


def _blas_threads(n):
    """Set OpenBLAS's thread count; returns the old one, 0 if unknown."""
    try:
        import ctypes
        path = next(ln.split()[-1] for ln in open("/proc/self/maps")
                    if "libopenblas" in ln)
        lib = ctypes.CDLL(path)
        was = int(lib.openblas_get_num_threads())
        lib.openblas_set_num_threads(int(n))
        return was
    except Exception:
        return 0


def embed(reads, edges, *, perplexity, seed, n_iter):
    """t-SNE coordinates of the reads in the graph; others stay at (0, 0)."""
    i, j, w, _ = edges
    n = len(reads["ids"])
    keep = np.flatnonzero(reads["ingraph"])
    at = np.full(n, -1, np.int64)
    at[keep] = np.arange(keep.size)
    ok = (at[i] >= 0) & (at[j] >= 0) if len(i) else np.zeros(0, bool)
    log(f"tsne: {int(ok.sum()):,} edges over {keep.size:,} of {n:,} reads, "
        f"perplexity {perplexity:g}")
    xy, deg = np.zeros((n, 2)), np.zeros(n, np.int64)
    was = _blas_threads(BLAS_THREADS)
    try:
        P, d = joint_p(keep.size, at[i[ok]], at[j[ok]], w[ok], perplexity)
        xy[keep] = tsne(P, seed=seed, n_iter=n_iter)
        deg[keep] = d
        return xy, deg
    finally:
        if was:
            _blas_threads(was)


def leaf_order(pts):
    """Order of pts along an average-linkage dendrogram."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return list(range(len(pts)))
    from scipy.cluster.hierarchy import leaves_list, linkage
    return [int(k) for k in leaves_list(linkage(pts, method="average"))]


def block_order(cluster, xy, ingraph):
    """Row order for the reads tab: clusters, and reads within them, by
    t-SNE position; unclustered groups last. Also (cid, start, size) per
    block."""
    cids = sorted({int(c) for c in cluster if c >= 0})
    cen = np.array([xy[cluster == c].mean(axis=0) for c in cids]) \
        if cids else np.zeros((0, 2))
    blocks = [(cids[k], np.flatnonzero(cluster == cids[k]))
              for k in leaf_order(cen)]
    for c in sorted({int(c) for c in cluster if c < 0}, reverse=True):
        m = np.flatnonzero(cluster == c)
        if m.size:
            blocks.append((c, m))

    order, starts = [], []
    for cid, members in blocks:
        starts.append((cid, len(order), len(members)))
        if ingraph[members].all():
            order.extend(int(members[k]) for k in leaf_order(xy[members]))
        else:
            order.extend(int(m) for m in members)
    return np.array(order, np.int64), starts


def pack_windows(seqs, b0, t_lo, t_hi):
    """Each read's bases at t = b0 - position over [t_lo, t_hi], t
    ascending, in one byte string; base t of read k is at
    off[k] + t - tmin[k]."""
    blob = bytearray()
    n = len(seqs)
    tmin = np.zeros(n, np.int64)
    off = np.zeros(n, np.int64)
    length = np.zeros(n, np.int64)
    for k, s in enumerate(seqs):
        if not s:
            off[k], tmin[k] = len(blob), t_lo
            continue
        lo = max(t_lo, b0[k] - (len(s) - 1))
        hi = min(t_hi, b0[k])
        off[k], tmin[k] = len(blob), lo
        if hi < lo:
            continue
        blob += s[b0[k] - hi:b0[k] - lo + 1][::-1].encode()
        length[k] = hi - lo + 1
    return bytes(blob), tmin, off, length


# numpy dtype -> page dtype
_DT = {"int8": "i1", "uint8": "u1", "int16": "i2", "int32": "i4",
       "float32": "f4"}


class Blob:
    """Arrays and text in one buffer, with a manifest for the page."""
    def __init__(self):
        self.buf = bytearray()
        self.man = []

    def add(self, name, arr, dtype=None):
        a = np.ascontiguousarray(arr, dtype or arr.dtype)
        if a.dtype.name not in _DT:
            raise ValueError(f"{name}: {a.dtype} is not a page dtype")
        self.man.append({"name": name, "dt": _DT[a.dtype.name],
                         "off": len(self.buf), "n": int(a.size)})
        self.buf += a.tobytes()
        return self

    def add_text(self, name, lines):
        b = "\n".join(lines).encode()
        self.man.append({"name": name, "dt": "txt",
                         "off": len(self.buf), "n": len(b)})
        self.buf += b
        return self


def gz_b64(data):
    """gzip (reproducibly, mtime 0) then base64."""
    return base64.b64encode(
        gzip.compress(bytes(data), 9, mtime=0)).decode()


CSS = """
:root{--fg:#1b1b1b;--mut:#666;--dim:#999;--line:#e0e0e0;--bg:#fff;
      --sel:#2b6cb0;--warn:#c0392b;--panel:#fbfbfb}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",
     Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);
     display:flex;flex-direction:column;overflow:hidden}
header{padding:9px 14px 0;border-bottom:1px solid var(--line);flex:0 0 auto}
h1{font-size:14px;margin:0 0 8px;font-weight:600}
h1 small{font-weight:400;color:var(--mut);margin-left:8px;
         font:12px ui-monospace,Menlo,Consolas,monospace}
#tabs{display:flex;gap:2px;align-items:flex-end}
.tab{padding:5px 15px;border:1px solid var(--line);border-bottom:none;
     border-radius:5px 5px 0 0;background:var(--panel);cursor:pointer;
     color:var(--mut);font-size:12.5px;position:relative;top:1px}
.tab.on{background:var(--bg);color:var(--fg);font-weight:600}
#tabs .grow{flex:1}
#find{border:1px solid #d8d8d8;border-radius:4px;padding:3px 7px;width:210px;
      font:11.5px ui-monospace,Menlo,Consolas,monospace;margin-bottom:4px}
#panels{flex:1 1 auto;position:relative;min-height:0}
.panel{position:absolute;inset:0;display:flex;flex-direction:column}
.panel[hidden]{display:none}
.ctl{flex:0 0 auto;display:flex;gap:14px;align-items:center;flex-wrap:wrap;
     padding:6px 14px;border-bottom:1px solid var(--line);
     font-size:11.5px;color:var(--mut);background:var(--panel)}
.ctl label{display:inline-flex;gap:5px;align-items:center;cursor:pointer}
.ctl input[type=range]{width:110px;vertical-align:middle}
.ctl b{font:11.5px ui-monospace,Menlo,Consolas,monospace;color:var(--fg);
       min-width:3.2em;display:inline-block}
button{font:inherit;font-size:11.5px;padding:2px 9px;border:1px solid #d0d0d0;
       border-radius:4px;background:#fff;cursor:pointer;color:var(--fg)}
button:hover{background:#f2f2f2}
.ctl select{font:inherit;font-size:11.5px;padding:1px 4px;border-radius:4px;
            border:1px solid #d0d0d0;background:#fff;color:var(--fg)}
.cv{flex:1 1 auto;min-height:0;position:relative}
#mapCv{display:block;width:100%;height:100%;cursor:grab}
#mapCv.drag{cursor:grabbing}
#readsScroll,#statsScroll{position:absolute;inset:0;overflow-y:auto;
     overflow-x:hidden}
#readsCv,#statsCv{display:block;position:sticky;top:0;z-index:1;width:100%}
#tip{position:fixed;pointer-events:none;background:#111;color:#fff;
     padding:6px 9px;border-radius:4px;font:11px ui-monospace,Menlo,Consolas,
     monospace;line-height:1.5;white-space:pre;z-index:9;display:none;
     max-width:46ch}
#boot{position:fixed;left:20px;bottom:20px;color:var(--mut);font-size:12px}
.key{display:inline-flex;gap:4px;align-items:center}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
body.editing header{background:#fff8f0;box-shadow:inset 0 3px 0 #d9822b}
#editbar{display:flex;gap:10px;align-items:center;padding:5px 14px;
         background:#fff3e4;border-bottom:1px solid #f0d5b4;font-size:11.5px;
         color:#8a4b0d;flex:0 0 auto}
#editbar[hidden]{display:none}
#editbar b{font:11.5px ui-monospace,Menlo,Consolas,monospace}
#editbar .grow{flex:1}
#editbar .hint{color:#a97546}
#editmsg{color:#1f6f43;font-weight:600}
#newcl{border-color:#d9822b;color:#8a4b0d;background:#fff}
#newcl:hover{background:#ffeeda}
#dlcsv{border-color:#d9822b;color:#8a4b0d;background:#fff}
#dlcsv:hover{background:#ffeeda}
body.editing #mapCv,body.editing #readsCv{cursor:crosshair}
#ghost{position:fixed;pointer-events:none;z-index:10;
       background:#fff;border:1px solid #d9822b;border-radius:4px;
       padding:3px 7px;font:11px ui-monospace,Menlo,Consolas,monospace;
       box-shadow:0 2px 8px rgba(0,0,0,0.18);white-space:pre}
#ghost[hidden]{display:none}
#ghost .sw{margin-right:5px;vertical-align:-1px}
#ghost.ok{border-color:#2f855a;background:#f0fff4;color:#22543d}
#modal{position:fixed;inset:0;background:rgba(20,20,20,0.34);z-index:20;
       display:flex;align-items:center;justify-content:center}
#modal[hidden]{display:none}
#mcard{background:#fff;border-radius:8px;padding:18px 20px 15px;width:430px;
       box-shadow:0 10px 40px rgba(0,0,0,0.3);font-size:12.5px;line-height:1.5}
#mtitle{font-size:14px;font-weight:600;margin-bottom:8px}
#mbody{color:#444}
#mbtns{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
#myes{border-color:#2b6cb0;color:#1a4d80}
#myes.danger{border-color:#c0392b;color:#c0392b}
#cmenu{position:fixed;z-index:30;width:214px;max-height:360px;overflow:auto;
       background:#fff;border:1px solid #d0d0d0;border-radius:6px;
       box-shadow:0 6px 24px rgba(0,0,0,0.22);font-size:11.5px;padding:4px 0}
#cmenu[hidden]{display:none}
#cmenu .mh{padding:6px 10px;border-bottom:1px solid #eee;
           font:11px ui-monospace,Menlo,Consolas,monospace;color:#333}
#cmenu .mh span{color:#999}
#cmenu .ms{padding:6px 10px 3px;color:#999;font-size:10.5px;
           text-transform:uppercase;letter-spacing:0.04em}
#cmenu .mi{display:flex;gap:7px;align-items:center;padding:3px 10px;
           cursor:pointer}
#cmenu .mi:hover{background:#eef4fb}
#cmenu .mi.on{background:#f3f3f3}
#cmenu .mi.on .mn{font-weight:600}
#cmenu .mn{flex:1}
#cmenu .mx{color:#aaa;font-size:10.5px}
#mfilt{width:calc(100% - 20px);margin:2px 10px 4px;padding:3px 6px;
       border:1px solid #ddd;border-radius:4px;
       font:11px ui-monospace,Menlo,Consolas,monospace}
"""

# page script, in sections: core, map tab, reads tab, stats tab, controls,
# edit mode
JS = r"""
const DPR = Math.min(2, window.devicePixelRatio || 1);
const $ = s => document.querySelector(s);
// A: the unpacked arrays; SEQ: the packed bases; N reads, M edges
let A = null, SEQ = null;
let N = 0, M = 0;
// sel: the focused read; hov: the read under the cursor on the map
let sel = -1, hov = -1;

async function gunzip(b64){
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) u8[i] = bin.charCodeAt(i);
  const s = new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}
// split the buffer into typed arrays by the manifest
function unpack(bytes, man){
  const out = {}, ab = bytes.buffer, b0 = bytes.byteOffset;
  for(const m of man){
    const a = b0 + m.off;
    if(m.dt === 'txt')
      out[m.name] = new TextDecoder().decode(
        bytes.subarray(m.off, m.off + m.n)).split('\n');
    else if(m.dt === 'i4') out[m.name] = new Int32Array(ab.slice(a, a + 4*m.n));
    else if(m.dt === 'f4') out[m.name] = new Float32Array(ab.slice(a, a+4*m.n));
    else out[m.name] = new Uint8Array(ab.slice(a, a + m.n));
  }
  return out;
}

// cluster colour by golden-angle hue; grey when unclustered
function ccol(c, l){ return c < 0 ? 'hsl(210,6%,72%)'
                     : 'hsl(' + ((c*137.508)%360).toFixed(1) + ',62%,' +
                       (l||48) + '%)'; }
// base colour by byte
const LR = new Uint8Array(256).fill(208), LG = new Uint8Array(256).fill(208),
      LB = new Uint8Array(256).fill(208);
for(const [b,c] of Object.entries(D.base))
  for(const ch of [b, b.toLowerCase()]){
    const k = ch.charCodeAt(0); LR[k]=c[0]; LG[k]=c[1]; LB[k]=c[2];
  }

// cluster codes: >= 0 cluster, -1 unclustered, -2 - k for D.reasons[k]
const UNCL = -1;
const RNAME = c => (c >= 0 || c === UNCL) ? '' : (D.reasons[-2 - c] || '');
const CNAME = c => c >= 0 ? String(c)
                 : (RNAME(c) ? 'unclustered:' + RNAME(c) : 'unclustered');
const CLABEL = c => c >= 0 ? 'cluster ' + c : CNAME(c);
const RWHY = c => D.reasonText[RNAME(c)] || '';
// a CSV cluster label as a code; NaN if unreadable
function codeOf(raw){
  if(raw.slice(0, 11) === 'unclustered'){
    const w = raw.slice(11).replace(/^:/, '');
    const k = w ? D.reasons.indexOf(w) : -1;
    return k < 0 ? UNCL : -2 - k;
  }
  const c = parseInt(raw, 10);
  return Number.isFinite(c) ? c : NaN;
}

let tab = 'map';
function show(t){
  tab = t;
  for(const el of document.querySelectorAll('.tab'))
    el.classList.toggle('on', el.dataset.tab === t);
  $('#mapPanel').hidden   = t !== 'map';
  $('#readsPanel').hidden = t !== 'reads';
  $('#statsPanel').hidden = t !== 'stats';
  if(t === 'map'){ sizeMap(); drawMap(); }
  else if(t === 'stats'){
    sizeStats(); if(sel >= 0) scrollToStats(sel); drawStats();
  }
  else { sizeReads(); if(sel >= 0) scrollToRead(sel); drawReads(); }
}

// the selection as a mask; sel is the read added last
let selMask = null, selN = 0;
const isSel = i => i >= 0 && selMask !== null && selMask[i] === 1;
function selected(){
  const out = [];
  if(selMask) for(let i = 0; i < N; i++) if(selMask[i]) out.push(i);
  return out;
}
// select list, or with add, extend the selection by it
function selPut(list, add){
  if(!selMask) selMask = new Uint8Array(N);
  if(!add){ selMask.fill(0); selN = 0; sel = -1; }
  for(const i of list){
    if(i < 0 || selMask[i]) continue;
    selMask[i] = 1; selN++; sel = i;
  }
  if(!selN) sel = -1;
}
function selDrop(i){
  if(!isSel(i)) return;
  selMask[i] = 0; selN--;
  if(sel === i){ sel = -1; for(let k = 0; k < N; k++) if(selMask[k]) sel = k; }
}
// redraw; scroll the open tab to sel unless the change came from it
function redrawSel(from){
  const i = sel;
  if(i >= 0 && tab === 'reads' && from !== 'reads') scrollToRead(i);
  if(i >= 0 && tab === 'stats' && from !== 'stats') scrollToStats(i);
  drawMap(); drawReads(); drawStats(); status();
}
// select one read; a read outside the graph opens the reads tab
function select(i, from){
  selPut(i < 0 ? [] : [i], false);
  anchorRead = i;
  if(i >= 0 && !A.ingraph[i] && tab !== 'reads'){
    show('reads'); status(); return;
  }
  redrawSel(from);
}
function selectMany(list, from, add){
  selPut(list, add);
  redrawSel(from);
}
function selectToggle(i, from){
  if(isSel(i)) selDrop(i); else selPut([i], true);
  redrawSel(from);
}
function status(){
  const el = $('#status');
  if(selN > 1){
    el.innerHTML = '<b>' + fmt(selN) + '</b> reads selected';
    return;
  }
  if(sel < 0){ el.textContent = 'no read selected'; return; }
  const c = cl(sel);
  el.innerHTML = '<b>' + A.ids[sel].slice(0,8) + '</b> → ' + CLABEL(c) +
    (A.ingraph[sel] ? '  strength ' + A.strength[sel].toFixed(1) +
                      '  degree ' + A.degree[sel] + SCORE(sel, 3)
                    : '  not in the graph') +
    '  array ' + bp(sel, 'b0') + '  sub ' + bp(sel, 'sub_bp');
}
const fmt = v => v.toLocaleString();
const SCORE = (i, dp) => (D.scoreName && A.score[i] >= 0)
  ? '  ' + D.scoreName + ' ' + A.score[i].toFixed(dp) : '';
const bp = (i, k) => A.hasb0[i] ? fmt(A[k][i]) : 'NA';
// tooltip text for read i
function detail(i){
  const c = cl(i);
  const why = RWHY(c);
  return A.ids[i] + '\n' + CLABEL(c) + '  (' + fmt(csize(c)) + ' reads)'
    + (why ? '\n' + why : '')
    + (A.ingraph[i]
       ? '\nstrength ' + A.strength[i].toFixed(2) + '   degree ' + A.degree[i]
         + SCORE(i, 4)
         + '\ninformative k-mers ' + fmt(A.nkmer[i])
       : '\nnot in the graph: no edges, no embedding, no box plot')
    + '\ntelomere b0 ' + bp(i, 'b0') + ' bp' +
    '\nsub ' + bp(i, 'sub_bp') + ' bp   read ' + fmt(A.read_bp[i]) + ' bp' +
    '   ' + (A.hasb0[i] ? (A.orient[i] ? 'tail' : 'head') : 'unoriented') +
    (A.qs[i] >= 0 ? '\nqs ' + A.qs[i].toFixed(1) : '');
}
"""

JS += r"""
// map tab: the t-SNE embedding
const mcv = $('#mapCv'), mx = mcv.getContext('2d');
let MW = 0, MH = 0;
// view centre in t-SNE units, and px per unit
let view = {cx:0, cy:0, s:1};
let byCluster = null, centroid = null, ebuck = null;

function sizeMap(){
  const r = mcv.parentElement.getBoundingClientRect();
  MW = Math.max(1, Math.floor(r.width)); MH = Math.max(1, Math.floor(r.height));
  mcv.width = MW*DPR; mcv.height = MH*DPR;
}
// fit the view to the reads in the graph
function fitMap(){
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  for(let i=0;i<N;i++){
    if(!A.ingraph[i]) continue;
    const x=A.x[i], y=A.y[i];
    if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y;
  }
  view.cx = (x0+x1)/2; view.cy = (y0+y1)/2;
  view.s = 0.92*Math.min(MW/Math.max(x1-x0,1e-9), MH/Math.max(y1-y0,1e-9));
}
// t-SNE -> screen
const SX = x => (x - view.cx)*view.s + MW/2;
const SY = y => MH/2 - (y - view.cy)*view.s;

// reads by cluster, cluster centroids, and edges in four weight buckets
function prep(){
  byCluster = new Map();
  for(let i=0;i<N;i++){
    if(!A.ingraph[i]) continue;
    const c = cl(i);
    if(!byCluster.has(c)) byCluster.set(c, []);
    byCluster.get(c).push(i);
  }
  centroid = [];
  for(const [c, ix] of byCluster){
    if(c < 0) continue;
    let sx=0, sy=0;
    for(const i of ix){ sx += A.x[i]; sy += A.y[i]; }
    centroid.push([c, sx/ix.length, sy/ix.length, ix.length]);
  }
  const NB = 4;
  ebuck = Array.from({length:NB}, () => []);
  for(let e=0;e<M;e++){
    const w = A.ew[e];
    ebuck[Math.min(NB-1, Math.max(0, Math.floor(w*NB)))].push(e);
  }
}

function drawMap(){
  if(!A || tab !== 'map') return;
  mx.setTransform(DPR,0,0,DPR,0,0);
  mx.clearRect(0,0,MW,MH);
  const floor = +$('#wfloor').value, r = +$('#psize').value;

  // edges, darker in heavier buckets
  if($('#edges').checked){
    mx.lineWidth = 1;
    for(let b=0;b<ebuck.length;b++){
      const a = 0.035 + 0.20*Math.pow((b+0.5)/ebuck.length, 2);
      mx.strokeStyle = 'rgba(70,80,95,' + a.toFixed(3) + ')';
      mx.beginPath();
      let drew = false;
      for(const e of ebuck[b]){
        if(A.ew[e] < floor) continue;
        const i = A.ei[e], j = A.ej[e];
        mx.moveTo(SX(A.x[i]), SY(A.y[i]));
        mx.lineTo(SX(A.x[j]), SY(A.y[j]));
        drew = true;
      }
      if(drew) mx.stroke();
    }
  }
  // reads, skipping those off screen
  for(const [c, ix] of byCluster){
    mx.fillStyle = ccol(c);
    mx.beginPath();
    for(const i of ix){
      const x = SX(A.x[i]), y = SY(A.y[i]);
      if(x < -8 || y < -8 || x > MW+8 || y > MH+8) continue;
      mx.moveTo(x+r, y); mx.arc(x, y, r, 0, 6.2832);
    }
    mx.fill();
  }
  // cluster ids at centroids
  if($('#labels').checked){
    mx.font = '600 10px ui-monospace,Menlo,Consolas,monospace';
    mx.textAlign = 'center'; mx.textBaseline = 'middle';
    for(const [c, x, y] of centroid){
      const sx = SX(x), sy = SY(y);
      if(sx < 0 || sy < 0 || sx > MW || sy > MH) continue;
      mx.lineWidth = 3; mx.strokeStyle = 'rgba(255,255,255,0.85)';
      mx.strokeText(String(c), sx, sy);
      mx.fillStyle = ccol(c, 30); mx.fillText(String(c), sx, sy);
    }
  }
  // rings around a multi-read selection
  if(selN > 1){
    mx.strokeStyle = '#2b6cb0'; mx.lineWidth = 1; mx.globalAlpha = 0.9;
    mx.beginPath();
    for(let i = 0; i < N; i++){
      if(!selMask[i] || !A.ingraph[i]) continue;
      const x = SX(A.x[i]), y = SY(A.y[i]);
      if(x < -8 || y < -8 || x > MW+8 || y > MH+8) continue;
      mx.moveTo(x + r + 2.5, y); mx.arc(x, y, r + 2.5, 0, 6.2832);
    }
    mx.stroke(); mx.globalAlpha = 1;
  }
  // hovered, selected and dragged reads: their edges and a ring
  const rings = [[hov,'#111',1.5],[sel,'#2b6cb0',2.5]];
  if(drag){
    rings.push([drag.i, '#d9822b', 2.5]);
    const over = drag.t ? drag.t.i : -1;
    if(over >= 0 && over !== drag.i) rings.push([over, '#2f855a', 3]);
  }
  for(const [i, col, w] of rings){
    if(i < 0 || !A.ingraph[i]) continue;
    mx.strokeStyle = col; mx.lineWidth = w === 2.5 ? 1.2 : 0.9;
    mx.globalAlpha = 0.55; mx.beginPath();
    for(let e=0;e<M;e++){
      const a = A.ei[e], b = A.ej[e];
      if(a !== i && b !== i) continue;
      const o = a === i ? b : a;
      mx.moveTo(SX(A.x[i]), SY(A.y[i]));
      mx.lineTo(SX(A.x[o]), SY(A.y[o]));
    }
    mx.stroke(); mx.globalAlpha = 1;
    mx.beginPath(); mx.arc(SX(A.x[i]), SY(A.y[i]), r+3.5, 0, 6.2832);
    mx.strokeStyle = col; mx.lineWidth = w; mx.stroke();
  }
  // shift-drag selection box
  if(band){
    const x = Math.min(band.x0, band.x1), y = Math.min(band.y0, band.y1);
    const w = Math.abs(band.x1 - band.x0), h = Math.abs(band.y1 - band.y0);
    mx.fillStyle = 'rgba(43,108,176,0.08)';
    mx.fillRect(x, y, w, h);
    mx.strokeStyle = '#2b6cb0'; mx.lineWidth = 1;
    mx.setLineDash([4, 3]);
    mx.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, w, h);
    mx.setLineDash([]);
  }
}

// nearest read in the graph within 12 px; preferSel favours the selection
function pick(px, py, preferSel){
  let best = -1, bd = 144, bsel = -1, bsd = 144;
  for(let i=0;i<N;i++){
    if(!A.ingraph[i]) continue;
    const dx = SX(A.x[i]) - px, dy = SY(A.y[i]) - py, d = dx*dx + dy*dy;
    if(d < bd){ bd = d; best = i; }
    if(preferSel && isSel(i) && d < bsd){ bsd = d; bsel = i; }
  }
  return bsel >= 0 ? bsel : best;
}

// band: shift-drag box; mdrag: pan
let band = null;
let mdrag = null;
// shift-drag selects a box; in edit mode dragging a read moves it; else pan
mcv.addEventListener('mousedown', e => {
  if(e.button === 0 && e.shiftKey){
    const r = mcv.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    band = {x0:px, y0:py, x1:px, y1:py};
    drawMap(); return;
  }
  if(edit && e.button === 0){
    const r = mcv.getBoundingClientRect();
    const i = pick(e.clientX - r.left, e.clientY - r.top, selN > 1);
    if(i >= 0){ dragStart(i, e); return; }
  }
  mdrag = {x:e.clientX, y:e.clientY, cx:view.cx, cy:view.cy, moved:false};
  mcv.classList.add('drag');
});
// reads in the graph inside a screen box
function inBand(b){
  const xl = Math.min(b.x0, b.x1), xh = Math.max(b.x0, b.x1);
  const yl = Math.min(b.y0, b.y1), yh = Math.max(b.y0, b.y1);
  const out = [];
  for(let i = 0; i < N; i++){
    if(!A.ingraph[i]) continue;
    const x = SX(A.x[i]), y = SY(A.y[i]);
    if(x >= xl && x <= xh && y >= yl && y <= yh) out.push(i);
  }
  return out;
}
window.addEventListener('mousemove', e => {
  if(drag && tab === 'map'){ dragMove(e); return; }
  const r = mcv.getBoundingClientRect();
  if(band){
    band.x1 = e.clientX - r.left; band.y1 = e.clientY - r.top;
    tipOff(); drawMap(); return;
  }
  if(mdrag){
    const dx = e.clientX - mdrag.x, dy = e.clientY - mdrag.y;
    if(Math.abs(dx) + Math.abs(dy) > 3) mdrag.moved = true;
    view.cx = mdrag.cx - dx/view.s; view.cy = mdrag.cy + dy/view.s;
    drawMap(); return;
  }
  if(tab !== 'map') return;
  const i = pick(e.clientX - r.left, e.clientY - r.top);
  if(i !== hov){ hov = i; drawMap(); }
  if(i >= 0) tipAt(e, detail(i)); else tipOff();
});
window.addEventListener('mouseup', e => {
  if(drag && tab === 'map'){ dragEnd(e); return; }
  if(band){
    const b = band; band = null;
    selectMany(inBand(b), 'map', true);
    return;
  }
  if(!mdrag) return;
  const moved = mdrag.moved;
  mdrag = null; mcv.classList.remove('drag');
  if(!moved && tab === 'map'){
    const r = mcv.getBoundingClientRect();
    select(pick(e.clientX - r.left, e.clientY - r.top), 'map');
  }
});
// wheel zooms about the cursor
mcv.addEventListener('wheel', e => {
  e.preventDefault();
  const r = mcv.getBoundingClientRect();
  const px = e.clientX - r.left, py = e.clientY - r.top;
  const wx = (px - MW/2)/view.s + view.cx, wy = view.cy - (py - MH/2)/view.s;
  const f = Math.exp(-e.deltaY*0.0016);
  view.s *= f;
  view.cx = wx - (px - MW/2)/view.s; view.cy = wy + (py - MH/2)/view.s;
  drawMap();
}, {passive:false});
"""

JS += r"""
// reads tab: each read's bases against t = b0 - position (t < 0 is
// subtelomere), one block of rows per cluster
const AXIS_H = 22, GID_W = 104, CHIP_W = 10, PADX = 6;
const READS_X = GID_W + CHIP_W + PADX;
const rcv = $('#readsCv'), rx = rcv.getContext('2d'),
      scroller = $('#readsScroll'), spacer = $('#readsSpacer');
let RW = 0, RH = 0, img = null;
// rowTop: px offset per row; blockOf: row -> block; ORDER: row -> read;
// RANK: read -> row
let rowTop = null, blockOf = null, totalPx = 0;
let BLOCKS = null, ORDER = null, RANK = null;
const EMPTY_ROWS = 3;
// visible t range
let xLo = 0, xHi = 1;

const tToX = t => READS_X + (t - xLo)/(xHi - xLo)*(RW - READS_X);
const xToT = x => xLo + (x - READS_X)/(RW - READS_X)*(xHi - xLo);

// blocks in display order; while editing: run clusters, new ones, then
// unclustered groups
function blockList(){
  if(!manual)
    return D.blocks.map(([cid, start, size]) => {
      const m = [];
      for(let r = start; r < start + size; r++) m.push(A.order[r]);
      return {cid, members:m};
    });
  const memb = new Map();
  for(let r = 0; r < N; r++){
    const i = A.order[r], c = manual[i];
    if(!memb.has(c)) memb.set(c, []);
    memb.get(c).push(i);
  }
  const out = [], seen = new Set();
  const put = c => { if(!seen.has(c)){ seen.add(c); out.push(c); } };
  for(const [c] of D.blocks) if(c >= 0) put(c);
  const made = [];
  for(const c of memb.keys()) if(c >= 0 && !seen.has(c)) made.push(c);
  for(const c of newClusters)
    if(c >= 0 && !seen.has(c) && made.indexOf(c) < 0) made.push(c);
  made.sort((a, b) => a - b).forEach(put);
  if(memb.has(UNCL) || newClusters.indexOf(UNCL) >= 0) put(UNCL);
  for(const [c] of D.blocks) if(c < 0) put(c);
  for(const c of memb.keys()) if(c < 0) put(c);
  return out.map(cid => ({cid, members: memb.get(cid) || []}));
}

// row and block offsets in px
function layout(){
  BLOCKS = blockList();
  ORDER = new Int32Array(N); RANK = new Int32Array(N);
  rowTop = new Float64Array(N + 1);
  blockOf = new Int32Array(N);
  let y = 0, r = 0;
  BLOCKS.forEach((blk, b) => {
    blk.b = b; blk.top = y; blk.start = r;
    for(const i of blk.members){
      ORDER[r] = i; RANK[i] = r; rowTop[r] = y; blockOf[r] = b;
      y += D.rowPx; r++;
    }
    if(!blk.members.length) y += EMPTY_ROWS*D.rowPx;
    blk.bot = y;
    y += D.gapRows*D.rowPx;
  });
  rowTop[N] = y;
  totalPx = y;
}
function blockAtY(y){
  if(!BLOCKS) return -1;
  for(const blk of BLOCKS) if(y >= blk.top && y < blk.bot) return blk.b;
  return -1;
}
// block whose trailing gap holds y, or -1
function gapAtY(y){
  if(!BLOCKS || y < 0) return -1;
  let g = -1;
  for(const blk of BLOCKS){
    if(y >= blk.bot) g = blk.b;
    else if(y >= blk.top) return -1;
    else break;
  }
  return g;
}
// t range: the window plus 6% each side
function winView(){
  const pad = (D.telo_bp + D.sub_bp)*0.06;
  xLo = -D.sub_bp - pad; xHi = D.telo_bp + pad;
}
// last row starting at or above y
function rowFloor(y){
  let lo = 0, hi = N - 1, r = -1;
  while(lo <= hi){
    const m = (lo + hi) >> 1;
    if(rowTop[m] <= y){ r = m; lo = m + 1; } else hi = m - 1;
  }
  return r;
}
function rowAtY(y){
  const r = rowFloor(y);
  return (r >= 0 && y < rowTop[r] + D.rowPx) ? r : -1;
}
function sizeReads(){
  RW = Math.max(1, scroller.clientWidth); RH = Math.max(1, scroller.clientHeight);
  rcv.style.height = RH + 'px';
  rcv.width = Math.floor(RW*DPR); rcv.height = Math.floor(RH*DPR);
  img = rx.createImageData(rcv.width, rcv.height);
  spacer.style.height = Math.max(0, totalPx - (RH - AXIS_H)) + 'px';
}
function scrollToRead(i){
  const r = RANK[i];
  scroller.scrollTop = Math.max(0, rowTop[r] - (RH - AXIS_H)/2);
}

// visible rows only; each pixel averages the bases it spans, faded
// outside the window
function drawReads(){
  if(!A || !SEQ || tab !== 'reads') return;
  const cw = rcv.width, ch = rcv.height;
  const seqX = Math.round(READS_X*DPR), seqW = cw - seqX;
  const st = scroller.scrollTop;
  const axis = Math.round(AXIS_H*DPR);
  const bpp = (xHi - xLo)/seqW;
  const flankLo = -D.sub_bp, flankHi = D.telo_bp;
  const d = img.data;
  d.fill(255);

  const rowH = Math.max(1, Math.round(D.rowPx*DPR));
  const line = new Uint8ClampedArray(seqW*4);
  const r0 = Math.max(0, rowFloor(st) - 1);
  const rEnd = Math.min(N - 1, Math.max(r0, rowFloor(st + RH)) + 1);
  for(let r = r0; r <= rEnd; r++){
    const yTop = Math.round((rowTop[r] - st)*DPR) + axis;
    if(yTop + rowH <= axis || yTop >= ch) continue;
    const i = ORDER[r], off = A.off[i], len = A.slen[i], t0 = A.tmin[i];
    for(let px = 0; px < seqW; px++){
      const ta = xLo + px*bpp, tb = ta + bpp;
      let m0 = Math.floor(ta) - t0, m1 = Math.ceil(tb) - t0;
      if(m1 <= m0) m1 = m0 + 1;
      if(m0 < 0) m0 = 0;
      if(m1 > len) m1 = len;
      const o = px*4;
      let R, G, B;
      if(m1 <= m0){ R = D.gap[0]; G = D.gap[1]; B = D.gap[2]; }
      else{
        R = 0; G = 0; B = 0;
        for(let m = m0; m < m1; m++){
          const c = SEQ[off + m]; R += LR[c]; G += LG[c]; B += LB[c];
        }
        const k = m1 - m0;
        R /= k; G /= k; B /= k;
        const tc = ta + bpp/2;
        if(tc < flankLo || tc > flankHi){
          R = 255 - (255 - R)*D.flank;
          G = 255 - (255 - G)*D.flank;
          B = 255 - (255 - B)*D.flank;
        }
      }
      line[o] = R; line[o+1] = G; line[o+2] = B; line[o+3] = 255;
    }
    for(let s = 0; s < rowH; s++){
      const y = yTop + s;
      if(y < axis || y >= ch) continue;
      d.set(line, (y*cw + seqX)*4);
    }
  }
  rx.putImageData(img, 0, 0);

  rx.setTransform(DPR, 0, 0, DPR, 0, 0);
  const X = tToX;

  rx.save();
  rx.beginPath(); rx.rect(READS_X, AXIS_H, RW - READS_X, RH - AXIS_H);
  rx.clip();
  // window edges dashed, t = 0 solid
  rx.setLineDash([4, 3]); rx.lineWidth = 1;
  rx.strokeStyle = '#2b6cb0';
  for(const t of [flankLo, flankHi]){
    const x = Math.round(X(t)) + 0.5;
    rx.beginPath(); rx.moveTo(x, AXIS_H); rx.lineTo(x, RH); rx.stroke();
  }
  rx.setLineDash([]); rx.strokeStyle = 'rgba(27,27,27,0.35)';
  const xz = Math.round(X(0)) + 0.5;
  rx.beginPath(); rx.moveTo(xz, AXIS_H); rx.lineTo(xz, RH); rx.stroke();
  rx.restore();

  // gutter: cluster chips, edit marks, block labels
  rx.save();
  rx.beginPath(); rx.rect(0, AXIS_H, READS_X - PADX, RH - AXIS_H); rx.clip();
  rx.fillStyle = '#fff'; rx.fillRect(0, AXIS_H, READS_X - PADX, RH - AXIS_H);
  for(let r = r0; r <= rEnd; r++){
    const y = rowTop[r] - st + AXIS_H;
    if(y + D.rowPx <= AXIS_H || y >= RH) continue;
    const i = ORDER[r];
    rx.fillStyle = ccol(cl(i));
    rx.fillRect(GID_W, y, CHIP_W - 2, Math.max(1, D.rowPx - 0.5));
    if(manual && manual[i] !== A.cluster[i]){
      rx.fillStyle = '#d9822b';
      rx.fillRect(GID_W - 3, y, 1.5, Math.max(1, D.rowPx - 0.5));
    }
    const over = drag && drag.t ? drag.t.i : -1;
    if(drag && (isHeld(i) || i === over)){
      rx.strokeStyle = (i === over && over !== drag.i) ? '#2f855a' : '#d9822b';
      rx.lineWidth = 1.5;
      rx.strokeRect(GID_W - 2.5, y - 0.5, CHIP_W + 3, Math.max(1, D.rowPx));
    }
  }
  rx.font = '600 10.5px ui-monospace,Menlo,Consolas,monospace';
  rx.textBaseline = 'top';
  for(const blk of BLOCKS){
    const top = blk.top - st + AXIS_H, bot = blk.bot - st + AXIS_H;
    if(bot < AXIS_H || top > RH) continue;
    const cid = blk.cid, size = blk.members.length;
    const tgt = !!(drag && drag.t && drag.t.blk === blk.b);
    const y = Math.min(Math.max(top, AXIS_H + 2), Math.max(AXIS_H + 2, bot - 13));
    rx.fillStyle = ccol(cid, 34);
    rx.fillText(CNAME(cid) + '  ' + size, 6, y);
    rx.fillStyle = tgt ? '#2f855a' : ccol(cid, 70);
    rx.fillRect(GID_W - 6, Math.max(top, AXIS_H), tgt ? 4 : 2,
                Math.min(bot, RH) - Math.max(top, AXIS_H) - 1);
    if(!size){
      rx.save();
      rx.setLineDash([3, 3]); rx.lineWidth = 1;
      rx.strokeStyle = tgt ? '#2f855a' : '#d9b892';
      rx.strokeRect(4.5, Math.max(top, AXIS_H) + 0.5, READS_X - PADX - 9,
                    Math.max(2, Math.min(bot, RH) - Math.max(top, AXIS_H) - 2));
      rx.restore();
    }
  }
  rx.restore();

  // drop line for a new cluster
  if(drag && drag.t && drag.t.gap >= 0){
    const g = BLOCKS[drag.t.gap];
    const y = Math.round(g.bot + D.gapRows*D.rowPx/2 - st) + AXIS_H + 0.5;
    if(y > AXIS_H && y < RH){
      rx.save();
      rx.strokeStyle = '#2f855a'; rx.lineWidth = 2; rx.setLineDash([5, 3]);
      rx.beginPath(); rx.moveTo(0, y); rx.lineTo(RW, y); rx.stroke();
      rx.setLineDash([]);
      rx.font = '600 10.5px ui-monospace,Menlo,Consolas,monospace';
      rx.textBaseline = 'middle';
      const label = 'new cluster';
      const w = rx.measureText(label).width + 8;
      rx.fillStyle = '#2f855a';
      rx.fillRect(4, y - 7, w, 14);
      rx.fillStyle = '#fff';
      rx.fillText(label, 8, y);
      rx.restore();
    }
  }

  // selection outlines
  if(selN){
    rx.strokeStyle = '#2b6cb0'; rx.lineWidth = 1;
    for(let r = r0; r <= rEnd; r++){
      if(!selMask[ORDER[r]]) continue;
      const y = rowTop[r] - st + AXIS_H;
      if(y + D.rowPx <= AXIS_H || y >= RH) continue;
      rx.strokeRect(0.5, Math.round(y) - 0.5, RW - 1,
                    Math.max(2, D.rowPx) + 1);
    }
  }

  // axis, ticks at 1/2/5 steps
  rx.fillStyle = '#fff'; rx.fillRect(0, 0, RW, AXIS_H);
  rx.fillStyle = 'rgba(43,108,176,0.09)';
  const bx0 = Math.max(READS_X, X(flankLo)), bx1 = Math.min(RW, X(flankHi));
  if(bx1 > bx0) rx.fillRect(bx0, 0, bx1 - bx0, AXIS_H);
  rx.strokeStyle = '#e0e0e0'; rx.lineWidth = 1;
  rx.beginPath(); rx.moveTo(0, AXIS_H - 0.5); rx.lineTo(RW, AXIS_H - 0.5);
  rx.stroke();
  rx.font = '10px ui-monospace,Menlo,Consolas,monospace';
  rx.textBaseline = 'middle'; rx.textAlign = 'center';
  const span = xHi - xLo, raw = span/((RW - READS_X)/110);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].find(m => m*mag >= raw)*mag;
  rx.fillStyle = '#888';
  for(let t = Math.ceil(xLo/step)*step; t <= xHi; t += step){
    const x = X(t);
    if(x < READS_X + 12) continue;
    rx.fillText(Math.abs(t) >= 1000 ? (t/1000) + ' kb' : t + '', x, AXIS_H/2);
    rx.strokeStyle = '#eee';
    rx.beginPath(); rx.moveTo(x, AXIS_H - 5); rx.lineTo(x, AXIS_H); rx.stroke();
  }
  rx.textAlign = 'left';
  rx.fillStyle = '#666';
  rx.fillText('subtelomere ←', READS_X + 4, AXIS_H/2);
  rx.textAlign = 'right';
  rx.fillText('→ telomere', RW - 4, AXIS_H/2);
  rx.textAlign = 'left';
}

scroller.addEventListener('scroll', () => drawReads(), {passive:true});
function readsHit(e){
  const b = rcv.getBoundingClientRect();
  const px = e.clientX - b.left, py = e.clientY - b.top;
  if(py < AXIS_H) return {row:-1, blk:-1, cy:-1, px, py};
  const cy = py - AXIS_H + scroller.scrollTop;
  return {row:rowAtY(cy), blk:blockAtY(cy), cy, px, py, t:xToT(px)};
}
// rdrag: x pan; anchorRead: start of a shift-click range
let rdrag = null;
let anchorRead = -1;
// shift-click selects a range, ctrl-click toggles a read; in edit mode
// dragging the gutter or a selection moves reads; else pan x
rcv.addEventListener('mousedown', e => {
  const h = readsHit(e);
  const i0 = h.row >= 0 ? ORDER[h.row] : -1;
  if(e.button === 0 && h.row >= 0 && (e.shiftKey || e.ctrlKey || e.metaKey)){
    const ar = anchorRead >= 0 ? RANK[anchorRead] : -1;
    if(e.shiftKey && ar >= 0){
      const lo = Math.min(ar, h.row), hi = Math.max(ar, h.row);
      const run = [];
      for(let r = lo; r <= hi; r++) run.push(ORDER[r]);
      if(ar > h.row) run.reverse();
      selectMany(run, 'reads', true);
    } else if(e.shiftKey){
      selectMany([i0], 'reads', true);
    } else {
      selectToggle(i0, 'reads');
    }
    anchorRead = i0;
    return;
  }
  if(edit && e.button === 0 && i0 >= 0 &&
     (h.px < READS_X || (selN > 1 && isSel(i0)))){
    dragStart(i0, e); return;
  }
  rdrag = {x:e.clientX, lo:xLo, hi:xHi, moved:false, row:h.row,
           pan:h.px >= READS_X};
});
rcv.addEventListener('mousemove', e => {
  if(drag && tab === 'reads'){ dragMove(e); return; }
  if(rdrag && rdrag.pan){
    const dx = e.clientX - rdrag.x;
    if(Math.abs(dx) > 3) rdrag.moved = true;
    const dt = dx/(RW - READS_X)*(rdrag.hi - rdrag.lo);
    xLo = rdrag.lo - dt; xHi = rdrag.hi - dt;
    drawReads(); return;
  }
  const h = readsHit(e);
  if(h.row < 0){ tipOff(); return; }
  const i = ORDER[h.row];
  let s = detail(i);
  if(h.px >= READS_X){
    const m = Math.round(h.t) - A.tmin[i];
    const base = (m >= 0 && m < A.slen[i])
      ? String.fromCharCode(SEQ[A.off[i] + m]) : '–';
    s += '\nt ' + fmt(Math.round(h.t)) + ' bp   ' + base +
         (h.t < -D.sub_bp || h.t > D.telo_bp ? '   (outside the window)' : '');
  }
  tipAt(e, s);
});
window.addEventListener('mouseup', e => {
  if(drag && tab === 'reads'){ dragEnd(e); return; }
  if(!rdrag) return;
  const {moved, row} = rdrag; rdrag = null;
  if(!moved && row >= 0 && tab === 'reads') select(ORDER[row], 'reads');
});
rcv.addEventListener('wheel', e => {
  if(!e.shiftKey) return;
  e.preventDefault();
  const b = rcv.getBoundingClientRect();
  const px = Math.max(READS_X, e.clientX - b.left);
  zoomX(Math.exp(e.deltaY*0.0015), xToT(px));
}, {passive:false});
// zoom x by f about t, to no less than 40 bp
function zoomX(f, about){
  const lo = about - (about - xLo)*f, hi = about + (xHi - about)*f;
  if(hi - lo < 40) return;
  xLo = lo; xHi = hi;
  drawReads();
}
"""

JS += r"""
// stats tab: per-cluster box plots of one metric over the reads in the graph
const S_AXIS = 34, S_GID = 138, S_ROW = 28, S_PAD = 9, S_RPAD = 12;
const SM = [{k:'read_bp', t:'read length'},
            {k:'b0',      t:'telomere length (b0)'}];
const scv = $('#statsCv'), sx = scv.getContext('2d'),
      sscroll = $('#statsScroll'), sspacer = $('#statsSpacer');
let SW = 0, SH = 0, sTotal = 0, smet = 0, shov = -1;
let SROWS = null, SDOM = null;

// box colour; grey for unclustered groups
const bcol = (c, l) => c < 0 ? 'hsl(210,6%,' + l + '%)' : ccol(c, l);

// quantile p of sorted v, interpolated
function quant(v, p){
  if(!v.length) return NaN;
  const h = (v.length - 1)*p, lo = Math.floor(h), hi = Math.ceil(h);
  return v[lo] + (v[hi] - v[lo])*(h - lo);
}
// box summary of sorted v; whiskers at the last value within 1.5 IQR
function boxOf(v){
  if(!v.length) return {n:0, min:0, max:0, q1:0, med:0, q3:0, wl:0, wh:0};
  const q1 = quant(v, 0.25), med = quant(v, 0.5), q3 = quant(v, 0.75);
  const iqr = q3 - q1, fLo = q1 - 1.5*iqr, fHi = q3 + 1.5*iqr;
  let wl = v[0], wh = v[v.length - 1];
  for(let i = 0; i < v.length; i++) if(v[i] >= fLo){ wl = v[i]; break; }
  for(let i = v.length - 1; i >= 0; i--) if(v[i] <= fHi){ wh = v[i]; break; }
  return {n:v.length, min:v[0], max:v[v.length - 1], q1, med, q3, wl, wh};
}
// SROWS: one row per block with reads in the graph; SROWI: block -> row;
// SDOM: each metric's range and median over the graph
let SROWI = null;
function statsPrep(){
  SROWI = new Int32Array(BLOCKS.length).fill(-1);
  SROWS = [];
  BLOCKS.forEach((blk, b) => {
    const idx = Int32Array.from(blk.members.filter(i => A.ingraph[i]));
    if(!idx.length) return;
    SROWI[b] = SROWS.length;
    const bx = SM.map(m => {
      const v = Array.from(idx, i => A[m.k][i]).sort((p, q) => p - q);
      return boxOf(v);
    });
    SROWS.push({cid: blk.cid, size: idx.length, idx, b: bx});
  });
  SDOM = SM.map(m => {
    const v = [];
    for(let i = 0; i < N; i++) if(A.ingraph[i]) v.push(A[m.k][i]);
    v.sort((p, q) => p - q);
    return {lo: v[0], hi: v[v.length - 1], med: quant(v, 0.5)};
  });
  sTotal = SROWS.length*S_ROW;
  sspacer.style.height = Math.max(0, sTotal - (SH - S_AXIS)) + 'px';
  sstat();
}
function sstat(){
  smet = +($('#smetric').value || 0);
  $('#sstatus').innerHTML =
    '<b>' + SM[smet].t + '</b>  ·  sample median ' +
    fmt(Math.round(SDOM[smet].med)) + ' bp  ·  ' +
    fmt(SDOM[smet].lo) + '–' + fmt(SDOM[smet].hi) + ' bp over ' + D.nGraph +
    ' reads in the graph' +
    (N > D.nGraph ? '  ·  ' + fmt(N - D.nGraph) + ' dropped before it, on '
                    + 'the reads tab only' : '');
}

function spanel(){ return {x0: S_GID, w: SW - S_GID - S_RPAD}; }
function sX(v){
  const d = SDOM[smet], g = spanel();
  const f = (v - d.lo)/(d.hi - d.lo);
  return g.x0 + S_PAD + Math.max(0, Math.min(1, f))*(g.w - 2*S_PAD);
}
function sticks(){
  const d = SDOM[smet], g = spanel(), out = [];
  const raw = (d.hi - d.lo)/Math.max(1, (g.w - 2*S_PAD)/70);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = ([1, 2, 5, 10].find(m => m*mag >= raw) || 10)*mag;
  for(let v = Math.ceil(d.lo/step)*step; v <= d.hi; v += step) out.push(v);
  return out;
}
function kb(v){
  if(v < 1000) return String(Math.round(v));
  const k = v/1000;
  return (k < 10 ? k.toFixed(1) : Math.round(k)) + ' kb';
}
// fixed jitter in [-1, 1) per read
function jit(i){
  let h = Math.imul(i + 1, 2654435761) >>> 0;
  h ^= h >>> 15; h = Math.imul(h, 2246822519) >>> 0; h ^= h >>> 13;
  return (h >>> 8)/8388608 - 1;
}

function sizeStats(){
  SW = Math.max(1, sscroll.clientWidth); SH = Math.max(1, sscroll.clientHeight);
  scv.style.height = SH + 'px';
  scv.width = Math.floor(SW*DPR); scv.height = Math.floor(SH*DPR);
  sspacer.style.height = Math.max(0, sTotal - (SH - S_AXIS)) + 'px';
}
// stats row of read i, or -1
function rowOf(i){
  if(!blockOf || !SROWI) return -1;
  return SROWI[blockOf[RANK[i]]];
}
function scrollToStats(i){
  if(!SROWS || !blockOf) return;
  const q = rowOf(i);
  if(q < 0) return;
  sscroll.scrollTop = Math.max(0, q*S_ROW - (SH - S_AXIS)/2);
}

function drawStats(){
  if(!A || !SROWS || tab !== 'stats') return;
  sx.setTransform(DPR, 0, 0, DPR, 0, 0);
  sx.clearRect(0, 0, SW, SH);
  sx.fillStyle = '#fff'; sx.fillRect(0, 0, SW, SH);
  const st = sscroll.scrollTop, dots = $('#sdots').checked;
  const col = A[SM[smet].k];
  const q0 = Math.max(0, Math.floor(st/S_ROW) - 1);
  const q1 = Math.min(SROWS.length - 1, Math.ceil((st + SH)/S_ROW));
  const half = S_ROW*0.5 - 5, jhalf = S_ROW*0.5 - 9;
  const selRow = sel >= 0 ? rowOf(sel) : -1;

  sx.save();
  sx.beginPath(); sx.rect(0, S_AXIS, SW, SH - S_AXIS); sx.clip();

  // sample median
  sx.strokeStyle = 'rgba(27,27,27,0.16)'; sx.lineWidth = 1;
  sx.setLineDash([3, 3]);
  const xmed = Math.round(sX(SDOM[smet].med)) + 0.5;
  sx.beginPath(); sx.moveTo(xmed, S_AXIS); sx.lineTo(xmed, SH); sx.stroke();
  sx.setLineDash([]);

  // rows: whiskers, box, median, then dots
  for(let q = q0; q <= q1; q++){
    const row = SROWS[q];
    const y = q*S_ROW - st + S_AXIS, mid = y + S_ROW/2;
    if(y + S_ROW <= S_AXIS || y >= SH) continue;
    if(q & 1){
      sx.fillStyle = 'rgba(27,27,27,0.028)';
      sx.fillRect(S_GID - 8, y, SW - S_GID + 8, S_ROW);
    }
    if(q === selRow){
      sx.fillStyle = 'rgba(43,108,176,0.07)';
      sx.fillRect(S_GID - 8, y, SW - S_GID + 8, S_ROW);
    }
    const b = row.b[smet];
    const xl = sX(b.wl), xh = sX(b.wh), x1 = sX(b.q1), x3 = sX(b.q3);
    sx.strokeStyle = bcol(row.cid, 66); sx.lineWidth = 1;
    sx.beginPath(); sx.moveTo(xl, mid); sx.lineTo(xh, mid); sx.stroke();
    for(const x of [xl, xh]){
      const xr = Math.round(x) + 0.5;
      sx.beginPath(); sx.moveTo(xr, mid - 4.5); sx.lineTo(xr, mid + 4.5);
      sx.stroke();
    }
    sx.fillStyle = bcol(row.cid, 89);
    sx.fillRect(x1, mid - half*0.66, Math.max(1, x3 - x1), half*1.32);
    sx.strokeStyle = bcol(row.cid, 50);
    sx.strokeRect(Math.round(x1) + 0.5, Math.round(mid - half*0.66) + 0.5,
                  Math.max(1, Math.round(x3 - x1)), Math.round(half*1.32));
    sx.strokeStyle = bcol(row.cid, 30); sx.lineWidth = 2;
    const xm = Math.round(sX(b.med)) + 0.5;
    sx.beginPath(); sx.moveTo(xm, mid - half*0.72); sx.lineTo(xm, mid + half*0.72);
    sx.stroke();
    sx.lineWidth = 1;
    if(dots){
      sx.fillStyle = bcol(row.cid, 40); sx.globalAlpha = 0.5;
      for(let z = 0; z < row.idx.length; z++){
        const i = row.idx[z];
        sx.beginPath();
        sx.arc(sX(col[i]), mid + jit(i)*jhalf, 1.7, 0, 6.2831853);
        sx.fill();
      }
      sx.globalAlpha = 1;
      if(selN > 1){
        sx.fillStyle = '#2b6cb0';
        for(let z = 0; z < row.idx.length; z++){
          const i = row.idx[z];
          if(!selMask[i]) continue;
          sx.beginPath();
          sx.arc(sX(col[i]), mid + jit(i)*jhalf, 2.2, 0, 6.2831853);
          sx.fill();
        }
      }
      for(const i of [shov, sel]){
        if(i < 0 || rowOf(i) !== q) continue;
        const cx = sX(col[i]), cy = mid + jit(i)*jhalf;
        sx.fillStyle = i === sel ? '#2b6cb0' : '#1b1b1b';
        sx.beginPath(); sx.arc(cx, cy, 2.6, 0, 6.2831853); sx.fill();
        sx.strokeStyle = '#fff'; sx.lineWidth = 1;
        sx.beginPath(); sx.arc(cx, cy, 4.2, 0, 6.2831853); sx.stroke();
      }
    }
  }
  sx.restore();

  // gutter: cluster ids and sizes
  sx.save();
  sx.beginPath(); sx.rect(0, S_AXIS, S_GID - 10, SH - S_AXIS); sx.clip();
  sx.fillStyle = '#fff'; sx.fillRect(0, S_AXIS, S_GID - 10, SH - S_AXIS);
  sx.font = '600 10.5px ui-monospace,Menlo,Consolas,monospace';
  sx.textBaseline = 'middle';
  for(let q = q0; q <= q1; q++){
    const row = SROWS[q], mid = q*S_ROW - st + S_AXIS + S_ROW/2;
    if(mid < S_AXIS - S_ROW || mid > SH + S_ROW) continue;
    sx.fillStyle = bcol(row.cid, 62);
    sx.fillRect(4, mid - 5, 8, 10);
    sx.textAlign = 'left';
    sx.fillStyle = bcol(row.cid, 32);
    sx.fillText(CNAME(row.cid), 17, mid);
    sx.textAlign = 'right';
    sx.fillStyle = '#aaa';
    sx.fillText(row.size + '', S_GID - 14, mid);
  }
  sx.textAlign = 'left';
  sx.restore();

  // axis
  sx.fillStyle = '#fff'; sx.fillRect(0, 0, SW, S_AXIS);
  sx.strokeStyle = '#e0e0e0'; sx.lineWidth = 1;
  sx.beginPath(); sx.moveTo(0, S_AXIS - 0.5); sx.lineTo(SW, S_AXIS - 0.5);
  sx.stroke();
  const g = spanel();
  sx.font = '600 11px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,' +
            'Helvetica,Arial,sans-serif';
  sx.fillStyle = '#444'; sx.textAlign = 'left'; sx.textBaseline = 'top';
  sx.fillText(SM[smet].t + '  ·  bp', g.x0 + S_PAD, 3);
  sx.font = '10px ui-monospace,Menlo,Consolas,monospace';
  sx.textAlign = 'center'; sx.textBaseline = 'middle';
  for(const v of sticks()){
    const x = sX(v);
    sx.fillStyle = '#888'; sx.fillText(kb(v), x, S_AXIS - 10);
    sx.strokeStyle = '#eee';
    sx.beginPath(); sx.moveTo(x, S_AXIS - 4); sx.lineTo(x, S_AXIS); sx.stroke();
  }
  sx.textAlign = 'left'; sx.textBaseline = 'alphabetic';
}

// row and nearest dot under the cursor
function statsHit(e){
  if(!SROWS) return null;
  const b = scv.getBoundingClientRect();
  const px = e.clientX - b.left, py = e.clientY - b.top;
  if(py < S_AXIS) return null;
  const q = Math.floor((py - S_AXIS + sscroll.scrollTop)/S_ROW);
  if(q < 0 || q >= SROWS.length) return null;
  const row = SROWS[q], col = A[SM[smet].k];
  const mid = q*S_ROW - sscroll.scrollTop + S_AXIS + S_ROW/2;
  const jhalf = S_ROW*0.5 - 9, g = spanel();
  let best = -1, bd = 30;
  if(px >= g.x0 - 2)
    for(let z = 0; z < row.idx.length; z++){
      const i = row.idx[z];
      const dx = sX(col[i]) - px, dy = mid + jit(i)*jhalf - py;
      const d = dx*dx + dy*dy;
      if(d < bd){ bd = d; best = i; }
    }
  return {row, q, i:best, px, py};
}
function sdetail(row){
  let s = CLABEL(row.cid) + '   ' + row.size + ' reads';
  for(let p = 0; p < 2; p++){
    const b = row.b[p], r = v => fmt(Math.round(v));
    s += '\n' + SM[p].t + (p === smet ? '   (drawn)' : '') +
         '\n  median ' + r(b.med) + '   IQR ' + r(b.q1) + '–' + r(b.q3) +
         '\n  range ' + r(b.min) + '–' + r(b.max);
  }
  return s;
}
sscroll.addEventListener('scroll', () => drawStats(), {passive:true});
scv.addEventListener('mousemove', e => {
  const h = statsHit(e);
  if(!h){ tipOff(); if(shov >= 0){ shov = -1; drawStats(); } return; }
  tipAt(e, h.i >= 0 ? detail(h.i) : sdetail(h.row));
  if(h.i !== shov){ shov = h.i; drawStats(); }
});
scv.addEventListener('mouseleave', () => {
  tipOff(); if(shov >= 0){ shov = -1; drawStats(); }
});
scv.addEventListener('click', e => {
  const h = statsHit(e);
  if(h && h.i >= 0) select(h.i, 'stats');
});
"""
JS += r"""
// tooltip, controls, keys and boot
const tip = $('#tip');
function tipAt(e, text){
  tip.textContent = text;
  tip.style.display = 'block';
  tip.style.left = Math.min(window.innerWidth - 380, e.clientX + 14) + 'px';
  tip.style.top  = Math.min(window.innerHeight - 130, e.clientY + 16) + 'px';
}
function tipOff(){ tip.style.display = 'none'; }
window.addEventListener('mouseleave', tipOff);

for(const el of document.querySelectorAll('.tab'))
  el.addEventListener('click', () => show(el.dataset.tab));
for(const id of ['edges','labels','wfloor','psize'])
  $('#' + id).addEventListener('input', () => {
    $('#wfloorv').textContent = (+$('#wfloor').value).toFixed(2);
    drawMap();
  });
$('#fitmap').addEventListener('click', () => { fitMap(); drawMap(); });
$('#zin').addEventListener('click', () => zoomX(0.6, (xLo + xHi)/2));
$('#zout').addEventListener('click', () => zoomX(1/0.6, (xLo + xHi)/2));
$('#zfit').addEventListener('click',
  () => { xLo = D.tLo; xHi = D.tHi; drawReads(); });
$('#zwin').addEventListener('click', () => { winView(); drawReads(); });
$('#smetric').addEventListener('change', () => { sstat(); drawStats(); });
$('#sdots').addEventListener('change', () => drawStats());
// find: the first id starting with the query, else the first containing it
$('#find').addEventListener('keydown', e => {
  if(e.key !== 'Enter') return;
  const q = $('#find').value.trim().toLowerCase();
  if(!q) return;
  let hit = -1;
  for(let i=0;i<N;i++) if(A.ids[i].toLowerCase().startsWith(q)){ hit = i; break; }
  if(hit < 0)
    for(let i=0;i<N;i++) if(A.ids[i].toLowerCase().includes(q)){ hit = i; break; }
  if(hit >= 0) select(hit, 'find');
  else $('#status').textContent = 'no read id matches ' + q;
});
// keys: ctrl-shift-E edit mode, ctrl-Z undo, U and N while editing, 1-3
// tabs, Esc closes a menu or dialog, else clears the selection
window.addEventListener('keydown', e => {
  if(e.ctrlKey && e.shiftKey && (e.key === 'E' || e.key === 'e')){
    e.preventDefault(); toggleEdit(); return;
  }
  if(e.key === 'Escape'){
    if(!$('#cmenu').hidden){ closeMenu(); return; }
    if(!$('#modal').hidden){ askClose(); return; }
  }
  if(edit && (e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')){
    e.preventDefault(); undo(); return;
  }
  if(e.target.tagName === 'INPUT') return;
  if(edit && $('#modal').hidden && !e.ctrlKey && !e.metaKey && !e.altKey){
    if(e.key === 'u' || e.key === 'U'){
      e.preventDefault();
      if(selN) assignMany(selected(), UNCL, sel);
      else editNote('nothing selected -- U moves the selection to unclustered');
      return;
    }
    if(e.key === 'n' || e.key === 'N'){
      e.preventDefault(); newCluster(); return;
    }
  }
  if(e.key === '1') show('map');
  if(e.key === '2') show('reads');
  if(e.key === '3') show('stats');
  if(e.key === 'Escape'){
    select(-1, tab);
  }
});
let rt = null;
window.addEventListener('resize', () => {
  clearTimeout(rt);
  rt = setTimeout(() => {
    sizeMap(); sizeReads(); sizeStats();
    if(tab === 'map') drawMap();
    else if(tab === 'stats') drawStats();
    else drawReads();
  }, 80);
});

// boot: unpack the data, then lay out and draw
(async function(){
  const el = $('#boot');
  if(typeof DecompressionStream === 'undefined'){
    el.textContent = 'This browser has no DecompressionStream, so the page '
      + 'cannot unpack itself. Chrome 80+, Firefox 113+ or Safari 16.4+ will.';
    return;
  }
  try{
    A = unpack(await gunzip(ARR_B64), D.man);
    A.ids = A.idtext;
    SEQ = await gunzip(SEQ_B64);
  }catch(err){ el.textContent = 'Could not unpack the page: ' + err; return; }
  N = A.x.length; M = A.ew.length;
  selMask = new Uint8Array(N);
  el.remove();
  $('#panels').style.visibility = 'visible';
  prep(); layout(); winView(); statsPrep();
  sizeMap(); sizeReads(); sizeStats(); fitMap();
  $('#wfloorv').textContent = (+$('#wfloor').value).toFixed(2);
  status();
  show('map');
})();
"""



JS += r"""
// edit mode: manual reassignment, held in the page until downloaded as CSV
const RUN_CLIDS = D.blocks.map(b => b[0]).filter(c => c >= 0)
                          .sort((a, b) => a - b);
// manual: cluster per read while editing; undoStack: batches of [read, old]
let edit = false, manual = null, nEdits = 0, undoStack = [];
let drag = null;
let newClusters = [];

// the run's cluster ids, new ones, and any in manual
function clusterIds(){
  const s = new Set(RUN_CLIDS);
  for(const c of newClusters) if(c >= 0) s.add(c);
  if(manual) for(let i = 0; i < N; i++) if(manual[i] >= 0) s.add(manual[i]);
  return [...s].sort((a, b) => a - b);
}

function cl(i){ return manual ? manual[i] : A.cluster[i]; }

let sizeCache = null;
function csize(c){
  if(!manual) return D.size[c] || 0;
  if(!sizeCache){
    sizeCache = new Map();
    for(let i = 0; i < N; i++)
      sizeCache.set(manual[i], (sizeCache.get(manual[i]) || 0) + 1);
  }
  return sizeCache.get(c) || 0;
}
// rebuild after an edit, keeping anchor's row where it was on screen
function afterEdit(anchor){
  sizeCache = null;
  nEdits = countEdits();
  const held = (anchor >= 0 && RANK) ? rowTop[RANK[anchor]] - scroller.scrollTop
                                     : null;
  prep(); layout(); statsPrep(); sizeReads();
  if(held !== null)
    scroller.scrollTop = Math.max(0, Math.min(
      Math.max(0, totalPx - (RH - AXIS_H)), rowTop[RANK[anchor]] - held));
  editStatus();
  redraw();
}
function editOn(){
  edit = true;
  manual = Int32Array.from(A.cluster);
  undoStack = []; newClusters = [];
  document.body.classList.add('editing');
  $('#editbar').hidden = false;
  afterEdit();
}
function editOff(){
  edit = false; manual = null; undoStack = []; drag = null; newClusters = [];
  editNote('');
  document.body.classList.remove('editing');
  $('#editbar').hidden = true;
  $('#ghost').hidden = true;
  closeMenu();
  afterEdit();
}
// dropping onto any unclustered read means plain unclustered
const dropCluster = c => c < 0 ? UNCL : c;
// move reads to c as one undoable edit
function assignMany(list, c, anchor){
  if(!manual) return;
  const batch = [];
  for(const i of list)
    if(i >= 0 && manual[i] !== c){ batch.push([i, manual[i]]); manual[i] = c; }
  if(!batch.length) return;
  undoStack.push(batch);
  afterEdit(anchor === undefined ? batch[0][0] : anchor);
}
// a read in a multi-selection acts for the whole selection
const actOn = i => (isSel(i) && selN > 1) ? selected() : [i];
function undo(){
  const u = undoStack.pop();
  if(!u) return;
  for(const [i, c] of u) manual[i] = c;
  afterEdit(u[0][0]);
}
// reads whose cluster differs from the run's
function countEdits(){
  if(!manual) return 0;
  let n = 0;
  for(let i = 0; i < N; i++) if(manual[i] !== A.cluster[i]) n++;
  return n;
}
function newClusterId(){
  const ids = clusterIds();
  const c = (ids.length ? ids[ids.length - 1] : -1) + 1;
  newClusters.push(c);
  return c;
}
function clusterFrom(list, anchor){
  if(!list.length) return -1;
  const c = newClusterId();
  assignMany(list, c, anchor);
  const n = csize(c);
  editNote((n === 1 ? '1 read' : fmt(n) + ' reads') + ' moved into cluster '
           + c + ', new');
  return c;
}
// N: a new cluster from the selection, or an empty one to drop reads into
function newCluster(){
  if(!edit) return;
  const last = newClusters.length ? newClusters[newClusters.length - 1] : -1;
  if(!selN && last >= 0 && csize(last) === 0){
    gotoCluster(last);
    editNote('cluster ' + last + ' is still empty');
    return;
  }
  if(selN){ clusterFrom(selected(), sel); return; }
  const c = newClusterId();
  afterEdit();
  gotoCluster(c);
  editNote('cluster ' + c + ' is empty -- drop reads on its band');
}
function gotoCluster(c){
  if(tab !== 'reads') show('reads');
  const blk = BLOCKS.find(b => b.cid === c);
  if(blk) scroller.scrollTop = Math.max(0, Math.min(
    Math.max(0, totalPx - (RH - AXIS_H)), blk.top - (RH - AXIS_H)/3));
  drawReads();
}
let noteT = null;
function editNote(s){
  $('#editmsg').textContent = s;
  clearTimeout(noteT);
  if(s) noteT = setTimeout(() => { $('#editmsg').textContent = ''; }, 5000);
}
function editStatus(){
  $('#editn').textContent = nEdits === 1 ? '1 read moved'
                                         : fmt(nEdits) + ' reads moved';
}
function redraw(){ drawMap(); drawReads(); drawStats(); status(); }

// read_id, run cluster, manual cluster, changed
function csv(){
  const out = ['read_id,cluster,manual_cluster,changed'];
  for(let i = 0; i < N; i++){
    const a = A.cluster[i], b = manual ? manual[i] : a;
    out.push(A.ids[i] + ',' + CNAME(a) + ',' + CNAME(b) + ',' + (a === b ? 0 : 1));
  }
  return out.join('\n') + '\n';
}
function download(){
  const blob = new Blob([csv()], {type:'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = D.sample + '.browser.manual.csv';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

// confirm/alert dialog
let modalYes = null;
function ask(title, body, yes, onYes, danger){
  modalYes = onYes;
  $('#mno').hidden = false;
  $('#mtitle').textContent = title;
  $('#mbody').innerHTML = body;
  $('#myes').textContent = yes;
  $('#myes').className = danger ? 'danger' : '';
  $('#modal').hidden = false;
  $('#myes').focus();
}
function tell(title, body){
  ask(title, body, 'OK', null);
  $('#mno').hidden = true;
}
function askClose(){
  $('#modal').hidden = true; modalYes = null; $('#mno').hidden = false;
}
$('#myes').addEventListener('click', () => {
  const f = modalYes; askClose(); if(f) f();
});
$('#mno').addEventListener('click', askClose);
$('#modal').addEventListener('mousedown', e => {
  if(e.target === $('#modal')) askClose();
});

// entering asks first; so does leaving with edits
function toggleEdit(){
  if(!edit){
    ask('Enter edit mode?',
        'Drag a read onto another read to take that read&rsquo;s cluster, or '
        + 'onto a cluster&rsquo;s label in the left gutter to join that '
        + 'cluster. Drop reads in the white space BETWEEN two clusters and a '
        + 'new cluster is made to hold them. Drag a read that is part of a '
        + 'selection and the whole selection moves with it, as one edit that '
        + 'one ctrl-Z undoes.<br><br>'
        + '<b>N</b> makes a new cluster &mdash; the selection&rsquo;s, or an '
        + 'empty band on the reads tab to drop reads into. <b>U</b> moves the '
        + 'selection to <b>unclustered</b>. Right-click a read for '
        + '<b>move to&hellip;</b>.<br><br>'
        + 'On the reads tab the blocks are the clusters as they now stand, so '
        + 'a read you move is drawn in its new cluster from then on.<br><br>'
        + 'Reads dropped before the graph was built are on the reads tab '
        + 'and can be moved into a cluster like any other &mdash; that is '
        + 'how this page says a gate lost a read it should not have.<br><br>'
        + 'Nothing is written to the run. The edits live in this page until you '
        + '<b>Download CSV</b>, and leaving edit mode discards them.',
        'Enter edit mode', editOn);
    return;
  }
  if(nEdits){
    ask('Leave edit mode?',
        '<b>' + nEdits + '</b> edit' + (nEdits === 1 ? '' : 's') + ' '
        + (nEdits === 1 ? 'has' : 'have') + ' not been downloaded. Leaving '
        + 'discards ' + (nEdits === 1 ? 'it' : 'them') + ' &mdash; the CSV is '
        + 'the only record.',
        'Discard and leave', editOff, true);
    return;
  }
  editOff();
}
$('#newcl').addEventListener('click', newCluster);
$('#dlcsv').addEventListener('click', download);
$('#editx').addEventListener('click', toggleEdit);

// drag: the ghost follows the cursor; the target is picked per tab
function dragStart(i, e){
  const list = actOn(i);
  drag = {i, list, set:new Set(list), x:e.clientX, y:e.clientY, t:null,
          moved:false};
  const g = $('#ghost');
  g.innerHTML = ghostHtml();
  g.hidden = false;
  dragMove(e);
}
function ghostHtml(){
  const i = drag.i, t = drag.t;
  return '<span class="sw" style="background:' + ccol(cl(i)) + '"></span>'
       + (drag.list.length > 1 ? fmt(drag.list.length) + ' reads'
                               : A.ids[i].slice(0, 16))
       + (t && t.gap >= 0 ? '  \u2192 new cluster' : '');
}
const isHeld = i => !!(drag && drag.set.has(i));
function dragMove(e){
  if(!drag) return;
  if(Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3)
    drag.moved = true;
  const g = $('#ghost');
  g.style.left = (e.clientX + 12) + 'px';
  g.style.top  = (e.clientY + 12) + 'px';
  const key = t => t ? t.i + '/' + t.blk + '/' + t.gap : '-';
  const was = key(drag.t);
  drag.t = dropTarget(e);
  const t = drag.t;
  g.className = (t && t.i !== drag.i) ? 'ok' : '';
  if(key(t) !== was){
    g.innerHTML = ghostHtml();
    if(tab === 'map') drawMap(); else drawReads();
  }
}
// a read, a block, or the gap after a block (a new cluster); null if none
function dropTarget(e){
  if(tab === 'map'){
    const r = mcv.getBoundingClientRect();
    const i = pick(e.clientX - r.left, e.clientY - r.top);
    return i >= 0 ? {i, blk:-1, gap:-1, c:dropCluster(cl(i))} : null;
  }
  const h = readsHit(e);
  if(h.row >= 0 && h.px >= GID_W){
    const i = ORDER[h.row];
    return {i, blk:-1, gap:-1, c:dropCluster(cl(i))};
  }
  if(h.blk >= 0)
    return {i:-1, blk:h.blk, gap:-1, c:dropCluster(BLOCKS[h.blk].cid)};
  const g = gapAtY(h.cy);
  if(g >= 0) return {i:-1, blk:-1, gap:g, c:null};
  return null;
}
// gap -> new cluster, read or block -> its cluster, no movement -> select
function dragEnd(e){
  if(!drag) return;
  const {i, list, t, moved} = drag;
  drag = null;
  $('#ghost').hidden = true;
  if(moved && t && t.gap >= 0) clusterFrom(list, i);
  else if(moved && t && t.i !== i) assignMany(list, t.c, i);
  else if(!moved) select(i, tab);
  redraw();
}

// right-click menu: the clusters a read is attached to, then every cluster
let menuRead = -1;
// clusters of i's neighbours by summed edge weight, heaviest first
function neighbourClusters(i){
  const w = new Map();
  for(let e = 0; e < M; e++){
    let o = -1;
    if(A.ei[e] === i) o = A.ej[e];
    else if(A.ej[e] === i) o = A.ei[e];
    else continue;
    const c = cl(o);
    w.set(c, (w.get(c) || 0) + A.ew[e]);
  }
  return [...w.entries()].sort((a, b) => b[1] - a[1]);
}
function menuRow(c, i, note){
  const here = cl(i) === c;
  return '<div class="mi' + (here ? ' on' : '') + '" data-c="' + c + '">'
       + '<span class="sw" style="background:' + ccol(c) + '"></span>'
       + '<span class="mn">' + CLABEL(c)
       + '</span><span class="mx">' + (note || (c < 0 ? ''
       : fmt(csize(c)) + ' reads')) + '</span></div>';
}
function openMenu(i, cx, cy){
  if(!edit || i < 0) return;
  menuRead = i;
  const near = neighbourClusters(i);
  const many = actOn(i).length;
  let h = '<div class="mh">'
        + (many > 1 ? fmt(many) + ' reads selected' : A.ids[i].slice(0, 18))
        + '<br><span>now ' + CLABEL(cl(i)) + '</span></div>';
  if(near.length){
    h += '<div class="ms">attached to</div>';
    for(const [c, wsum] of near.slice(0, 6))
      h += menuRow(c, i, wsum.toFixed(2) + ' weight');
  }
  h += '<div class="ms">move to</div>'
     + '<input id="mfilt" placeholder="cluster id" spellcheck="false">'
     + '<div id="mlist">' + menuRow(UNCL, i) + allRows(i, '') + '</div>';
  const m = $('#cmenu');
  m.innerHTML = h;
  m.hidden = false;
  const w = 214, hh = Math.min(360, m.scrollHeight || 360);
  m.style.left = Math.min(cx, window.innerWidth - w - 8) + 'px';
  m.style.top  = Math.min(cy, window.innerHeight - hh - 8) + 'px';
  const f = $('#mfilt');
  if(f){
    f.addEventListener('input', () => {
      $('#mlist').innerHTML = menuRow(UNCL, menuRead) +
                              allRows(menuRead, f.value.trim());
    });
    f.focus();
  }
}
// a menu row per cluster whose id starts with q
function allRows(i, q){
  let h = '';
  for(const c of clusterIds()){
    if(q && !String(c).startsWith(q)) continue;
    h += menuRow(c, i);
  }
  return h;
}
function closeMenu(){ $('#cmenu').hidden = true; menuRead = -1; }
$('#cmenu').addEventListener('mousedown', e => {
  const row = e.target.closest ? e.target.closest('.mi') : null;
  if(!row) return;
  e.preventDefault();
  assignMany(actOn(menuRead), +row.dataset.c, menuRead);
  closeMenu();
});
window.addEventListener('mousedown', e => {
  if($('#cmenu').hidden) return;
  if(!$('#cmenu').contains(e.target)) closeMenu();
}, true);

// load read_id + manual_cluster (or cluster) over the run's assignment
function applyCsv(text){
  const lines = text.split(/\r?\n/).filter(s => s.length);
  if(!lines.length) return {err:'the file is empty'};
  const hdr = lines[0].split(',').map(s => s.trim().toLowerCase());
  const ci = hdr.indexOf('read_id');
  let cc = hdr.indexOf('manual_cluster');
  if(cc < 0) cc = hdr.indexOf('cluster');
  if(ci < 0 || cc < 0)
    return {err:'no read_id and manual_cluster columns in the header'};
  const pos = new Map();
  for(let i = 0; i < N; i++) pos.set(A.ids[i], i);
  const next = Int32Array.from(A.cluster);
  let applied = 0, unknown = 0, bad = 0, seen = 0;
  for(let r = 1; r < lines.length; r++){
    const f = lines[r].split(',');
    if(f.length <= Math.max(ci, cc)) continue;
    seen++;
    const i = pos.get(f[ci].trim());
    if(i === undefined){ unknown++; continue; }
    const c = codeOf(f[cc].trim());
    if(!Number.isFinite(c)){ bad++; continue; }
    next[i] = c;
    applied++;
  }
  manual = next;
  afterEdit();
  return {rows:seen, applied, unknown, bad, edits:nEdits};
}
$('#upcsv').addEventListener('change', e => {
  const f = e.target.files && e.target.files[0];
  if(!f) return;
  const fr = new FileReader();
  fr.onload = () => {
    const r = applyCsv(String(fr.result));
    tell(r.err ? 'That file could not be read' : 'Loaded ' + f.name,
        r.err ? r.err
              : '<b>' + fmt(r.applied) + '</b> of ' + fmt(r.rows)
                + ' rows matched a read on this page, leaving <b>' + r.edits
                + '</b> read' + (r.edits === 1 ? '' : 's')
                + ' assigned away from the run.'
                + (r.unknown ? '<br>' + fmt(r.unknown) + ' read ids are not on '
                   + 'this page and were skipped.' : '')
                + (r.bad ? '<br>' + fmt(r.bad) + ' rows had no readable '
                   + 'cluster.' : ''));
  };
  fr.readAsText(f);
  e.target.value = '';
});
$('#upbtn').addEventListener('click', () => $('#upcsv').click());

// right-click a read for the move menu while editing
for(const cv of [mcv, rcv])
  cv.addEventListener('contextmenu', e => {
    if(!edit) return;
    e.preventDefault();
    const r = cv === mcv ? mcv.getBoundingClientRect() : null;
    const i = cv === mcv
      ? pick(e.clientX - r.left, e.clientY - r.top, selN > 1)
      : (() => { const h = readsHit(e);
                 return h.row >= 0 ? ORDER[h.row] : -1; })();
    if(i >= 0) openMenu(i, e.clientX, e.clientY);
  });
"""

# __NAME__ placeholders are filled by _page
PAGE = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>__CSS__</style>
<header>
  <h1>__SAMPLE__ <small>__HEAD__</small></h1>
  <div id="tabs">
    <div class="tab on" data-tab="map">map</div>
    <div class="tab" data-tab="reads">reads</div>
    <div class="tab" data-tab="stats">cluster stats</div>
    <span class="grow"></span>
    <input id="find" placeholder="read id, then Enter" spellcheck="false">
  </div>
</header>
  <div id="editbar" hidden>
    <b>EDIT MODE</b>
    <span id="editn">0 edits</span>
    <button id="newcl">+ cluster</button>
    <span class="hint">drag reads onto a read, onto a cluster's label, or
      into the space between two clusters for a new one · <b>N</b> new cluster
      · <b>U</b> unclustered · right-click for <b>move to…</b> · ctrl-Z
      undoes</span>
    <span id="editmsg"></span>
    <span class="grow"></span>
    <button id="upbtn">Load CSV…</button>
    <button id="dlcsv">Download CSV</button>
    <button id="editx">Exit</button>
    <input type="file" id="upcsv" accept=".csv,text/csv" hidden>
  </div>
<div id="panels" style="visibility:hidden">
  <div class="panel" id="mapPanel">
    <div class="ctl">
      <label><input type="checkbox" id="edges" checked> shared-k-mer edges</label>
      <label>weight ≥ <input type="range" id="wfloor" min="0" max="0.99"
             step="0.01" value="0"> <b id="wfloorv">0.00</b></label>
      <label>point <input type="range" id="psize" min="1" max="6" step="0.5"
             value="2.5"></label>
      <label><input type="checkbox" id="labels" checked> cluster ids</label>
      <button id="fitmap">fit</button>
      <span style="color:#999">· shift-drag selects a square</span>
      <span id="status"></span>
    </div>
    <div class="cv"><canvas id="mapCv"></canvas></div>
  </div>
  <div class="panel" id="readsPanel" hidden>
    <div class="ctl">
      <span>x: <button id="zout">−</button> <button id="zin">+</button>
        <button id="zwin">window</button> <button id="zfit">all</button>
        <span style="color:#999">· shift-wheel zooms, drag pans, wheel
        scrolls · shift-click selects a range, ctrl-click one
        read</span></span>
      <span class="key">__LEGEND__</span>
      <span style="color:#999">dashed: the window the graph was built from</span>
    </div>
    <div class="cv"><div id="readsScroll"><canvas id="readsCv"></canvas>
      <div id="readsSpacer"></div></div></div>
  </div>
  <div class="panel" id="statsPanel" hidden>
    <div class="ctl">
      <label>stat <select id="smetric">
        <option value="0">read length</option>
        <option value="1">telomere length (b0)</option></select></label>
      <label><input type="checkbox" id="sdots" checked> reads</label>
      <span style="color:#999">box: IQR, whiskers 1.5×IQR, dashed: the
        sample median</span>
      <span id="sstatus"></span>
    </div>
    <div class="cv"><div id="statsScroll"><canvas id="statsCv"></canvas>
      <div id="statsSpacer"></div></div></div>
  </div>
</div>
<div id="modal" hidden><div id="mcard">
  <div id="mtitle"></div>
  <div id="mbody"></div>
  <div id="mbtns"><button id="mno">Cancel</button><button id="myes"></button></div>
</div></div>
<div id="cmenu" hidden></div>
<div id="ghost" hidden></div>
<div id="tip"></div>
<div id="boot">unpacking __NREADS__ reads…</div>
<script>
const D = __D__;
const ARR_B64 = "__ARR__";
const SEQ_B64 = "__SEQ__";
__JS__
</script>
"""


def _swatch(rgb, label):
    return (f'<span class="sw" style="background:rgb{rgb}"></span>'
            f'<span style="margin-right:7px">{label}</span>')


# t-SNE defaults
PERPLEXITY = 5.0
N_ITER = 1000


def _page(sample, reads, edges, seqs, *, telo_bp, sub_bp, out_path,
          perplexity, seed, n_iter, row_px, flank):
    """Embed, lay out, pack and write the page; returns its path."""
    ids = reads["ids"]
    n = len(ids)
    if edges[3]:
        log(f"browser: {edges[3]:,} of {edges[3] + len(edges[2]):,} edges "
            f"touch a read that is not in the read table and were dropped -- "
            f"the edge list and the table may be from different runs")
    blank = sum(1 for q in seqs if not q)
    if blank:
        log(f"browser: {blank:,} reads have no sequence and draw a blank row")
    xy, _ = embed(reads, edges, perplexity=perplexity, seed=seed,
                  n_iter=n_iter)

    order, starts = block_order(reads["cluster"], xy, reads["ingraph"])
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    t_lo, t_hi = -(sub_bp + flank), telo_bp + flank
    blob, tmin, off, length = pack_windows(seqs, np.maximum(reads["b0"], 0),
                                           t_lo, t_hi)
    log(f"browser: {len(blob) / 1e6:.1f} Mbp packed over "
        f"t {t_lo:,} to {t_hi:,}")

    B = Blob()
    B.add("x", xy[:, 0], "float32").add("y", xy[:, 1], "float32")
    B.add("cluster", reads["cluster"], "int32")
    B.add("strength", reads["strength"], "float32")
    B.add("score", reads["score"], "float32")
    B.add("qs", reads["qs"], "float32")
    for c in ("degree", "nkmer", "b0", "sub_bp", "read_bp"):
        B.add(c, reads[c], "int32")
    B.add("orient", np.array([1 if o == "tail" else 0
                              for o in reads["orient"]]), "uint8")
    B.add("ingraph", reads["ingraph"].astype(np.uint8), "uint8")
    B.add("hasb0", reads["hasb0"].astype(np.uint8), "uint8")
    B.add("order", order, "int32").add("rank", rank, "int32")
    B.add("tmin", tmin, "int32").add("off", off, "int32")
    B.add("slen", length, "int32")
    B.add("ei", edges[0], "int32").add("ej", edges[1], "int32")
    B.add("ew", edges[2], "float32")
    B.add_text("idtext", ids)

    sizes = {int(c): int(s) for c, _, s in starts}
    n_clusters = sum(1 for c, _, _ in starts if c >= 0)
    n_graph = int(reads["ingraph"].sum())
    data = {
        "sample": sample, "man": B.man,
        "scoreName": reads.get("score_name", ""),
        "reasons": list(UNPLACED_REASONS),
        "reasonText": {k: REASON_TEXT[k] for k in UNPLACED_REASONS},
        "nGraph": n_graph,
        "tLo": int(t_lo), "tHi": int(t_hi),
        "rowPx": int(row_px), "gapRows": GAP_ROWS,
        "base": {b: list(c) for b, c in BASE_RGB.items()},
        "gap": list(GAP_RGB), "flank": FLANK_ALPHA,
        "sub_bp": sub_bp, "telo_bp": telo_bp,
        "blocks": [[int(c), int(s), int(m)] for c, s, m in starts],
        "size": sizes,
    }
    head = (f"{n:,} reads · {n_graph:,} in the graph · "
            f"{n_clusters:,} clusters")
    legend = "".join(_swatch(BASE_RGB[b], b) for b in "ACGT")

    page = (PAGE
            .replace("__TITLE__", f"{sample} t-SNE read browser")
            .replace("__CSS__", CSS)
            .replace("__SAMPLE__", html.escape(sample))
            .replace("__HEAD__", head)
            .replace("__LEGEND__", legend)
            .replace("__NREADS__", f"{n:,}")
            .replace("__D__", json.dumps(data, separators=(",", ":")))
            .replace("__ARR__", gz_b64(B.buf))
            .replace("__SEQ__", gz_b64(blob))
            .replace("__JS__", JS))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = f"{out_path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(page)
    os.replace(tmp, out_path)
    log(f"browser: {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


def build(outdir, *, sample=None, out_path=None, perplexity=PERPLEXITY,
          seed=0, n_iter=N_ITER, row_px=ROW_PX, flank=FLANK_BP):
    """Page for a finished run directory; needs its cache at outdir/cache."""
    sample, f = find_run(outdir, sample)
    reads = load_reads(f["reads"])
    ids = reads["ids"]
    pos = {r: k for k, r in enumerate(ids)}
    if len(pos) != len(ids):
        raise SystemExit(f"{f['reads']} has duplicate read ids")
    log(f"browser: {sample}, {len(ids):,} reads from "
        f"{os.path.basename(f['reads'])}")
    cfg = {}
    if f["json"]:
        with open(f["json"]) as fh:
            cfg = json.load(fh).get("params", {})
    seqs, _ = load_intake(f["intake"], f["seqs"], ids)
    return _page(sample, reads, load_edges(f["edges"], pos), seqs,
                 telo_bp=int(cfg.get("telo_bp", 2500)),
                 sub_bp=int(cfg.get("sub_bp", 1000)),
                 out_path=out_path or os.path.join(
                     os.path.abspath(outdir), f"{sample}.browser.html"),
                 perplexity=perplexity, seed=seed, n_iter=n_iter,
                 row_px=row_px, flank=flank)


def build_from_run(out_path, sample, rows, reads, dropped, G, *, telo_bp,
                   sub_bp, seed=0, perplexity=PERPLEXITY, n_iter=N_ITER,
                   row_px=ROW_PX, flank=FLANK_BP):
    """Page from the run in memory (trc --browser)."""
    table = _reads_table(rows)
    ids = table["ids"]
    pos = {r: k for k, r in enumerate(ids)}
    if len(pos) != len(ids):
        raise SystemExit("the read table has duplicate read ids")
    log(f"browser: {sample}, {len(ids):,} reads")
    seq = {r.read_id: r.seq for r in list(reads) + list(dropped)}
    edges = _edges(((reads[int(i)].read_id, reads[int(j)].read_id, float(w))
                    for (i, j), w in zip(G.pairs, G.weights)), pos)
    return _page(sample, table, edges, [seq.get(r, "") for r in ids],
                 telo_bp=telo_bp, sub_bp=sub_bp, out_path=out_path,
                 perplexity=perplexity, seed=seed, n_iter=n_iter,
                 row_px=row_px, flank=flank)


def main(argv=None):
    """Build a page for a run directory from the command line."""
    ap = argparse.ArgumentParser(
        description="A three-tab HTML browser over one trc run: the "
                    "read/k-mer graph as t-SNE, the same reads as sequence, "
                    "and the clusters as distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("outdir", help="a trc run directory (needs the run's "
                                   "--cache DIR beside its tables)")
    ap.add_argument("--sample", default=None,
                    help="which sample, if the directory holds more than one")
    ap.add_argument("-o", "--out", default=None, metavar="HTML",
                    help="the page; default is <sample>.browser.html in the "
                         "run directory")
    ap.add_argument("--perplexity", type=float, default=PERPLEXITY,
                    help="t-SNE perplexity.  Keep it well under the graph's "
                         "degree -- a read has at least --n-neighbors "
                         "neighbours and there is no more neighbourhood "
                         "than that to match")
    ap.add_argument("--iterations", type=int, default=N_ITER, dest="n_iter",
                    help="t-SNE gradient steps")
    ap.add_argument("--seed", type=int, default=0,
                    help="the embedding's seed; the same seed gives the same "
                         "picture")
    ap.add_argument("--row-px", type=int, default=ROW_PX, dest="row_px",
                    help="height of one read on the read tab")
    ap.add_argument("--flank", type=int, default=FLANK_BP,
                    help="bp drawn past each edge of the clustering window, "
                         "faded")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="no progress on stderr")
    a = ap.parse_args(argv)
    if a.quiet:
        from .util import verbose
        verbose(False)
    build(a.outdir, sample=a.sample, out_path=a.out, perplexity=a.perplexity,
          seed=a.seed, n_iter=a.n_iter, row_px=a.row_px, flank=a.flank)


if __name__ == "__main__":
    main()
