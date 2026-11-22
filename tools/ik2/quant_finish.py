#!/usr/bin/env python3
"""tools/ik2/quant_finish.py - the last step of a one-T4 quant of Inkling-2-Mini: rewrites a GGUF with the named
Q8_0 expert tensors as NVFP4 (64-value blocks of four 16-value sub-blocks, each with a ue4m3 scale, the scale chosen
by a search around max / 12 for the least squared error - llama.cpp's quantize_row_nvfp4_ref takes max / 12 as it
is) and a `.scale` tensor of ones beside each (saguaro-ik2 wants one beside an NVFP4 expert tensor).  Every other
tensor and the metadata are copied byte for byte.

    quant_finish.py --in A.gguf --out B.gguf --nvfp4 'blk\\.2[0-3]\\.ffn_down_exps\\.weight' [--imatrix I.gguf] [--jobs 16]
    quant_finish.py --in A.gguf --header-only --out B.gguf ...   (a sparse file with B's table: for --plan)
    quant_finish.py --check A.gguf     (one expert of the first match: error of max / 12 against the search)
"""
from __future__ import annotations

import argparse
import mmap
import os
import re
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import gguf_reader  # noqa: E402

NVFP4, Q8_0, F32 = 40, 8, 0
LEVELS = np.array([0, 1, 2, 3, 4, 6, 8, 12], np.float32)       # kvalues_mxfp4 (codes 8.. are the negatives)
EDGES = np.array([0.5, 1.5, 2.5, 3.5, 5, 7, 10], np.float32)


def ue4m3(x: int) -> float:
    if x == 0 or x == 0x7F:
        return 0.0
    e, m = (x >> 3) & 15, x & 7
    return (m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)) * 0.5


_codes = sorted((ue4m3(c), c) for c in range(1, 127))
SCALE_V = np.array([v for v, _ in _codes], np.float32)          # ascending, all > 0
SCALE_C = np.array([c for _, c in _codes], np.uint8)


def deq_q8(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)


def deq_nvfp4(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 36)
    tab = np.array([ue4m3(c) for c in range(256)], np.float32)
    kv = np.concatenate([LEVELS, -LEVELS])
    d = tab[b[:, :4]]                                           # [nb, 4]
    qs = b[:, 4:].reshape(-1, 4, 8)
    out = np.empty((len(b), 4, 16), np.float32)
    out[:, :, :8] = kv[qs & 15] * d[:, :, None]
    out[:, :, 8:] = kv[qs >> 4] * d[:, :, None]
    return out.reshape(-1)


def quant_nvfp4(x: np.ndarray, w: np.ndarray | None, span=(-2, 4)):
    """x [n] float32 (n % 64 == 0), w its weights or None -> (block bytes, sse of the search, sse of max / 12)."""
    X = x.reshape(-1, 16)
    W = None if w is None else w.reshape(-1, 16)
    amax = np.abs(X).max(axis=1)
    i0 = np.clip(np.searchsorted(SCALE_V, amax / 12.0), 0, len(SCALE_V) - 1)   # (first scale >= max / 12)
    best_e = np.full(len(X), np.inf, np.float32)
    best_i = i0.copy()
    best_q = np.zeros(X.shape, np.uint8)
    naive = None
    A = np.abs(X)
    neg = (X < 0)
    # llama.cpp's own pick: the nearest-below code of max / 12 (fp32_to_ue4m3 truncates the mantissa)
    for k in list(range(span[0], span[1])) + ["naive"]:
        if k == "naive":
            i = np.clip(np.searchsorted(SCALE_V, amax / 12.0, side="right") - 1, 0, len(SCALE_V) - 1)
        else:
            i = np.clip(i0 + k, 0, len(SCALE_V) - 1)
        d = SCALE_V[i][:, None]
        q = np.searchsorted(EDGES, (A / d).reshape(-1)).reshape(X.shape).astype(np.uint8)
        err = (LEVELS[q] * d - A) ** 2
        if W is not None:
            err *= W
        e = err.sum(axis=1)
        if k == "naive":
            naive = float(e.sum())
            continue
        better = e < best_e
        best_e[better] = e[better]
        best_i[better] = i[better]
        best_q[better] = q[better]
    zero = amax == 0
    code = best_q | (neg.astype(np.uint8) << 3)
    code[(best_q == 0)] = 0
    blk = np.empty((len(X) // 4, 36), np.uint8)
    dc = SCALE_C[best_i]
    dc[zero] = 0
    blk[:, :4] = dc.reshape(-1, 4)
    c = code.reshape(-1, 4, 16)
    blk[:, 4:] = (c[:, :, :8] | (c[:, :, 8:] << 4)).reshape(-1, 32)
    return blk.reshape(-1), float(best_e.sum()), naive


def _job(args):
    path, off, nbytes, wt = args
    with open(path, "rb") as fh:
        fh.seek(off)
        raw = np.frombuffer(fh.read(nbytes), np.uint8)
    x = deq_q8(raw)
    w = None if wt is None else np.tile(wt, len(x) // len(wt))
    blk, e, en = quant_nvfp4(x, w)
    y = deq_nvfp4(blk)
    return blk.tobytes(), float(((y - x) ** 2).sum()), float((x ** 2).sum()), e, en


def imatrix_rows(path, name, n_per_row, n_exp):
    """llama-imatrix's GGUF: NAME.in_sum2 [n_per_row * n_exp] and NAME.counts -> [n_exp, n_per_row] mean squares."""
    if not path:
        return None
    g = gguf_reader.GGUFFile(path)
    t = {x.name: x for x in g.tensors}
    s, c = t.get(name + ".in_sum2"), t.get(name + ".counts")
    if s is None:
        raise SystemExit("imatrix: no " + name)
    with open(path, "rb") as fh:
        fh.seek(g.data_start + s.offset)
        v = np.frombuffer(fh.read(4 * s.elements), np.float32).reshape(n_exp, n_per_row).copy()
        fh.seek(g.data_start + c.offset)
        cnt = np.frombuffer(fh.read(4 * c.elements), np.float32).copy()
    cnt = np.broadcast_to(cnt.reshape(-1, 1) if cnt.size == n_exp else cnt.reshape(1, -1), (n_exp, 1))
    v = v / np.maximum(cnt, 1)
    bad = ~(v.sum(axis=1) > 0)
    v[bad] = 1.0                                                # an expert never seen: plain squared error
    return v.astype(np.float32)


def tinfo(name, shape, ty, off):
    nb = name.encode()
    return struct.pack("<Q", len(nb)) + nb + struct.pack("<I%dQIQ" % len(shape), len(shape), *shape, ty, off)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src")
    ap.add_argument("--out")
    ap.add_argument("--nvfp4", default=r"blk\.2[0-3]\.ffn_down_exps\.weight")
    ap.add_argument("--imatrix")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--header-only", action="store_true")
    ap.add_argument("--check")
    a = ap.parse_args()
    if a.check:
        g = gguf_reader.GGUFFile(a.check)
        t = next(t for t in g.tensors if re.search(a.nvfp4, t.name))
        n = t.elements // t.shape[-1]
        r = _job((a.check, g.data_start + t.offset, n // 32 * 34, None))
        print("%s expert 0: rel rms error search %.4f, max/12 %.4f" % (t.name, (r[3] / r[2]) ** 0.5, (r[4] / r[2]) ** 0.5))
        return 0
    g = gguf_reader.GGUFFile(a.src)
    al = g.alignment
    src_size = os.path.getsize(a.src)
    offs = sorted(t.offset for t in g.tensors) + [src_size - g.data_start]
    size_of = {o: offs[i + 1] - o for i, o in enumerate(offs[:-1])}        # (with the padding after it)
    plan = []                                                               # (name, shape, type, bytes, source)
    for t in g.tensors:
        nbytes = t.expected_bytes() or size_of[t.offset]
        if re.search(a.nvfp4, t.name) and t.type_id == Q8_0:
            n_exp = t.shape[-1]
            plan.append((t.name[:-len("weight")] + "scale", [n_exp], F32, 4 * n_exp, ("ones", n_exp)))
            plan.append((t.name, t.shape, NVFP4, t.elements // 64 * 36, ("nvfp4", t)))
        else:
            plan.append((t.name, t.shape, t.type_id, nbytes, ("copy", t)))
    infos, off = b"", 0
    starts = []
    for name, shape, ty, nbytes, _ in plan:
        infos += tinfo(name, shape, ty, off)
        starts.append(off)
        off = (off + nbytes + al - 1) // al * al
    old_infos = sum(len(tinfo(t.name, t.shape, t.type_id, t.offset)) for t in g.tensors)
    kv_end = g.header_end - old_infos
    with open(a.src, "rb") as fi, open(a.out, "wb") as fo:
        head = fi.read(kv_end)
        fo.write(head[:8] + struct.pack("<Q", len(plan)) + head[16:])
        fo.write(infos)
        data0 = (fo.tell() + al - 1) // al * al
        fo.write(b"\0" * (data0 - fo.tell()))
        if a.header_only:
            fo.truncate(data0 + off)
            print("header only: %d tensors, %.3f GiB" % (len(plan), (data0 + off) / 2 ** 30))
            return 0
        mm = mmap.mmap(fi.fileno(), 0, access=mmap.ACCESS_READ)
        with ProcessPoolExecutor(a.jobs) as ex:
            for (name, shape, ty, nbytes, (kind, t)), st in zip(plan, starts):
                fo.seek(data0 + st)
                if kind == "copy":
                    p = g.data_start + t.offset
                    for o in range(0, nbytes, 1 << 26):
                        fo.write(mm[p + o:p + min(nbytes, o + (1 << 26))])
                elif kind == "ones":
                    fo.write(np.ones(t, np.float32).tobytes())
                else:
                    n_exp = t.shape[-1]
                    per = t.elements // n_exp
                    wt = imatrix_rows(a.imatrix, t.name, t.shape[0], n_exp)
                    jobs = [(a.src, g.data_start + t.offset + e * (per // 32 * 34), per // 32 * 34,
                             None if wt is None else wt[e]) for e in range(n_exp)]
                    se = sx = e1 = e0 = 0.0
                    for blk, d2, x2, es, en in ex.map(_job, jobs):
                        fo.write(blk)
                        se, sx, e1, e0 = se + d2, sx + x2, e1 + es, e0 + en
                    print("%s: Q8_0 -> NVFP4, rel rms error %.4f (%sobjective: search %.4g, max/12 %.4g)" %
                          (name, (se / sx) ** 0.5, "weighted " if wt is not None else "", e1, e0), flush=True)
        fo.truncate(data0 + off)
    print("wrote %s: %d tensors, %.3f GiB" % (a.out, len(plan), (data0 + off) / 2 ** 30))
    return 0


if __name__ == "__main__":
    sys.exit(main())
