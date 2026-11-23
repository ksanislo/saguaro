"""Strata .strata v1.x pack reader and reference decoders (numpy), for converters and exactness checks.

Format and decode arithmetic ported from Strata (MIT, Copyright (c) 2026 Strata contributors):
engine/src/artifact/strata_reader.cpp, engine/src/quant/sk_codecs.cpp, engine/src/inkling2/inkling2_loop.cpp.
Model-independent: Inkling 2 Mini packs today, Hy4-Flash SQ packs later (same container, same S_K codecs).
"""
from __future__ import annotations

import json
import mmap
import struct
import subprocess

import numpy as np

MAGIC = b"STRATA\x00\x01"
PAGE = 4096
NSEC = 6
SEC_FMT = "<QQ32sIIQ"
EXP_FMT = "<HBBHHQIIII"

# S_K / pack codec ids (sk_codecs.hpp SkCode) and trunk formats (ink_manifest.h WFmt)
S1_25, S1_5, S2_K, S3_K, S4_K, S5_K, S6_K, S8_K, SF4, MX8, GGUF_RAW = 0, 1, 2, 3, 4, 5, 6, 8, 15, 16, 20
SK_BITS = {S2_K: 2, S3_K: 3, S4_K: 4, S5_K: 5, S6_K: 6}
SK_SB_BYTES = {S2_K: 70, S3_K: 102, S4_K: 134, S5_K: 166, S6_K: 198, S8_K: 262}
SK_HALF = {S2_K: 1.5, S3_K: 3.5, S4_K: 7.5, S5_K: 15.5, S6_K: 31.5}
CODE_NAMES = {S1_25: "S1_25", S1_5: "S1_5", S2_K: "S2_K", S3_K: "S3_K", S4_K: "S4_K", S5_K: "S5_K", S6_K: "S6_K",
              S8_K: "S8_K", SF4: "SF4", MX8: "MX8", GGUF_RAW: "GGUF_RAW"}
FMT_CODES = {v: k for k, v in CODE_NAMES.items()}


class Pack:
    def __init__(self, path):
        self.path = str(path)
        self.fh = open(self.path, "rb")
        self.mm = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
        hdr = self.mm[:PAGE]
        if hdr[:8] != MAGIC:
            raise ValueError(f"{path}: not a .strata pack (magic {hdr[:8]!r})")
        self.flags, self.profile, nsec = struct.unpack_from("<IHH", hdr, 8)
        if nsec != NSEC:
            raise ValueError(f"{path}: {nsec} sections")
        self.sections = [struct.unpack_from(SEC_FMT, hdr, 0x10 + 64 * i) for i in range(NSEC)]
        for i, s in enumerate(self.sections):
            if s[1] and s[0] % PAGE:
                raise ValueError(f"section {i} not 4 KiB aligned")
        off, n = self.sections[0][:2]
        raw = self.mm[off:off + n]
        try:   # (the zstd program; without it the manifest is not read - inkling_expert_bytes knows Inkling's shape)
            self.manifest = json.loads(subprocess.run(["zstd", "-dc"], input=raw, capture_output=True,
                                                      check=True).stdout)
        except (OSError, subprocess.CalledProcessError, ValueError):
            self.manifest = {}
        off, n = self.sections[3][:2]
        s3 = self.mm[off:off + n]
        if self.sections[3][4] & 1:   # v1.1: 32 B entries then a u16 VRAM load order
            ne = n // 34
            self.vram_order = list(struct.unpack_from(f"<{ne}H", s3, 32 * ne))
        else:
            ne = n // 32
            self.vram_order = []
        self.experts = [struct.unpack_from(EXP_FMT, s3, 32 * i) for i in range(ne)]
        self.trunk = {t["name"]: t for t in self.manifest.get("trunk_tensors", [])}
        self.dspark = {t["name"]: t for t in self.manifest.get("dspark_tensors", [])}

    def section_view(self, i) -> memoryview:
        off, n = self.sections[i][:2]
        return memoryview(self.mm)[off:off + n]

    def tensor_bytes(self, t) -> memoryview:
        s1 = self.sections[1][0]
        return memoryview(self.mm)[s1 + t["offset"]: s1 + t["offset"] + t["bytes"]]

    def expert(self, layer: int, rank: int):
        """(entry, body) of directory slot layer * n_experts + rank; entry fields as EXP_FMT."""
        e = self.experts[layer * self.manifest["num_experts_per_layer"] + rank]
        s4 = self.sections[4][0]
        return e, memoryview(self.mm)[s4 + e[5]: s4 + e[5] + e[6]]


# ---- scalar formats

def fp16_to_f32(u16: np.ndarray) -> np.ndarray:
    return u16.view(np.float16).astype(np.float32)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def e4m3_table() -> np.ndarray:
    """sk_codecs.cpp E4M3Table (F8_E4M3FN; 0x7F / 0xFF NaN -> 0)."""
    v = np.zeros(256, np.float32)
    for c in range(256):
        e, m = (c & 0x7F) >> 3, c & 7
        if (c & 0x7F) == 0x7F:
            val = 0.0
        elif e == 0:
            val = np.float32(m) * np.float32(1.0 / 512.0)
        else:
            val = np.float32(1.0 + m * 0.125) * np.float32(2.0 ** (e - 7))
        v[c] = -val if c & 0x80 else val
    return v


E4M3 = e4m3_table()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


# ---- S_K super-blocks (256 weights: fp16 D, 4 x u8 sub-scales, packed codes)

def sk_codes(raw: np.ndarray, code: int) -> np.ndarray:
    """raw [nsb, sb_bytes] uint8 -> unsigned codes [nsb, 256] (unpack_s*k_sb)."""
    pl = raw[:, 6:]
    n = raw.shape[0]

    def two_bit(p):   # byte i -> q[4i..4i+3]
        return np.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(n, -1)

    def nib(p):       # byte i -> q[2i], q[2i+1]
        return np.stack([p & 15, p >> 4], axis=-1).reshape(n, -1)

    def bits(p):      # byte i bit b -> q[8i + b]
        return np.unpackbits(p, axis=-1, bitorder="little")

    if code == S2_K:
        return two_bit(pl[:, :64])
    if code == S3_K:
        return two_bit(pl[:, :64]) | (bits(pl[:, 64:96]) << 2)
    if code == S4_K:
        return nib(pl[:, :128])
    if code == S5_K:
        return nib(pl[:, :128]) | (bits(pl[:, 128:160]) << 4)
    if code == S6_K:
        return nib(pl[:, :128]) | (two_bit(pl[:, 128:192]) << 4)
    raise ValueError(f"S_K code {code} not decoded here")


def sk_dequant(buf, code: int, n: int) -> np.ndarray:
    """dequant_sk_row: n weights (n % 256 == 0) -> f32, Strata's arithmetic order: (q - half) * (D * (u / 255))."""
    sbb = SK_SB_BYTES[code]
    raw = np.frombuffer(buf, np.uint8, count=n // 256 * sbb).reshape(-1, sbb)
    D = fp16_to_f32(raw[:, 0:2].copy().view(np.uint16)[:, 0])
    u = raw[:, 2:6].astype(np.float32)
    eff = (D[:, None] * (u / np.float32(255.0))).astype(np.float32)          # [nsb, 4]
    q = sk_codes(raw, code).astype(np.float32) - np.float32(SK_HALF[code])   # [nsb, 256]
    return (q.reshape(-1, 4, 64) * eff[:, :, None]).reshape(-1)


def sk_fields(buf, code: int, n: int):
    """(D fp16 bits [nsb], u [nsb, 4], codes [nsb, 256]) - the fields a lossless re-layout must keep."""
    sbb = SK_SB_BYTES[code]
    raw = np.frombuffer(buf, np.uint8, count=n // 256 * sbb).reshape(-1, sbb)
    return raw[:, 0:2].copy().view(np.uint16)[:, 0], raw[:, 2:6].copy(), sk_codes(raw, code)


def q8row_dequant(buf, rows: int, cols: int) -> np.ndarray:
    sc = np.frombuffer(buf, np.float32, count=rows)
    q = np.frombuffer(buf, np.int8, count=rows * cols, offset=rows * 4).reshape(rows, cols)
    return q.astype(np.float32) * sc[:, None]


def trunk_dequant(pack: Pack, t) -> np.ndarray:
    b = pack.tensor_bytes(t)
    shape = t["shape"]
    n = int(np.prod(shape))
    f = t["format"]
    if f == "F32":
        return np.frombuffer(b, np.float32, count=n).copy()
    if f == "BF16":
        return bf16_to_f32(np.frombuffer(b, np.uint16, count=n))
    if f == "F16":
        return fp16_to_f32(np.frombuffer(b, np.uint16, count=n))
    if f == "Q8_ROW":
        return q8row_dequant(b, shape[0], int(np.prod(shape[1:]))).reshape(-1)
    if f in FMT_CODES and FMT_CODES[f] in SK_SB_BYTES:
        return sk_dequant(b, FMT_CODES[f], n)
    raise ValueError(f"{t['name']}: format {f}")


# ---- SF4 (lossless NVFP4) expert parts: E2M1 nibbles [rows][cols/2] then E4M3 scales [rows][cols/16]

def sf4_part(body, off: int, rows: int, cols: int):
    """(nibbles [rows, cols] uint8 codes, scales [rows, cols/16] uint8)."""
    p = np.frombuffer(body, np.uint8, count=rows * cols // 2, offset=off).reshape(rows, cols // 2)
    s = np.frombuffer(body, np.uint8, count=rows * cols // 16, offset=off + rows * cols // 2).reshape(rows, cols // 16)
    q = np.stack([p & 15, p >> 4], axis=-1).reshape(rows, cols)   # low nibble = even column
    return q, s


def sf4_dequant(q, s, outer: np.float32) -> np.ndarray:
    """dequant_sf4_row's scalar arithmetic: e2m1 * (e4m3 * outer)."""
    sc = (E4M3[s] * np.float32(outer)).astype(np.float32)
    return E2M1[q] * np.repeat(sc, 16, axis=-1)


def sf4_header(body):
    """amax_13, denom_13, amax_2, denom_2 (outer scales = amax / denom, f32)."""
    return np.frombuffer(body, np.float32, count=4)


class InklingExpert:
    """One Inkling 2 expert body (inkling2_loop.cpp eval_inkling2_expert_batch layout).
    gate_up: [2F][C] rows interleaved (even = gate, odd = up); down: [C][F]."""

    def __init__(self, entry, body, C: int, F: int):
        self.entry, self.body, self.C, self.F = entry, body, C, F
        self.gu, self.dn, self.dn_off = entry[1], entry[2], entry[7]

    def gate_up(self) -> np.ndarray:
        C, F = self.C, self.F
        if self.gu == SF4:
            h = sf4_header(self.body)
            q, s = sf4_part(self.body, 16, 2 * F, C)
            return sf4_dequant(q, s, np.float32(h[0] / h[1]))
        if self.gu in SK_SB_BYTES:
            n = 2 * F * C
            return sk_dequant(self.body[2 * F: 2 * F + n // 256 * SK_SB_BYTES[self.gu]], self.gu, n).reshape(2 * F, C)
        raise ValueError(f"gate_up code {self.gu}")

    def down(self) -> tuple[np.ndarray, np.ndarray | None]:
        """(W [C][F] as stored, macaq [F] f32 or None): the effective weight is W[:, j] / macaq[j] (the activation is
        multiplied by 1 / macaq before the product)."""
        C, F = self.C, self.F
        if self.dn == SF4:
            h = sf4_header(self.body)
            q, s = sf4_part(self.body, self.dn_off, C, F)
            return sf4_dequant(q, s, np.float32(h[2] / h[3])), None
        if self.dn in SK_SB_BYTES:
            n = C * F   # paired 2-row layout = the flat [C][F] run (F % 256 == 128, a pair is whole super-blocks)
            w = sk_dequant(self.body[self.dn_off: self.dn_off + n // 256 * SK_SB_BYTES[self.dn]], self.dn, n)
            sq = fp16_to_f32(np.frombuffer(self.body, np.uint16, count=F))
            return w.reshape(C, F), sq
        raise ValueError(f"down code {self.dn}")


def inkling_expert_bytes(pack: Pack) -> int:
    """The experts' bytes as saguaro-ik2 holds them resident (src/strata_fmt/strata_inkling.cpp's layout: an all-SF4
    layer as NVFP4, an all-MX8 layer as 36-byte blocks of 32, any other layer each expert packed)."""
    tc = (pack.manifest.get("config") or {}).get("text_config") or \
        {"hidden_size": 2304, "intermediate_size": 1152, "n_routed_experts": 128, "num_hidden_layers": 24}
    C, F, E = tc["hidden_size"], tc["intermediate_size"], tc["n_routed_experts"]

    def rb(code, K):
        if code == SF4:
            return K // 64 * 36
        if code == MX8:
            return K // 32 * 36
        return K // 128 * (4 + 16 * code)

    def a16(x):
        return (x + 15) // 16 * 16

    total = 0
    for l in range(tc["num_hidden_layers"]):
        ents = pack.experts[l * E:(l + 1) * E]
        codes = {(e[1], e[2]) for e in ents}
        if codes == {(SF4, SF4)} or codes == {(MX8, MX8)}:
            c = next(iter(codes))[0]
            per = 2 * F * rb(c, C) + C * rb(c, F)
            total += per * E
        else:   # (each expert packed: its own parts, its pool's slot rounded to 256 - EC.init_packed)
            for e in ents:
                g, d = rb(e[1], C), rb(e[2], F)
                mq = 2 * F if S2_K <= e[2] <= S6_K else 0
                up = a16(16 + F * g)
                dn = a16(up + 16 + F * g + mq)
                total += (a16(dn + 16 + C * d) + 255) // 256 * 256
    return total
