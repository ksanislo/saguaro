"""serve/test_context_panel.py - the Monitor's Context panel as the ENGINE's context (PLAN-context-panel): every chat
in a slot of its positions, split VRAM / RAM / PARKED / FREE, against the binding capacity (positional
max_context x slots, or the engine's live K/V budgets over kv_bytes_token) - all against fake engines that print the
VRAM lines with the new fields.  No GPU, no pack.

    python -m unittest serve.test_context_panel -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import gpu_map  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KBT = 12672        # a position's K/V bytes, QFN Q8's own figure (12 QSA layers x 1056 B)
CTX = 1048576      # 1M positions a slot


def vram_line(dev=0, kv_room=9_000_000_000_000, kv_ram_room=0, parked=0, kv=8_000_000_000):
    """A `VRAM dev=...` line as the four engines re-print them with the K/V budget fields appended (PLAN-context-panel)."""
    return (f"VRAM dev={dev} pci=0000:0{dev + 1}:00.0 total=25296896000 free=751828992 weights=6678575104 "
            f"kv={kv} kv_ram=0 act=107374182 drafter=0 window=0 pool=15536793108 tail=0 borrow=0 vision=0 "
            f"slots=1231 ctx={CTX} layers=0-45 kv_room={kv_room} kv_ram_room={kv_ram_room} parked={parked}")


class FakeEngine:
    """An engine as the server sees it: its INFO facts, its VRAM lines, its chat slots."""

    def __init__(self, max_context=CTX, info=None, vram=None, multi=0, batch=0):
        self.max_context = max_context
        self.info = info if info is not None else {"kv_bytes_token": KBT, "kv_resident": 0}
        self.vram = vram if vram is not None else [gpu_map.parse(vram_line())]
        self.multi, self.batch = multi, batch
        self.proc = SimpleNamespace(pid=4242)
        self.progress = None
        self.stat = None


def service(engine):
    tok = ByteTokenizer()
    return Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))


class Lines(unittest.TestCase):
    """The VRAM line's new fields reach the server: parse, and the split parts of a card merged."""

    def test_parse_carries_the_budget_fields(self):
        r = gpu_map.parse(vram_line(kv_room=40 * KBT, kv_ram_room=9 * KBT, parked=3 * KBT))
        self.assertEqual((r["kv_room"], r["kv_ram_room"], r["parked"]), (40 * KBT, 9 * KBT, 3 * KBT))

    def test_merge_adds_kv_room_and_keeps_the_engine_wide_figure(self):
        a = {"dev": 0, "kv_room": 10, "kv_ram_room": 5, "parked": 100, "free": 9}
        b = {"dev": 0, "kv_room": 20, "kv_ram_room": 7, "parked": 100, "free": 7}
        (m,) = gpu_map.merge([a, b])
        self.assertEqual(m["kv_room"], 30)               # (two split parts of one card add up)
        self.assertEqual(m["kv_ram_room"], 7)            # (the engine's own figure: the biggest part's, never a sum)
        self.assertEqual(m["parked"], 100)

    def test_a_re_printed_line_moves_the_panel(self):
        """PLAN-vram-live: the engine re-prints a card's line when a park happened; the panel follows it."""
        eng = FakeEngine(vram=[])
        svc = service(eng)
        self.assertIsNone(svc._engine_context([]))
        StrataEngine._vram_line(eng, vram_line(parked=0))
        self.assertEqual(svc._engine_context([])["parked"]["bytes"], 0)
        StrataEngine._vram_line(eng, vram_line(parked=10 * KBT))
        self.assertEqual(svc._engine_context([])["parked"], {"bytes": 10 * KBT, "tokens": 10})


class Capacity(unittest.TestCase):
    """The bar's end: the BINDING limit, labelled with which one it is."""

    def test_positions_bind(self):
        """8 slots x 1M positions with budgets that hold far more: the positions are the limit."""
        eng = FakeEngine(multi=8, vram=[gpu_map.parse(vram_line(kv_room=1000 * CTX * KBT))])
        c = service(eng)._engine_context([])
        self.assertEqual(c["capacity"]["tokens"], 8 * CTX)
        self.assertEqual(c["capacity"]["kind"], "positional")
        self.assertEqual(c["capacity"]["slots"], 8)

    def test_batch_slots_count_as_slots(self):
        """QFN --batch 8: the batch slots are the engine's chat slots (it says batch_slots=, not READY multi=)."""
        eng = FakeEngine(batch=8, vram=[gpu_map.parse(vram_line(kv_room=1000 * CTX * KBT))])
        self.assertEqual(service(eng)._engine_context([])["capacity"]["slots"], 8)

    def test_memory_binds_vram_only(self):
        """An engine without RAM streaming (kv_ram_room=0): the memory limit is the VRAM budget alone."""
        eng = FakeEngine(multi=8, vram=[gpu_map.parse(vram_line(kv_room=500_000 * KBT, kv_ram_room=0))])
        c = service(eng)._engine_context([])
        self.assertEqual(c["capacity"]["kind"], "memory")
        self.assertEqual(c["capacity"]["which"], "VRAM")
        self.assertEqual(c["capacity"]["tokens"], 500_000)

    def test_memory_binds_with_the_ram_budget(self):
        """Streaming engines: (VRAM K/V budget + RAM K/V budget available) / kv_bytes_token."""
        eng = FakeEngine(multi=8, vram=[gpu_map.parse(vram_line(kv_room=200_000 * KBT, kv_ram_room=300_000 * KBT))])
        c = service(eng)._engine_context([])
        self.assertEqual(c["capacity"]["kind"], "memory")
        self.assertEqual(c["capacity"]["which"], "VRAM+RAM")
        self.assertEqual(c["capacity"]["tokens"], 500_000)

    def test_memory_binds_on_ram_alone(self):
        eng = FakeEngine(multi=8, vram=[gpu_map.parse(vram_line(kv_room=0, kv_ram_room=300_000 * KBT))])
        c = service(eng)._engine_context([])
        self.assertEqual((c["capacity"]["kind"], c["capacity"]["which"]), ("memory", "RAM"))

    def test_split_parts_add_up_to_one_budget(self):
        a = vram_line(dev=0, kv_room=200_000 * KBT)
        b = vram_line(dev=1, kv_room=300_000 * KBT)
        eng = FakeEngine(multi=2, vram=[gpu_map.parse(a), gpu_map.parse(b)])
        c = service(eng)._engine_context([])
        self.assertEqual(c["capacity"]["tokens"], 500_000)
        self.assertEqual(c["capacity"]["kv_room"], 500_000 * KBT)

    def test_no_budget_lines_no_claim(self):
        """An engine printing no VRAM lines: no aggregate - the page keeps its per-request headline."""
        self.assertIsNone(service(FakeEngine(vram=[]))._engine_context([]))


class Used(unittest.TestCase):
    """The bar's start: every chat in a slot, its positions split into VRAM, RAM and what the engine parked."""

    def conv(self, used):
        return {"context_used": used}

    def test_chats_sum_and_split_on_the_window(self):
        """Two chats in a slot: each one's positions inside the VRAM window are VRAM, the rest streamed RAM."""
        eng = FakeEngine(max_context=32768, multi=2,
                         info={"kv_bytes_token": KBT, "kv_resident": 4096},
                         vram=[gpu_map.parse(vram_line(kv_room=500_000 * KBT))])
        c = service(eng)._engine_context([self.conv(6000), self.conv(2000)])
        self.assertEqual(c["used"]["tokens"], 8000)
        self.assertEqual(c["used"]["vram_tokens"], 4096 + 2000)     # (the first chat's positions past the window...)
        self.assertEqual(c["used"]["ram_tokens"], 1904)             # (...are the engine's RAM segment)
        self.assertEqual(c["used"]["vram_bytes"], 6096 * KBT)
        self.assertEqual(c["used"]["ram_bytes"], 1904 * KBT)

    def test_no_streaming_means_vram_alone(self):
        eng = FakeEngine(max_context=32768, multi=2, info={"kv_bytes_token": KBT, "kv_resident": 32768},
                         vram=[gpu_map.parse(vram_line(kv_room=500_000 * KBT))])
        c = service(eng)._engine_context([self.conv(6000)])
        self.assertEqual((c["used"]["vram_tokens"], c["used"]["ram_tokens"]), (6000, 0))

    def test_parked_sits_outside_the_slots(self):
        """Conversations the engine parked in RAM: their bytes and positions, and FREE counts them off the capacity."""
        eng = FakeEngine(multi=8, info={"kv_bytes_token": KBT, "kv_resident": 0},
                         vram=[gpu_map.parse(vram_line(kv_room=1000 * CTX * KBT, parked=10_000 * KBT))])
        c = service(eng)._engine_context([self.conv(5000)])
        self.assertEqual(c["parked"], {"bytes": 10_000 * KBT, "tokens": 10_000})
        self.assertEqual(c["free"], 8 * CTX - 5000 - 10_000)

    def test_parked_without_a_bytes_per_position_says_only_bytes(self):
        eng = FakeEngine(multi=8, info={"kv_bytes_token": None, "kv_resident": 0},
                         vram=[gpu_map.parse(vram_line(kv_room=0, parked=1 << 30))])
        c = service(eng)._engine_context([self.conv(5000)])
        self.assertEqual(c["parked"], {"bytes": 1 << 30})
        self.assertEqual(c["free"], 8 * CTX - 5000)                 # (no per-position figure: no positions claimed)

    def test_metrics_carries_the_context_section(self):
        eng = FakeEngine(multi=8, vram=[gpu_map.parse(vram_line(parked=KBT))])
        svc = service(eng)
        m = svc.metrics()
        json.dumps(m)
        self.assertEqual(m["context"]["capacity"]["kind"], "positional")
        self.assertEqual(m["context"]["parked"], {"bytes": KBT, "tokens": 1})
        eng.vram = []
        self.assertIsNone(service(eng).metrics()["context"])


if __name__ == "__main__":
    unittest.main()
