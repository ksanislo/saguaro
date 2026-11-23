"""SAGUARO_SLOT_SOLO: two long conversations that alternate without overlapping stop swapping through the engine's
main context.  The first stays there; the second, arriving alone, gets an empty batch slot and keeps it
(SAGUARO_SLOT_KEEP), so neither is parked and restored (copied whole) at every turn.

The real QFN driver (serve/qfn_engine.py) runs against a fake engine that logs which path each request took."""
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve import qfn_engine as QE

FAKE = r'''import sys, threading, queue, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1])
log = open(args[args.index("--log") + 1], "a")
def note(s):
    log.write(s + "\n"); log.flush()
print(f"INFO engine=0.1.49 batch_slots={slots} slot_cache=1 batch_groups={slots} slot_admit=1 slot_keep=1", flush=True)
print("READY 1048576 stop", flush=True)
held = {}                                   # slot -> ids it holds
REPLY = [11, 12, 13, 257]
for line in sys.stdin:
    line = line.strip()
    if line == "QUIT":
        break
    if not line or line == "STOP" or line.startswith(("BSTOP", "BYIELD")):
        continue
    f = line.split()
    ids = [int(x) for x in f[-1].split(",")]
    if f[0] == "GEN":
        note(f"GEN {len(ids)}")
        for t in REPLY:
            print(f"T {t}", flush=True)
        print(f"DONE {len(REPLY)} {len(ids)} 5.0 1.0 stop 0 0 0", flush=True)
        continue
    slot = int(f[1])
    h = held.get(slot, [])
    own = 0
    while own < min(len(h), len(ids)) and h[own] == ids[own]:
        own += 1
    direct = own * 2 >= len(ids)             # the engine's SAGUARO_SLOT_KEEP rule: the slot's own conversation
    note(f"BGEN {slot} {len(ids)} direct={int(direct)}")
    print(f"BQ {slot} {int(direct)}", flush=True)
    if direct:
        print(f"BRA {slot} 1 prompt={len(ids)} reused={own} read_ms=5.0 wait_ms=1.0", flush=True)
        rest = REPLY
    else:
        print(f"T {REPLY[0]}", flush=True)
        print(f"DONE 1 {len(ids)} 5.0 1.0 length 0 0 0", flush=True)
        print(f"BADM {slot} 1", flush=True)
        rest = REPLY[1:]
    for t in rest:
        print(f"BT {slot} {t}", flush=True)
    print(f"BDONE {slot} {len(REPLY)} stop 1.0", flush=True)
    held[slot] = ids + REPLY[:-1]
'''

TURN = QE.StrataEngine.TURN_ID


def chat(seed: int, turns: int, n: int = 200) -> list[int]:
    """A prompt of `turns` turns, each opened by the turn token; `seed` tells conversations apart."""
    out: list[int] = []
    for t in range(turns):
        out += [TURN] + [1000 + seed * 100 + (t * 7 + k) % 90 for k in range(n)]
    return out + [TURN, 3]


class SlotSolo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_qfn.py"
        script.write_text(FAKE, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = QE.subprocess.Popen
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ("SAGUARO_SLOT_SOLO", "SAGUARO_SLOT_KEEP"):
            os.environ.pop(k, None)
        with mock.patch.object(QE.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.eng = QE.StrataEngine("saguaro-qfn", ["--batch", "3", "--log", str(self.log)])
        self.eng.SLOT_KEEP_MIN_TOKENS = 256      # (the rule's own threshold is 16384 tokens)

    def tearDown(self):
        self.eng.unload()
        self.env.stop()
        self.tmp.cleanup()

    def run_one(self, ids):
        out = [t for t in self.eng.generate_batched(list(ids), 16, {}, threading.Event()) if t is not None]
        self.assertEqual(out, [11, 12, 13, 257])
        self.assertFalse(any(self.eng.slot_busy))
        return out

    def paths(self):
        return [ln.split()[0] + (" " + ln.split()[1] if ln.startswith("BGEN") else "")
                for ln in self.log.read_text().splitlines() if ln.startswith(("GEN", "BGEN"))]

    def test_two_alternating_chats_settle_in_their_own_places(self):
        a1, b1 = chat(1, 3), chat(2, 3)
        self.run_one(a1)                                   # A alone on an idle engine: the main context
        self.assertEqual(self.eng.main_held, a1 + [11, 12, 13])
        a2 = a1 + [21, 22] + chat(1, 1)                    # its next turn, the reply rendered again
        self.run_one(a2)
        self.run_one(b1)                                   # B alone, the main context holds A: an empty slot
        self.assertEqual(self.eng.main_held, b1)           # (the old way read it through the main context)
        self.run_one(a2 + [21] + chat(1, 1))               # A again: the main copy of B is the slot's, so solo
        self.run_one(b1 + [21] + chat(2, 1))               # B's next turn: in its slot, read beside (SLOT_KEEP)
        self.run_one(a2 + [21] + chat(1, 2))               # and A stays in the main context
        self.assertEqual(self.paths(), ["GEN", "GEN", "BGEN 0", "GEN", "BGEN 0", "GEN"])
        lines = [ln for ln in self.log.read_text().splitlines() if ln.startswith("BGEN")]
        self.assertTrue(lines[0].endswith("direct=0") and lines[1].endswith("direct=1"), lines)

    def test_off_restores_the_solo_path(self):
        os.environ["SAGUARO_SLOT_SOLO"] = "0"
        self.run_one(chat(1, 3))
        self.run_one(chat(2, 3))
        self.assertEqual(self.paths(), ["GEN", "GEN"])

    def test_the_rule(self):
        e = self.eng
        a, b = chat(1, 3), chat(2, 3)
        e.main_held = a + [11, 12, 13]
        self.assertEqual(e._solo_to_slot(b), 0)                              # another long conversation
        self.assertIsNone(e._solo_to_slot(a + [21] + chat(1, 1)))            # the main context's own
        self.assertIsNone(e._solo_to_slot(chat(2, 1)[:100]))                 # a short prompt
        e.main_held = a[:100]
        self.assertIsNone(e._solo_to_slot(b))                                # the main context holds little
        e.main_held = a + [11, 12, 13]
        e.slot_held[1] = e.main_held + [5, 6]
        self.assertIsNone(e._solo_to_slot(b))                                # a slot has the main conversation too
        e.slot_held[1] = [1, 2, 3]
        e.slot_busy[0] = True
        self.assertEqual(e._solo_to_slot(b), 2)                              # the next empty slot
        e.slot_held[2] = [4]
        self.assertIsNone(e._solo_to_slot(b))                                # none empty: the solo path, as before
        e.slot_busy[0] = False
        e.slot_held[2] = []
        e.slot_kept[0] = (0, 1e18)                                           # kept for another request's next pass
        self.assertEqual(e._solo_to_slot(b), 2)
        e.slot_kept.clear()
        e.slot_admit = False
        self.assertIsNone(e._solo_to_slot(b))                                # needs the engine's slot admission
        e.slot_admit = True
        # a sibling that shares only the system prompt is not the main context's conversation
        sib = a[:300] + chat(3, 2)
        self.assertEqual(e._solo_to_slot(sib), 0)
        with mock.patch.dict(os.environ, {"SAGUARO_SLOT_KEEP": "0"}):
            self.assertIsNone(e._solo_to_slot(b))

    def test_a_request_that_ends_early_forgets_the_main_context(self):
        a = chat(1, 3)
        gen = self.eng.generate_batched(list(a), 16, {}, threading.Event())
        self.assertEqual(next(gen), 11)
        gen.close()                                                           # the consumer left: STOP, drained
        self.assertEqual(self.eng.main_held, [])
        self.assertFalse(self.eng.ctl.locked())


if __name__ == "__main__":
    unittest.main()
