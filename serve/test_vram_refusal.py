"""P124e round 6 (manager card): the engine's context-grow refusal says which memory ran out.  "no RAM is left"
stays the server's RAM error; "no VRAM is left" (the device state after the expert cache gave all it can) ends as
no-VRAM - retry once / 503 - and never with the RAM text."""
import unittest

from serve.qfn_engine import EngineNonFinite, EngineOutOfRam, EngineOutOfVram, engine_refusal


class RefusalWording(unittest.TestCase):
    def test_vram_refusal_is_no_vram_not_no_ram(self):
        e = engine_refusal("no VRAM is left for this request's context (546145 tokens): the card's expert cache "
                           "gave all it can")
        self.assertIsInstance(e, EngineOutOfVram)
        self.assertIsInstance(e, EngineNonFinite)          # Service.run's retry-once path
        self.assertNotIsInstance(e, EngineOutOfRam)
        self.assertIn("no VRAM is left", str(e))
        self.assertNotIn("no RAM is left", str(e))

    def test_ram_refusal_unchanged(self):
        e = engine_refusal("no RAM is left to pin 300000 cells of context in slot 2")
        self.assertIsInstance(e, EngineOutOfRam)
        self.assertIn("300000 cells", str(e))

    def test_ordinary_refusal_stays_the_requests(self):
        e = engine_refusal("a token id is outside the vocabulary")
        self.assertNotIsInstance(e, EngineNonFinite)


if __name__ == "__main__":
    unittest.main()
