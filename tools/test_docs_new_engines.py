"""tools/test_docs_new_engines.py - the user-facing text for Hy4-Flash and Inkling 2 Mini says what the code does: the
setup ids in the docs are ones the setup takes, the speeds are the measured ones the setup and the page quote
(hy4_fit.py, ik2_fit.py), the flags named are the layouts', the links resolve, and no page names upstream's programs.

    python -m unittest tools.test_docs_new_engines -v
"""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import maya_quants as Q  # noqa: E402
import hy4_fit as HY4F  # noqa: E402
import ik2_fit as IK2F  # noqa: E402

PAGES = {n: (ROOT / n).read_text(encoding="utf-8") for n in
         ("README.md", "CHANGELOG.md", "NOTICE.md", "docs/MODELS.md", "docs/SETUP.md", "docs/HY4-FLASH.md",
          "docs/INKLING2-MINI.md")}


class Docs(unittest.TestCase):
    def test_setup_ids_are_the_setups(self):
        for name, text in PAGES.items():
            for ident in set(re.findall(r"--model ((?:hy4|inkling)[a-z0-9_.-]*)", text)):
                self.assertIsNotNone(Q.resolve(ident), f"{name}: --model {ident}")
        for ident in ("hy4-flash-maya-s", "hy4-flash-maya-xs", "hy4-flash-maya-m", "hy4-flash-maya-l", "inkling-2-mini"):
            self.assertIn(ident, PAGES["README.md"])
            self.assertIsNotNone(Q.resolve(ident))

    def test_speeds_are_the_measured_ones(self):
        hy4 = PAGES["docs/HY4-FLASH.md"]
        dec, read = HY4F.MEASURED["maya-s"]
        self.assertIn(f"about {dec} tok/s", hy4)
        self.assertIn(f"about {read} tok/s", hy4)
        self.assertIn("4x Tesla T4", hy4)
        self.assertRegex(hy4, r"clocks\s+locked")
        ik2 = PAGES["docs/INKLING2-MINI.md"]
        for n in (1, 2):
            dec, read = IK2F.MEASURED[n]
            self.assertIn(f"about {dec} tok/s", ik2)
            self.assertIn(f"about {read} tok/s", ik2)
            self.assertIn(f"{n}x Tesla T4", ik2)
        for text in (PAGES["README.md"], PAGES["CHANGELOG.md"], PAGES["docs/MODELS.md"]):
            for n in (HY4F.MEASURED["maya-s"][0], IK2F.MEASURED[1][0], IK2F.MEASURED[2][0]):
                self.assertRegex(text, rf"(about |~){n}\b")

    def test_layout_flags_are_the_setups(self):
        self.assertIn("`" + " ".join(HY4F.T4_ARGS) + "`", PAGES["docs/HY4-FLASH.md"])
        self.assertIn("HY4_HOT=1", PAGES["docs/HY4-FLASH.md"])
        t4 = lambda i: {"index": i, "name": "Tesla T4", "vram_gb": 16.0, "arch": "75"}   # noqa: E731
        for n in (1, 2):
            lay = IK2F.layout([t4(i) for i in range(n)])
            self.assertIn("`" + " ".join(lay["args"][:4]) + "`", PAGES["docs/INKLING2-MINI.md"])
        self.assertIn("nvidia-smi -lgc 1590,1590", PAGES["docs/HY4-FLASH.md"])

    def test_inkling_on_amd_is_said_alike_everywhere(self):
        """README, SETUP, the setup's --help, the web catalog and AMD_SAGUARO.md say what the setup does (the 9070 XT
        box, 2026-11-22): Inkling 2 Mini runs on NVIDIA and on one AMD RDNA 3 / 4 GPU of 12 GB+ (the offload path),
        and is
        recommended there; the RX 9070 XT speeds quoted are ik2_fit's."""
        flat = lambda t: " ".join(t.split())   # noqa: E731
        readme, setup = flat(PAGES["README.md"]), flat(PAGES["docs/SETUP.md"])
        amd = flat((ROOT / "docs" / "AMD_SAGUARO.md").read_text(encoding="utf-8"))
        self.assertNotIn("Hy4-Flash and Inkling 2 Mini on NVIDIA under Linux", readme)
        self.assertIn("one AMD RDNA 3 / 4 GPU of 12 GB or more", readme)
        self.assertIn("16-23 GB AMD RDNA 3 / 4 GPU", readme)
        self.assertIn("16-23 GB RDNA 3 / 4", amd)
        self.assertNotIn("NVIDIA Tesla T4 / RTX 20 and newer with 16 GB or more", setup)
        self.assertIn("one AMD RDNA 3 / 4 GPU of 12 GB or more", setup)
        self.assertIn("16-23 GB AMD RDNA 3 / 4 GPU", setup)
        self.assertNotIn("(20-40 minutes plus the download)", readme)
        src = (ROOT / "saguaro.py").read_text(encoding="utf-8")
        head = flat(src[:src.index(chr(34) * 3, src.index(chr(34) * 3) + 3)])
        self.assertIn("one AMD RDNA 3 / 4 GPU of 12 GB or more", head)
        md = flat(src[src.index('ap.add_argument("--model-dir"'):][:900])
        self.assertIn("{IK2_SHOWN}", md)
        pull = flat((ROOT / "serve" / "model_pull.py").read_text(encoding="utf-8"))
        self.assertNotIn("that holds every expert", pull)
        ik2 = flat(PAGES["docs/INKLING2-MINI.md"])
        for dec, read in IK2F.MEASURED_AMD.values():
            self.assertIn(f"about {dec} tok/s", ik2)
            self.assertIn(f"about {read}", ik2)

    def test_limits_and_licenses_are_said(self):
        for name in ("docs/HY4-FLASH.md", "docs/INKLING2-MINI.md"):
            text = PAGES[name]
            for part in ("One conversation at a time", "## Install", "## Speed", "## Limits"):
                self.assertIn(part, text, name)
        self.assertIn("Tencent Hy License", PAGES["docs/HY4-FLASH.md"])
        self.assertIn("LICENSE.hy4-flash", PAGES["docs/HY4-FLASH.md"])
        self.assertTrue((ROOT / "LICENSE.hy4-flash").is_file())
        self.assertIn("Apache 2.0", PAGES["docs/INKLING2-MINI.md"])
        self.assertIn("HY4_VISION_F16=1", PAGES["docs/HY4-FLASH.md"])   # (pictures: measured on T4s, the FP16 path)
        self.assertIn("LICENSE.hy4-flash", PAGES["NOTICE.md"])

    def test_links_resolve(self):
        for name, text in PAGES.items():
            base = (ROOT / name).parent
            for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
                if "://" in target or target.startswith("mailto:"):
                    continue
                if any(k in target for k in ("HY4", "INKLING", "hy4", "MODELS.md", "README.md")):
                    self.assertTrue((base / target).exists(), f"{name}: {target}")

    def test_no_upstream_program_names(self):
        """saguaro-hy4 / saguaro-ik2 everywhere a user reads; upstream's names only where the lineage is told."""
        for name, text in PAGES.items():
            self.assertNotIn("maya-hy4", text.replace("maya-hy4-flash-maya-s.json", ""), name)
            self.assertNotIn("maya-ik2", text, name)
        for name in ("docs/INKLING2.md", "tools/ik2/README.md", "tools/ik2/download.json"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("maya-vision", text, name)
            self.assertLessEqual(text.count("maya-ik2"), 1, name)           # (INKLING2.md: "upstream's maya-ik2")
        for src in ("saguaro.py", "serve/models.py", "serve/model_pull.py", "tools/hy4_fit.py", "tools/ik2_fit.py"):
            text = (ROOT / src).read_text(encoding="utf-8")
            self.assertNotIn("maya-ik2", text, src)
            self.assertNotIn("build/maya-hy4", text, src)


if __name__ == "__main__":
    unittest.main()
