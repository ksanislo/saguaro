"""The QFN sub-build (engines/qfn, an ExternalProject) is always made with ninja: it must get a ninja, never this build's
make program - a build folder made with make (an install from before 1.2.1) failed its update ("Ninja 1.3 or newer")."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class QfnSubBuild(unittest.TestCase):
    def test_the_sub_build_gets_a_ninja(self):
        text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}", text)       # (make's path when this build is make's)
        self.assertIn("find_program(SAGUARO_QFN_NINJA NAMES ninja ninja-build", text)
        self.assertIn('if(CMAKE_GENERATOR MATCHES "Ninja")', text)                 # (its own ninja: the setup's, off the PATH)
        self.assertEqual(text.count("${_qfn_make_arg}"), 1)                        # (CUDA) ...
        self.assertEqual(text.count("CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release ${_qfn_make} -DCMAKE_HIP_COMPILER"), 1)   # (HIP)


if __name__ == "__main__":
    unittest.main()
