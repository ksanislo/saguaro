#!/usr/bin/env python3
"""tools/ik2/bench_model.py - bench.py on another model file:  bench_model.py --model FILE [bench.py's options]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import t4_check  # noqa: E402

i = sys.argv.index("--model")
bench.MODEL = t4_check.MODEL = sys.argv[i + 1]
del sys.argv[i:i + 2]
sys.exit(bench.main())
