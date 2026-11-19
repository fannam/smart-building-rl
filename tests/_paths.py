"""Test helpers: import path and optional real-data locations."""
import os
import sys
from pathlib import Path

# Jupyter kernels (e.g. Kaggle) export MPLBACKEND=module://matplotlib_inline.backend_inline; subprocesses using
# another environment cannot import that backend, and the simulator imports matplotlib -> force a headless backend.
if os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
SAMPLE = ROOT / "data" / "phase1_sample"
HAVE_SAMPLE = (SAMPLE / "manifest.jsonl").exists()
