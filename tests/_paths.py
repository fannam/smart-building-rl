"""Test helpers: import path and optional real-data locations."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
SAMPLE = ROOT / "data" / "phase1_sample"
HAVE_SAMPLE = (SAMPLE / "manifest.jsonl").exists()
