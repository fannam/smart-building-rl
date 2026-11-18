"""Run scripts/validate_phase1.py unchanged, but with its data and results
directories redirected into scratch/phase1_audit/out (no existing outputs touched).
The full-horizon baseline episode is read (read-only) from data/phase1_validation/full_baseline."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import validate_phase1 as v
OUT = ROOT / "scratch/phase1_5/validation"
v.OUT = OUT / "validation_data"
v.RESULTS = OUT / "validation_results"
sys.argv = [sys.argv[0], "--full-baseline", str(ROOT / "data/phase1_validation/full_baseline")]
v.main()
