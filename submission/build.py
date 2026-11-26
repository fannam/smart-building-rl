"""Build the Kaggle submission: one self-contained inference module plus the notebook that embeds it.

    python -m submission.build                      # reference controller (M1)
    python -m submission.build --params best.json   # TunablePolicy with tuned parameters (M3+)

Writes
- submission/inference.py: controller source + action sanitizer + SafePolicy wrapper (NumPy only). It is also
  importable locally, so `python -m sbx.evaluate --policies submission.inference:SafePolicy ...` tests exactly
  what is submitted.
- submission/notebook.ipynb: attach the competition data, paste inference.py, start the gateway.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from sbx import env as sbx_env

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONTROLLER = ROOT / "sbx" / "controller" / "reference.py"
CONTROLLER_CLASS = "WideHotStorageReservePolicy"
TUNABLE = ROOT / "sbx" / "controller" / "tunable.py"
TUNABLE_BASE = "SmoothRefinedRecoveryPolicy"

WRAPPER = '''

# ---------------------------------------------------------------------------------------------------------------
# Submission safety layer (ours). One invalid action fails the whole submission, so every action goes through
# Sanitizer, and an exception inside the controller repeats the last valid action instead of propagating.
# ---------------------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

{sanitizer}

class SafePolicy({controller}):
    """{controller} behind the sanitizer, with a last-valid-action fallback."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._sanitize = Sanitizer(action_spec)
        self._last_action = None
        self.controller_errors = 0

    def _neutral_action(self):
        spec = self._sanitize.spec
        return {{k: (np.asarray(spec[k].low, np.float64) + np.asarray(spec[k].high, np.float64)) / 2.0
                for k in ("continuous_action", "discrete_action")}}

    def get_action(self, obs):
        try:
            action = super().get_action(obs)
        except Exception:
            self.controller_errors += 1
            action = self._last_action if self._last_action is not None else self._neutral_action()
        action = self._sanitize(action)
        self._last_action = action
        return action
'''

SETUP_CELL = """import sys
from pathlib import Path
candidates = [Path('/kaggle/input/competitions/neurips-2026-smart-buildings'), Path('/kaggle/input/neurips-2026-smart-buildings')]
DATA_ROOT = next((path for path in candidates if (path / 'smart_buildings').exists()), None)
if DATA_ROOT is None:
    raise FileNotFoundError('Attach the Smart Buildings competition data')
sys.path.insert(0, str(DATA_ROOT))
sys.path.insert(0, str(DATA_ROOT / 'kaggle_evaluation'))"""

SERVER_CELL = """from kaggle_evaluation import smart_buildings_inference_server

class SubmissionPolicy(SafePolicy, smart_buildings_inference_server.Policy):
    pass

smart_buildings_inference_server.SmartBuildingsInferenceServer(SubmissionPolicy).run(
    competition_data_folder=str(DATA_ROOT))"""


def _tunable_source(params: dict) -> str:
    """tunable.py without its package imports, plus a TunedPolicy fixing the parameters."""
    drop = ("from __future__ import", "from sbx.controller.reference import")
    body = "".join(line for line in TUNABLE.read_text().splitlines(keepends=True) if not line.startswith(drop))
    fixed = json.dumps(params, indent=4, sort_keys=True)
    return (f"\n\n# {'-' * 111}\n# Tunable wrapper (ours): sbx/controller/tunable.py\n# {'-' * 111}\n{body.rstrip()}\n\n"
            f"TUNED_PARAMS = {fixed}\n\n\nclass TunedPolicy(TunablePolicy):\n"
            f"    def __init__(self, building_meta_data, action_spec=None):\n"
            f"        super().__init__(building_meta_data, action_spec, TUNED_PARAMS)\n")


def load_params(path: Path | None) -> dict | None:
    if path is None:
        return None
    params = json.loads(Path(path).read_text())
    params = params.get("params", params)
    from sbx.controller.tunable import PARAMS
    unknown = set(params) - set(PARAMS)
    if unknown:
        raise ValueError(f"unknown parameters in {path}: {sorted(unknown)}")
    return {k: float(v) for k, v in params.items()}


def build_inference(params: dict | None = None) -> str:
    sanitizer = inspect.getsource(sbx_env.Sanitizer)
    code, controller = CONTROLLER.read_text().rstrip() + "\n", CONTROLLER_CLASS
    if params is not None:
        code, controller = code + _tunable_source(params), "TunedPolicy"
    return code + WRAPPER.format(sanitizer=sanitizer.rstrip(), controller=controller)


def cell(kind: str, src: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, default=None, help="TunablePolicy parameters (JSON); default: M1 reference")
    a = ap.parse_args(argv)
    params = load_params(a.params)
    code = build_inference(params)
    (HERE / "inference.py").write_text(code)
    sha = hashlib.sha256(code.encode()).hexdigest()
    what = CONTROLLER_CLASS if params is None else f"TunablePolicy over {TUNABLE_BASE} ({len(params)} tuned parameters)"
    header = (f"# Smart Buildings submission: `{what}` + safety layer\n\n"
              "The controller is a public analytical controller written by other competitors (Daniel Wu, jazivxt, "
              "penguin069 and the author of the *Hot Storage340 Nov14* notebook), embedded unmodified; see the "
              "attribution comments at the top of the code cell. The tunable wrapper, sanitizer and fallback "
              f"wrapper after it are ours.\n\nInference code SHA256: `{sha}`.")
    nb = {"cells": [cell("markdown", header), cell("code", SETUP_CELL), cell("code", code),
                    cell("code", SERVER_CELL)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (HERE / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    print(f"inference.py {len(code)} chars, sha256 {sha[:16]}…  -> {HERE / 'notebook.ipynb'}")


if __name__ == "__main__":
    main()
