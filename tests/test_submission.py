import numpy as np
from gymnasium.spaces import Box

from submission import build

SPEC = {
    "continuous_action": Box(-1.0, 1.0, (4,), np.float32),
    "discrete_action": Box(0, 1, (2,), np.int32),
}


def _safe_policy_class(controller_src: str):
    """Exec the submission wrapper around a stand-in controller, the way inference.py is assembled."""
    ns: dict = {}
    code = ("import numpy as np\n" + controller_src
            + build.WRAPPER.format(sanitizer=build.inspect.getsource(build.sbx_env.Sanitizer).rstrip(),
                                   controller="Controller"))
    exec(compile(code, "<inference>", "exec"), ns)
    return ns["SafePolicy"]


CONTROLLER = '''
class Controller:
    def __init__(self, meta, spec):
        self.calls = 0
    def get_action(self, obs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")
        if self.calls == 3:
            return {"continuous_action": [np.nan, 5.0, -5.0, 0.1], "discrete_action": [0.7, 3]}
        return {"continuous_action": np.full(4, 0.25), "discrete_action": [1, 0]}
'''


def test_safe_policy_falls_back_and_sanitizes():
    p = _safe_policy_class(CONTROLLER)({}, SPEC)
    a1 = p.get_action(None)
    a2 = p.get_action(None)  # controller raises -> repeat the last valid action
    a3 = p.get_action(None)  # garbage -> sanitized
    assert np.array_equal(a2["continuous_action"], a1["continuous_action"])
    assert p.controller_errors == 1
    assert np.all(np.isfinite(a3["continuous_action"])) and np.all(np.abs(a3["continuous_action"]) <= 1)
    assert a3["discrete_action"].tolist() == [1, 1] and a3["discrete_action"].dtype == np.int32


def test_safe_policy_first_step_error_gives_neutral_action():
    src = "class Controller:\n    def __init__(self, m, s): pass\n    def get_action(self, obs): raise ValueError\n"
    a = _safe_policy_class(src)({}, SPEC).get_action(None)
    assert np.allclose(a["continuous_action"], 0.0) and a["discrete_action"].tolist() == [0, 0]


def test_built_inference_is_current():
    """submission/inference.py must be regenerated after changing the controller or the sanitizer."""
    assert (build.HERE / "inference.py").read_text() == build.build_inference()
