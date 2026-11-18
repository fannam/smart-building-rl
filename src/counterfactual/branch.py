"""Simulator branching for paired counterfactual experiments (Phase 2B, offline only).

Why plain `copy.deepcopy(env)` is not enough
--------------------------------------------
1. `FiniteDifferenceSolver._factorization` (scipy SuperLU) cannot be pickled or
   deep-copied. It is a read-only cache of a constant matrix (refactorized only if
   the wind convection coefficient changes, which no shipped weather model does),
   so branches may share it (`memo[id(factorization)] = factorization`).
2. The stochastic convection kernel draws from *global* random streams: numba's
   internal `np.random` state (the jitted kernel) and CPython's `random` (the
   p=1 path). Global streams are not part of the env object graph, so deepcopy
   does not copy them, and two branches stepped one after another consume
   *different* random numbers. This is the cause of the Phase 0 copy divergence
   (2.7e-4 .. 1.3e-2), not the shared factorization.

`RNGState` snapshots every global stream (numba np/py, NumPy legacy global,
CPython `random`). Restoring it before stepping a branch gives that branch the same
random numbers the original would have drawn (common random numbers across arms).
Per-object generators (e.g. occupancy `RandomState`) are inside the env graph and
are copied by deepcopy.

Nothing here modifies the simulator; it is only driven from outside.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RNGState:
    numba_np: Any
    numba_py: Any
    numpy_global: Any
    python: Any

    @classmethod
    def capture(cls) -> "RNGState":
        from numba import _helperlib
        return cls(numba_np=_helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr()),
                   numba_py=_helperlib.rnd_get_state(_helperlib.rnd_get_py_state_ptr()),
                   numpy_global=np.random.get_state(), python=random.getstate())

    def restore(self) -> None:
        from numba import _helperlib
        _helperlib.rnd_set_state(_helperlib.rnd_get_np_state_ptr(), self.numba_np)
        _helperlib.rnd_set_state(_helperlib.rnd_get_py_state_ptr(), self.numba_py)
        np.random.set_state(self.numpy_global)
        random.setstate(self.python)

    def equals(self, other: "RNGState") -> bool:
        a, b = self.numba_np, other.numba_np
        same_numba = a[0] == b[0] and list(a[1]) == list(b[1])
        c, d = self.numba_py, other.numba_py
        same_numba_py = c[0] == d[0] and list(c[1]) == list(d[1])
        g, h = self.numpy_global, other.numpy_global
        same_np = g[0] == h[0] and np.array_equal(g[1], h[1]) and tuple(g[2:]) == tuple(h[2:])
        return bool(same_numba and same_numba_py and same_np and self.python == other.python)


def _solvers(env):
    """FiniteDifferenceSolver instances reachable from this env (no global gc scan)."""
    from smart_buildings.smart_control.simulator.finite_difference_solver import FiniteDifferenceSolver
    found, seen, stack = [], set(), [env]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, FiniteDifferenceSolver):
            found.append(o)
            continue
        if isinstance(o, (str, bytes, int, float, bool, type(None), np.ndarray)):
            continue
        if isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)
        elif hasattr(o, "__dict__"):
            stack.extend(vars(o).values())
    return found


def env_copy(env):
    """deepcopy sharing the read-only SuperLU factorization(s)."""
    memo = {}
    for s in _solvers(env):
        f = getattr(s, "_factorization", None)
        if f is not None:
            memo[id(f)] = f
    return copy.deepcopy(env, memo)


@dataclass
class Branch:
    """A frozen simulator state: env copy + global RNG state at the branch point."""
    env: Any
    rng: RNGState

    @classmethod
    def take(cls, env) -> "Branch":
        return cls(env=env_copy(env), rng=RNGState.capture())

    def spawn(self):
        """A fresh, independent env positioned at the branch point; global RNG restored."""
        e = env_copy(self.env)
        self.rng.restore()
        return e
