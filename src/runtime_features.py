"""Runtime feature boundary: what a deployed policy may consume.

Backed by `configs/runtime_feature_schema.json` (whitelist). Use
`validate_runtime_features()` on every model-input feature list, and
`public_metadata()` whenever scenario metadata is handed to a model.

Feature syntax:  '<source>'  or  '<source>:<sub>'
  * obs:<field>                    an observation field (must exist in the building's field list
                                   when `field_names` is given)
  * action_continuous, action_discrete, effective_action[:<sub>]
  * meta:<key>[.<subkey>...]        a meta.json key path
  * anything else (zone_info, calendar:month, sim_timestamp, ...) is classified by its source
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = ROOT / "configs" / "runtime_feature_schema.json"


class PrivilegedFeatureError(ValueError):
    """A proposed runtime input is privileged, restricted or unknown."""


@dataclass(frozen=True)
class FeatureDecision:
    feature: str
    category: str
    allowed: bool
    reason: str


class RuntimeFeatureSchema:
    def __init__(self, schema: Mapping[str, Any]):
        self.schema = schema
        self.version = schema["schema_version"]
        self.categories = schema["categories"]
        self.sources = schema["sources"]
        meta = schema["meta"]
        self.meta_category = {}
        for cat in ("public_metadata", "privileged_episode", "identifier", "bookkeeping"):
            for k in meta.get(cat, []):
                self.meta_category[k] = cat

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SCHEMA_PATH) -> "RuntimeFeatureSchema":
        return cls(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------ classification
    def classify(self, feature: str, field_names: Sequence[str] | None = None,
                 allow_restricted: bool = False) -> FeatureDecision:
        if not isinstance(feature, str) or not feature:
            return FeatureDecision(str(feature), "unknown", False, "empty or non-string feature")
        source, _, sub = feature.partition(":")
        if source == "meta":
            if not sub:
                return FeatureDecision(feature, "unknown", False, "meta requires a key path")
            top = sub.split(".")[0]
            cat = self.meta_category.get(top)
            if cat is None:
                return FeatureDecision(feature, "unknown", False, f"meta key {top!r} not in whitelist")
            ok = cat == "public_metadata"
            return FeatureDecision(feature, cat, ok, "public metadata" if ok else f"meta key {top!r} is {cat}")
        spec = self.sources.get(source)
        if spec is None:
            return FeatureDecision(feature, "unknown", False, f"source {source!r} not in whitelist")
        cat = spec["category"]
        cat_spec = self.categories[cat]
        if source == "obs" and sub and field_names is not None and sub not in set(field_names):
            return FeatureDecision(feature, cat, False, f"obs field {sub!r} does not exist for this building")
        if cat_spec.get("runtime_input"):
            return FeatureDecision(feature, cat, True, cat)
        if cat == "restricted" and allow_restricted:
            return FeatureDecision(feature, cat, True, "restricted, explicitly allowed")
        return FeatureDecision(feature, cat, False, f"{source!r} is {cat}")

    def validate(self, features: Iterable[str], field_names: Sequence[str] | None = None,
                 allow_restricted: bool = False) -> list[FeatureDecision]:
        decisions = [self.classify(f, field_names, allow_restricted) for f in features]
        bad = [d for d in decisions if not d.allowed]
        if bad:
            raise PrivilegedFeatureError(
                "forbidden runtime input(s): " + "; ".join(f"{d.feature} ({d.reason})" for d in bad))
        return decisions

    def is_runtime_source(self, source: str, allow_restricted: bool = False) -> bool:
        return self.classify(source, allow_restricted=allow_restricted).allowed

    def privileged_target_sources(self) -> list[str]:
        return [s for s, v in self.sources.items() if v["category"] == "privileged_target"]

    # ------------------------------------------------------------ metadata filter
    def public_metadata(self, scenario_meta: Mapping[str, Any]) -> dict[str, Any]:
        """Only whitelisted public keys; unknown keys are dropped (and reported)."""
        return {k: v for k, v in scenario_meta.items() if self.meta_category.get(k) == "public_metadata"}

    def unclassified_meta_keys(self, scenario_meta: Mapping[str, Any]) -> list[str]:
        return sorted(k for k in scenario_meta if k not in self.meta_category)


_DEFAULT: RuntimeFeatureSchema | None = None


def default_schema() -> RuntimeFeatureSchema:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RuntimeFeatureSchema.load()
    return _DEFAULT


def validate_runtime_features(features: Iterable[str], field_names: Sequence[str] | None = None,
                              allow_restricted: bool = False,
                              schema: RuntimeFeatureSchema | None = None) -> list[FeatureDecision]:
    """Raise PrivilegedFeatureError if any proposed runtime input is not whitelisted."""
    return (schema or default_schema()).validate(features, field_names, allow_restricted)


def public_metadata(scenario_meta: Mapping[str, Any], schema: RuntimeFeatureSchema | None = None) -> dict[str, Any]:
    return (schema or default_schema()).public_metadata(scenario_meta)
