"""Registry of disaster analyzers (Phase 6B.1).

Adding a hazard later (landslide, earthquake, wildfire, cyclone, avalanche,
industrial) means adding one module + one ``register()`` call; the job
pipeline does not change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from .base import ArtifactBundle, HazardAnalyzer
from .flood import FloodAnalyzer

_ANALYZERS: Dict[str, Type[HazardAnalyzer]] = {
    FloodAnalyzer.name: FloodAnalyzer,
}


def register(analyzer_cls: Type[HazardAnalyzer]) -> Type[HazardAnalyzer]:
    """Register an analyzer class (usable as a decorator)."""
    _ANALYZERS[analyzer_cls.name] = analyzer_cls
    return analyzer_cls


def available() -> List[str]:
    """Names of the hazard analyzers that are implemented."""
    return sorted(_ANALYZERS)


def get(name: str) -> Type[HazardAnalyzer]:
    """Return the analyzer class registered under ``name``."""
    try:
        return _ANALYZERS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown hazard analyzer '{name}'. Available: {available()}"
        ) from exc


def run(name: str, bundle: ArtifactBundle, **kwargs: Any) -> Dict[str, Any]:
    """Instantiate and run one analyzer against ``bundle``."""
    return get(name)(**kwargs).analyze(bundle)
