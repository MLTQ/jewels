"""Research spikes for structured, editable spacetime-jewel video."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GridSpec",
    "OccupancyGrid",
    "Parallelepiped",
    "RasterFlowPrior",
    "StructuredJewelAutoencoder",
    "masked_flow_inpaint",
    "translate_selected",
]

_EXPORTS = {
    "GridSpec": ("sol.token_grid", "GridSpec"),
    "OccupancyGrid": ("sol.token_grid", "OccupancyGrid"),
    "Parallelepiped": ("sol.geometry", "Parallelepiped"),
    "RasterFlowPrior": ("sol.latent_prior", "RasterFlowPrior"),
    "StructuredJewelAutoencoder": ("sol.autoencoder", "StructuredJewelAutoencoder"),
    "masked_flow_inpaint": ("sol.inpaint", "masked_flow_inpaint"),
    "translate_selected": ("sol.geometry", "translate_selected"),
}


def __getattr__(name: str) -> Any:
    """Resolve public training-stack exports only when a caller requests them."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
