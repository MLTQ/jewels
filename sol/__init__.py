"""Research spikes for structured, editable spacetime-jewel video."""

from sol.autoencoder import StructuredJewelAutoencoder
from sol.geometry import Parallelepiped, translate_selected
from sol.inpaint import masked_flow_inpaint
from sol.latent_prior import RasterFlowPrior
from sol.token_grid import GridSpec, OccupancyGrid

__all__ = [
    "GridSpec",
    "OccupancyGrid",
    "Parallelepiped",
    "RasterFlowPrior",
    "StructuredJewelAutoencoder",
    "masked_flow_inpaint",
    "translate_selected",
]
