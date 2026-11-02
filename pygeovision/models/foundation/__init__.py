"""Real, precomputed-embedding foundation model wrappers (TESSERA, AlphaEarth, etc.)."""

from pygeovision.models.foundation.alphaearth_geo import AlphaEarthGeo
from pygeovision.models.foundation.tessera import TesseraGeo

__all__ = ["TesseraGeo", "AlphaEarthGeo"]
