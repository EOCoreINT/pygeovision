"""Colours used by the maps, charts and vector attribute tables.

Class colours are fixed per product so a map made today looks like a map made
next year. The WorldCover palette is ESA's own; the rest were chosen to stay
distinguishable for common colour-vision deficiencies (no red/green pairs
carrying the only difference).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassDef:
    """One value of a class raster: its number, a human name and a display colour."""
    value: int
    name: str
    color: str  # "#rrggbb"


# ESA WorldCover legend (official product colours)
WORLDCOVER = (
    ClassDef(10, "Tree cover", "#006400"),
    ClassDef(20, "Shrubland", "#ffbb22"),
    ClassDef(30, "Grassland", "#ffff4c"),
    ClassDef(40, "Cropland", "#f096ff"),
    ClassDef(50, "Built-up", "#fa0000"),
    ClassDef(60, "Bare / sparse vegetation", "#b4b4b4"),
    ClassDef(70, "Snow and ice", "#f0f0f0"),
    ClassDef(80, "Permanent water", "#0064c8"),
    ClassDef(90, "Herbaceous wetland", "#0096a0"),
    ClassDef(95, "Mangroves", "#00cf75"),
    ClassDef(100, "Moss and lichen", "#fae6a0"),
)

# Google Dynamic World legend
DYNAMIC_WORLD = (
    ClassDef(0, "Water", "#419bdf"),
    ClassDef(1, "Trees", "#397d49"),
    ClassDef(2, "Grass", "#88b053"),
    ClassDef(3, "Flooded vegetation", "#7a87c6"),
    ClassDef(4, "Crops", "#e49635"),
    ClassDef(5, "Shrub and scrub", "#dfc35a"),
    ClassDef(6, "Built area", "#c4281b"),
    ClassDef(7, "Bare ground", "#a59b8f"),
    ClassDef(8, "Snow and ice", "#b39fe1"),
)

# Okabe-Ito colour-blind-safe qualitative palette, for classes that have no
# standard legend (user-trained models).
CATEGORICAL = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999")

NO_CHANGE = "#d9dee3"       # "nothing happened here" -- deliberately quiet
NEUTRAL_BG = "#eef1f4"      # map background where there is no data
INK = "#1f2933"
MUTED = "#52606d"
ACCENT = "#1b4965"          # report headings and table headers
ACCENT_SOFT = "#e3edf4"
WARN = "#b45309"
WARN_SOFT = "#fdf3e1"


def categorical(index: int) -> str:
    """The ``index``-th qualitative colour (cycles)."""
    return CATEGORICAL[index % len(CATEGORICAL)]
