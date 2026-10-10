"""Format-neutral report model: the writers (PDF, Word, Excel) all render this."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Col:
    name: str
    fmt: str = "text"          # a fmt.py code
    align: str = "left"        # left | right | center
    weight: float = 1.0        # relative width


@dataclass
class Table:
    title: str
    cols: list[Col]
    rows: list[list[Any]]
    colors: list[str | None] | None = None   # optional per-row swatch colour ("#rrggbb")
    note: str = ""
    sheet: str = ""                           # Excel sheet name (empty = not exported as its own sheet)
    total_row: bool = False                   # sum the numeric columns marked in ``sum_cols`` in Excel
    sum_cols: tuple[int, ...] = ()
    raw: list[list[Any]] | None = None        # (label, value, format code, note) rows, for Excel numeric cells


@dataclass
class Figure:
    path: Path
    caption: str


@dataclass
class Block:
    kind: str                  # "p" | "bullets" | "table" | "figure" | "callout" | "kv"
    text: str = ""
    items: list[Any] = field(default_factory=list)
    table: Table | None = None
    figure: Figure | None = None
    level: str = "info"        # callout level: info | warn


def para(text: str) -> Block:
    return Block("p", text=text)


def bullets(items: list[str]) -> Block:
    return Block("bullets", items=list(items))


def table_block(t: Table) -> Block:
    return Block("table", table=t)


def figure_block(path: Path, caption: str) -> Block:
    return Block("figure", figure=Figure(Path(path), caption))


def callout(text: str, level: str = "info") -> Block:
    return Block("callout", text=text, level=level)


def kv_block(pairs: list[tuple[str, str]]) -> Block:
    return Block("kv", items=list(pairs))


@dataclass
class Section:
    title: str
    blocks: list[Block] = field(default_factory=list)
    page_break: bool = False


@dataclass
class Report:
    title: str
    subtitle: str
    pipeline: str
    meta: list[tuple[str, str]]                     # shown on the cover
    kpis: list[tuple[str, str, str]]                # label, value, sub-label
    sections: list[Section]
    warnings: list[str] = field(default_factory=list)
    cover_image: Path | None = None
    generated: str = ""
    version: str = ""
    # data for the Excel workbook
    timeseries: list[dict[str, Any]] = field(default_factory=list)
    feature_tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)   # vector layer -> attribute rows (capped)
    feature_totals: dict[str, int] = field(default_factory=dict)                    # vector layer -> true feature count
    charts: list[dict[str, Any]] = field(default_factory=list)                      # native Excel chart hints
