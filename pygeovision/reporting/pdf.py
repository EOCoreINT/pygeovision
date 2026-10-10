"""PDF report (reportlab)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from pygeovision.reporting import fmt as F
from pygeovision.reporting.model import Block, Report, Table
from pygeovision.reporting.palettes import ACCENT, ACCENT_SOFT, INK, MUTED, WARN, WARN_SOFT

_FONT = "DejaVu"
_registered = False


def _register_fonts() -> None:
    """DejaVu Sans ships with matplotlib, so every Unicode symbol used in the text renders."""
    global _registered
    if _registered:
        return
    import matplotlib
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    files = {"": "DejaVuSans.ttf", "-Bold": "DejaVuSans-Bold.ttf", "-Oblique": "DejaVuSans-Oblique.ttf", "-BoldOblique": "DejaVuSans-BoldOblique.ttf"}
    for suffix, fn in files.items():
        pdfmetrics.registerFont(TTFont(f"{_FONT}{suffix}", str(d / fn)))
    addMapping(_FONT, 0, 0, _FONT)
    addMapping(_FONT, 1, 0, f"{_FONT}-Bold")
    addMapping(_FONT, 0, 1, f"{_FONT}-Oblique")
    addMapping(_FONT, 1, 1, f"{_FONT}-BoldOblique")
    _registered = True


def _esc(text: Any) -> str:
    return escape(str(text)).replace("\n", "<br/>")


def write_pdf(report: Report, path: Path) -> Path:
    _register_fonts()
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        Image,
        KeepTogether,
        NextPageTemplate,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        TableStyle,
    )
    from reportlab.platypus import (
        Table as RLTable,
    )

    accent, ink, muted = colors.HexColor(ACCENT), colors.HexColor(INK), colors.HexColor(MUTED)
    page_w, page_h = A4
    margin = 18 * mm
    avail = page_w - 2 * margin

    body = ParagraphStyle("body", fontName=_FONT, fontSize=9.2, leading=13.4, textColor=ink, alignment=TA_LEFT, spaceAfter=4)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10.5, textColor=muted, spaceAfter=0)
    cap = ParagraphStyle("cap", parent=small, fontName=f"{_FONT}-Oblique", spaceBefore=3, spaceAfter=8)
    h1 = ParagraphStyle("h1", fontName=f"{_FONT}-Bold", fontSize=15, leading=19, textColor=accent, spaceBefore=12, spaceAfter=5, keepWithNext=1)
    h2 = ParagraphStyle("h2", fontName=f"{_FONT}-Bold", fontSize=10.5, leading=14, textColor=ink, spaceBefore=8, spaceAfter=3, keepWithNext=1)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.2, leading=10.6, spaceAfter=0)
    cell_r = ParagraphStyle("cell_r", parent=cell, alignment=2)
    cell_h = ParagraphStyle("cell_h", parent=cell, fontName=f"{_FONT}-Bold", textColor=colors.white)
    cell_hr = ParagraphStyle("cell_hr", parent=cell_h, alignment=2)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)
    kpi_v = ParagraphStyle("kpi_v", fontName=f"{_FONT}-Bold", fontSize=15, leading=18, textColor=accent)
    kpi_l = ParagraphStyle("kpi_l", parent=small, fontSize=7.6, leading=9.6)

    def table_flowable(t: Table):
        cols = t.cols
        has_sw = t.colors is not None
        header = ([Paragraph("", cell_h)] if has_sw else []) + [Paragraph(_esc(c.name), cell_hr if c.align == "right" else cell_h) for c in cols]
        if all(len(r) > 2 for r in t.rows) and False:
            pass
        data = [header]
        for r in t.rows:
            row = ([""] if has_sw else [])
            for c, v in zip(cols, r):
                txt = F.fmt_value(v, c.fmt) if c.fmt != "text" else (F.NA if v is None else str(v))
                row.append(Paragraph(_esc(txt), cell_r if c.align == "right" else cell))
            data.append(row)
        total_w = sum(c.weight for c in cols)
        sw_w = 5 * mm if has_sw else 0
        widths = ([sw_w] if has_sw else []) + [(avail - sw_w) * c.weight / total_w for c in cols]
        style = [("BACKGROUND", (0, 0), (-1, 0), accent), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5dbe1")),
                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                 ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f7f9")))
            if has_sw and t.colors and t.colors[i - 1]:
                style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(t.colors[i - 1])))
                style.append(("BOX", (0, i), (0, i), 0.4, ink))
        tbl = RLTable(data, colWidths=widths, repeatRows=1)
        tbl.setStyle(TableStyle(style))
        tail = Paragraph(_esc(t.note), cap) if t.note else Spacer(1, 6)
        if len(t.rows) <= 18:
            return [KeepTogether([Paragraph(_esc(t.title), h2), tbl, tail])]
        return [Paragraph(_esc(t.title), h2), tbl, tail]

    def callout_flowable(text: str, level: str):
        bg, bar = (WARN_SOFT, WARN) if level == "warn" else (ACCENT_SOFT, ACCENT)
        tbl = RLTable([[Paragraph(_esc(text), body)]], colWidths=[avail])
        tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)), ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(bar)),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return [tbl, Spacer(1, 6)]

    def kv_flowable(pairs):
        data = [[Paragraph(_esc(k), small), Paragraph(_esc(v), cell)] for k, v in pairs]
        tbl = RLTable(data, colWidths=[avail * 0.34, avail * 0.66])
        tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5dbe1")),
                                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
        return [tbl, Spacer(1, 8)]

    def image_flowable(p: Path, caption: str, max_h: float = 150 * mm):
        from PIL import Image as PILImage
        with PILImage.open(p) as im:
            iw, ih = im.size
        w = avail
        h = w * ih / iw
        if h > max_h:
            h, w = max_h, max_h * iw / ih
        return [KeepTogether([Image(str(p), width=w, height=h), Paragraph(_esc(caption), cap)])]

    def block_flowables(b: Block):
        if b.kind == "p":
            return [Paragraph(_esc(b.text), body)]
        if b.kind == "bullets":
            return [Paragraph(_esc(i), bullet, bulletText="•") for i in b.items] + [Spacer(1, 4)]
        if b.kind == "table" and b.table is not None:
            return table_flowable(b.table)
        if b.kind == "figure" and b.figure is not None and b.figure.path.exists():
            return image_flowable(b.figure.path, b.figure.caption)
        if b.kind == "callout":
            return callout_flowable(b.text, b.level)
        if b.kind == "kv":
            return kv_flowable(b.items)
        return []

    class _Numbered(rl_canvas.Canvas):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._saved: list[dict] = []

        def showPage(self):
            self._saved.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            n = len(self._saved)
            for state in self._saved:
                self.__dict__.update(state)
                if self._pageNumber > 1:
                    self.setFont(_FONT, 7.5)
                    self.setFillColor(muted)
                    self.drawRightString(page_w - margin, 10 * mm, f"Page {self._pageNumber} of {n}")
                super().showPage()
            super().save()

    def on_body(canv, doc):
        canv.saveState()
        canv.setStrokeColor(colors.HexColor("#d5dbe1"))
        canv.setLineWidth(0.5)
        canv.line(margin, page_h - 14 * mm, page_w - margin, page_h - 14 * mm)
        canv.setFont(_FONT, 7.5)
        canv.setFillColor(muted)
        canv.drawString(margin, page_h - 11.5 * mm, report.title)
        canv.drawRightString(page_w - margin, page_h - 11.5 * mm, f"PyGeoVision {report.version}")
        canv.drawString(margin, 10 * mm, f"Generated {report.generated}")
        canv.restoreState()

    def on_cover(canv, doc):
        canv.saveState()
        canv.setFillColor(accent)
        canv.rect(0, page_h - 62 * mm, page_w, 62 * mm, stroke=0, fill=1)
        canv.setFillColor(colors.HexColor("#8fb8d3"))
        canv.rect(0, page_h - 64 * mm, page_w, 2 * mm, stroke=0, fill=1)
        canv.setFillColor(colors.white)
        canv.setFont(f"{_FONT}-Bold", 9)
        canv.drawString(margin, page_h - 14 * mm, "PYGEOVISION  ·  GEOSPATIAL ANALYSIS REPORT")
        from reportlab.lib.utils import simpleSplit
        y = page_h - 28 * mm
        canv.setFont(f"{_FONT}-Bold", 24)
        for line in simpleSplit(report.title, f"{_FONT}-Bold", 24, avail)[:2]:
            canv.drawString(margin, y, line)
            y -= 29
        canv.setFont(_FONT, 11)
        canv.setFillColor(colors.HexColor("#dbe9f3"))
        for line in simpleSplit(report.subtitle, _FONT, 11, avail)[:2]:
            canv.drawString(margin, y - 2, line)
            y -= 15
        canv.setFillColor(muted)
        canv.setFont(_FONT, 7.5)
        canv.drawString(margin, 10 * mm, f"Generated {report.generated}")
        canv.restoreState()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=20 * mm, bottomMargin=18 * mm,
                          title=report.title, author="PyGeoVision", subject=report.subtitle, creator=f"PyGeoVision {report.version}")
    cover_frame = Frame(margin, 16 * mm, avail, page_h - 62 * mm - 16 * mm - 10 * mm, id="cover", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    body_frame = Frame(margin, 18 * mm, avail, page_h - 20 * mm - 18 * mm, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate("cover", [cover_frame], onPage=on_cover), PageTemplate("body", [body_frame], onPage=on_body)])

    story: list[Any] = []
    if report.kpis:
        cells = [[Paragraph(_esc(v), kpi_v), Paragraph(_esc(l), kpi_l)] for l, v, _ in report.kpis[:4]]
        n = len(cells)
        data = [[c for c in cells]]
        k = RLTable(data, colWidths=[avail / n] * n)
        k.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(ACCENT_SOFT)), ("LINEBEFORE", (0, 0), (-1, -1), 2, accent),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story += [k, Spacer(1, 4)]
    story += [NextPageTemplate("body")]

    first = True
    for sec in report.sections:
        if sec.page_break and not first:
            story.append(PageBreak())
        first = False
        head = Paragraph(_esc(sec.title), h1)
        flows = [f for b in sec.blocks for f in block_flowables(b)]
        if flows:                       # a heading never sits alone at the bottom of a page
            story.append(KeepTogether([head, flows[0]]))
            story.extend(flows[1:])
        else:
            story.append(head)
    if report.warnings:
        story.append(Paragraph("Notes on this run", h1))
        story.append(Paragraph("Parts of the deliverables that could not be produced, or were changed, and why:", body))
        story.append(Paragraph("", body))
        for w in report.warnings:
            story.append(Paragraph(_esc(w), bullet, bulletText="•"))
    doc.build(story, canvasmaker=_Numbered)
    return out
