# ruff: noqa: E702, N802, C408, E731
"""Draw the documentation's flow figures as animated, theme-aware SVG.

    python tools/make_flow_figures.py            # write docs/_static/img/flow-*.svg
    python tools/make_flow_figures.py --check    # exit 1 if a file differs from what would be written

Each figure is a row of numbered step cards joined by animated arrows, with an optional command pill under each card and optional
dashed side cards (an alternative way into a step). The figures are data below; the layout is computed, so a step can be added or
reworded without redrawing anything. They replace the ASCII diagrams the pages used to carry. Colours follow the system light/dark
setting; the motion stops when the reader asks for reduced motion.
"""
from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "_static" / "img"
W = 1040          # viewBox width
PAD = 18
GAP = 30
LINE_H = 15
MAX_CHARS = 23    # a line longer than this would not fit in a card of six

STYLE = """
  <style>
    :root{--card:#ffffff;--line:#d7dbe0;--title:#1c1e21;--sub:#5a5c63;--pill:#eef3f8;--pillt:#1b4f72;--flow:#1b6a9c;--soft:#f6f8fa;--shadow:rgba(16,24,40,.10)}
    @media (prefers-color-scheme:dark){:root{--card:#242526;--line:#3a3b3d;--title:#e8e8e8;--sub:#b0b3b8;--pill:#2f3a44;--pillt:#a9d6ee;--flow:#6cc4dd;--soft:#1d1e1f;--shadow:rgba(0,0,0,.45)}}
    text{font-family:Ubuntu,"Segoe UI",system-ui,sans-serif}
    .card{fill:var(--card);stroke:var(--line);stroke-width:1.2;filter:drop-shadow(0 2px 3px var(--shadow))}
    .side{fill:var(--soft);stroke:var(--flow);stroke-width:1.3;stroke-dasharray:5 4}
    .t{font-weight:600;font-size:14px;fill:var(--title)}
    .s{font-size:11.5px;fill:var(--sub)}
    .n{font-weight:700;font-size:12px;fill:#fff}
    .pill{fill:var(--pill)}
    .cmd{font-family:"Ubuntu Mono",ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;fill:var(--pillt)}
    .arrow{stroke:var(--flow);stroke-width:2;fill:none;stroke-dasharray:6 6;animation:dash 1.1s linear infinite}
    .head{fill:var(--flow)}
    .step{animation:rise .6s ease-out both}
    .q{fill:var(--card);stroke:var(--line);stroke-width:1.2;filter:drop-shadow(0 2px 3px var(--shadow))}
    .leaf{fill:var(--pill);stroke:var(--flow);stroke-width:1.2}
    .qa{fill:#1b6a9c}
    .mono{font-family:"Ubuntu Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;fill:var(--pillt);font-weight:600}
    .lab{font-size:10.5px;font-weight:700;fill:var(--flow)}
    .labbg{fill:var(--card);stroke:var(--flow);stroke-width:1}
    .band{fill:var(--card);stroke:var(--line);stroke-width:1.2;filter:drop-shadow(0 2px 3px var(--shadow))}
    .bt{font-weight:700;font-size:15px;fill:#fff}
    .bs{font-size:11.5px;fill:rgba(255,255,255,.86)}
    .quote{font-size:12px;font-style:italic;fill:var(--sub)}
    .a0{fill:#1b6a9c}.a1{fill:#2386b0}.a2{fill:#1f9bb0}.a3{fill:#23a58e}.a4{fill:#3aa66b}.a5{fill:#6aa84f}
    @keyframes dash{to{stroke-dashoffset:-12}}
    @keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
    @media (prefers-reduced-motion:reduce){.arrow{animation:none}.step{animation:none}}
  </style>"""


def _check_lines(fig: str, where: str, lines: list[str], limit: int = MAX_CHARS) -> None:
    for ln in lines:
        if len(ln) > limit:
            raise SystemExit(f"{fig}: '{ln}' ({where}) is {len(ln)} characters; the card fits {limit}")


def render(fig: str, alt: str, steps: list[dict], sides: list[dict] | None = None) -> str:
    n = len(steps)
    cw = (W - 2 * PAD - GAP * (n - 1)) / n
    for i, s in enumerate(steps):
        _check_lines(fig, f"step {i + 1}", s["lines"] + [s["title"]] if len(s["title"]) > 20 else s["lines"])
        if s.get("cmd") and len(s["cmd"]) > 22:
            raise SystemExit(f"{fig}: command '{s['cmd']}' is longer than 22 characters")
    top = 18
    body = max(len(s["lines"]) for s in steps)
    ch = 84 + LINE_H * body + (34 if any(s.get("cmd") for s in steps) else 12)
    midy = top + 24
    parts: list[str] = []
    for i, s in enumerate(steps):
        x = PAD + i * (cw + GAP)
        cx = x + cw / 2
        g = [f'<g class="step" style="animation-delay:{i * 0.12:.2f}s">']
        g.append(f'<rect class="card" x="{x:.1f}" y="{top}" width="{cw:.1f}" height="{ch}" rx="12"/>')
        g.append(f'<circle class="a{min(i, 5)}" cx="{cx:.1f}" cy="{midy}" r="13"/>')
        g.append(f'<text class="n" x="{cx:.1f}" y="{midy + 4.2}" text-anchor="middle">{i + 1}</text>')
        g.append(f'<text class="t" x="{cx:.1f}" y="{top + 66}" text-anchor="middle">{escape(s["title"])}</text>')
        for k, ln in enumerate(s["lines"]):
            g.append(f'<text class="s" x="{cx:.1f}" y="{top + 86 + k * LINE_H}" text-anchor="middle">{escape(ln)}</text>')
        if s.get("cmd"):
            pw = min(cw - 16, 7.0 * len(s["cmd"]) + 18)
            py = top + ch - 28
            g.append(f'<rect class="pill" x="{cx - pw / 2:.1f}" y="{py}" width="{pw:.1f}" height="20" rx="10"/>')
            g.append(f'<text class="cmd" x="{cx:.1f}" y="{py + 14}" text-anchor="middle">{escape(s["cmd"])}</text>')
        g.append("</g>")
        parts.append("\n".join(g))
        if i < n - 1:
            x1, x2 = x + cw + 3, x + cw + GAP - 7
            parts.append(f'<path class="arrow" d="M{x1:.1f} {midy} H{x2:.1f}"/>'
                         f'<path class="head" d="M{x2 + 1:.1f} {midy - 5} L{x2 + 8:.1f} {midy} L{x2 + 1:.1f} {midy + 5} Z"/>')
    height = top + ch + 16
    for sd in sides or []:
        _check_lines(fig, "side card", sd["lines"], 40 if sd.get("wide") else MAX_CHARS)
        j = sd["to"] - 1
        x = PAD + j * (cw + GAP)
        cx = x + cw / 2
        sw = max(cw, 2 * cw + GAP) if sd.get("wide") else cw
        sx = cx - sw / 2 if not sd.get("wide") else min(max(PAD, cx - sw / 2), W - PAD - sw)
        sy = top + ch + 34
        sh = 46 + LINE_H * len(sd["lines"])
        parts.append(f'<path class="arrow" d="M{cx:.1f} {sy - 2} V{top + ch + 6}"/>'
                     f'<path class="head" d="M{cx - 5:.1f} {top + ch + 7} L{cx:.1f} {top + ch} L{cx + 5:.1f} {top + ch + 7} Z"/>')
        parts.append(f'<g class="step" style="animation-delay:{(n + 1) * 0.12:.2f}s">'
                     f'<rect class="side" x="{sx:.1f}" y="{sy}" width="{sw:.1f}" height="{sh}" rx="12"/>'
                     f'<text class="t" x="{sx + sw / 2:.1f}" y="{sy + 24}" text-anchor="middle">{escape(sd["title"])}</text>'
                     + "".join(f'<text class="s" x="{sx + sw / 2:.1f}" y="{sy + 43 + k * LINE_H}" text-anchor="middle">{escape(ln)}</text>'
                               for k, ln in enumerate(sd["lines"])) + "</g>")
        height = max(height, sy + sh + 14)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height:.0f}" role="img" aria-label="{escape(alt)}">'
            f"<title>{escape(alt)}</title>{STYLE}\n" + "\n".join(parts) + "\n</svg>\n")


# ---------------------------------------------------------------------------------------------------------------------------
# Decision trees: {"lines": [...], "kids": [(edge label, node), ...]}; a node without kids is an answer.
# ---------------------------------------------------------------------------------------------------------------------------
def _node_h(node: dict) -> float:
    n = len(node["lines"])
    return 26 + LINE_H * n if n > 1 else 36


def _tree_layout(node: dict, depth: int, state: dict) -> None:
    node["depth"] = depth
    kids = node.get("kids") or []
    if not kids:
        h = _node_h(node)
        node["y"] = state["cursor"] + h / 2
        state["cursor"] += h + 16
    else:
        for _, k in kids:
            _tree_layout(k, depth + 1, state)
        node["y"] = (kids[0][1]["y"] + kids[-1][1]["y"]) / 2
        state["gaps"][depth] = max(state["gaps"].get(depth, 0), max(len(lb) for lb, _ in kids))
    state["cols"][depth] = max(state["cols"].get(depth, 0), max(len(x) for x in node["lines"]))


def render_tree(fig: str, alt: str, root: dict) -> str:
    state = {"cursor": 0.0, "cols": {}, "gaps": {}}
    _tree_layout(root, 0, state)
    cw = {d: n * (7.4 if d == max(state["cols"]) else 7.0) + 34 for d, n in state["cols"].items()}
    gaps = {d: 64 + 9.0 * state["gaps"].get(d, 0) for d in cw}
    xs, x = {}, 18
    for d in sorted(cw):
        xs[d] = x
        x += cw[d] + gaps[d]
    width = x - gaps[max(cw)] + 18
    height = state["cursor"] + 8
    parts: list[str] = []
    edges: list[str] = []

    def draw(node: dict, idx: list[int]) -> None:
        d, y = node["depth"], node["y"] + 12
        lines = node["lines"]
        h = _node_h(node)
        w = cw[d]
        leaf = not node.get("kids")
        cls = "leaf" if leaf else "q"
        g = [f'<g class="step" style="animation-delay:{idx[0] * 0.05:.2f}s">',
             f'<rect class="{cls}" x="{xs[d]:.1f}" y="{y - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" rx="10"/>']
        if not leaf:
            g.append(f'<rect class="qa" x="{xs[d]:.1f}" y="{y - h / 2 + 8:.1f}" width="4" height="{h - 16:.1f}" rx="2"/>')
        for k, ln in enumerate(lines):
            ty = y - (len(lines) - 1) * LINE_H / 2 + k * LINE_H + 4.5
            text_class = "mono" if leaf else "t"
            text_style = "" if leaf else ' style="font-size:13px"'
            text_x = xs[d] + w / 2 + (2 if not leaf else 0)
            g.append(f'<text class="{text_class}" x="{text_x:.1f}" y="{ty:.1f}" '
                     f'text-anchor="middle"{text_style}>{escape(ln)}</text>')
        g.append("</g>")
        parts.append("\n".join(g))
        idx[0] += 1
        for label, kid in node.get("kids") or []:
            xp, yp = xs[d] + w, y
            xc, yc = xs[kid["depth"]], kid["y"] + 12
            xm = xp + 18
            edges.append(f'<path class="arrow" d="M{xp:.1f} {yp:.1f} H{xm:.1f} V{yc:.1f} H{xc - 7:.1f}"/>'
                         f'<path class="head" d="M{xc - 8:.1f} {yc - 5:.1f} L{xc:.1f} {yc:.1f} L{xc - 8:.1f} {yc + 5:.1f} Z"/>')
            if label:
                lw = 7.0 * len(label) + 14
                lx = xm + (xc - 8 - xm) / 2
                edges.append(f'<rect class="labbg" x="{lx - lw / 2:.1f}" y="{yc - 9:.1f}" width="{lw:.1f}" height="18" rx="9"/>'
                             f'<text class="lab" x="{lx:.1f}" y="{yc + 4:.1f}" text-anchor="middle">{escape(label)}</text>')
            draw(kid, idx)

    draw(root, [0])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{escape(alt)}">'
            f"<title>{escape(alt)}</title>{STYLE}\n" + "\n".join(edges) + "\n" + "\n".join(parts) + "\n</svg>\n")


# ---------------------------------------------------------------------------------------------------------------------------
# Layer stacks: bands from top to bottom, each with a coloured name strip, feature lines and a quoted one-line purpose.
# ---------------------------------------------------------------------------------------------------------------------------
def render_stack(fig: str, alt: str, bands: list[dict]) -> str:
    bh, gap, strip = 98, 26, 210
    height = len(bands) * bh + (len(bands) - 1) * gap + 24
    parts: list[str] = []
    for i, b in enumerate(bands):
        y = 12 + i * (bh + gap)
        a = f"a{min(i * 2, 5)}"
        g = [f'<g class="step" style="animation-delay:{i * 0.15:.2f}s">',
             f'<rect class="band" x="18" y="{y}" width="{W - 36}" height="{bh}" rx="14"/>',
             f'<path class="{a}" d="M32 {y} H{18 + strip} V{y + bh} H32 A14 14 0 0 1 18 {y + bh - 14} V{y + 14} A14 14 0 0 1 32 {y} Z"/>',
             f'<text class="bt" x="{18 + strip / 2}" y="{y + 40}" text-anchor="middle">{escape(b["name"])}</text>',
             f'<text class="bs" x="{18 + strip / 2}" y="{y + 62}" text-anchor="middle">{escape(b["tag"])}</text>']
        for k, ln in enumerate(b["lines"]):
            g.append(f'<text class="s" x="{18 + strip + 28}" y="{y + 32 + k * LINE_H}" style="font-size:12.5px">{escape(ln)}</text>')
        g.append(f'<text class="quote" x="{18 + strip + 28}" y="{y + bh - 16}">{escape(b["quote"])}</text></g>')
        parts.append("\n".join(g))
        if i < len(bands) - 1:
            cy = y + bh + gap / 2
            for dx in (-22, 22):
                cx = W / 2 + dx
                parts.append(f'<path class="arrow" d="M{cx} {cy + 11} V{cy - 8}"/>'
                             f'<path class="head" d="M{cx - 5} {cy - 7} L{cx} {cy - 13} L{cx + 5} {cy - 7} Z"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" role="img" aria-label="{escape(alt)}">'
            f"<title>{escape(alt)}</title>{STYLE}\n" + "\n".join(parts) + "\n</svg>\n")


def _leaf(*lines: str) -> dict:
    return {"lines": list(lines)}


def _q(*lines: str, kids: list) -> dict:
    return {"lines": list(lines), "kids": kids}


FIGURES: dict[str, dict] = {
    "flow-batch": dict(
        alt="Batch inference: find the files, skip finished ones, predict each in tiles, collect successes and failures, report",
        steps=[
            dict(title="Find files", lines=["Every scene that", "matches the pattern,", "subfolders too"], cmd="--pattern"),
            dict(title="Skip done", lines=["Leave scenes whose", "output file already", "exists"], cmd="skip_existing"),
            dict(title="Predict", lines=["Tile, run the model,", "blend, write a", "GeoTIFF"], cmd="TiledInference"),
            dict(title="Record", lines=["Success, or the", "error text of the", "failure"], cmd="on_error"),
            dict(title="Report", lines=["Counts, timing, and", "the list of failed", "scenes"], cmd="n_failed"),
        ],
        sides=[dict(to=3, wide=True, title="Several workers", lines=["Scenes run side by side; each", "holds a whole scene in memory"])]),
    "flow-ensemble": dict(
        alt="Ensemble inference: every model predicts probabilities for the same scene, they are weighted and fused, and the most likely class is written",
        steps=[
            dict(title="One scene", lines=["The same imagery", "goes to every model"]),
            dict(title="Predict", lines=["Each model writes", "class probabilities", "(one run per model)"], cmd="TiledInference"),
            dict(title="Weight", lines=["Multiply by the", "model's relative", "importance"], cmd="weights"),
            dict(title="Fuse", lines=["Average the votes,", "or take the loudest", "per class"], cmd="mean | max"),
            dict(title="Map", lines=["The likeliest class", "per pixel; 255 where", "there is no data"]),
        ]),
    "flow-dinov3": dict(
        alt="DINOv3 workflow: read a raster as three bands, resize and pad to patches, run the frozen backbone, get dense features, a global vector and attention maps",
        steps=[
            dict(title="Raster", lines=["Three bands of your", "choice as red, green,", "blue"], cmd="rgb_bands"),
            dict(title="Prepare", lines=["Shrink if large, pad", "to whole 16 px", "patches"], cmd="max_size"),
            dict(title="Backbone", lines=["Frozen network,", "satellite or web", "statistics"], cmd="dinov3_vitl16_sat"),
            dict(title="Features", lines=["One vector per", "patch, registered", "with the raster"], cmd="extract_features"),
            dict(title="Use", lines=["Cluster, find look-", "alikes, or train a", "small head"], cmd="build_classifier"),
        ],
        sides=[dict(to=4, wide=True, title="Also from the same pass", lines=["Global vector (1, D) and attention maps"])]),
    "flow-dofa": dict(
        alt="DOFA workflow: name the sensor or list the wavelengths, check the band count, the model builds its first layer from the wavelengths, get an embedding",
        steps=[
            dict(title="Your bands", lines=["Any sensor, any", "number of bands, in", "a known order"]),
            dict(title="Wavelengths", lines=["Sensor name, or a", "list in micrometres", "per band"], cmd="sensor="),
            dict(title="Check", lines=["Band count must", "equal wavelength", "count"], cmd="ValueError"),
            dict(title="Model", lines=["First layer is built", "from the wavelengths"], cmd="load_dofa_hf"),
            dict(title="Embedding", lines=["One vector per", "image: 768 numbers", "for the base model"], cmd="extract_dofa_..."),
        ]),
    "flow-sam": dict(
        alt="SAM workflow: the raster becomes a three-band picture, prompts go to SAM, candidate masks come back, you filter and name them",
        steps=[
            dict(title="Picture", lines=["Three bands as red,", "green, blue, stretched", "for display"], cmd="rgb_bands"),
            dict(title="Prompts", lines=["One click, or a", "grid of clicks over", "the scene"], cmd="points_per_side"),
            dict(title="SAM", lines=["Frozen model gives", "candidate masks, no", "class names"], cmd="sam-vit-b"),
            dict(title="Filter", lines=["Drop small or", "low-quality masks"], cmd="min_area_m2"),
            dict(title="Name", lines=["Give shapes classes", "from a map or a", "person"], cmd="assign_classes"),
        ]),
    "flow-clip": dict(
        alt="CLIP workflow: a tile becomes an RGB picture, an image encoder and a text encoder place picture and sentences in one space, similarity gives scores or a ranking",
        steps=[
            dict(title="Tile", lines=["Your raster", "or a folder of tiles"]),
            dict(title="Picture", lines=["Three bands, shared", "stretch, nodata", "shown black"], cmd="bands=(3, 2, 1)"),
            dict(title="Embed", lines=["Image encoder and", "text encoder, one", "shared space"], cmd="remoteclip-b32"),
            dict(title="Compare", lines=["Cosine similarity", "between picture and", "each sentence"]),
            dict(title="Result", lines=["Probabilities over", "your list, or a", "ranked folder"], cmd="zero_shot | search"),
        ],
        sides=[dict(to=3, wide=True, title="Your words", lines=["Descriptions or a query,", "as full sentences"])]),
    "flow-moondream": dict(
        alt="Moondream workflow: raster to RGB picture, with or without a question, to the model, text comes out",
        steps=[
            dict(title="Raster", lines=["A GeoTIFF or an", "ordinary picture"]),
            dict(title="Picture", lines=["Three bands, shared", "stretch, nodata black"], cmd="bands=(3, 2, 1)"),
            dict(title="Question", lines=["None for a caption;", "a closed question", "for an answer"], cmd="caption | vqa"),
            dict(title="Moondream2", lines=["About 1.8 B", "parameters, pinned", "revision"], cmd="2025-06-21"),
            dict(title="Text", lines=["A sentence or a", "short answer: a note", "for a person"]),
        ]),
    "flow-monitoring": dict(
        alt="Monitoring a deployed model: fit a reference from training data once, check each new batch of imagery for input drift, check accuracy on a fresh labelled sample, alert someone, then retrain or investigate",
        steps=[
            dict(title="Reference", lines=["Fit once on the data", "the model was built on"], cmd="DriftDetector.fit"),
            dict(title="New data", lines=["Each run: this week's", "imagery"], cmd="your pipeline"),
            dict(title="Inputs", lines=["Did the pixels change?", "No labels needed"], cmd="check()"),
            dict(title="Accuracy", lines=["Did the score fall?", "Needs fresh labels"], cmd="PerformanceDrift"),
            dict(title="Alert", lines=["Severity, then log,", "chat or e-mail"], cmd="AlertManager"),
            dict(title="Act", lines=["Investigate, then", "retrain if confirmed"]),
        ],
        sides=[dict(to=3, wide=True, title="A warning light, not a verdict", lines=["Drift can occur with the model fine"])]),
    "flow-alerts": dict(
        alt="From drift report to a person: check_drift_report turns a report into alerts with a severity; trigger stores each in history and calls the handler of every channel; handlers write a file, post to chat or send e-mail",
        steps=[
            dict(title="Check", lines=["The drift check runs", "and returns a report"], cmd="det.check(...)"),
            dict(title="Report", lines=["drift_detected,", "incomplete, errors"], cmd="report dict"),
            dict(title="Alerts", lines=["One alert per finding", "with a severity"], cmd="check_drift_report"),
            dict(title="Route", lines=["Stored in history,", "sent to each channel"], cmd="trigger()"),
            dict(title="Handlers", lines=["log, file, chat,", "e-mail: yours"], cmd="register_handler"),
            dict(title="A person", lines=["Reads it and", "decides"]),
        ],
        sides=[dict(to=2, wide=True, title="An incomplete report sends nothing", lines=["Check report[\"incomplete\"] yourself"])]),
    "flow-prithvi": dict(
        alt="Using Prithvi-EO: prepare six HLS bands as reflectance, load the encoder, take features or build a segmentation head, train the head on labels, run a task, check against reference data",
        steps=[
            dict(title="Prepare", lines=["Six HLS bands as", "reflectance 0 to 1"], cmd="prepare_for_ai"),
            dict(title="Load", lines=["Pinned download from", "the Hub, or local files"], cmd="Prithvi().load()"),
            dict(title="Features", lines=["Scene vector, or one", "vector per 16 px patch"], cmd="extract_features"),
            dict(title="Head", lines=["Random conv head on", "the frozen encoder"], cmd="seg head"),
            dict(title="Train", lines=["Your labels, your", "GeoTrainer run"], cmd="save_prithvi_head"),
            dict(title="Run", lines=["Task output, with", "the method named"], cmd="PrithviTasks"),
        ],
        sides=[dict(to=6, wide=True, title="No head: rules, not Prithvi", lines=["result[\"method\"] says which ran"])]),
    "flow-radar": dict(
        alt="Using radar with an optical foundation model: calibrate and despeckle, convert to decibels, normalise to 0 to 1, map VV and VH to six channels, validate, then threshold or fine-tune",
        steps=[
            dict(title="Prepare", lines=["Calibrate, despeckle", "on linear power, then", "convert to dB"], cmd="SARPreprocessor"),
            dict(title="Normalise", lines=["Fixed -35 to +5 dB", "bounds onto 0 to 1"], cmd="S8 normalise"),
            dict(title="Map", lines=["VV and VH become", "six pseudo-HLS bands"], cmd="sar_to_hls_6ch"),
            dict(title="Validate", lines=["Shape, range, NaN,", "size at least 64"], cmd="validate_sar_ai"),
            dict(title="Route", lines=["Threshold now, or", "fine-tune on labels"], cmd="SARPrithviAdapter"),
            dict(title="Check", lines=["Compare with a", "reference flood map"]),
        ],
        sides=[dict(to=4, wide=True, title="zero_shot is a threshold", lines=["It does not run Prithvi"])]),
    "flow-embeddings": dict(
        alt="Working with ready-made embeddings: fetch the embedding stack for a box, sample the vectors at labelled points, train a simple classifier, predict every pixel, check on points you held back",
        steps=[
            dict(title="Fetch", lines=["Embedding stack for", "a box and a year"], cmd="embeddings_for_bbox"),
            dict(title="Label", lines=["A few hundred points", "with known classes"], cmd="your field data"),
            dict(title="Sample", lines=["The vector at each", "labelled pixel"], cmd="embedding[:, r, c]"),
            dict(title="Train", lines=["A simple classifier;", "no deep network"], cmd="LogisticRegression"),
            dict(title="Predict", lines=["Every pixel gets", "a class id"], cmd="classify_pixels"),
            dict(title="Check", lines=["Score on points you", "held back"], cmd="accuracy_score"),
        ],
        sides=[dict(to=3, wide=True, title="Also: similarity search", lines=["Pixels that look like a reference one"])]),
    "flow-tracking": dict(
        alt="Experiment tracking: the config names the trackers, the trainer starts them, logs the validation numbers every epoch, and ends the run; each tracker has its own place to look",
        steps=[
            dict(title="Choose", lines=["tracker= names one", "or more backends"], cmd="TrainingConfig"),
            dict(title="Start", lines=["Settings logged once;", "a missing package is", "skipped, not fatal"], cmd="start_run"),
            dict(title="Log", lines=["Every epoch: loss,", "IoU, F1, learning", "rate, per class"], cmd="val_metrics"),
            dict(title="End", lines=["Run closed, duration", "recorded"], cmd="end_run"),
            dict(title="Look", lines=["TensorBoard, MLflow", "UI or the W&B page"], cmd="mlflow ui"),
        ]),
    "flow-resume": dict(
        alt="Resuming a run: every finished epoch writes last.pth atomically; after an interruption the same script with resume=True restores weights, optimiser, schedule, history and early-stopping state and continues",
        steps=[
            dict(title="Train", lines=["Each finished epoch", "is a safe point"], cmd="fit()"),
            dict(title="Save", lines=["last.pth written", "atomically, every", "epoch"], cmd="save_last=True"),
            dict(title="Interrupted", lines=["Ctrl-C, reboot,", "time limit"]),
            dict(title="Re-run", lines=["Same script, same", "output_dir"], cmd="resume=True"),
            dict(title="Restore", lines=["Weights, optimiser,", "schedule, history,", "early stopping"], cmd="load_state_dict"),
            dict(title="Continue", lines=["From the next epoch", "to max_epochs"]),
        ],
        sides=[dict(to=4, wide=True, title="Keep max_epochs the same", lines=["The schedule is stored with its length"])]),
    "flow-azure": dict(
        alt="Azure ML deployment: export and check ONNX, write a scoring script, register the model, create the endpoint and deployment with an environment, test, delete",
        steps=[
            dict(title="Export + check", lines=["ONNX file, compared", "with PyTorch"], cmd="export_onnx"),
            dict(title="Score script", lines=["init() and run():", "JSON in, JSON out"], cmd="azure_score.py"),
            dict(title="Register", lines=["The model goes into", "the workspace"], cmd="deploy()"),
            dict(title="Deploy", lines=["Endpoint plus a", "deployment with", "environment + script"], cmd="CodeConfiguration"),
            dict(title="Test", lines=["Send a chip, read", "predictions"], cmd="predict()"),
            dict(title="Delete", lines=["Stop the hourly", "charge when done"], cmd="delete_endpoint"),
        ],
        sides=[dict(to=4, wide=True, title="Not covered by the shipped deploy()", lines=["The scoring script and environment"])]),
    "flow-vertex": dict(
        alt="Vertex AI deployment: export and check ONNX, build a serving container, upload, register the model with the container, create the endpoint, deploy, test, delete",
        steps=[
            dict(title="Export + check", lines=["ONNX file, compared", "with PyTorch"], cmd="export_onnx"),
            dict(title="Container", lines=["Server that answers", "/health and /predict"], cmd="vertex_server.py"),
            dict(title="Upload", lines=["Model to Cloud", "Storage, image to", "Artifact Registry"], cmd="deploy()"),
            dict(title="Register", lines=["Model.upload with", "your container image"], cmd="Model.upload"),
            dict(title="Deploy", lines=["Endpoint, 1 to 5", "replicas"], cmd="model.deploy"),
            dict(title="Delete", lines=["Undeploy and delete", "when done"], cmd="delete_endpoint"),
        ],
        sides=[dict(to=2, wide=True, title="Not covered by the shipped deploy()", lines=["The custom serving container"])]),
    "flow-training": dict(
        alt="Training workflow: labels and imagery, chips, model, train, check, use",
        steps=[
            dict(title="Labels + imagery", lines=["GeoTIFF pairs on", "the same grid"], cmd="labeling"),
            dict(title="Chips", lines=["Image and label", "tiles, split into", "train and val"], cmd="GeoSegDataset"),
            dict(title="Model", lines=["Pretrained, or from", "scratch"], cmd="get_model(...)"),
            dict(title="Train", lines=["Fit on the train", "chips, watch the", "val chips"], cmd="GeoTrainer.fit"),
            dict(title="Check", lines=["Read the log and", "look at maps"], cmd="metrics"),
            dict(title="Use", lines=["Predict, run a", "pipeline, or call", "from Python"], cmd="infer predict"),
        ]),
    "flow-model-sources": dict(
        alt="Where models come from: pretrained weights go into get_model, GeoTrainer.fit trains them on your chips, best.pth is used for prediction",
        steps=[
            dict(title="Public weights", lines=["ImageNet, ADE20K,", "HLS and others"]),
            dict(title="Build", lines=["Pick an architecture", "from the catalogue"], cmd="get_model(...)"),
            dict(title="Train", lines=["Fine-tune on your", "chips, or train from", "scratch"], cmd="GeoTrainer.fit"),
            dict(title="Checkpoint", lines=["Your own .pth", "file: best.pth"], cmd="best.pth"),
            dict(title="Use", lines=["infer predict, a", "pipeline's", "--checkpoint, Python"], cmd="--checkpoint"),
        ],
        sides=[dict(to=3, wide=True, title="Your labelled chips", lines=["GeoTIFF image and label pairs"])]),

    "flow-segmentation": dict(
        alt="Segmentation workflow: get labels, cut chips, train, check, predict tiles, deliver a map",
        steps=[
            dict(title="Labels", lines=["Mask per scene:", "OSM, WorldCover,", "SAM or your own"], cmd="labeling"),
            dict(title="Chips", lines=["Cut image and mask", "into matching", "tiles"], cmd="GeoSegDataset"),
            dict(title="Train", lines=["Fit a model with", "a loss that suits", "your classes"], cmd="GeoTrainer.fit"),
            dict(title="Check", lines=["mIoU and IoU per", "class on tiles the", "model never saw"], cmd="SegmentationMetrics"),
            dict(title="Predict", lines=["Slide over the", "whole scene, blend", "the overlaps"], cmd="TiledInference"),
            dict(title="Deliver", lines=["GeoTIFF, polygons", "and a report"], cmd="vectorise"),
        ],
        sides=[dict(to=3, wide=True, title="Few labels?", lines=["Start from a foundation model", "and train only a small head"])]),
    "flow-classification": dict(
        alt="Classification workflow: labelled chips, train, check with a confusion matrix, predict, map or table",
        steps=[
            dict(title="Labelled chips", lines=["One folder per", "class, or a CSV", "of file and label"]),
            dict(title="Train", lines=["Fine-tune a", "pretrained network", "on your chips"], cmd="GeoTrainer.fit"),
            dict(title="Check", lines=["Confusion matrix:", "which classes get", "mixed up"], cmd="confusion matrix"),
            dict(title="Predict", lines=["A class and a", "probability for", "each chip"], cmd="ai classify"),
            dict(title="Use", lines=["Table, map of", "chips, or a filter", "for later steps"]),
        ],
        sides=[dict(to=4, wide=True, title="No labels at all?", lines=["Zero-shot with CLIP: describe", "the classes in words"])]),
    "flow-detection": dict(
        alt="Detection workflow: boxes, train a detector, score with mAP, predict on tiles, merge duplicates, georeferenced objects",
        steps=[
            dict(title="Boxes", lines=["An image and a", "box per object", "(COCO format)"], cmd="labeling"),
            dict(title="Train", lines=["Fit a detector;", "background is", "added for you"], cmd="GeoTrainer.fit"),
            dict(title="Score", lines=["Average precision", "at the overlap you", "choose"], cmd="mAP"),
            dict(title="Predict", lines=["Detect tile by", "tile with overlap", "at the seams"], cmd="TiledInference"),
            dict(title="Merge", lines=["Remove duplicate", "boxes where tiles", "overlap"], cmd="NMS"),
            dict(title="Objects", lines=["Boxes in map", "coordinates, as", "GeoJSON"], cmd="vectorise"),
        ]),
    "flow-change-detection": dict(
        alt="Change-detection workflow: two dates, align and match, choose a method, change map, polygons and a report",
        steps=[
            dict(title="Two dates", lines=["Same place, same", "sensor, same", "season"]),
            dict(title="Align", lines=["Same grid, same", "size, same band", "order"], cmd="preprocess"),
            dict(title="Method", lines=["No labels: CVA", "Labelled pairs:", "siamese network"], cmd="ai change"),
            dict(title="Change map", lines=["Per pixel: changed", "or not, with a", "threshold"]),
            dict(title="Deliver", lines=["Polygons of change", "and area totals"], cmd="vectorise"),
        ],
        sides=[dict(to=3, wide=True, title="Never skip the check", lines=["Look at false changes from clouds,", "shadows and seasons"])]),
    "flow-super-resolution": dict(
        alt="Super-resolution workflow: a low resolution scene is cut into tiles, upscaled four times, blended, and gives a sharper image",
        steps=[
            dict(title="Scene", lines=["Your image at its", "original resolution"]),
            dict(title="Tiles", lines=["Cut into small", "overlapping pieces"], cmd="tile size 32"),
            dict(title="Upscale", lines=["SwinIR makes each", "tile four times", "larger"], cmd="swinir-m"),
            dict(title="Blend", lines=["Stitch the tiles", "and hide the seams"], cmd="overlap"),
            dict(title="Sharper image", lines=["For people to look", "at, not to measure"]),
        ]),
    "flow-experiments": dict(
        alt="Experiment workflow: configure, run, track metrics, compare runs on a leaderboard, search settings, pick the best",
        steps=[
            dict(title="Configure", lines=["One TrainingConfig", "per experiment"], cmd="TrainingConfig"),
            dict(title="Run", lines=["Train and save", "checkpoints"], cmd="GeoTrainer.fit"),
            dict(title="Track", lines=["Metrics and settings", "written to a file", "per run"], cmd="tracking"),
            dict(title="Compare", lines=["Same data, same", "metric, one table"], cmd="Leaderboard"),
            dict(title="Search", lines=["Let a search try", "settings for you"], cmd="HPO, AutoML"),
            dict(title="Pick", lines=["Keep the best run", "and its config"]),
        ]),
    "flow-case-flood": dict(
        alt="Multisensor flood mapping: choose a dry reference window and a flood window, compare Sentinel-1 radar, compute Sentinel-2 MNDWI, class each pixel by what each sensor says, remove ground high above the nearest drainage, write the maps and reports",
        steps=[
            dict(title="Two windows", lines=["A dry reference and", "a flood window, as", "date ranges"], cmd="--date-before/after"),
            dict(title="Radar", lines=["Sentinel-1 median in", "dB per window; water", "threshold calibrated"], cmd="water_db=auto"),
            dict(title="Optical", lines=["Sentinel-2 MNDWI,", "cloud and shadow", "masked, never dry"], cmd="mndwi_threshold"),
            dict(title="Agree", lines=["Both, radar only,", "optical only or", "disputed, per pixel"], cmd="flood_extent.tif"),
            dict(title="Terrain", lines=["Drop floods more than", "15 m above the", "nearest drainage"], cmd="max_hand_m=15"),
            dict(title="Deliver", lines=["GeoTIFFs, vectors,", "PDF, Excel, Word,", "run_summary.json"], cmd="deliverables/"),
        ],
        sides=[dict(to=5, wide=True, title="Box over 6 million cells", lines=["Terrain is skipped and the", "report says so"])]),
    "flow-case-fire": dict(
        alt="Burn severity: pick a pre-fire and a post-fire date, compute NBR for each, subtract to get dNBR, class it with the USGS thresholds, measure areas over valid pixels, write the maps and reports",
        steps=[
            dict(title="Two dates", lines=["Before and after,", "same season if", "you can"], cmd="--date-before/after"),
            dict(title="NBR twice", lines=["(NIR - SWIR2) over", "(NIR + SWIR2), cloud", "pixels left empty"], cmd="nir, swir2"),
            dict(title="Subtract", lines=["dNBR is NBR before", "minus NBR after"], cmd="dnbr.tif"),
            dict(title="Classes", lines=["Unburned below 0.10,", "low, moderate, high", "at 0.27 and 0.66"], cmd="severity_class.tif"),
            dict(title="Measure", lines=["Areas and shares over", "valid pixels only"], cmd="burned_area_ha"),
            dict(title="Deliver", lines=["Maps, polygons,", "PDF, Excel, Word"], cmd="deliverables/"),
        ],
        sides=[dict(to=2, wide=True, title="Cloud or smoke", lines=["Cloud pixels are not dry;", "check the valid share"])]),
    "flow-case-drainage": dict(
        alt="Drainage analysis: download a Copernicus 30 m elevation model, fill depressions with Priority-Flood, route water D8, accumulate flow, mark streams above a catchment threshold and number them by Strahler order, compute height above the nearest drainage",
        steps=[
            dict(title="Elevation", lines=["Copernicus 30 m", "surface model for", "the box"], cmd="dem_source"),
            dict(title="Fill", lines=["Priority-Flood: sinks", "filled, flats resolved"], cmd="cells_filled"),
            dict(title="Route", lines=["Each cell drains to", "its steepest downhill", "neighbour (D8)"], cmd="flow direction"),
            dict(title="Accumulate", lines=["Draining area through", "every cell, in km²"], cmd="flow_accumulation"),
            dict(title="Streams", lines=["Cells above the", "threshold, numbered", "by Strahler order"], cmd="stream_threshold_km2"),
            dict(title="HAND", lines=["Height of each cell", "above the stream it", "drains into"], cmd="hand_m.tif"),
        ],
        sides=[dict(to=4, wide=True, title="Edge of the box", lines=["Areas are cut at the edge, so", "border streams read short"])]),
    "flow-case-water": dict(
        alt="Water quality indicators: one cloud-masked Sentinel-2 scene, water mask from NDWI above zero, NDCI and red reflectance inside the water, terciles of NDCI, report with the valid share",
        steps=[
            dict(title="One scene", lines=["Sentinel-2 L2A for", "the month, cloud", "pixels left empty"], cmd="--date 2024-06"),
            dict(title="Water", lines=["NDWI (green, NIR)", "above zero"], cmd="pct_water"),
            dict(title="Indicators", lines=["NDCI and red", "reflectance, water", "pixels only"], cmd="water_quality.tif"),
            dict(title="Terciles", lines=["Lower, middle, upper", "third of NDCI in", "this water"], cmd="ndci_zones"),
            dict(title="Report", lines=["Valid share and", "relative-indicator", "warning"], cmd="valid_fraction"),
        ],
        sides=[dict(to=3, wide=True, title="Red or red-edge at zero", lines=["NDCI is undefined there and", "the pixel drops out"])]),
}


STACKS: dict[str, dict] = {
    "stack-layers": dict(
        alt="PyGeoVision in three layers: the data layer at the bottom, the AI layer above it, the task layer on top",
        bands=[
            dict(name="3  Task layer", tag="finished products",
                 lines=["41 pipelines  ·  reporting  ·  the command line (channel, report)"],
                 quote="\u201cgive me flood extent for this box and this date\u201d"),
            dict(name="2  AI layer", tag="optional: needs PyTorch",
                 lines=["Model registry  ·  training  ·  tiled inference  ·  labelling",
                        "OBIA  ·  explainability  ·  monitoring  ·  edge  ·  cloud"],
                 quote="\u201ctrain, run, explain or ship a network\u201d"),
            dict(name="1  Data layer", tag="always available",
                 lines=["22 providers (through pygeofetch)  ·  reference products",
                        "Spectral indices  ·  radiometry  ·  scene selection"],
                 quote="\u201cfind, download and calibrate imagery\u201d"),
        ]),
}

TREES: dict[str, dict] = {
    "tree-case-london": dict(
        alt="One box in central London, five questions: land cover and habitat mix come from the WorldCover 2021 map; buildings and roads are published datasets; vegetation change compares two Sentinel-2 dates",
        root=_q("One box,", "-0.15 51.47 -0.10 51.52", kids=[
            ("what is on the ground?", _q("Which map?", kids=[
                ("classes", _leaf("land_cover", "WorldCover 2021")),
                ("how mixed", _leaf("biodiversity_hotspot", "Shannon index")),
            ])),
            ("what is built?", _q("Which dataset?", kids=[
                ("buildings", _leaf("building_footprints", "Microsoft ML")),
                ("roads", _leaf("road_network", "OpenStreetMap")),
            ])),
            ("what changed?", _leaf("crop_health", "NDVI, May to Aug 2024")),
        ])),
    "tree-ai-ready": dict(
        alt="Which AI-ready gate to use: a raw scene folder goes to prepare_for_ai; clipped optical data to prepare_optical_for_ai; clipped radar to prepare_sar_for_ai",
        root=_q("A scene folder, or an", "un-clipped raster?", kids=[
            ("yes", _leaf("client.prepare_for_ai(...)")),
            ("no", _q("Already clipped, masked", "and scaled. Which sensor?", kids=[
                ("optical", _leaf("prepare_optical_for_ai(...)")),
                ("radar (VV, VH)", _leaf("prepare_sar_for_ai(...)")),
            ])),
        ])),
    "tree-vlm-chooser": dict(
        alt="Choosing a vision-language tool: ranking many tiles by a description leads to CLIP; a sentence or an answer about one tile leads to Moondream; a mask from words is not available",
        root=_q("What do you need", "back?", kids=[
            ("tiles ranked by words", _leaf("CLIPGeo.search", "GeoImageRetrieval")),
            ("a label for a tile", _leaf("CLIPGeo.zero_shot")),
            ("a sentence about a tile", _leaf("MoondreamGeo.caption")),
            ("an answer about a tile", _leaf("MoondreamGeo.vqa")),
            ("a mask from words", _leaf("not available here", "SAM + CLIP by hand")),
            ("a number", _leaf("a pipeline, not a VLM")),
        ])),
    "tree-foundation-chooser": dict(
        alt="Choosing a foundation model: with no labels, CLIP ranks tiles by words, SAM outlines objects, DINOv3 gives features to cluster; with a few labels, Prithvi suits six HLS bands, DOFA any sensor, DINOv3 plus a head suits ordinary colour images, radar needs mapping first; ready-made yearly vectors come from TESSERA or AlphaEarth",
        root=_q("What do you have?", kids=[
            ("no labels", _q("What do you want?", kids=[
                ("find or label tiles by words", _leaf("CLIP / RemoteCLIP")),
                ("outlines of objects", _leaf("SAM / SAM 2")),
                ("features to cluster", _leaf("DINOv3")),
            ])),
            ("a few hundred labels", _q("Which imagery?", kids=[
                ("six HLS bands", _leaf("Prithvi-EO", "train a head")),
                ("any sensor or bands", _leaf("DOFA", "train a head")),
                ("ordinary colour", _leaf("DINOv3", "train a head")),
                ("radar", _leaf("map to six bands", "then Prithvi")),
            ])),
            ("an area and a year", _leaf("TESSERA / AlphaEarth", "ready-made vectors")),
        ])),
    "tree-deploy-chooser": dict(
        alt="Choosing where to run a trained model: one-off or batch scenes use tiled inference; an app needing a web address uses a cloud endpoint on AWS, Azure or Google; a machine without PyTorch uses ONNX; an NVIDIA Jetson uses TensorRT",
        root=_q("Who calls the model?", kids=[
            ("me, on scenes", _leaf("tiled inference", "no deployment needed")),
            ("an app, by web address", _q("Which cloud do you", "already pay for?", kids=[
                ("AWS", _leaf("SageMaker", "most complete path")),
                ("Azure", _leaf("Azure ML", "add a scoring script")),
                ("Google", _leaf("Vertex AI", "add a container")),
            ])),
            ("a machine in the field", _q("What hardware?", kids=[
                ("any CPU or GPU", _leaf("ONNX Runtime")),
                ("NVIDIA Jetson", _leaf("TensorRT engine")),
            ])),
        ])),
    "tree-pipeline-chooser": dict(
        alt="Choosing a pipeline: change between two dates leads to flood, fire, forest or built-up, or generic change pipelines; a map of what is there leads to land cover, water, terrain, roads, buildings, temperature, snow or vegetation pipelines",
        root=_q("Is the question about", "change between two dates?", kids=[
            ("yes", _q("What changed?", kids=[
                ("a flood", _q("Is it cloudy?", kids=[
                    ("yes", _leaf("multisensor_flood_mapping", "or sar_flood_mapping")),
                    ("no", _leaf("flood_mapping")),
                ])),
                ("fire", _leaf("wildfire_severity (scar)", "active_fire (burning)")),
                ("forest or built-up", _leaf("forest_change", "urban_expansion", "landcover_change")),
                ("anything else", _leaf("change_detection")),
            ])),
            ("no", _q("A map of what is", "there?", kids=[
                ("land cover", _leaf("land_cover")),
                ("water", _leaf("water_bodies")),
                ("terrain", _leaf("terrain_analysis", "drainage_analysis")),
                ("roads, buildings", _leaf("road_network", "building_footprints")),
                ("temperature, snow, plants", _leaf("land_surface_temperature", "snow_cover", "vegetation_indices")),
            ])),
        ])),
}


def build() -> dict[Path, str]:
    out = {OUT / f"{name}.svg": render(name, f["alt"], f["steps"], f.get("sides")) for name, f in FIGURES.items()}
    out.update({OUT / f"{name}.svg": render_stack(name, f["alt"], f["bands"]) for name, f in STACKS.items()})
    out.update({OUT / f"{name}.svg": render_tree(name, f["alt"], f["root"]) for name, f in TREES.items()})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    stale = []
    for path, text in build().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if stale:
        print("out of date:", *[str(p.relative_to(ROOT)) for p in stale], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
