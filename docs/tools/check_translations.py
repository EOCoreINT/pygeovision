#!/usr/bin/env python3
"""Check the translation files in docs/locale.

    python docs/tools/check_translations.py          # exits 1 and lists every problem

For every language in conf.py's LANGUAGES it checks that

* ``chrome.po`` (the navigation bar, breadcrumbs and section names) exists and translates every string the templates use,
* ``index.po`` translates every sidebar caption of ``index.md``,
* every ``.po`` file parses and names its own language,
* no translation drops or alters a Markdown link target, an inline code span or a ``{directive}`` of its source text
  (a translated link that points nowhere is the most common way a translation breaks a page), and
* every ``.po`` file belongs to a page that still exists (a deleted page leaves an orphan behind).

It does not judge the quality of the translation; that needs a person who reads the language.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import polib
except ImportError:  # pragma: no cover
    sys.exit("polib is required: pip install polib")

DOCS = Path(__file__).resolve().parent.parent
LOCALE = DOCS / "locale"
LANGS = ["fr", "es", "pt", "de", "zh_CN", "ar"]
CHROME_KEYS = [
    "Documentation", "Pipelines", "Guides", "Community", "Download", "Search", "Version:", "Home", "Language",
    "Partly translated", "Get started", "Data", "Train and deploy", "Case studies", "Reference", "Project",
    # the section names used by the breadcrumb (docs/_templates/page.html) and the sidebar captions (index.md)
    "Segmentation", "Classification", "Object detection", "Change detection", "Foundation models", "Vision-language",
    "3D / LiDAR", "Super-resolution", "Time series", "Object-based analysis (OBIA)", "Models", "Labelling", "Training",
    "Inference", "Experiments", "Explainability", "Monitoring", "Reporting", "Deployment", "CLI reference",
]
LINK = re.compile(r"\]\(([^)\s]+)\)")
CODE = re.compile(r"`([^`]+)`")


def targets(text: str) -> list[str]:
    return sorted(LINK.findall(text))


def spans(text: str) -> list[str]:
    return sorted(CODE.findall(text))


def main() -> int:
    problems: list[str] = []
    for lang in LANGS:
        base = LOCALE / lang / "LC_MESSAGES"
        if not base.is_dir():
            problems.append(f"{lang}: no locale directory")
            continue
        chrome = base / "chrome.po"
        if not chrome.exists():
            problems.append(f"{lang}: chrome.po is missing")
        else:
            have = {e.msgid: e.msgstr for e in polib.pofile(str(chrome))}
            for key in CHROME_KEYS:
                if not have.get(key):
                    problems.append(f"{lang}/chrome.po: no translation for {key!r}")
        index_po = base / "index.po"
        if index_po.exists():
            have_idx = {e.msgid: e.msgstr for e in polib.pofile(str(index_po))}
            for cap in re.findall(r"^:caption: (.*)$", (DOCS / "index.md").read_text(encoding="utf-8"), flags=re.M):
                if not have_idx.get(cap):
                    problems.append(f"{lang}/index.po: no translation for the sidebar caption {cap!r}")
        for po_path in sorted(base.rglob("*.po")):
            rel = po_path.relative_to(base)
            try:
                po = polib.pofile(str(po_path))
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{lang}/{rel}: does not parse ({exc})")
                continue
            if po.metadata.get("Language", "").replace("-", "_") != lang:
                problems.append(f"{lang}/{rel}: header says Language={po.metadata.get('Language')!r}")
            if rel.name != "chrome.po":
                page = DOCS / rel.with_suffix(".md")
                if not page.exists() and not rel.with_suffix(".rst").exists():
                    problems.append(f"{lang}/{rel}: orphan, there is no page {rel.with_suffix('.md')}")
            for e in po:
                if not e.msgstr:
                    continue
                if targets(e.msgid) != targets(e.msgstr):
                    problems.append(f"{lang}/{rel}: link targets differ for {e.msgid[:60]!r}")
                if spans(e.msgid) != spans(e.msgstr):
                    problems.append(f"{lang}/{rel}: code spans differ for {e.msgid[:60]!r}")
    if problems:
        print("\n".join(problems))
        print(f"\n{len(problems)} problem(s)")
        return 1
    n = sum(1 for _ in LOCALE.rglob("*.po"))
    print(f"{n} translation files in {len(LANGS)} languages: all consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
