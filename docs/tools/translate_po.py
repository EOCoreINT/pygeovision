#!/usr/bin/env python3
"""Translate the whole documentation into one language, page by page, with the Claude API.

Only the navigation, the home page and the quickstart are translated by hand today. This tool does the rest, with the
standard Sphinx workflow:

    1. ``sphinx-build -b gettext`` extracts every paragraph, heading, table cell and list item of every page into a ``.pot``
       file (28,000 messages, 270,000 words at the time of writing).
    2. This script translates the messages that have no translation yet and writes ``locale/<lang>/LC_MESSAGES/<page>.po``.
    3. The normal build turns the ``.po`` files into the translated pages. A message with no translation stays English.

    export ANTHROPIC_API_KEY=...
    python docs/tools/translate_po.py --lang de --dry-run            # how many messages and words, nothing is sent
    python docs/tools/translate_po.py --lang de --pages 'getting-started/*'
    python docs/tools/translate_po.py --lang de                      # every page (resumable: run it again after an error)
    python docs/tools/check_translations.py                          # links and code spans must survive translation

What it keeps exactly as written: inline code, link targets, ``{directive}`` and ``{role}`` names, URLs, and the names in
``GLOSSARY``. A translation that changes any of them is rejected, asked for once more, and left in English if it fails again,
so a bad answer can never break a link or a command. Nothing here replaces a native speaker: the output is machine
translation and is stamped as such in the ``.po`` header (``X-Translator``).
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import fnmatch
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

try:
    import polib
except ImportError:  # pragma: no cover
    sys.exit("polib is required: pip install polib")

DOCS = Path(__file__).resolve().parent.parent
LOCALE = DOCS / "locale"
POT_DIR = DOCS / "_build" / "gettext"
API = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"
LANGUAGE_NAMES = {
    "fr": "French", "es": "Spanish (neutral, Latin America and Spain)", "pt": "Portuguese (Brazil)", "de": "German",
    "zh_CN": "Simplified Chinese", "ar": "Modern Standard Arabic",
}
GLOSSARY = ("PyGeoVision", "PyGeoFetch", "Sentinel-1", "Sentinel-2", "Landsat", "NDVI", "NDWI", "NDCI", "dNBR", "HAND", "SAR",
            "GeoTIFF", "GeoPackage", "STAC", "OBIA", "LiDAR", "UNet", "SegFormer", "SAM", "CLIP", "Prithvi", "ONNX", "Docker",
            "PyTorch", "NumPy", "GitHub", "Read the Docs")

_spec = importlib.util.spec_from_file_location("check_translations", Path(__file__).with_name("check_translations.py"))
_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_check)

LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)
CODE_SPAN = re.compile(r"`[^`]*`")
URL_OR_PATH = re.compile(r"^(https?://\S+|[\w./-]+\.(md|py|yaml|yml|json|toml|txt))$")
ROLES = re.compile(r"\{[a-zA-Z:_-]+\}")
PLACEHOLDER = re.compile(r"%\(\w+\)[sd]|\{\{[^}]*\}\}|\$\{[^}]*\}")


def translatable(text: str) -> bool:
    """False for messages with nothing to translate: code, a bare URL or path, numbers, table rules."""
    stripped = CODE_SPAN.sub("", text)
    stripped = re.sub(r"\]\([^)]*\)", "]", stripped)
    stripped = re.sub(r"https?://\S+", "", stripped)
    if URL_OR_PATH.match(text.strip()):
        return False
    return len(LETTERS.findall(stripped)) >= 3


def valid(source: str, translation: str) -> bool:
    """A translation is acceptable only if everything that must not change is exactly as in the source."""
    return (
        bool(translation.strip())
        and _check.targets(source) == _check.targets(translation)
        and _check.spans(source) == _check.spans(translation)
        and sorted(ROLES.findall(source)) == sorted(ROLES.findall(translation))
        and sorted(PLACEHOLDER.findall(source)) == sorted(PLACEHOLDER.findall(translation))
        and source.count("```") == translation.count("```")
    )


SYSTEM = """You translate technical documentation for PyGeoVision, a Python package for Earth observation, into {language}.

You receive a JSON array of strings. Each is one paragraph, heading, list item or table cell of a Markdown (MyST) page.
Return a JSON array of exactly the same length, in the same order, and nothing else: no commentary, no code fence.

Rules, all of them strict:
- Translate the prose naturally and precisely, in the register of good technical documentation. Use the formal address where the language has one.
- Keep unchanged, character for character: everything inside `backticks`, link targets (the part in parentheses after ]), URLs, file names, MyST roles and directives such as {{doc}} {{ref}} {{eval-rst}}, and placeholders such as %(name)s.
- Translate link text and headings, but keep the Markdown syntax: **bold**, *italic*, [text](target), lists, tables with | and line breaks stay as they are.
- Do not translate these names: {glossary}. Keep product, library, model and pipeline names as they are.
- Keep numbers, units, and decimal and thousand separators as in the source.
- If a string needs no translation (it is code, a name or a number), return it unchanged.
- Do not add or drop sentences."""


def build_prompt(language: str) -> str:
    return SYSTEM.format(language=language, glossary=", ".join(GLOSSARY))


Translator = Callable[[Sequence[str]], list[str]]


def api_translator(language: str, model: str, api_key: str, timeout: float = 180.0) -> Translator:
    """Translate a batch of strings with the Claude Messages API (plain HTTPS; no extra dependency)."""
    system = build_prompt(language)

    def call(batch: Sequence[str]) -> list[str]:
        body = json.dumps({
            "model": model, "max_tokens": 16000, "system": system,
            "messages": [{"role": "user", "content": json.dumps(list(batch), ensure_ascii=False)}],
        }).encode()
        request = urllib.request.Request(
            API, data=body, method="POST",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    reply = json.load(response)
                text = "".join(part.get("text", "") for part in reply["content"]).strip()
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
                out = json.loads(text)
                if isinstance(out, list) and len(out) == len(batch) and all(isinstance(x, str) for x in out):
                    return out
                raise ValueError("the reply is not a list of the same length")
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 409, 429, 500, 502, 503, 529) or attempt == 5:
                    raise RuntimeError(f"the API answered {exc.code}: {exc.read()[:300]!r}") from exc
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
                if attempt == 5:
                    raise
            time.sleep(min(60, 2 ** attempt * 2))
        raise RuntimeError("unreachable")

    return call


def batches(items: Sequence[str], max_words: int = 2500, max_items: int = 60) -> list[list[int]]:
    """Group message indexes so that a request stays below ``max_words`` words and ``max_items`` messages."""
    groups: list[list[int]] = []
    current: list[int] = []
    words = 0
    for i, text in enumerate(items):
        n = len(text.split())
        if current and (words + n > max_words or len(current) >= max_items):
            groups.append(current)
            current, words = [], 0
        current.append(i)
        words += n
    if current:
        groups.append(current)
    return groups


def translate_messages(messages: Sequence[str], translator: Translator, max_words: int = 2500) -> list[str]:
    """Translate ``messages``; the result has the same length, with "" where no valid translation was obtained."""
    result = [""] * len(messages)
    for group in batches(messages, max_words):
        source = [messages[i] for i in group]
        try:
            answer = translator(source)
        except Exception as exc:  # noqa: BLE001 - one bad batch must not lose the rest of the page
            print(f"    batch failed ({exc}); its messages stay in English", file=sys.stderr)
            continue
        retry = []
        for i, text in zip(group, answer):
            if valid(messages[i], text):
                result[i] = text
            else:
                retry.append(i)
        for i in retry:                                   # one more try, alone, so one bad message cannot hide behind a batch
            try:
                (text,) = translator([messages[i]])
            except Exception:  # noqa: BLE001
                continue
            if valid(messages[i], text):
                result[i] = text
    return result


def new_catalog(pot: polib.POFile, lang: str) -> polib.POFile:
    po = polib.POFile()
    po.metadata = {
        "Project-Id-Version": "PyGeoVision", "Language": lang, "Language-Team": lang, "MIME-Version": "1.0",
        "Content-Type": "text/plain; charset=UTF-8", "Content-Transfer-Encoding": "8bit",
        "Last-Translator": "PyGeoVision Contributors", "X-Translator": "machine translation, not reviewed",
    }
    for entry in pot:
        po.append(polib.POEntry(msgid=entry.msgid, msgstr=""))
    return po


def page_name(pot_path: Path, pot_dir: Path) -> str:
    return pot_path.relative_to(pot_dir).with_suffix("").as_posix()


def process(pot_path: Path, pot_dir: Path, lang: str, translator: Translator | None, locale: Path = LOCALE,
            dry_run: bool = False, max_words: int = 2500) -> tuple[int, int, int]:
    """Translate one page's untranslated messages into ``locale/<lang>/LC_MESSAGES/<page>.po``.

    Returns (messages translated, messages left in English, words of the messages that needed translating).
    """
    name = page_name(pot_path, pot_dir)
    po_path = locale / lang / "LC_MESSAGES" / (name + ".po")
    pot = polib.pofile(str(pot_path))
    po = polib.pofile(str(po_path)) if po_path.exists() else new_catalog(pot, lang)
    have = {e.msgid: e for e in po}
    for entry in pot:                                        # a page that changed since: add its new messages
        if entry.msgid not in have:
            po.append(polib.POEntry(msgid=entry.msgid, msgstr=""))
    todo = [e for e in po if not e.msgstr and e.msgid and translatable(e.msgid)]
    words = sum(len(e.msgid.split()) for e in todo)
    if dry_run or translator is None or not todo:
        return 0, len(todo), words
    out = translate_messages([e.msgid for e in todo], translator, max_words)
    done = 0
    for entry, text in zip(todo, out):
        if text:
            entry.msgstr, done = text, done + 1
    po_path.parent.mkdir(parents=True, exist_ok=True)
    po.save(str(po_path))
    po.save_as_mofile(str(po_path.with_suffix(".mo")))
    return done, len(todo) - done, words


DOC_REQUIREMENTS = ("sphinx", "furo", "myst_parser", "sphinx_design", "sphinx_copybutton", "sphinx_reredirects", "polib")


def missing_doc_requirements(find_spec: Callable[[str], object] = importlib.util.find_spec) -> list[str]:
    """Names of the packages the documentation build needs that this Python cannot import."""
    return [name for name in DOC_REQUIREMENTS if find_spec(name) is None]


def make_pot(pot_dir: Path) -> None:
    """Extract every message of every page with ``sphinx-build -b gettext``. This needs the documentation's own requirements."""
    missing = missing_doc_requirements()
    if missing:
        sys.exit(
            f"The messages are extracted with Sphinx, and this Python ({sys.executable}) cannot import: {', '.join(missing)}.\n"
            f"Install what the documentation needs, then run this again:\n\n"
            f"    {sys.executable} -m pip install -r {DOCS / 'requirements.txt'}\n"
            f"    {sys.executable} -m pip install -e '{DOCS.parent}[geo,report]'\n")
    print(f"extracting messages with sphinx-build -b gettext into {pot_dir} (about a minute) ...")
    env = dict(os.environ, DOCS_SKIP_OTHER_LANGS="1")
    done = subprocess.run([sys.executable, "-m", "sphinx", "-b", "gettext", "-q", str(DOCS), str(pot_dir)], env=env)
    if done.returncode:
        sys.exit(f"sphinx-build failed (exit code {done.returncode}); the messages above say why. Nothing was translated.")


def main(argv: Sequence[str] | None = None, translator_factory: Callable[[str, str], Translator] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lang", required=True, choices=sorted(LANGUAGE_NAMES))
    ap.add_argument("--pages", nargs="*", default=["*"], help="page names or globs, e.g. 'getting-started/*' (default: all)")
    ap.add_argument("--model", default=os.environ.get("PGV_TRANSLATE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--jobs", type=int, default=3, help="pages translated at the same time (default 3)")
    ap.add_argument("--pot-dir", type=Path, default=POT_DIR)
    ap.add_argument("--refresh", action="store_true", help="extract the messages again even if --pot-dir exists")
    ap.add_argument("--dry-run", action="store_true", help="count what would be translated; send nothing")
    ap.add_argument("--max-words", type=int, default=2500, help="words per request")
    args = ap.parse_args(argv)

    if args.refresh or not args.pot_dir.is_dir():
        make_pot(args.pot_dir)
    pots = sorted(p for p in args.pot_dir.rglob("*.pot") if any(fnmatch.fnmatch(page_name(p, args.pot_dir), g) for g in args.pages))
    if not pots:
        print("no page matches", args.pages)
        return 1
    translator: Translator | None = None
    if not args.dry_run:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if translator_factory is None and not key:
            print("set ANTHROPIC_API_KEY (or use --dry-run to count)", file=sys.stderr)
            return 2
        factory = translator_factory or (lambda language, model: api_translator(language, model, key or ""))
        translator = factory(LANGUAGE_NAMES[args.lang], args.model)

    total = [0, 0, 0]

    def one(pot: Path) -> tuple[str, tuple[int, int, int]]:
        return page_name(pot, args.pot_dir), process(pot, args.pot_dir, args.lang, translator, dry_run=args.dry_run,
                                                     max_words=args.max_words)

    with futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for name, (done, left, words) in pool.map(one, pots):
            for i, v in enumerate((done, left, words)):
                total[i] += v
            if args.dry_run:
                if left:
                    print(f"  {name}: {left} messages, {words} words")
            else:
                print(f"  {name}: {done} translated" + (f", {left} left in English" if left else ""))
    done, left, words = total
    if args.dry_run:
        print(f"\n{left} messages ({words} words) to translate into {LANGUAGE_NAMES[args.lang]} on {len(pots)} pages. Nothing was sent.")
    else:
        print(f"\n{done} translated, {left} left in English. Run docs/tools/check_translations.py, then build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
