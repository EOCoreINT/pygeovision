# Translating the PyGeoVision documentation

Languages: French (`fr`), Spanish (`es`), Portuguese (`pt`), German (`de`), Simplified Chinese (`zh_CN`) and Arabic (`ar`).
English is the source. Text without a translation falls back to English, so every language build is complete; a page with no
translation file shows a "partly translated" note.

## What is translated

| File | Holds |
|---|---|
| `<lang>/LC_MESSAGES/chrome.po` | The top navigation bar, breadcrumbs, the language menu and the section names. Read by `conf.py` |
| `<lang>/LC_MESSAGES/<page>.po` | One file per translated page, mirroring the page's path (`index.po`, `getting-started/quickstart.po`) |

Code blocks, command lines, file names, link targets and `{directive}` names are never translated.

## Add or update a page

```bash
# 1. Extract the current English text of every page into .pot templates
python -m sphinx -b gettext docs /tmp/pot

# 2. Create or update one language's file for one page
pip install sphinx-intl
sphinx-intl update -p /tmp/pot -l de --locale-dir docs/locale     # all pages, or copy just the .pot you want

# 3. Fill in the msgstr entries, keeping every Markdown link target and `code span` exactly as in the msgid
$EDITOR docs/locale/de/LC_MESSAGES/getting-started/quickstart.po

# 4. Check, then build
python docs/tools/check_translations.py
DOCS_SKIP_OTHER_LANGS=1 python -m sphinx -b html -D language=de docs /tmp/out-de
```

`check_translations.py` fails when a translation drops or changes a link target or an inline code span, when `chrome.po` lacks a
string the templates use, and when a `.po` file belongs to a page that no longer exists.

## Translate every page with the Claude API

Only the navigation, the home page and the quickstart are translated by hand, so in a language build most of the text is still
English. `tools/translate_po.py` translates the rest. It uses the same `.pot` and `.po` files as above, so what it writes is
reviewed and corrected the same way afterwards.

```bash
pip install -r docs/requirements.txt && pip install -e '.[geo,report]'   # once: extracting the messages runs Sphinx
export ANTHROPIC_API_KEY=...                                   # your own key
python docs/tools/translate_po.py --lang de --dry-run          # counts messages and words; nothing is sent
python docs/tools/translate_po.py --lang de --pages 'getting-started/*' 'pipelines/*'   # a few sections first
python docs/tools/translate_po.py --lang de                    # every page; run it again after an error, it resumes
python docs/tools/check_translations.py
DOCS_SKIP_OTHER_LANGS=1 python -m sphinx -b html -D language=de docs /tmp/out-de
```

At the time of writing the English docs hold about 22,000 messages (260,000 words) beyond what is already translated, so a
full language is a few hundred thousand tokens in and out. Use `--pages` to start with the pages people read first.

What it protects: inline code, link targets, `{role}` and `{directive}` names, URLs and product names are checked after every
answer. A translation that changes one is rejected, asked for once more, and left in English if it fails again, so a bad answer
cannot break a link or a command. The `.po` header says `X-Translator: machine translation, not reviewed`. It is a starting
point for a native speaker, not a replacement, and the rules for translators below still apply.

## Add a language

1. Add `("xx", "Name in that language")` to `LANGUAGES` in `docs/conf.py` and `docs/tools/check_translations.py`.
2. Create `docs/locale/xx/LC_MESSAGES/chrome.po` from another language's file and translate it.
3. Right-to-left scripts: add the code to the `pgv_rtl` line in `conf.py` and a rule like the `html[lang^="ar"]` ones in `_static/custom.css`.
4. Translate `index.po` first; it carries the landing page and the section names in the sidebar.

## Rules for translators

- Keep product and command names in English: PyGeoVision, pygeofetch, `pygeovision doctor`, `--checkpoint`, `ready`, `needs_weights`.
- Keep numbers that the code produces (`7/7 components healthy`) exactly as printed.
- When a sentence says something is *not verified* or *not measured*, the translation must say the same. Do not soften it.
