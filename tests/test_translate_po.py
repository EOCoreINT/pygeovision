"""The translation tool (docs/tools/translate_po.py) with a stand-in translator: nothing is sent anywhere."""
import importlib.util
from pathlib import Path

import polib

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("translate_po", ROOT / "docs" / "tools" / "translate_po.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

MESSAGES = [
    "Run the **pipeline** with `pygeovision channel land_cover` and read [the guide](guide.md).",
    "Install it.",
    "`pygeovision doctor`",
    "42",
    "See {doc}`installation` for the extras.",
    "https://example.org/data.tif",
]


def fake(batch):
    return ["[de] " + text for text in batch]


def test_what_is_translatable():
    assert [tp.translatable(m) for m in MESSAGES] == [True, True, False, False, True, False]


def test_a_translation_must_keep_code_links_roles_and_fences():
    src = MESSAGES[0]
    assert tp.valid(src, "Starten Sie die **Pipeline** mit `pygeovision channel land_cover` und lesen Sie [die Anleitung](guide.md).")
    assert not tp.valid(src, "Starten Sie die **Pipeline** mit `pygeovision kanal land_cover` und lesen Sie [die Anleitung](guide.md).")
    assert not tp.valid(src, "Starten Sie die Pipeline mit `pygeovision channel land_cover` und lesen Sie [die Anleitung](anleitung.md).")
    assert not tp.valid(MESSAGES[4], "Siehe :doc:`installation` für die Extras.")
    assert not tp.valid(src, "   ")


def test_batches_respect_the_word_and_item_limits():
    items = ["word " * 10] * 25
    groups = tp.batches(items, max_words=60, max_items=4)
    assert sum(groups, []) == list(range(25)) and all(len(g) <= 4 for g in groups)
    assert tp.batches([], 10) == []


def test_a_rejected_translation_is_asked_again_alone_and_then_left_in_english():
    calls = []

    def flaky(batch):
        calls.append(len(batch))
        return [t.replace("`pygeovision channel land_cover`", "`falsch`") if "land_cover" in t else "[de] " + t for t in batch]

    out = tp.translate_messages(MESSAGES[:2], flaky)
    assert out[0] == "" and out[1] == "[de] Install it."
    assert calls == [2, 1]                                 # the batch, then the rejected message alone


def test_a_failing_request_leaves_its_messages_in_english_and_keeps_the_rest():
    def broken(batch):
        raise RuntimeError("the API answered 500")

    assert tp.translate_messages(MESSAGES[:2], broken) == ["", ""]


def _pot(tmp_path, messages, name="guide"):
    pot_dir = tmp_path / "gettext"
    pot_dir.mkdir(exist_ok=True)
    pot = polib.POFile()
    for m in messages:
        pot.append(polib.POEntry(msgid=m))
    pot.save(str(pot_dir / (name + ".pot")))
    return pot_dir


def test_a_page_is_written_to_the_locale_folder_and_is_resumable(tmp_path):
    pot_dir = _pot(tmp_path, MESSAGES)
    locale = tmp_path / "locale"
    pot = pot_dir / "guide.pot"
    done, left, words = tp.process(pot, pot_dir, "de", fake, locale=locale)
    assert (done, left) == (3, 0) and words > 0
    po = polib.pofile(str(locale / "de" / "LC_MESSAGES" / "guide.po"))
    by_id = {e.msgid: e.msgstr for e in po}
    assert by_id[MESSAGES[1]] == "[de] Install it."
    assert by_id[MESSAGES[2]] == ""                         # code is not translated and not sent
    assert po.metadata["Language"] == "de" and "machine" in po.metadata["X-Translator"]
    assert (locale / "de" / "LC_MESSAGES" / "guide.mo").exists()
    sent = []
    again = tp.process(pot, pot_dir, "de", lambda b: sent.append(b) or fake(b), locale=locale)
    assert again[0] == 0 and sent == []                    # nothing left to translate, nothing sent


def test_a_message_added_to_a_page_later_is_picked_up(tmp_path):
    pot_dir = _pot(tmp_path, MESSAGES[:2])
    locale = tmp_path / "locale"
    tp.process(pot_dir / "guide.pot", pot_dir, "de", fake, locale=locale)
    _pot(tmp_path, MESSAGES[:2] + ["A new paragraph of text."])
    done, left, _ = tp.process(pot_dir / "guide.pot", pot_dir, "de", fake, locale=locale)
    assert (done, left) == (1, 0)


def test_dry_run_counts_and_writes_nothing(tmp_path):
    pot_dir = _pot(tmp_path, MESSAGES)
    done, left, words = tp.process(pot_dir / "guide.pot", pot_dir, "de", None, locale=tmp_path / "locale", dry_run=True)
    assert (done, left) == (0, 3) and words > 5
    assert not (tmp_path / "locale").exists()


def test_main_translates_the_pages_that_match(tmp_path, monkeypatch, capsys):
    pot_dir = _pot(tmp_path, MESSAGES, "guide")
    _pot(tmp_path, MESSAGES, "other")
    monkeypatch.setattr(tp, "LOCALE", tmp_path / "locale")
    monkeypatch.setattr(tp, "process", lambda *a, **k: _orig(*a, **{**k, "locale": tmp_path / "locale"}))
    rc = tp.main(["--lang", "de", "--pot-dir", str(pot_dir), "--pages", "guide"], translator_factory=lambda lang, model: fake)
    assert rc == 0
    assert (tmp_path / "locale" / "de" / "LC_MESSAGES" / "guide.po").exists()
    assert not (tmp_path / "locale" / "de" / "LC_MESSAGES" / "other.po").exists()
    assert "3 translated" in capsys.readouterr().out


_orig = tp.process


def test_missing_documentation_requirements_are_named_before_sphinx_is_started():
    found = {"polib"}
    missing = tp.missing_doc_requirements(lambda name: object() if name in found else None)
    assert "sphinx" in missing and "furo" in missing and "polib" not in missing
    assert tp.missing_doc_requirements(lambda name: object()) == []


def test_make_pot_stops_with_the_install_command_when_sphinx_is_missing(monkeypatch):
    import pytest
    monkeypatch.setattr(tp, "missing_doc_requirements", lambda: ["sphinx", "furo"])
    with pytest.raises(SystemExit) as stop:
        tp.make_pot(Path("/nonexistent"))
    assert "pip install -r" in str(stop.value) and "sphinx, furo" in str(stop.value)
