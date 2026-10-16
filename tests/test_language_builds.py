"""The language builds must never leave a page pointing at static files that are not there."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("langstatic", ROOT / "docs" / "tools" / "langstatic.py")
langstatic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(langstatic)


def _site(tmp_path, with_lang_static):
    root = tmp_path / "site" / "_static"
    root.mkdir(parents=True)
    for name in langstatic.NEEDED:
        (root / name).write_text("root " + name)
    (root / "img").mkdir()
    (root / "img" / "logo.svg").write_text("<svg/>")
    lang = tmp_path / "site" / "de"
    lang.mkdir()
    if with_lang_static:
        (lang / "_static").mkdir()
        (lang / "_static" / "documentation_options.js").write_text("german options")
    return tmp_path / "site"


def test_a_language_build_that_stopped_before_copying_its_static_files_is_filled_in(tmp_path):
    site = _site(tmp_path, with_lang_static=False)
    assert langstatic.fill_missing_static(site, "de") == []
    for name in langstatic.NEEDED:
        assert (site / "de" / "_static" / name).is_file()
    assert (site / "de" / "_static" / "img" / "logo.svg").is_file()


def test_files_the_language_build_wrote_itself_are_never_replaced(tmp_path):
    site = _site(tmp_path, with_lang_static=True)
    assert langstatic.fill_missing_static(site, "de") == []
    assert (site / "de" / "_static" / "documentation_options.js").read_text() == "german options"


def test_what_cannot_be_filled_in_is_reported(tmp_path):
    site = _site(tmp_path, with_lang_static=False)
    (site / "_static" / "pgv.js").unlink()
    assert langstatic.fill_missing_static(site, "de") == ["pgv.js"]
    assert langstatic.fill_missing_static(tmp_path / "empty", "de") == list(langstatic.NEEDED)


def test_conf_uses_the_helper_and_reports_a_failed_language_build():
    text = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    assert "fill_missing_static" in text and "language build failed" in text
