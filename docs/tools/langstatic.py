"""Make sure a language build has the static files its pages load.

Every page of ``<site>/<lang>/`` loads ``<site>/<lang>/_static/pgv.js`` (the sidebar accordion), the stylesheet and the logo.
If that language build stopped early, the pages exist but those files do not, and the page loses its styling and its
sidebar. ``fill_missing_static`` copies whatever is missing from the English build and says what is still not there.
"""
from __future__ import annotations

import shutil
from pathlib import Path

NEEDED = ("pgv.js", "custom.css", "logo.svg", "documentation_options.js")


def fill_missing_static(site: Path, code: str) -> list[str]:
    """Copy every file of ``<site>/_static`` that ``<site>/<code>/_static`` lacks; return the needed files still missing.

    Files the language build wrote itself (its own ``documentation_options.js``, its search index) are never replaced.
    """
    site = Path(site)
    root_static = site / "_static"
    lang_static = site / code / "_static"
    if root_static.is_dir() and (site / code).is_dir():
        for path in root_static.rglob("*"):
            target = lang_static / path.relative_to(root_static)
            if path.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    return [name for name in NEEDED if not (lang_static / name).is_file()]
