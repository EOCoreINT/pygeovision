#!/usr/bin/env bash
# Build the site: English at _build/html (the site root) and every other language in _build/html/<lang>/.
# The English build starts the other languages itself (see conf.py), so a plain
#   python -m sphinx -b html docs docs/_build/html
# (as Read the Docs runs it) gives the same result. -W makes a warning fail the English build.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf _build/html
python3 -m sphinx -b html -W --keep-going -q . _build/html
echo "built: $(pwd)/_build/html  (languages: $(ls -d _build/html/*/ | xargs -n1 basename | grep -E '^(fr|es|pt|de|zh_CN|ar)$' | tr '\n' ' '))"
