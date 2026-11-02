#!/usr/bin/env bash
# Run from the root of your pygeovision repo:  bash /path/to/fixed_files/apply.sh
set -e
here="$(cd "$(dirname "$0")" && pwd)"
(cd "$here" && find . -type f ! -name apply.sh ! -name DELETE_THESE.txt -print0) | while IFS= read -r -d '' f; do
  mkdir -p "$(dirname "$f")"; cp -v "$here/$f" "$f"
done
while read -r f; do [ -n "$f" ] && [ -e "$f" ] && rm -rfv "$f"; done < "$here/DELETE_THESE.txt"
echo done
