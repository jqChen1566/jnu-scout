#!/usr/bin/env bash
# Build the JNUScout manual.
# Pass sequence (project rule for documents with a bibliography + index):
#   xelatex -> biber -> makeindex -> xelatex -> xelatex
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n== %s ==\n' "$*"; "$@"; }

step xelatex -interaction=nonstopmode -halt-on-error main.tex
step biber main
if [ -f main.idx ]; then
  step makeindex -q main.idx
fi
step xelatex -interaction=nonstopmode -halt-on-error main.tex
step xelatex -interaction=nonstopmode -halt-on-error main.tex

printf '\n== acceptance summary ==\n'
fail=0
if grep -E "LaTeX Warning: (There were undefined references|Label\(s\) may have changed|Citation .* undefined|Reference .* undefined)" main.log; then
  fail=1
else
  echo "  no undefined references / citation problems"
fi
pages=$(grep -oE "Output written on main.pdf \(([0-9]+) pages" main.log | grep -oE "[0-9]+" | head -1)
echo "  pages: ${pages:-?}"
if [ -s main.pdf ]; then
  echo "  main.pdf built: $(stat -c %s main.pdf 2>/dev/null || wc -c < main.pdf) bytes"
else
  echo "  main.pdf missing"; fail=1
fi
exit $fail
