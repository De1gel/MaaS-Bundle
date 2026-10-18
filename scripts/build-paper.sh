#!/bin/bash
set -euo pipefail

# Keep every generated file outside the Overleaf local replica.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$project_dir/Paper/MaaS bundle design"
build_dir="$project_dir/Paper/build"
tex_file="${1:-MaaS bundle design under uncertain trip frequency.tex}"

# GUI terminals on macOS may not inherit the TeX distribution's PATH.
export PATH="/Library/TeX/texbin:$PATH"
if ! command -v latexmk >/dev/null 2>&1; then
    echo "latexmk is required. Install a TeX distribution with XeLaTeX and BibTeX." >&2
    exit 1
fi

cd -- "$source_dir"
if [[ ! -f "$tex_file" ]]; then
    echo "LaTeX source not found: $tex_file" >&2
    exit 1
fi
mkdir -p -- "$build_dir"
exec latexmk -xelatex -synctex=1 -interaction=nonstopmode \
    -halt-on-error -file-line-error "-outdir=$build_dir" "$tex_file"
