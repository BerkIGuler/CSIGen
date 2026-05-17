#!/usr/bin/env bash
# Compile docs/main.tex to docs/main.pdf and remove build artifacts (keep only the PDF among outputs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "${SCRIPT_DIR}/../docs" && pwd)"
MAIN_TEX="${DOCS_DIR}/main.tex"
MAIN_PDF="${DOCS_DIR}/main.pdf"

if [[ ! -f "${MAIN_TEX}" ]]; then
  echo "error: ${MAIN_TEX} not found" >&2
  exit 1
fi

cd "${DOCS_DIR}"

# minted requires -shell-escape; run twice for stable references if hyperref updates .aux
PDFLATEX=(pdflatex -shell-escape -interaction=nonstopmode -halt-on-error -file-line-error main.tex)
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -pdflatex="pdflatex -shell-escape %O %S" \
    main.tex
  latexmk -c main.tex >/dev/null 2>&1 || true
else
  "${PDFLATEX[@]}"
  "${PDFLATEX[@]}"
fi

if [[ ! -f "${MAIN_PDF}" ]]; then
  echo "error: ${MAIN_PDF} was not produced" >&2
  exit 1
fi

# Remove auxiliary files latexmk -c may miss + minted cache; keep main.tex, assets, and main.pdf
shopt -s nullglob
rm -f main.{acn,acr,alg,aux,bbl,bcf,blg,fdb_latexmk,fls,glg,glo,gls,idx,ilg,ind,ist,lof,log,lot,nav,out,run.xml,snm,spl,synctex.gz,toc,xdv,vrb}
rm -rf _minted-*

echo "Wrote ${MAIN_PDF}"
