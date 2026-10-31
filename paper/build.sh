#!/usr/bin/env bash
# Rebuilds every table and figure of the paper from results/ (paper/README.md):
#
#   paper/build.sh       # Tables 2-7 and S1-S4, Figs. 1-7 and the ORCID icon (needs uv and pdflatex)
#   paper/build.sh pdf   # also main.pdf and supplementary.pdf (needs the IEEE Access template files)
#
# Table 1 is written by hand (tables/tab01_design.tex); everything else is generated here.
set -euo pipefail
cd "$(dirname "$0")"

pdftex_once() {  # one pdflatex run; on failure, show the end of its log
  pdflatex -interaction=nonstopmode -halt-on-error "$1" > /dev/null || { tail -n 20 "${1%.tex}.log"; exit 1; }
}

uv run python tables/adapt_tables.py ../results/tables  # Tables 2-7, S1-S4 from results/tables (T1-T10)
cp ../results/figures/F1_accuracy_vs_train_frac.pdf figures/fig2.pdf  # Figs. 2-6: the analysis figures, unchanged
cp ../results/figures/F2_rank_rank.pdf figures/fig3.pdf
cp ../results/figures/F6_sensitivity_heatmap.pdf figures/fig4.pdf
cp ../results/figures/F5_qubit_sweep.pdf figures/fig5.pdf
cp ../results/figures/F3_delta_cal_vs_G.pdf figures/fig6.pdf
uv run python figures/fig7_make.py figures            # Fig. 7: F4 redrawn with p_dep on the x-axis
(cd figures && pdftex_once fig1.tex && pdftex_once orcid.tex)  # Fig. 1 (TikZ) and the ORCID icon

if [ "${1:-}" = pdf ]; then
  [ -f ieeeaccess.cls ] || { echo "paper/build.sh pdf: copy the IEEE Access template files into paper/ first (paper/README.md)" >&2; exit 2; }
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex           # pdfLaTeX + BibTeX
  latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex  # reads the paper's numbers from main.aux
fi
