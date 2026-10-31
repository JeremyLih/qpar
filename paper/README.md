# Paper sources

LaTeX sources of "Q-Par: A Resource-Aware Composite Metric for Comparing Hybrid CNN–QNN Models
under NISQ Constraints" (`main.tex`, `references.bib`) and of its supplementary material
(`supplementary.tex`), with the scripts that regenerate their tables and figures from `../results/`.

| In the paper | Made from |
|---|---|
| Table 1 (design matrix) | `tables/tab01_design.tex`, written by hand |
| Tables 2–7, S1–S4 | `results/tables/T1–T10.tex`, adapted by `tables/adapt_tables.py`: LaTeX form and the author's captions only; the script checks that every number survives unchanged and lists which table comes from which file |
| Fig. 1 (architecture) | `figures/fig1.tex` (TikZ) |
| Figs. 2–6 | `results/figures/` F1, F2, F6, F5 and F3, unchanged |
| Fig. 7 | `figures/fig7_make.py`: F4 redrawn from the run records with the depolarizing strength written p_dep |

## Build

```bash
paper/build.sh
```

regenerates every table and figure. It needs uv (the environment of `../uv.lock`) and a TeX Live
installation with pdfLaTeX. The generated files are not committed.

```bash
paper/build.sh pdf
```

also builds `main.pdf` and then `supplementary.pdf`, which reads the paper's section, figure and
table numbers from `main.aux`. This needs latexmk, BibTeX and the class, style, font and logo files
of the official IEEE Access LaTeX template (`ACCESS_latex_template_20260513`, linked from
ieeeaccess.ieee.org under Submission Guidelines), which are not redistributed here. Copy
`ieeeaccess.cls`, `IEEEtran.cls`, `IEEEtran.bst`, `spotcolor.sty`, the font files (`t1-*.pfb`,
`t1-*.tfm`, `t1-*.map`, `t1*.fd`), `logo.png`, `notaglinelogo.png` and `bullet.png` from the template
into `paper/` first.
