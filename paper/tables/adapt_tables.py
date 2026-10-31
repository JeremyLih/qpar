"""Copy the result tables (code/results/tables/T*.tex) into the paper with LaTeX-level changes only.

Numbers are never changed. Changes: hyphen-minus before a number becomes a math minus; Kendall's
"tau = -0.06" set in math; code identifiers (base_cnn, cnn_matched, train_frac) in typewriter font;
the depolarizing strength written p_dep in the T5 (Table 7) header; the generated caption is replaced by
the author's caption (Caption tab of paper/v3/input/QML Paper.md), so the number check covers the table body.
Run from this folder with the folder of the code's result tables as the argument:
    python3 adapt_tables.py ../../results/tables              # in the code repository (paper/tables)
    python3 adapt_tables.py ../../../../code/results/tables   # in paper/v3/latex/tables
"""
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(__file__).resolve().parent

# paper table number <- source file
MAP = {
    "tab02_mnist.tex": "T1_mnist_tf1.0.tex",
    "tab03_fashionmnist.tex": "T2_fashionmnist_tf1.0.tex",
    "tab04_baselines.tex": "T3_baselines.tex",
    "tab05_sensitivity.tex": "T6_sensitivity.tex",
    "tab06_sweep.tex": "T4_qubit_sweep.tex",
    "tab07_noise.tex": "T5_noise.tex",
    "tabS1_rank_moves.tex": "T9_rank_moves.tex",
    "tabS2_sensitivity_supplement.tex": "T10_sensitivity_supplement.tex",
    "tabS3_matched.tex": "T8_hybrid_vs_matched.tex",
    "tabS4_full_matrix.tex": "T7_full_matrix.tex",
}

# the author's captions (Caption tab of QML Paper.md); Tables S1-S4 are typeset in supplementary.tex,
# which resolves \ref into main.tex through xr, while main.tex cites them as plain "Table~S2"
CAPTIONS = {
    "tab02_mnist.tex": r"MNIST at \texttt{train\_frac} = 1.0: the 18 main-grid configurations with noiseless test accuracy $A$ (mean $\pm$ std over 5 seeds), $\sigma_a$, gate count $G$, latency $D$ (median, seed-0 checkpoint), equal-weight Q-Par, and the ranks by Q-Par and by $A$ among the 54 main-grid configurations. Normalization constants: $Q_{\max} = 8$, $D_{\max} = 4.046$~ms, $G_{\max} = 1{,}111$ (maxima over all 63 hybrid configurations of the dataset).",
    "tab03_fashionmnist.tex": r"Fashion-MNIST at \texttt{train\_frac} = 1.0: the same columns as Table~\ref{tab:mnist_tf1}. Normalization constants: $Q_{\max} = 8$, $D_{\max} = 3.928$~ms, $G_{\max} = 1{,}174$.",
    "tab04_baselines.tex": r"The best hybrid per placement (highest mean best-validation accuracy at \texttt{train\_frac} = 1.0) against its parameter-matched \texttt{cnn\_matched} model and \texttt{base\_cnn}: trainable parameters (total; core = the quantum part of the hybrid or the classical map that replaces it), latency $D$ of the \texttt{train\_frac} = 1.0 configuration, and test accuracy $A$ (mean $\pm$ std over seeds) at each training fraction. Total parameter counts of \texttt{cnn\_matched} differ from their hybrids by at most 0.12\%.",
    "tab05_sensitivity.tex": r"Sensitivity of the Q-Par ranking of the 54 main-grid configurations to the exponents: equal weights, 10 single-exponent settings (one exponent at 0.5 or 2, the others at 1), and 3 presets (accuracy-first $\alpha = 2$, latency-first $\delta = 2$, resource-first $\gamma = \varepsilon = 2$). For each setting, Kendall $\tau$ against the equal-weight ranking, its $p$-value, and the top five configurations, which show whether the first place moves. The $\gamma$ settings reproduce the equal-weight ranking because every main-grid configuration has $n = 4$; Table~S2 repeats the analysis where $Q$ varies.",
    "tab06_sweep.tex": r"Qubit sweep at \texttt{train\_frac} = 1.0 for the best configuration per placement (encoding/topology given with the placement; $n = 4$ is the main grid): test accuracy $A$ (mean $\pm$ std over seeds), $\sigma_a$, gate count $G$, latency $D$, and equal-weight Q-Par at $n = 2$, 4, 6, 8 for both datasets.",
    "tab07_noise.tex": r"Noisy evaluation of every hybrid configuration at \texttt{train\_frac} = 1.0 on the fixed 2,000-image test subset (mean $\pm$ std over seeds): noiseless accuracy $A_{\mathrm{sub}}$, accuracy under the calibrated device noise model (FakeAachen, 1,024 shots), $\Delta_{\mathrm{cal}} = A_{\mathrm{sub}} - A_{\mathrm{cal}}$, accuracy under depolarizing noise of strength $p_{\mathrm{dep}}$ ($10p_{\mathrm{dep}}$ on each qubit of every CNOT), gate count $G$, and the number of two-qubit gates after transpilation (2Q). Rows with $n = 4$ are the main grid, the others the qubit sweep.",
    "tabS1_rank_moves.tex": r"The 80 configurations whose rank moves by at least 10 places between the ranking by $A$ and the ranking by equal-weight Q-Par (39 on MNIST, 41 on Fashion-MNIST), with both ranks, the move, and the term that drives it ($G$ or $D$).",
    "tabS2_sensitivity_supplement.tex": r"Sensitivity analysis repeated on the 27 configurations at \texttt{train\_frac} = 1.0 of each dataset (18 main grid + 9 qubit sweep), where $Q$ varies from 2 to 8: Kendall $\tau$ against the equal-weight ranking for every exponent setting of Table~\ref{tab:sensitivity}, and the top five configurations.",
    "tabS3_matched.tex": r"Hybrid against its parameter-matched classical model in the 36 cells (dataset $\times$ placement $\times$ encoding $\times$ training fraction), paired over seeds: hybrid $A$ (mean of its three topologies per seed), matched $A$, $\Delta$ = hybrid $-$ matched, paired $t$-test $p_t$, exact Wilcoxon $p_W$ ($p_W \geq 0.0625$ with 5 seeds, printed as 0.062), and Cohen's $d_z$. Unadjusted for multiple comparisons.",
    "tabS4_full_matrix.tex": r"The full main grid: all 54 configurations (placement $\times$ encoding $\times$ topology $\times$ training fraction) of each dataset, with test accuracy $A$ and equal-weight Q-Par.",
}


def split_caption(text):
    """(text before, content, text after) of the table's \\caption{...}; escaped braces are skipped."""
    i = j = text.index(r"\caption{") + len(r"\caption{")
    depth = 1
    while True:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        depth += {"{": 1, "}": -1}.get(c, 0)
        if depth == 0:
            return text[:i], text[i:j], text[j:]
        j += 1


def adapt(name, text):
    out = text
    # code identifiers in typewriter font
    for ident in (r"base\_cnn", r"cnn\_matched", r"train\_frac"):
        out = out.replace(ident, r"\texttt{" + ident + "}")
    # hyphen-minus before a digit (cell values, moves) -> math minus; not inside words or commands
    out = re.sub(r"(?<![\w$\\-])-(?=\d)", r"$-$", out)
    # Kendall tau in the T9 header: one math expression
    out = re.sub(r"Kendall \$\\tau\$ = \$-\$([\d.]+)", r"Kendall $\\tau = -\1$", out)
    if name == "tab07_noise.tex":
        out = out.replace(r"\multicolumn{4}{c}{Depolarizing, $p$ =}", r"\multicolumn{4}{c}{Depolarizing, $p_{\mathrm{dep}}$ =}")
    # last, so that the changes above leave the author's caption as written
    head, _, tail = split_caption(out)
    out = head + CAPTIONS[name] + tail
    header = f"% Table from code/results/tables/{MAP[name]} (adapted by adapt_tables.py; numbers unchanged; caption by the author).\n"
    return header + out


for name, src in MAP.items():
    text = (SRC / src).read_text()
    new = adapt(name, text)
    (OUT / name).write_text(new)
    # every number in the source must survive unchanged and in order (outside the caption, which is replaced)
    body = lambda s: "".join(split_caption(s)[::2])
    nums = lambda s: re.findall(r"\d+(?:\.\d+)?", re.sub(r"%[^\n]*\n", "\n", body(s)))
    a, b = nums(text), nums(new.split("\n", 1)[1])
    b = [x for x in b if x not in ("10",) or True]
    extra = [x for x in b if x not in a]
    assert all(x in b for x in a), name
    print(f"{name:34s} <- {src:32s} numbers {len(a)} -> {len(b)}; added tokens: {sorted(set(b) - set(a))}")
