"""Fig. 7 of the paper: code/results/figures/F4_depolarizing, redrawn with the depolarizing strength
written p_dep. Same data path and plotting helpers as qpar.analysis.figure_depolarizing; only the
x-axis label and the legend title ("Main grid", the paper's term, for "Main matrix") differ.
Writes <out>/fig7.pdf and fig7.png; nothing under code/results is touched."""
import sys
from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from qpar import analysis as an
from qpar.train import load_base

out = Path(sys.argv[1])
depol_p = load_base()["noise"]["depolarizing_p"]
runs = an.load_runs(an.RESULTS, an.CONFIGS, depol_p)
with matplotlib.rc_context(an.STYLE):
    fig, axes = an.figure(an.DOUBLE, 2.9, 1, 2)
    for ax, ds in zip(axes[0], an.DATASETS):
        p, curves = an.depolarizing_curves(runs, ds, depol_p)
        for i, pl in enumerate(an.PLACEMENTS):
            mean, std = curves[pl]
            an.dodged_errorbar(ax, i, len(an.PLACEMENTS), p, 100 * mean, 100 * std, elinewidth=0.7, **an.marker_style(pl))
        ax.set_xscale("symlog", linthresh=min(depol_p), linscale=0.5)
        ax.set_xlim(-0.25 * min(depol_p), 1.15 * max(depol_p))
        ax.set_xticks(p, ["0"] + [f"{v:g}" for v in depol_p])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"Depolarizing strength $p_{\mathrm{dep}}$ (two-qubit gates $10p_{\mathrm{dep}}$; 0 = noiseless)")
        ax.set_ylabel("Accuracy on the noisy-eval subset (%)")
        ax.set_title(an.NAME[ds], loc="left")
    handles = [Line2D([], [], linestyle="-", **an.marker_style(pl), label=an.NAME[pl]) for pl in an.PLACEMENTS]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, title="Main grid, train_frac = 1.0, "
               "mean over encodings and topologies; error bars ±1 std over seeds")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig7.pdf")
    fig.savefig(out / "fig7.png", dpi=300)
    for ds in an.DATASETS:
        p, curves = an.depolarizing_curves(runs, ds, depol_p)
        print(ds, {pl: [round(100 * v, 2) for v in curves[pl][0]] for pl in an.PLACEMENTS})
