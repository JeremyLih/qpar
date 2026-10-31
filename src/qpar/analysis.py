"""Summary tables, statistics and figures (protocol Sections 8, 12, 13).

    uv run python -m qpar.analysis      # after the Mac latency step (protocol 7)

Inputs: results/runs/*.json (noise results merged, or still in results/noise/), results/latency.csv,
results/sweep_selection_<dataset>.json. Outputs in results/: runs.csv, summary.csv, qpar_maxima.csv,
sensitivity.csv (+ sensitivity_summary.csv), sensitivity_supplement_tf1*.csv, rank_comparison.csv,
rank_moves.csv, stats_*.csv, tables/T1-T10.tex, figures/F1-F6.{pdf,png}. Every number in the paper
comes from these files; the analysis choices are recorded in results/DECISIONS.md §14.
"""

import json
import math

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator
from matplotlib.transforms import offset_copy
from scipy import stats

from qpar import ROOT
from qpar.qpar import EQUAL, EXPONENTS, qpar, sensitivity_settings
from qpar.train import RESULTS, config_id, load_base

CONFIGS = ROOT / "configs"
CONFIG_DIRS = ("matrix", "baselines", "sweep")
DATASETS = ("mnist", "fashionmnist")
PLACEMENTS = ("pre", "middle", "post")
ENCODINGS = ("angle", "amplitude")
TOPOLOGIES = ("linear", "fullyconnected", "random")
TRAIN_FRACS = (0.2, 0.5, 1.0)
QUBIT_SWEEP = (2, 4, 6, 8)  # n = 4 is the main matrix (9)
MAIN_N = 4
MAIN_BLOCKS = {"mnist": "M1", "fashionmnist": "M2"}
MOVE_THRESHOLD = 10  # 8.3

NAME = {"mnist": "MNIST", "fashionmnist": "Fashion-MNIST", "pre": "Pre", "middle": "Middle", "post": "Post",
        "angle": "Angle", "amplitude": "Amplitude", "linear": "Linear", "fullyconnected": "FC", "random": "Random"}
SHORT = {"pre": "Pre", "middle": "Mid", "post": "Post", "angle": "Ang", "amplitude": "Amp", "linear": "Lin",
         "fullyconnected": "FC", "random": "Rnd"}


# ============================================================================= loading (12.1)

CONFIG_KEYS = ("kind", "block", "dataset", "placement", "encoding", "topology", "n_qubits", "train_frac")
CAL_KEYS = ("acc_noise_calibrated", "twoq_transpiled", "depth_transpiled", "noise_backend", "shots")
MERGED_NOISE = (*CAL_KEYS, "delta_cal", "acc_depol")
NUMERIC = ("n_qubits", "epochs_to_threshold", "G", "G_variational", "G_encoding", "params_matched_map",
           "params_matched_target", "skipped_steps_total", "acc_noise_calibrated", "twoq_transpiled",
           "depth_transpiled")


def study_configs(configs=CONFIGS):
    """config_id of every configuration in the study: main matrices, baselines, selected sweep."""
    return {p.stem for d in CONFIG_DIRS for p in (configs / d).glob("*.yaml")}


def flatten(record):
    """A run JSON as one flat row: config fields as columns, nested dicts as parent.key, the
    per-epoch log as eNN.field, lists of scalars space-separated. Noise fields: noise_fields()."""
    row = {"run_id": record["run_id"], "config_id": record["config_id"], "seed": record["seed"],
           **{k: record["config"].get(k) for k in CONFIG_KEYS}}
    for key, value in record.items():
        if key in row or key == "config" or key in MERGED_NOISE:
            continue
        if key == "epochs":
            row.update({f"e{e['epoch']:02d}.{k}": v for e in value for k, v in e.items() if k != "epoch"})
        elif isinstance(value, dict):
            row.update({f"{key}.{k}": v for k, v in value.items()})
        elif isinstance(value, list):
            row[key] = " ".join(map(str, value))
        else:
            row[key] = value
    return row


def noise_fields(record, noise_dir, depol_p):
    """A run's 6.2/6.3 results: merged into its JSON (12.1) or, before the merge, in
    results/noise/<run>.<part>.json. Classical models and parts not yet run: missing (NaN)."""
    if "acc_depol" in record:
        cal, depol = record, record["acc_depol"] or {}
    else:
        cal = _read(noise_dir / f"{record['run_id']}.cal.json")
        depol = {str(p): _read(noise_dir / f"{record['run_id']}.depol{p}.json").get("acc_depol") for p in depol_p}
    fields = {**{k: cal.get(k) for k in CAL_KEYS}, **{f"acc_depol_{p}": depol.get(str(p)) for p in depol_p}}
    backend = fields.pop("noise_backend") or {}  # name, qubits, basis gates, calibration date
    return {**fields, **{f"noise_backend.{k}": " ".join(v) if isinstance(v, list) else v for k, v in backend.items()}}


def _read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def load_runs(results=RESULTS, configs=CONFIGS, depol_p=None):
    """One flat row per completed run of the study's configurations; run JSONs of any other
    configuration (a discarded sweep candidate) are skipped."""
    depol_p = load_base()["noise"]["depolarizing_p"] if depol_p is None else depol_p
    wanted = study_configs(configs)
    rows = []
    for path in sorted((results / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record["config_id"] in wanted:
            rows.append({**flatten(record), **noise_fields(record, results / "noise", depol_p)})
    runs = pd.DataFrame(rows)
    for col in (*NUMERIC, *(f"acc_depol_{p}" for p in depol_p)):
        runs[col] = pd.to_numeric(runs[col], errors="coerce") if col in runs else np.nan
    # Runs trained before the non-finite-gradient guard (DECISIONS.md §13) carry no count; a
    # non-finite gradient would have crashed them, so they completed with none skipped.
    runs["skipped_steps_total"] = runs["skipped_steps_total"].fillna(0).astype(int)
    runs["delta_cal"] = runs["A_subset_noiseless"] - runs["acc_noise_calibrated"]
    return runs


def check_complete(runs, seeds, depol_p, configs=CONFIGS):
    """The final analysis needs every run of every configuration and all noise results."""
    counts = runs.groupby("config_id")["seed"].nunique()
    missing = sorted(c for c in study_configs(configs) if counts.get(c, 0) != len(seeds))
    hybrid = runs[runs["kind"] == "hybrid"]
    cols = ["acc_noise_calibrated", *(f"acc_depol_{p}" for p in depol_p)]
    no_noise = hybrid.loc[hybrid[cols].isna().any(axis=1), "run_id"].tolist()
    if missing or no_noise:
        raise SystemExit(f"incomplete results: {len(missing)} configurations without all {len(seeds)} seeds "
                         f"(e.g. {missing[:3]}), {len(no_noise)} hybrid runs without every noise result "
                         f"(e.g. {no_noise[:3]}); sync first")


# ============================================================================= summary (12.1, 13)

CONSTANT = ("kind", "block", "dataset", "placement", "encoding", "topology", "n_qubits", "train_frac",
            "params_total", "params_quantum", "params_classical", "matched_map", "params_matched_map",
            "params_matched_target", "G", "G_variational", "G_encoding", "twoq_transpiled", "depth_transpiled")


def ci95(values):
    """Half-width of the 95 % confidence interval of the mean, t-distribution with n - 1 dof (13)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    return float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan


def rank_desc(values):
    """Rank 1 = highest; ties share the best rank."""
    return values.rank(ascending=False, method="min")


def main_mask(summary, dataset):
    return (summary["dataset"] == dataset) & (summary["block"] == MAIN_BLOCKS[dataset])


def qpar_of(rows, exponents):
    return qpar(rows["A_mean"], rows["sigma_a"], rows["Q"], rows["D_ms"], rows["G"], rows["Q_max"],
                rows["D_max"], rows["G_max"], **exponents)


def summarize(runs, latency=None, depol_p=None):
    """One row per configuration (12.1): mean/std/95 % CI of A over seeds, sigma_a, Q, D, G,
    equal-weight Q-Par with Q_max/D_max/G_max over every hybrid configuration of the dataset (main
    matrix + qubit sweep, 8.1), ranks among the dataset's 54 main configurations, noise means and
    stds, mean epochs_to_threshold over the seeds that reached it, parameter counts."""
    depol_p = load_base()["noise"]["depolarizing_p"] if depol_p is None else depol_p
    metrics = ("A", "best_val_acc", "A_subset_noiseless", "acc_noise_calibrated", "delta_cal",
               *(f"acc_depol_{p}" for p in depol_p))
    rows = []
    for cid, g in runs.groupby("config_id", sort=True):
        row = {"config_id": cid}
        for col in CONSTANT:
            values = g[col].dropna().unique() if col in g else []
            if len(values) > 1:
                raise ValueError(f"{cid}: {col} differs between seeds: {values}")
            row[col] = values[0] if len(values) else np.nan
        row["n_seeds"] = g["seed"].nunique()
        for m in metrics:
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std(ddof=1)
        row["A_ci95"] = ci95(g["A"])
        row["sigma_a"] = row["A_std"]
        reached = g["epochs_to_threshold"].dropna()
        row["epochs_to_threshold_mean"] = reached.mean() if len(reached) else np.nan
        row["threshold_reached"] = len(reached)
        row["skipped_steps"] = int(g["skipped_steps_total"].sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    hybrid = summary["kind"] == "hybrid"
    summary["Q"] = summary["n_qubits"].where(hybrid, 0)
    summary["D_ms"] = (summary["config_id"].map(latency.set_index("config_id")["D_ms"])
                       if latency is not None else np.nan)
    for col in ("Q_max", "D_max", "G_max", "qpar_rank", "A_rank"):
        summary[col] = np.nan
    for ds in DATASETS:
        of_dataset = hybrid & (summary["dataset"] == ds)
        for top, col in (("Q_max", "Q"), ("D_max", "D_ms"), ("G_max", "G")):
            summary.loc[of_dataset, top] = summary.loc[of_dataset, col].max()
    summary["qpar"] = qpar_of(summary, EQUAL).where(hybrid)
    for ds in DATASETS:
        main = main_mask(summary, ds)
        summary.loc[main, "qpar_rank"] = rank_desc(summary.loc[main, "qpar"])
        summary.loc[main, "A_rank"] = rank_desc(summary.loc[main, "A_mean"])
    return summary


def maxima(summary):
    """The Q-Par normalization per dataset (8.1: compute and record)."""
    hybrid = summary[summary["kind"] == "hybrid"]
    return pd.DataFrame([{"dataset": ds, **hybrid.loc[hybrid["dataset"] == ds, ["Q_max", "D_max", "G_max"]].iloc[0]}
                         for ds in DATASETS if (hybrid["dataset"] == ds).any()])


# ============================================================================= Q-Par analyses (8.2, 8.3)

def population(summary, dataset, name):
    """'main': the 54 main configurations (8.2, 8.3). 'tf1': the supplement Jeremy approved on
    2026-09-28 — every hybrid configuration at train_frac = 1.0, the 18 of the main matrix and the
    9 of the qubit sweep (27, like for like), the only population in which Q varies."""
    if name == "main":
        return summary[main_mask(summary, dataset)]
    return summary[(summary["kind"] == "hybrid") & (summary["dataset"] == dataset) & (summary["train_frac"] == 1.0)]


def gamma_inert(summary):
    """Q is the same for every main configuration (n = 4), so gamma cannot change their ranking."""
    return all(population(summary, ds, "main")["Q"].nunique() == 1 for ds in DATASETS)


def sensitivity(summary, name="main"):
    """8.2, long format: for every dataset and exponent setting, the full ranking of the
    population's configurations (one row each, in rank order) with Kendall tau against the
    equal-weight ranking. Q-Par keeps the 8.1 normalization in either population."""
    rows = []
    for ds in DATASETS:
        main = population(summary, ds, name)
        if main.empty:
            continue
        equal = qpar_of(main, EQUAL)
        for setting, exponents in sensitivity_settings().items():
            score = qpar_of(main, exponents)
            tau, p = stats.kendalltau(equal, score)
            rank = rank_desc(score)
            for idx in rank.sort_values(kind="stable").index:
                rows.append({"dataset": ds, "setting": setting, **exponents, "rank": int(rank[idx]),
                             "config_id": main.at[idx, "config_id"], "qpar": score[idx],
                             "kendall_tau": tau, "p_value": p})
    return pd.DataFrame(rows)


def sensitivity_summary(sens):
    """Per dataset and setting: Kendall tau (and p) vs equal weights and the top 5."""
    rows = []
    for (ds, name), g in sens.groupby(["dataset", "setting"], sort=False):
        first = g.iloc[0]
        rows.append({"dataset": ds, "setting": name, **{e: first[e] for e in EXPONENTS},
                     "kendall_tau": first["kendall_tau"], "p_value": first["p_value"],
                     "top5": " | ".join(g.nsmallest(5, "rank", keep="first")["config_id"])})
    return pd.DataFrame(rows)


def cost_terms(rows):
    """Each cost term's contribution to log Q-Par at equal weights (higher = better); log A plus
    these is log Q-Par."""
    return pd.DataFrame({"sigma_a": np.log(1 - rows["sigma_a"]), "Q": -np.log(rows["Q"] / rows["Q_max"]),
                         "D": -np.log(rows["D_ms"] / rows["D_max"]), "G": -np.log(rows["G"] / rows["G_max"])},
                        index=rows.index)


def rank_comparison(summary, threshold=MOVE_THRESHOLD):
    """8.3: Kendall tau between the ranking by A and by equal-weight Q-Par, per dataset; every
    configuration whose rank moves by >= threshold places (numbered per dataset, largest rise
    first; F2 labels and T9 use the numbers), with the cost term that drove the move. A
    configuration that rises passed configurations with a higher A; one that falls was passed by
    configurations with a lower A. Each pass is explained exactly by the cost terms (log Q-Par
    = log A + the terms), so the driver is the term with the largest mean log advantage over the
    configurations passed (rising) or passing (falling). Correlated terms (D and G) are thereby
    weighed against each other instead of masking each other."""
    taus, moves = [], []
    for ds in DATASETS:
        main = summary[main_mask(summary, ds)]
        if main.empty:
            continue
        tau, p = stats.kendalltau(main["A_mean"], main["qpar"])
        taus.append({"dataset": ds, "n": len(main), "kendall_tau": tau, "p_value": p})
        terms = cost_terms(main)
        found = []
        for idx, r in order(main).iterrows():
            move = r["A_rank"] - r["qpar_rank"]  # > 0: ranked higher by Q-Par than by A
            if abs(move) < threshold:
                continue
            if move > 0:
                passed = (main["A_rank"] < r["A_rank"]) & (main["qpar_rank"] > r["qpar_rank"])
                advantage = terms.loc[idx] - terms[passed].mean()
            else:
                passed = (main["A_rank"] > r["A_rank"]) & (main["qpar_rank"] < r["qpar_rank"])
                advantage = terms[passed].mean() - terms.loc[idx]
            found.append({"dataset": ds, "config_id": r["config_id"], "A_rank": int(r["A_rank"]),
                          "qpar_rank": int(r["qpar_rank"]), "move": int(move), "passed": int(passed.sum()),
                          "driver": advantage.idxmax(), **{f"log_advantage_{t}": v for t, v in advantage.items()}})
        found.sort(key=lambda m: -m["move"])  # stable: ties keep the factor order
        moves += [{"id": i, **m} for i, m in enumerate(found, start=1)]
    return pd.DataFrame(taus), pd.DataFrame(moves)


# ============================================================================= statistics (13)

def wilcoxon(diff):
    """Two-sided Wilcoxon signed-rank test, exact over all 2^n sign flips (zeros dropped, ties
    mid-ranked; never the normal approximation): with n = 5 the p-value is never below 0.0625."""
    diff = np.asarray(diff, dtype=float)
    if not np.any(diff != 0):
        return math.nan, 1.0
    result = stats.wilcoxon(diff, method=stats.PermutationMethod())
    return float(result.statistic), float(result.pvalue)


def paired(hybrid, matched):
    """Paired over seeds: t-test, Wilcoxon signed-rank, Cohen's d_z (mean / std of the differences)."""
    h, m = np.asarray(hybrid, dtype=float), np.asarray(matched, dtype=float)
    d = h - m
    sd = d.std(ddof=1)
    t, p_t = stats.ttest_rel(h, m) if sd > 0 else (math.nan, math.nan)
    w, p_w = wilcoxon(d)
    return {"n": len(d), "A_hybrid_mean": h.mean(), "A_hybrid_std": h.std(ddof=1), "A_matched_mean": m.mean(),
            "A_matched_std": m.std(ddof=1), "diff_mean": d.mean(), "diff_std": sd, "t": float(t),
            "p_t": float(p_t), "W": w, "p_wilcoxon": p_w, "cohens_dz": d.mean() / sd if sd > 0 else math.nan}


def hybrid_vs_matched(runs):
    """13: per dataset x placement x encoding x train_frac, the hybrid (per seed, the mean over its
    three topologies) against its cnn_matched model, paired by seed (same seed = same training
    subset and batch order). Also per topology (every hybrid configuration against its anchor)."""
    hybrid = runs[(runs["kind"] == "hybrid") & runs["block"].isin(MAIN_BLOCKS.values())]
    cells, per_topology = [], []
    for (ds, pl, enc, tf), m in runs[runs["kind"] == "matched"].groupby(
            ["dataset", "placement", "encoding", "train_frac"]):
        key = {"dataset": ds, "placement": pl, "encoding": enc, "train_frac": tf}
        m = m.set_index("seed")["A"]
        h = hybrid[(hybrid["dataset"] == ds) & (hybrid["placement"] == pl) & (hybrid["encoding"] == enc)
                   & (hybrid["train_frac"] == tf)].pivot(index="seed", columns="topology", values="A")
        for topo in TOPOLOGIES:
            if topo in h:
                seeds = h[topo].dropna().index.intersection(m.index)
                if len(seeds) > 1:
                    per_topology.append({**key, "topology": topo, **paired(h.loc[seeds, topo], m[seeds])})
        if set(TOPOLOGIES) <= set(h.columns):
            seeds = h[list(TOPOLOGIES)].dropna().index.intersection(m.index)
            if len(seeds) > 1:
                cells.append({**key, "topology": "mean of 3",
                              **paired(h.loc[seeds, list(TOPOLOGIES)].mean(axis=1), m[seeds])})
    return order(pd.DataFrame(cells)), order(pd.DataFrame(per_topology))


def noise_correlations(summary):
    """13: Spearman rho of G and of twoq_transpiled against mean Delta_cal across the hybrid
    configurations at train_frac = 1.0 (main matrix + qubit sweep; the main matrix alone, n = 4,
    as a check that the correlation is not only the qubit count)."""
    rows = []
    at1 = summary[(summary["kind"] == "hybrid") & (summary["train_frac"] == 1.0)].dropna(subset=["delta_cal_mean"])
    for ds in DATASETS:
        for scope, g in (("main+sweep", at1[at1["dataset"] == ds]), ("main", at1[main_mask(at1, ds)])):
            for x in ("G", "twoq_transpiled"):
                if len(g) > 2:
                    rho, p = stats.spearmanr(g[x], g["delta_cal_mean"])
                    rows.append({"dataset": ds, "configurations": scope, "n": len(g), "x": x, "y": "delta_cal_mean",
                                 "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


# ============================================================================= selections

def selection(dataset, results=RESULTS):
    """Best hybrid per placement (decision 6a: highest mean best-validation accuracy at
    train_frac = 1.0), as selected for the qubit sweep; T3/F1 track it across train_frac."""
    return json.loads((results / f"sweep_selection_{dataset}.json").read_text())["selection"]


def hybrid_id(dataset, placement, encoding, topology, n, train_frac):
    return config_id({"kind": "hybrid", "dataset": dataset, "placement": placement, "encoding": encoding,
                      "topology": topology, "n_qubits": n, "train_frac": train_frac})


def matched_id(dataset, placement, encoding, train_frac):
    return config_id({"kind": "matched", "dataset": dataset, "placement": placement, "encoding": encoding,
                      "train_frac": train_frac})


def base_id(dataset, train_frac):
    return config_id({"kind": "base_cnn", "dataset": dataset, "train_frac": train_frac})


def sweep_ids(dataset, sel, placement):
    enc, topo = sel[placement]["encoding"], sel[placement]["topology"]
    return [hybrid_id(dataset, placement, enc, topo, n, 1.0) for n in QUBIT_SWEEP]


ORDER = {"dataset": DATASETS, "placement": PLACEMENTS, "encoding": ENCODINGS, "topology": TOPOLOGIES}


def order(df, keys=("dataset", "placement", "encoding", "topology", "n_qubits", "train_frac")):
    """Rows in the study's factor order (not alphabetical)."""
    keys = [k for k in keys if k in df]
    if df.empty or not keys:
        return df
    return df.sort_values(keys, kind="stable", key=lambda s: s.map(
        {v: i for i, v in enumerate(ORDER[s.name])}).fillna(len(ORDER[s.name])) if s.name in ORDER else s)


def label(row, n=False):
    """Short configuration label, e.g. Mid/Amp/FC/0.2 (with n: Mid/Amp/FC/n6)."""
    tail = f"n{int(row['n_qubits'])}" if n else f"{row['train_frac']:.1f}"
    return "/".join([SHORT[row["placement"]], SHORT[row["encoding"]], SHORT[row["topology"]], tail])


# ============================================================================= tables (12.2)

def fixed(x, digits):
    """x with `digits` decimals; a value that rounds to zero prints unsigned (a mean of exact zeros can
    come out of floating point as -2e-17, which would otherwise print as -0.00)."""
    if pd.isna(x):
        return "--"
    return f"{0.0 if round(x, digits) == 0 else x:.{digits}f}"


def pct(mean, std=None, digits=2):
    """An accuracy fraction in percent, with its std over seeds."""
    if pd.isna(mean):
        return "--"
    return fixed(100 * mean, digits) + ("" if std is None else rf" $\pm$ {100 * std:.{digits}f}")


def sig(x, n=3):
    """n significant digits in fixed notation."""
    if pd.isna(x):
        return "--"
    if x == 0:
        return "0"
    return f"{x:.{max(n - 1 - math.floor(math.log10(abs(x))), 0)}f}"


def integer(x):
    return "--" if pd.isna(x) else f"{int(x)}"


def pvalue(p, latex=True):
    """Two significant digits; below 0.001 as '< 0.001'."""
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "$<$0.001" if latex else "< 0.001"
    return sig(p, 2)


def row(*cells):
    return " & ".join(map(str, cells)) + r" \\"


def write_table(path, caption, label_, colspec, header, body, size="footnotesize", tabcolsep=None):
    """An IEEE two-column-wide table (booktabs) for \\input{} in the paper."""
    lines = [r"\begin{table*}[!t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label_}}}", f"\\{size}"]
    if tabcolsep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule", *body, r"\bottomrule",
              r"\end{tabular}", r"\end{table*}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def table_main(summary, dataset, number, out):
    """T1/T2: train_frac = 1.0, the 18 configurations."""
    rows = order(summary[main_mask(summary, dataset) & (summary["train_frac"] == 1.0)])
    top = maxima(summary).set_index("dataset").loc[dataset]
    body = []
    for pl in PLACEMENTS:
        if body:
            body.append(r"\midrule")
        for i, (_, r) in enumerate(rows[rows["placement"] == pl].iterrows()):
            body.append(row(NAME[pl] if i == 0 else "", NAME[r["encoding"]], NAME[r["topology"]],
                            pct(r["A_mean"], r["A_std"]), fixed(100 * r["sigma_a"], 2), integer(r["G"]),
                            fixed(r["D_ms"], 3), sig(r["qpar"]), integer(r["qpar_rank"]), integer(r["A_rank"])))
    header = [row(*[""] * 8, r"\multicolumn{2}{c}{Rank}"), r"\cmidrule(lr){9-10}",
              row("Placement", "Encoding", "Topology", r"$A$ (\%)", r"$\sigma_a$ (pp)", "$G$", "$D$ (ms)", "Q-Par",
                  "Q-Par", "$A$")]
    caption = (f"{NAME[dataset]}, train\\_frac = 1.0: noiseless test accuracy $A$ (mean $\\pm$ std over "
               f"{int(rows['n_seeds'].min())} seeds), $\\sigma_a$, gate count $G$, latency $D$ (median, seed-0 "
               f"checkpoint), equal-weight Q-Par, and ranks among the 54 main-matrix configurations of the dataset. "
               f"Normalization over all hybrid configurations of the dataset (main matrix and qubit sweep): "
               f"$Q_{{\\max}} = {int(top['Q_max'])}$, $D_{{\\max}} = {top['D_max']:.3f}$~ms, "
               f"$G_{{\\max}} = {int(top['G_max'])}$.")
    write_table(out / f"T{number}_{dataset}_tf1.0.tex", caption, f"tab:{dataset}_tf1", "lllrrrrrrr", header, body)


def table_baselines(summary, results, out):
    """T3: best hybrid per placement vs cnn_matched vs base_cnn, both datasets, three train_frac."""
    s = summary.set_index("config_id")
    body, gaps = [], []
    for ds in DATASETS:
        sel = selection(ds, results)
        if body:
            body.append(r"\midrule")
        models = []
        for pl in PLACEMENTS:
            enc, topo = sel[pl]["encoding"], sel[pl]["topology"]
            models.append((f"{NAME[pl]} hybrid ({NAME[enc]}/{NAME[topo]})",
                           [hybrid_id(ds, pl, enc, topo, MAIN_N, tf) for tf in TRAIN_FRACS], "params_quantum"))
            models.append((f"{NAME[pl]} cnn\\_matched ({NAME[enc]})",
                           [matched_id(ds, pl, enc, tf) for tf in TRAIN_FRACS], "params_matched_map"))
        models.append(("base\\_cnn", [base_id(ds, tf) for tf in TRAIN_FRACS], None))
        for i, (name, ids, core) in enumerate(models):
            ref = s.loc[ids[-1]]
            body.append(row(NAME[ds] if i == 0 else "", name, integer(ref["params_total"]),
                            integer(ref[core]) if core else "--", fixed(ref["D_ms"], 3),
                            *(pct(s.at[c, "A_mean"], s.at[c, "A_std"]) for c in ids)))
    for _, m in summary[summary["kind"] == "matched"].iterrows():
        h = s.loc[hybrid_id(m["dataset"], m["placement"], m["encoding"], "linear", MAIN_N, m["train_frac"])]
        gaps.append(abs(m["params_total"] - h["params_total"]) / h["params_total"])
    header = [row("", "", "", "", "", r"\multicolumn{3}{c}{$A$ (\%) at train\_frac}"), r"\cmidrule(lr){6-8}",
              row("Dataset", "Model", "Params", "Core", "$D$ (ms)", *(f"{tf:.1f}" for tf in TRAIN_FRACS))]
    caption = ("Best hybrid per placement (highest mean best-validation accuracy at train\\_frac = 1.0) against "
               "its parameter-matched classical model and the unmodified CNN: test accuracy $A$ (mean $\\pm$ std "
               "over seeds), trainable parameters (total; core = quantum part of the hybrid or the classical map "
               "replacing it), and latency $D$ of the train\\_frac = 1.0 configuration. Total parameter counts of "
               f"cnn\\_matched differ from their hybrids by at most {100 * max(gaps):.2f}\\,\\%.")
    write_table(out / "T3_baselines.tex", caption, "tab:baselines", "llrrrccc", header, body)


def table_sweep(summary, results, out):
    """T4: qubit sweep per dataset and placement."""
    s = summary.set_index("config_id")
    body = []
    for ds in DATASETS:
        sel = selection(ds, results)
        for pl in PLACEMENTS:
            if body:
                body.append(r"\midrule")
            name = f"{NAME[pl]} ({NAME[sel[pl]['encoding']]}/{NAME[sel[pl]['topology']]})"
            for i, cid in enumerate(sweep_ids(ds, sel, pl)):
                r = s.loc[cid]
                body.append(row(NAME[ds] if i == 0 and pl == PLACEMENTS[0] else "", name if i == 0 else "",
                                integer(r["n_qubits"]), pct(r["A_mean"], r["A_std"]), fixed(100 * r["sigma_a"], 2),
                                integer(r["G"]), fixed(r["D_ms"], 3), sig(r["qpar"])))
    header = [row("Dataset", "Placement", "$n$", r"$A$ (\%)", r"$\sigma_a$ (pp)", "$G$", "$D$ (ms)", "Q-Par")]
    caption = ("Qubit sweep at train\\_frac = 1.0 for the best configuration per placement ($n = 4$ is the main "
               "matrix): test accuracy $A$ (mean $\\pm$ std over seeds), $\\sigma_a$, gate count $G$, latency $D$, "
               "and equal-weight Q-Par.")
    write_table(out / "T4_qubit_sweep.tex", caption, "tab:sweep", "llrrrrrr", header, body)


def noise_rows(summary, dataset):
    """Hybrid configurations at train_frac = 1.0: the 18 of the main matrix, then the sweep's."""
    at1 = summary[(summary["kind"] == "hybrid") & (summary["dataset"] == dataset) & (summary["train_frac"] == 1.0)]
    return pd.concat([order(at1[at1["block"] == MAIN_BLOCKS[dataset]]), order(at1[at1["block"] == "S"])])


def table_noise(summary, depol_p, out):
    """T5: noise results per hybrid configuration at train_frac = 1.0."""
    body = []
    for ds in DATASETS:
        if body:
            body.append(r"\midrule")
        body.append(row(f"\\multicolumn{{{6 + len(depol_p)}}}{{l}}{{\\textit{{{NAME[ds]}}}}}"))
        for _, r in noise_rows(summary, ds).iterrows():
            body.append(row(label(r, n=True), pct(r["A_subset_noiseless_mean"], r["A_subset_noiseless_std"]),
                            pct(r["acc_noise_calibrated_mean"], r["acc_noise_calibrated_std"]),
                            pct(r["delta_cal_mean"], r["delta_cal_std"]),
                            *(pct(r[f"acc_depol_{p}_mean"], r[f"acc_depol_{p}_std"]) for p in depol_p),
                            integer(r["G"]), integer(r["twoq_transpiled"])))
    header = [row("", "", "", "", rf"\multicolumn{{{len(depol_p)}}}{{c}}{{Depolarizing, $p$ =}}", "", ""),
              f"\\cmidrule(lr){{5-{4 + len(depol_p)}}}",
              row("Configuration", r"$A_\mathrm{sub}$ (\%)", r"Calibrated (\%)", r"$\Delta_\mathrm{cal}$ (pp)",
                  *(f"{p:g}" for p in depol_p), "$G$", "2Q")]
    caption = ("Noisy evaluation of every hybrid configuration at train\\_frac = 1.0 on the fixed 2,000-image test "
               "subset (mean $\\pm$ std over seeds): noiseless accuracy $A_\\mathrm{sub}$, accuracy under the "
               "calibrated device noise model (1,024 shots), $\\Delta_\\mathrm{cal} = A_\\mathrm{sub} - "
               "A_\\mathrm{cal}$, accuracy under depolarizing noise of strength $p$ (two-qubit gates $10p$), gate "
               "count $G$, and two-qubit gates after transpilation (2Q). Configuration: placement/encoding/"
               "topology/qubits; $n = 4$ rows are the main matrix, the others the qubit sweep.")
    write_table(out / "T5_noise.tex", caption, "tab:noise", "l" + "r" * (5 + len(depol_p)), header, body,
                size="scriptsize", tabcolsep="4pt")


GREEK = {"alpha": ("α", r"\alpha"), "beta": ("β", r"\beta"), "gamma": ("γ", r"\gamma"), "delta": ("δ", r"\delta"),
         "epsilon": ("ε", r"\varepsilon")}


def setting_label(name, latex=True):
    """equal -> Equal weights; alpha=0.5 -> alpha = 0.5 (Greek); presets keep their names."""
    if name == "equal":
        return "Equal weights"
    if "=" in name:
        exp, value = name.split("=")
        return f"${GREEK[exp][1]} = {value}$" if latex else f"{GREEK[exp][0]} = {value}"
    return name


SETTINGS_TEXT = ("one exponent at 0.5 or 2 with the others at 1, and three presets (accuracy-first $\\alpha = 2$, "
                 "latency-first $\\delta = 2$, resource-first $\\gamma = \\varepsilon = 2$). Kendall $\\tau$ against "
                 "the equal-weight ranking and the five top-ranked configurations")


def table_sensitivity(sens_summary, summary, out, supplement=False):
    """T6 (main matrix) / T10 (supplement, train_frac = 1.0 incl. the qubit sweep): Kendall tau vs
    equal weights and the top 5, every setting, both datasets."""
    labels = {r["config_id"]: label(r, n=supplement) for _, r in summary[summary["kind"] == "hybrid"].iterrows()}
    body = []
    for ds in DATASETS:
        if body:
            body.append(r"\midrule")
        body.append(row(r"\multicolumn{4}{l}{\textit{" + NAME[ds] + "}}"))
        for _, r in sens_summary[sens_summary["dataset"] == ds].iterrows():
            body.append(row(setting_label(r["setting"]), fixed(r["kendall_tau"], 3),
                            "--" if r["setting"] == "equal" else pvalue(r["p_value"]),
                            ", ".join(labels[c] for c in r["top5"].split(" | "))))
    header = [row("Setting", r"Kendall $\tau$", "$p$", "Top 5 (rank 1 first)")]
    if supplement:
        n = int(population(summary, DATASETS[0], "tf1").shape[0])
        caption = (f"Supplement to Table~\\ref{{tab:sensitivity}}: sensitivity of the Q-Par ranking of the {n} hybrid "
                   "configurations at train\\_frac = 1.0 of each dataset (the 18 of the main matrix and the 9 of the "
                   "qubit sweep, $n \\in \\{2, 4, 6, 8\\}$), where the qubit count $Q$ varies, to the exponents: "
                   f"{SETTINGS_TEXT} (placement/encoding/topology/qubits).")
        write_table(out / "T10_sensitivity_supplement.tex", caption, "tab:sensitivity_supplement", "lrrl", header,
                    body, size="scriptsize")
        return
    caption = (f"Sensitivity of the Q-Par ranking of the 54 main-matrix configurations to the exponents: "
               f"{SETTINGS_TEXT} (placement/encoding/topology/train\\_frac).")
    if gamma_inert(summary):
        caption += (" The exponent $\\gamma$ is inert here by construction: every main-matrix configuration has "
                    "$n = 4$ qubits, so $Q/Q_{\\max}$ is the same for all 54 and the $\\gamma$ settings reproduce the "
                    "equal-weight ranking ($\\tau = 1$; resource-first ranks as $\\varepsilon = 2$). "
                    "Table~\\ref{tab:sensitivity_supplement} repeats the analysis over the train\\_frac = 1.0 "
                    "configurations including the qubit sweep, where $Q$ varies.")
    write_table(out / "T6_sensitivity.tex", caption, "tab:sensitivity", "lrrl", header, body, size="scriptsize")


def table_full(summary, out):
    """T7 (appendix): the full 54-configuration matrices of both datasets."""
    s = summary.set_index("config_id")
    body = []
    for pl in PLACEMENTS:
        for enc in ENCODINGS:
            if body:
                body.append(r"\midrule")
            for topo in TOPOLOGIES:
                for tf in TRAIN_FRACS:
                    cells = []
                    for ds in DATASETS:
                        r = s.loc[hybrid_id(ds, pl, enc, topo, MAIN_N, tf)]
                        cells += [pct(r["A_mean"], r["A_std"]), sig(r["qpar"])]
                    body.append(row(NAME[pl], NAME[enc], NAME[topo], f"{tf:.1f}", *cells))
    header = [row("", "", "", "", *(f"\\multicolumn{{2}}{{c}}{{{NAME[ds]}}}" for ds in DATASETS)),
              r"\cmidrule(lr){5-6} \cmidrule(lr){7-8}",
              row("Placement", "Encoding", "Topology", "train\\_frac", *[r"$A$ (\%)", "Q-Par"] * len(DATASETS))]
    caption = ("All 54 main-matrix configurations per dataset: test accuracy $A$ (mean $\\pm$ std over seeds) and "
               "equal-weight Q-Par.")
    write_table(out / "T7_full_matrix.tex", caption, "tab:full", "llllrrrr", header, body, size="scriptsize")


def table_matched_stats(cells, out):
    """T8 (appendix; reports the Section 13 statistics): hybrid (topology mean) vs cnn_matched."""
    body = []
    for ds in DATASETS:
        if body:
            body.append(r"\midrule")
        for i, (_, r) in enumerate(cells[cells["dataset"] == ds].iterrows()):
            body.append(row(NAME[ds] if i == 0 else "", NAME[r["placement"]], NAME[r["encoding"]],
                            f"{r['train_frac']:.1f}", pct(r["A_hybrid_mean"], r["A_hybrid_std"]),
                            pct(r["A_matched_mean"], r["A_matched_std"]), fixed(100 * r["diff_mean"], 2),
                            pvalue(r["p_t"]), pvalue(r["p_wilcoxon"]), fixed(r["cohens_dz"], 2)))
    header = [row("Dataset", "Placement", "Encoding", "train\\_frac", r"Hybrid $A$ (\%)", r"Matched $A$ (\%)",
                  r"$\Delta$ (pp)", "$p_t$", "$p_W$", "$d_z$")]
    caption = ("Hybrid against its parameter-matched classical model, paired over seeds: hybrid $A$ is the mean of "
               "its three topologies per seed; $\\Delta$ = hybrid $-$ matched; paired $t$-test ($p_t$), exact "
               "Wilcoxon signed-rank test ($p_W$; with 5 seeds $p_W \\geq 0.0625$), and Cohen's $d_z$ (mean "
               "difference / std of the differences). Unadjusted for multiple comparisons.")
    write_table(out / "T8_hybrid_vs_matched.tex", caption, "tab:matched_stats", "llllrrrrrr", header, body,
                size="scriptsize")


DRIVER = {"sigma_a": r"$\sigma_a$", "Q": "$Q$", "D": "$D$", "G": "$G$"}


def table_rank_moves(moves, taus, summary, out):
    """T9 (reports the 8.3 list; its numbers label F2): every configuration whose rank moves by
    >= 10 places, both datasets side by side."""
    labels = {r["config_id"]: label(r) for _, r in summary[summary["kind"] == "hybrid"].iterrows()}
    columns = []
    for ds in DATASETS:
        m = moves[moves["dataset"] == ds] if len(moves) else moves
        columns.append([[str(r["id"]), labels[r["config_id"]], str(r["A_rank"]), str(r["qpar_rank"]),
                         f"{r['move']:+d}", DRIVER[r["driver"]]] for _, r in m.iterrows()])
    body = []
    for i in range(max(map(len, columns), default=0)):
        cells = []
        for col in columns:
            cells += col[i] if i < len(col) else [""] * 6
        body.append(row(*cells))
    tau = taus.set_index("dataset")
    header = [row(*(f"\\multicolumn{{6}}{{c}}{{{NAME[ds]} (Kendall $\\tau$ = {tau.at[ds, 'kendall_tau']:.2f})}}"
                    for ds in DATASETS)),
              r"\cmidrule(lr){1-6} \cmidrule(lr){7-12}",
              row(*["\\#", "Configuration", "$A$", "Q-Par", "Move", "Driver"] * len(DATASETS))]
    constant_q = len(moves) == 0 or (moves["log_advantage_Q"] == 0).all()
    caption = ("Configurations whose rank among the 54 main-matrix configurations moves by at least "
               f"{MOVE_THRESHOLD} places between the ranking by accuracy $A$ and by equal-weight Q-Par (the numbers "
               "label the rank--rank scatter; positive = ranked higher by Q-Par). Driver: the cost term with the "
               "largest mean advantage in log Q-Par over the configurations the configuration passed (rising) or "
               "that passed it (falling)"
               + (" ($Q$ is the same, $n = 4$, for every main-matrix configuration and cannot move a rank)."
                  if constant_q else "."))
    write_table(out / "T9_rank_moves.tex", caption, "tab:rank_moves", "rlrrrl" * len(DATASETS), header, body,
                size="scriptsize")


# ============================================================================= figures (12.3)

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"pre": "#0072b2", "middle": "#d55e00", "post": "#009e73"}  # Okabe-Ito, validated (DECISIONS.md §14)
MARKER = {"pre": "o", "middle": "s", "post": "^"}
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")  # one hue, light->dark
SINGLE, DOUBLE = 3.5, 7.16  # IEEE column widths, inches
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
         "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
         "legend.title_fontsize": 8, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
         "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
         "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
         "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
         "legend.frameon": False, "lines.linewidth": 1.3, "lines.markersize": 4.5, "errorbar.capsize": 1.5,
         "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
         "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.cal": "Arial", "mathtext.tt": "Arial",
         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300}


def figure(width, height, rows=1, cols=1):
    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    return fig, fig.subplots(rows, cols, squeeze=False)


def plain_log(axis, low, high):
    """Log-scale ticks as plain numbers: 1-2-5 steps within two decades, else decades only."""
    subs = (1.0, 2.0, 5.0) if high / low <= 100 else (1.0,)
    axis.set_major_locator(LogLocator(subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def dodged_errorbar(ax, i, k, x, y, err, step=2.5, **style):
    """Series i of k shifted sideways by a few points (any axis scale) so error bars do not overlap.
    The shifted transform bypasses autoscaling, so the data limits are registered here."""
    x, y, err = (np.asarray(v, dtype=float) for v in (x, y, err))
    shift = offset_copy(ax.transData, fig=ax.figure, x=(i - (k - 1) / 2) * step, y=0, units="points")
    ax.errorbar(x, y, yerr=err, transform=shift, **style)
    ax.update_datalim(np.column_stack([np.r_[x, x], np.r_[y - err, y + err]]))
    ax.autoscale_view()


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)


def marker_style(placement, filled=True):
    """Filled marks carry a thin white ring; open marks are the hue on white."""
    c = COLOR[placement]
    return {"color": c, "marker": MARKER[placement], "markerfacecolor": c if filled else "white",
            "markeredgecolor": "white" if filled else c, "markeredgewidth": 0.6 if filled else 0.9}


def figure_train_frac(summary, results, out):
    """F1: A vs train_frac - best hybrid per placement, cnn_matched, base_cnn; a panel per dataset."""
    s = summary.set_index("config_id")
    fig, axes = figure(DOUBLE, 2.9, 1, 2)
    x = np.array(TRAIN_FRACS)
    for ax, ds in zip(axes[0], DATASETS):
        sel = selection(ds, results)
        series = []
        for pl in PLACEMENTS:
            enc, topo = sel[pl]["encoding"], sel[pl]["topology"]
            series.append(([hybrid_id(ds, pl, enc, topo, MAIN_N, tf) for tf in TRAIN_FRACS],
                           {**marker_style(pl), "linestyle": "-"}))
            series.append(([matched_id(ds, pl, enc, tf) for tf in TRAIN_FRACS],
                           {**marker_style(pl, filled=False), "linestyle": (0, (4, 2))}))
        series.append(([base_id(ds, tf) for tf in TRAIN_FRACS],
                       {"color": INK2, "marker": "D", "markersize": 3.5, "markerfacecolor": INK2,
                        "markeredgecolor": "white", "markeredgewidth": 0.6, "linestyle": (0, (1, 1.5))}))
        for i, (ids, style) in enumerate(series):
            dodged_errorbar(ax, i, len(series), x, 100 * s.loc[ids, "A_mean"], 100 * s.loc[ids, "A_std"],
                            elinewidth=0.7, **style)
        chosen = ", ".join(f"{SHORT[pl]}: {SHORT[sel[pl]['encoding']]}/{SHORT[sel[pl]['topology']]}"
                           for pl in PLACEMENTS)
        ax.set_title(f"{NAME[ds]}  ({chosen})", loc="left")
        ax.set_xticks(x, [f"{tf:.1f}" for tf in TRAIN_FRACS])
        ax.set_xlabel("train_frac")
        ax.set_ylabel("Test accuracy A (%)")
    handles = [Line2D([], [], linestyle="-", **marker_style(pl), label=NAME[pl]) for pl in PLACEMENTS]
    handles += [Line2D([], [], color=INK2, linestyle="-", label="hybrid (filled)"),
                Line2D([], [], color=INK2, linestyle=(0, (4, 2)), marker="o", markerfacecolor="white",
                       markeredgecolor=INK2, label="cnn_matched (open)"),
                Line2D([], [], color=INK2, linestyle=(0, (1, 1.5)), marker="D", markersize=3.5, label="base_cnn")]
    fig.legend(handles=handles, loc="outside lower center", ncols=6)
    save(fig, out / "F1_accuracy_vs_train_frac")


def _overlap(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def place_labels(fig, ax, points, texts, obstacles):
    """Label points without overlapping each other, the panel's points (obstacles) or the frame:
    try offsets around each point, take the first free one (else the least crowded); offset
    labels get a thin leader line."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    dots = ax.transData.transform(np.asarray(obstacles, dtype=float))
    taken = []
    for (x, y), text in zip(np.asarray(points, dtype=float), texts):
        best, best_cost = None, math.inf
        for radius in (5, 10, 16, 24, 34):
            for angle in (45, -45, 135, -135, 0, 180, 90, -90):
                dx, dy = radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))
                ha = "left" if dx > 0.5 else "right" if dx < -0.5 else "center"
                va = "bottom" if dy > 0.5 else "top" if dy < -0.5 else "center"
                ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                  fontsize=8, annotation_clip=False)
                box = ann.get_window_extent(renderer).expanded(1.04, 1.1)
                ann.remove()
                cost = (sum(_overlap(box, b) for b in taken) + 1e3 * sum(box.contains(px, py) for px, py in dots)
                        + (0 if frame.contains(box.x0, box.y0) and frame.contains(box.x1, box.y1) else 1e6))
                if cost < best_cost:
                    best, best_cost = (dx, dy, ha, va, box, radius), cost
                if cost == 0:
                    break
            if best_cost == 0:
                break
        dx, dy, ha, va, box, radius = best
        leader = ({"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "shrinkA": 0, "shrinkB": 2.5}
                  if radius > 5 else None)
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=8, color=INK,
                    arrowprops=leader, annotation_clip=False).set_in_layout(False)
        taken.append(box)


def figure_ranks(summary, taus, moves, out):
    """F2: rank by A vs rank by equal-weight Q-Par, 54 points per dataset; the configurations whose
    rank changes by >= 10 places carry their number from T9 (the band marks smaller changes)."""
    fig, axes = figure(DOUBLE, 4.1, 1, 2)
    tau = taus.set_index("dataset")["kendall_tau"]
    labelled = []
    for ax, ds in zip(axes[0], DATASETS):
        main = summary[main_mask(summary, ds)]
        n = len(main)
        edge = np.array([0, n + 1])
        ax.fill_between(edge, edge - MOVE_THRESHOLD, edge + MOVE_THRESHOLD, color="#f0efec", linewidth=0, zorder=0)
        ax.plot(edge, edge, color=AXIS, linewidth=0.8, zorder=1)
        for pl in PLACEMENTS:
            g = main[main["placement"] == pl]
            ax.scatter(g["A_rank"], g["qpar_rank"], s=22, color=COLOR[pl], marker=MARKER[pl], edgecolors="white",
                       linewidths=0.6, zorder=3)
        ticks = [1, *range(10, n + 1, 10)]
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0, n + 1)
        ax.set_aspect("equal")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("Rank by accuracy A")
        ax.set_ylabel("Rank by Q-Par (equal weights)")
        ax.set_title(f"{NAME[ds]}  (Kendall τ = {tau[ds]:.2f})", loc="left")
        m = moves[moves["dataset"] == ds] if len(moves) else moves
        labelled.append((ax, m[["A_rank", "qpar_rank"]].to_numpy() if len(m) else [],
                         [str(i) for i in m["id"]] if len(m) else [], main[["A_rank", "qpar_rank"]].to_numpy()))
    handles = [Line2D([], [], linestyle="", **marker_style(pl), markersize=5, label=NAME[pl]) for pl in PLACEMENTS]
    handles.append(Patch(color="#f0efec", label=f"rank change < {MOVE_THRESHOLD}"))
    fig.legend(handles=handles, loc="outside lower center", ncols=4,
               title=f"Numbered: rank change ≥ {MOVE_THRESHOLD} places (listed with their drivers in the rank-move table)")
    for ax, points, texts, obstacles in labelled:  # after the layout is final
        if len(texts):
            place_labels(fig, ax, points, texts, obstacles)
    save(fig, out / "F2_rank_rank")


def figure_noise_vs_gates(summary, corr, out):
    """F3: Delta_cal vs G per configuration at train_frac = 1.0, Spearman rho and p in the legend."""
    fig, axes = figure(DOUBLE, 3.0, 1, 2)
    for ax, ds in zip(axes[0], DATASETS):
        rows = noise_rows(summary, ds)
        for pl in PLACEMENTS:
            for filled, g in ((True, rows[(rows["placement"] == pl) & (rows["block"] != "S")]),
                              (False, rows[(rows["placement"] == pl) & (rows["block"] == "S")])):
                ax.errorbar(g["G"], 100 * g["delta_cal_mean"], yerr=100 * g["delta_cal_std"], linestyle="",
                            elinewidth=0.6, **marker_style(pl, filled=filled))
        c = corr[(corr["dataset"] == ds) & (corr["x"] == "G") & (corr["configurations"] == "main+sweep")].iloc[0]
        p = pvalue(c["p_value"], latex=False)
        ax.axhline(0, color=AXIS, linewidth=0.6, zorder=1)
        ax.set_xscale("log")
        plain_log(ax.xaxis, rows["G"].min(), rows["G"].max())
        ax.set_xlabel("Gate count G (log scale)")
        ax.set_ylabel(r"$\Delta_\mathrm{cal}$ (percentage points)")
        ax.set_title(NAME[ds], loc="left")
        ax.legend(handles=[Line2D([], [], linestyle="", label=f"Spearman ρ = {c['spearman_rho']:.2f}, "
                                                              f"p {p if p.startswith('<') else '= ' + p} "
                                                              f"(n = {int(c['n'])})")],
                  loc="best", handlelength=0, handletextpad=0)
    handles = [Line2D([], [], linestyle="", **marker_style(pl), label=NAME[pl]) for pl in PLACEMENTS]
    handles += [Line2D([], [], linestyle="", marker="o", color=INK2, markeredgecolor="white", label="n = 4 (main)"),
                Line2D([], [], linestyle="", marker="o", markerfacecolor="white", markeredgecolor=INK2,
                       label="n = 2, 6, 8 (sweep)")]
    fig.legend(handles=handles, loc="outside lower center", ncols=5)
    save(fig, out / "F3_delta_cal_vs_G")


def depolarizing_curves(runs, dataset, depol_p):
    """Per placement: per seed, the mean over its main-matrix configurations at train_frac = 1.0
    (encodings x topologies) at p = 0 (noiseless, same subset) and each p; mean and std over seeds."""
    rows = runs[(runs["kind"] == "hybrid") & (runs["dataset"] == dataset) & (runs["block"] == MAIN_BLOCKS[dataset])
                & (runs["train_frac"] == 1.0)]
    cols = ["A_subset_noiseless", *(f"acc_depol_{p}" for p in depol_p)]
    curves = {}
    for pl in PLACEMENTS:
        per_seed = rows[rows["placement"] == pl].groupby("seed")[cols].mean()
        curves[pl] = (per_seed.mean().to_numpy(), per_seed.std(ddof=1).to_numpy())
    return np.array([0.0, *depol_p]), curves


def figure_depolarizing(runs, depol_p, out):
    """F4: acc_depol vs p, a line per placement (mean over encodings/topologies), both datasets."""
    fig, axes = figure(DOUBLE, 2.9, 1, 2)
    for ax, ds in zip(axes[0], DATASETS):
        p, curves = depolarizing_curves(runs, ds, depol_p)
        for i, pl in enumerate(PLACEMENTS):
            mean, std = curves[pl]
            dodged_errorbar(ax, i, len(PLACEMENTS), p, 100 * mean, 100 * std, elinewidth=0.7, **marker_style(pl))
        ax.set_xscale("symlog", linthresh=min(depol_p), linscale=0.5)  # p = 0 and a log-spaced sweep
        ax.set_xlim(-0.25 * min(depol_p), 1.15 * max(depol_p))
        ax.set_xticks(p, ["0"] + [f"{v:g}" for v in depol_p])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Depolarizing strength p (two-qubit gates 10p; 0 = noiseless)")
        ax.set_ylabel("Accuracy on the noisy-eval subset (%)")
        ax.set_title(NAME[ds], loc="left")
    handles = [Line2D([], [], linestyle="-", **marker_style(pl), label=NAME[pl]) for pl in PLACEMENTS]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, title="Main matrix, train_frac = 1.0, "
               "mean over encodings and topologies; error bars ±1 std over seeds")
    save(fig, out / "F4_depolarizing")


def figure_sweep(summary, results, out):
    """F5: qubit sweep - A, D, G, Q-Par vs n, lines per placement; a row per dataset."""
    s = summary.set_index("config_id")
    panels = (("A_mean", "Test accuracy A (%)", False), ("D_ms", "Latency D (ms, log)", True),
              ("G", "Gate count G (log)", True), ("qpar", "Q-Par, equal weights (log)", True))
    fig, axes = figure(DOUBLE, 4.4, 2, 4)
    for r, ds in enumerate(DATASETS):
        sel = selection(ds, results)
        for c, (col, title, log) in enumerate(panels):
            ax = axes[r][c]
            values = []
            for i, pl in enumerate(PLACEMENTS):
                rows = s.loc[sweep_ids(ds, sel, pl)]
                n = rows["n_qubits"].to_numpy()
                values += list(rows[col])
                if col == "A_mean":
                    dodged_errorbar(ax, i, len(PLACEMENTS), n, 100 * rows["A_mean"], 100 * rows["A_std"],
                                    elinewidth=0.7, **marker_style(pl))
                else:
                    ax.plot(n, rows[col], **marker_style(pl))
            if log:
                ax.set_yscale("log")
                plain_log(ax.yaxis, min(values), max(values))
            ax.set_xticks(QUBIT_SWEEP)
            ax.set_xlabel("Qubits n")
            if r == 0:
                ax.set_title(title, loc="left")
            if c == 0:
                ax.set_ylabel(NAME[ds], fontweight="bold")
    handles = [Line2D([], [], linestyle="-", **marker_style(pl), label=NAME[pl]) for pl in PLACEMENTS]
    fig.legend(handles=handles, loc="outside lower center", ncols=3,
               title="Best configuration per placement, train_frac = 1.0; A error bars ±1 std over seeds")
    save(fig, out / "F5_qubit_sweep")


def figure_sensitivity(sens_summary, out, gamma_note=False):
    """F6: Kendall tau vs equal weights for each exponent setting, both datasets; with gamma_note,
    the gamma settings are marked inert (every main configuration has n = 4)."""
    settings = list(sensitivity_settings())
    tau = sens_summary.pivot(index="setting", columns="dataset", values="kendall_tau").loc[settings, list(DATASETS)]
    values = tau.to_numpy()
    if np.nanmin(values) < 0:  # a reversed ranking: diverging scale with a neutral gray at 0
        cmap, vmin, vmax = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"]), -1.0, 1.0
    else:
        cmap, vmin, vmax = LinearSegmentedColormap.from_list("seq", SEQUENTIAL), \
            min(math.floor(10 * np.nanmin(values)) / 10, 0.9), 1.0
    fig, axes = figure(SINGLE, 4.2)
    ax = axes[0][0]
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    for i in range(len(settings)):
        for j in range(len(DATASETS)):
            rgb = to_rgb(cmap((values[i, j] - vmin) / (vmax - vmin)))
            light = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 0.5
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK if light else "white")
    ax.set_xticks(range(len(DATASETS)), [NAME[d] for d in DATASETS])
    inert = {"gamma=0.5", "gamma=2", "resource-first"} if gamma_note else set()
    ax.set_yticks(range(len(settings)), [setting_label(name, latex=False) + (" †" if name in inert else "")
                                         for name in settings])
    if gamma_note:
        ax.set_xlabel("† γ inert by construction: all 54 main-grid\nconfigurations have n = 4 (see the\n"
                      "train_frac = 1.0 supplement, where Q varies)", color=INK2, labelpad=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label("Kendall τ vs equal weights")
    bar.outline.set_visible(False)
    save(fig, out / "F6_sensitivity_heatmap")


# ============================================================================= main

def write_all(results=RESULTS, configs=CONFIGS):
    base = load_base()
    depol_p, seeds = base["noise"]["depolarizing_p"], base["seeds"]
    latency_path = results / "latency.csv"
    if not latency_path.exists():
        raise SystemExit("results/latency.csv is missing: run the Mac latency step (protocol 7) first")
    runs = load_runs(results, configs, depol_p)
    check_complete(runs, seeds, depol_p, configs)
    summary = summarize(runs, pd.read_csv(latency_path), depol_p)
    if summary["D_ms"].isna().any():
        raise SystemExit(f"latency.csv lacks D for {summary.loc[summary['D_ms'].isna(), 'config_id'].tolist()[:3]}")
    sens = sensitivity(summary)
    sens_sum = sensitivity_summary(sens)
    sens_tf1 = sensitivity(summary, "tf1")
    sens_tf1_sum = sensitivity_summary(sens_tf1)
    taus, moves = rank_comparison(summary)
    cells, per_topology = hybrid_vs_matched(runs)
    corr = noise_correlations(summary)

    runs.to_csv(results / "runs.csv", index=False)
    order(summary, ("block", "dataset", "placement", "encoding", "topology", "n_qubits", "train_frac")).to_csv(
        results / "summary.csv", index=False)
    maxima(summary).to_csv(results / "qpar_maxima.csv", index=False)
    sens.to_csv(results / "sensitivity.csv", index=False)
    sens_sum.to_csv(results / "sensitivity_summary.csv", index=False)
    sens_tf1.to_csv(results / "sensitivity_supplement_tf1.csv", index=False)
    sens_tf1_sum.to_csv(results / "sensitivity_supplement_tf1_summary.csv", index=False)
    taus.to_csv(results / "rank_comparison.csv", index=False)
    moves.to_csv(results / "rank_moves.csv", index=False)
    cells.to_csv(results / "stats_hybrid_vs_matched.csv", index=False)
    per_topology.to_csv(results / "stats_hybrid_vs_matched_per_topology.csv", index=False)
    corr.to_csv(results / "stats_noise_correlations.csv", index=False)

    tables, figures = results / "tables", results / "figures"
    table_main(summary, "mnist", 1, tables)
    table_main(summary, "fashionmnist", 2, tables)
    table_baselines(summary, results, tables)
    table_sweep(summary, results, tables)
    table_noise(summary, depol_p, tables)
    table_sensitivity(sens_sum, summary, tables)
    table_full(summary, tables)
    table_matched_stats(cells, tables)
    table_rank_moves(moves, taus, summary, tables)
    table_sensitivity(sens_tf1_sum, summary, tables, supplement=True)
    with matplotlib.rc_context(STYLE):
        figure_train_frac(summary, results, figures)
        figure_ranks(summary, taus, moves, figures)
        figure_noise_vs_gates(summary, corr, figures)
        figure_depolarizing(runs, depol_p, figures)
        figure_sweep(summary, results, figures)
        figure_sensitivity(sens_sum, figures, gamma_note=gamma_inert(summary))


if __name__ == "__main__":
    write_all()
