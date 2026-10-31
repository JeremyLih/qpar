"""Analysis (Sections 8, 12, 13) on a synthetic but complete study: 168 configurations x 5 seeds,
noise results (half merged into the run JSONs, half still in results/noise/), a synthetic
latency.csv and sweep selections. Nothing here reads the real results/."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

from qpar import analysis as an
from qpar.train import config_id, load_base, run_id

BASE = load_base()
SEEDS = BASE["seeds"]
DEPOL = BASE["noise"]["depolarizing_p"]
SELECTION = {"mnist": {"pre": ("angle", "fullyconnected"), "middle": ("amplitude", "random"),
                       "post": ("amplitude", "random")},
             "fashionmnist": {"pre": ("amplitude", "linear"), "middle": ("amplitude", "fullyconnected"),
                              "post": ("amplitude", "fullyconnected")}}
PERFECT = "mnist_post_amplitude_random_n8_tf1.0"  # sigma_a = 0 and Q = Q_max, D = D_max, G = G_max


def configurations():
    cfgs = {"matrix": [], "baselines": [], "sweep": []}
    for ds, block, bblock in (("mnist", "M1", "B1"), ("fashionmnist", "M2", "B2")):
        for tf in an.TRAIN_FRACS:
            cfgs["baselines"].append({"kind": "base_cnn", "block": bblock, "dataset": ds, "train_frac": tf})
            for pl in an.PLACEMENTS:
                for enc in an.ENCODINGS:
                    cfgs["baselines"].append({"kind": "matched", "block": bblock, "dataset": ds, "placement": pl,
                                              "encoding": enc, "n_qubits": 4, "train_frac": tf})
                    for topo in an.TOPOLOGIES:
                        cfgs["matrix"].append({"kind": "hybrid", "block": block, "dataset": ds, "placement": pl,
                                               "encoding": enc, "topology": topo, "n_qubits": 4, "train_frac": tf})
        for pl, (enc, topo) in SELECTION[ds].items():
            for n in (2, 6, 8):
                cfgs["sweep"].append({"kind": "hybrid", "block": "S", "dataset": ds, "placement": pl, "encoding": enc,
                                      "topology": topo, "n_qubits": n, "train_frac": 1.0})
    return cfgs


def gates(cfg):
    n = cfg["n_qubits"]
    edges = n * (n - 1) // 2 if cfg["topology"] == "fullyconnected" else n - 1
    return 3 * (3 * n + edges) + (n if cfg["encoding"] == "angle" else 2 ** (n + 1)), 3 * edges


def accuracy(cfg, rng):
    a = {"mnist": 0.985, "fashionmnist": 0.885}[cfg["dataset"]] - {0.2: 0.012, 0.5: 0.005, 1.0: 0.0}[cfg["train_frac"]]
    a += {"base_cnn": 0.003, "matched": -0.004, "hybrid": 0.0}[cfg["kind"]]
    a += 0.002 * (cfg.get("encoding") == "amplitude") - 0.003 * (cfg.get("placement") == "middle")
    return round(a + rng.normal(0, 0.002), 4)


def record(cfg, seed, rng):
    cid = config_id(cfg)
    a = accuracy({**cfg}, rng) if cid != PERFECT else 0.975
    epochs = [{"epoch": e, "train_loss": 1 / e, "train_acc": a - 0.01 / e, "val_acc": a - 0.005 / e,
               "seconds": 10.0, "train_seconds": 9.0, "skipped_steps": 0} for e in range(1, 16)]
    r = {"run_id": run_id(cfg, seed), "config_id": cid, "config": cfg, "seed": seed, "train_size": 55000,
         "epochs": epochs, "best_epoch": 15, "best_val_acc": a - 0.0003, "threshold": 0.95,
         "epochs_to_threshold": None if (cfg["kind"] == "matched" and seed == 0) else 2, "A": a,
         "A_subset_noiseless": round(a + rng.normal(0, 0.001), 4), "params_total": 12000, "params_quantum": 0,
         "params_classical": 12000, "timings": {"train_seconds": 135.0, "test_seconds": 1.0},
         "resumed_after_epochs": [], "environment": {"python": "3.11", "machine": "x86_64"}}
    if seed == 1:  # trained before the non-finite-gradient guard: no count
        for e in epochs:
            del e["skipped_steps"]
    else:
        r["skipped_steps_total"] = 1 if seed == 2 else 0
    if cfg["kind"] == "hybrid":
        g, twoq = gates(cfg)
        r.update(params_total=12036, params_quantum=36, params_classical=12000, G=g, G_variational=g // 2,
                 G_encoding=g - g // 2, gate_types={"RY": g // 2, "CNOT": g - g // 2}, simulator="lightning.qubit")
    if cfg["kind"] == "matched":
        r.update(params_total=12031, params_classical=12031, matched_map="mlp_h3", params_matched_map=31,
                 params_matched_target=36)
    return r


def noise(r, rng):
    g, twoq = gates(r["config"])
    cal = {"acc_noise_calibrated": round(r["A_subset_noiseless"] - 1e-4 * g + rng.normal(0, 0.001), 4),
           "twoq_transpiled": 2 * twoq, "depth_transpiled": 3 * g, "shots": 1024,
           "noise_backend": {"name": "fake_aachen", "num_qubits": 156, "basis_gates": ["cz", "rz", "sx", "x"],
                             "calibration_date": "2026-04-17 12:29:46+02:00"}}
    depol = {str(p): round(r["A_subset_noiseless"] - 0.05 * p * g, 4) for p in DEPOL}
    return cal, depol


def build_study(root):
    configs, results = root / "configs", root / "results"
    rng = np.random.default_rng(0)
    latency = []
    for folder, cfgs in configurations().items():
        (configs / folder).mkdir(parents=True)
        for cfg in cfgs:
            cid = config_id(cfg)
            (configs / folder / f"{cid}.yaml").write_text(yaml.safe_dump(cfg))
            d = 0.05 + (0.001 * gates(cfg)[0] + 0.01 * cfg["n_qubits"] if cfg["kind"] == "hybrid" else 0.0)
            latency.append({"config_id": cid, "D_ms": d + (5.0 if cid == PERFECT else 0.0)})
            for seed in SEEDS:
                r = record(cfg, seed, rng)
                merged = (seed + len(cid)) % 2 == 0
                if cfg["kind"] == "hybrid":
                    cal, depol = noise(r, rng)
                    if merged:
                        r.update(cal, delta_cal=r["A_subset_noiseless"] - cal["acc_noise_calibrated"], acc_depol=depol)
                    else:
                        (results / "noise").mkdir(parents=True, exist_ok=True)
                        (results / "noise" / f"{r['run_id']}.cal.json").write_text(json.dumps({**cal, "part": "cal"}))
                        for p in DEPOL:
                            (results / "noise" / f"{r['run_id']}.depol{p}.json").write_text(
                                json.dumps({"acc_depol": depol[str(p)], "p": p}))
                elif merged:
                    r.update(acc_noise_calibrated=None, delta_cal=None, acc_depol=None, twoq_transpiled=None,
                             depth_transpiled=None)
                (results / "runs").mkdir(parents=True, exist_ok=True)
                (results / "runs" / f"{r['run_id']}.json").write_text(json.dumps(r))
    # a discarded sweep candidate's run is not part of the study
    stray = record({**configurations()["sweep"][0], "n_qubits": 7}, 0, rng)
    (results / "runs" / f"{stray['run_id']}.json").write_text(json.dumps(stray))
    pd.DataFrame(latency).to_csv(results / "latency.csv", index=False)
    for ds, sel in SELECTION.items():
        (results / f"sweep_selection_{ds}.json").write_text(json.dumps(
            {"selection": {pl: {"encoding": enc, "topology": topo} for pl, (enc, topo) in sel.items()}}))
    an.write_all(results, configs)
    return results, configs


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    return build_study(tmp_path_factory.mktemp("study"))


def test_outputs_written(study):
    results, _ = study
    for name in ("runs.csv", "summary.csv", "qpar_maxima.csv", "sensitivity.csv", "sensitivity_summary.csv",
                 "rank_comparison.csv", "rank_moves.csv", "stats_hybrid_vs_matched.csv",
                 "stats_hybrid_vs_matched_per_topology.csv", "stats_noise_correlations.csv",
                 "sensitivity_supplement_tf1.csv", "sensitivity_supplement_tf1_summary.csv"):
        assert (results / name).stat().st_size > 0, name
    tables = sorted(p.name for p in (results / "tables").glob("*.tex"))
    assert sorted(int(t[1:].split("_")[0]) for t in tables) == list(range(1, 11))
    for t in tables:
        text = (results / "tables" / t).read_text()
        assert r"\toprule" in text and r"\bottomrule" in text and "nan" not in text.lower()
    for stem in ("F1_accuracy_vs_train_frac", "F2_rank_rank", "F3_delta_cal_vs_G", "F4_depolarizing",
                 "F5_qubit_sweep", "F6_sensitivity_heatmap"):
        assert (results / "figures" / f"{stem}.pdf").stat().st_size > 0
        assert (results / "figures" / f"{stem}.png").stat().st_size > 0


def test_runs_csv(study):
    results, _ = study
    runs = pd.read_csv(results / "runs.csv")
    assert len(runs) == 168 * len(SEEDS)  # the stray candidate is skipped
    assert {"e15.val_acc", "timings.train_seconds", "environment.machine", "gate_types.RY",
            "noise_backend.calibration_date"} <= set(runs.columns)
    hybrid = runs[runs["kind"] == "hybrid"]
    assert hybrid[["acc_noise_calibrated", *(f"acc_depol_{p}" for p in DEPOL)]].notna().all().all()
    assert runs.loc[runs["kind"] != "hybrid", "acc_noise_calibrated"].isna().all()
    assert (runs.loc[runs["seed"] == 1, "skipped_steps_total"] == 0).all()  # pre-guard runs: none


def test_merged_and_unmerged_noise_agree(study, tmp_path):
    results, _ = study
    path = next(p for p in (results / "runs").glob("*.json") if "acc_depol" not in json.loads(p.read_text())
                and json.loads(p.read_text())["config"]["kind"] == "hybrid")
    r = json.loads(path.read_text())
    unmerged = an.noise_fields(r, results / "noise", DEPOL)
    cal = json.loads((results / "noise" / f"{r['run_id']}.cal.json").read_text())
    merged = {**r, **{k: cal[k] for k in an.CAL_KEYS}, "acc_depol": {
        str(p): json.loads((results / "noise" / f"{r['run_id']}.depol{p}.json").read_text())["acc_depol"]
        for p in DEPOL}}
    assert an.noise_fields(merged, tmp_path, DEPOL) == unmerged


def test_summary(study):
    results, _ = study
    s = pd.read_csv(results / "summary.csv").set_index("config_id")
    assert len(s) == 168 and (s["n_seeds"] == 5).all()
    runs = pd.read_csv(results / "runs.csv")
    a = runs.loc[runs["config_id"] == "mnist_pre_angle_linear_n4_tf0.5", "A"].to_numpy()
    r = s.loc["mnist_pre_angle_linear_n4_tf0.5"]
    assert r["A_mean"] == pytest.approx(a.mean()) and r["sigma_a"] == pytest.approx(a.std(ddof=1))
    assert r["A_ci95"] == pytest.approx(stats.t.ppf(0.975, 4) * a.std(ddof=1) / np.sqrt(5))
    assert (s.loc[s["kind"] != "hybrid", "Q"] == 0).all() and s.loc[s["kind"] != "hybrid", "qpar"].isna().all()
    # 8.1 identity: sigma_a = 0 and Q, D, G at their maxima -> Q-Par = A
    assert s.loc[PERFECT, "sigma_a"] == 0 and s.loc[PERFECT, "qpar"] == pytest.approx(s.loc[PERFECT, "A_mean"])
    top = pd.read_csv(results / "qpar_maxima.csv").set_index("dataset")
    assert top.loc["mnist", "Q_max"] == 8 and top.loc["mnist", "D_max"] == s.loc[PERFECT, "D_ms"]
    for ds in an.DATASETS:  # ranks only among the 54 main configurations
        main = s[(s["dataset"] == ds) & (s["block"] == an.MAIN_BLOCKS[ds])]
        assert len(main) == 54 and main["qpar_rank"].min() == 1 and main["A_rank"].min() == 1
        assert s.loc[(s["dataset"] == ds) & (s["block"] == "S"), "qpar_rank"].isna().all()
    matched = s.loc["mnist_matched_post_angle_tf1.0"]  # seed 0 never reached the threshold
    assert matched["threshold_reached"] == 4 and matched["epochs_to_threshold_mean"] == 2


def test_sensitivity(study):
    results, _ = study
    sens = pd.read_csv(results / "sensitivity.csv")
    assert len(sens) == 2 * 14 * 54
    summary = pd.read_csv(results / "sensitivity_summary.csv")
    equal = summary[summary["setting"] == "equal"]
    assert np.allclose(equal["kendall_tau"], 1.0) and len(summary) == 28
    one = sens[(sens["dataset"] == "mnist") & (sens["setting"] == "delta=2")]
    assert list(one["rank"]) == sorted(one["rank"]) and one["qpar"].is_monotonic_decreasing
    # gamma is inert on the main matrix (n = 4 throughout), and the tables say so
    gamma = summary[summary["setting"].isin(["gamma=0.5", "gamma=2"])]
    assert np.allclose(gamma["kendall_tau"], 1.0)
    assert "inert here by construction" in (results / "tables" / "T6_sensitivity.tex").read_text()
    # the supplement (Jeremy, 2026-09-28): the 27 train_frac = 1.0 configurations, where Q varies
    supp = pd.read_csv(results / "sensitivity_supplement_tf1.csv")
    assert len(supp) == 2 * 14 * 27
    supp_sum = pd.read_csv(results / "sensitivity_supplement_tf1_summary.csv")
    configs = pd.read_csv(results / "summary.csv").set_index("config_id")
    assert set(configs.loc[supp["config_id"].unique(), "n_qubits"]) == {2, 4, 6, 8}  # Q varies here
    assert (supp_sum.loc[supp_sum["setting"].isin(["gamma=0.5", "gamma=2"]), "kendall_tau"] < 1).any()
    assert "tab:sensitivity_supplement" in (results / "tables" / "T10_sensitivity_supplement.tex").read_text()


def test_rank_driver():
    """A configuration pushed up only by a small D is attributed to D."""
    rows = [{"config_id": f"c{i}", "dataset": "mnist", "block": "M1", "kind": "hybrid", "A_mean": 0.99 - 0.001 * i,
             "sigma_a": 0.002, "Q": 4, "D_ms": 1.0, "G": 100.0} for i in range(54)]
    rows[40]["D_ms"] = 0.5
    s = pd.DataFrame(rows).assign(Q_max=8, D_max=1.0, G_max=100.0)
    s["qpar"] = an.qpar_of(s, an.EQUAL)
    s["A_rank"], s["qpar_rank"] = an.rank_desc(s["A_mean"]), an.rank_desc(s["qpar"])
    taus, moves = an.rank_comparison(s)
    m = moves.iloc[0]
    assert moves["config_id"].tolist() == ["c40"] and m["driver"] == "D" and m["move"] == 40 and m["passed"] == 40
    assert m["log_advantage_D"] == pytest.approx(np.log(2)) and m["log_advantage_G"] == 0 and m["log_advantage_Q"] == 0
    assert taus.iloc[0]["kendall_tau"] < 1


def test_wilcoxon_floor_and_ties():
    assert an.wilcoxon([0.001, 0.002, 0.003, 0.004, 0.005])[1] == pytest.approx(0.0625)
    assert an.wilcoxon([0.001, 0.001, 0.002, 0.002, 0.003])[1] == pytest.approx(0.0625)  # ties: still exact
    assert an.wilcoxon([0.001, 0.0, 0.002, 0.003, 0.004])[1] == pytest.approx(0.125)  # a zero is dropped
    assert an.wilcoxon([0.0] * 5) == (pytest.approx(np.nan, nan_ok=True), 1.0)
    d = np.array([0.003, -0.001, 0.004, 0.002, 0.006])
    assert an.wilcoxon(d)[1] == pytest.approx(stats.wilcoxon(d, method="exact").pvalue)


def test_zero_prints_without_sign():
    """An exact zero mean that floating point returns as -2.2e-17 prints as 0.00, not -0.00."""
    assert an.pct(-2.2e-17, 0.0004) == r"0.00 $\pm$ 0.04"
    assert an.fixed(-0.004, 2) == "0.00" and an.fixed(-0.0004, 3) == "0.000"
    assert an.pct(-0.0001) == "-0.01" and an.fixed(-0.006, 2) == "-0.01"  # real negatives keep their sign


def test_paired_by_seed(study):
    results, _ = study
    runs = pd.read_csv(results / "runs.csv")
    shuffled = runs.sample(frac=1, random_state=0)
    a, _ = an.hybrid_vs_matched(runs)
    b, _ = an.hybrid_vs_matched(shuffled)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    cells = pd.read_csv(results / "stats_hybrid_vs_matched.csv")
    assert len(cells) == 36 and len(pd.read_csv(results / "stats_hybrid_vs_matched_per_topology.csv")) == 108
    one = cells[(cells["dataset"] == "mnist") & (cells["placement"] == "pre") & (cells["encoding"] == "angle")
                & (cells["train_frac"] == 1.0)].iloc[0]
    h = runs[(runs["config_id"].str.startswith("mnist_pre_angle_")) & runs["config_id"].str.endswith("_n4_tf1.0")]
    h = h.groupby("seed")["A"].mean().sort_index().to_numpy()
    m = runs[runs["config_id"] == "mnist_matched_pre_angle_tf1.0"].sort_values("seed")["A"].to_numpy()
    assert one["diff_mean"] == pytest.approx((h - m).mean())
    assert one["p_t"] == pytest.approx(stats.ttest_rel(h, m).pvalue)
    assert one["cohens_dz"] == pytest.approx((h - m).mean() / (h - m).std(ddof=1))


def test_noise_correlations(study):
    results, _ = study
    corr = pd.read_csv(results / "stats_noise_correlations.csv")
    assert set(corr["configurations"]) == {"main+sweep", "main"}
    assert (corr.loc[corr["configurations"] == "main+sweep", "n"] == 27).all()
    assert (corr.loc[corr["configurations"] == "main", "n"] == 18).all()
    g = corr[(corr["dataset"] == "mnist") & (corr["x"] == "G") & (corr["configurations"] == "main+sweep")]
    assert g.iloc[0]["spearman_rho"] > 0.8  # the synthetic Delta_cal grows with G


def test_incomplete_study_stops(study):
    results, configs = study
    runs = an.load_runs(results, configs, DEPOL)
    with pytest.raises(SystemExit, match="1 configurations"):
        an.check_complete(runs[runs["run_id"] != "mnist_base_cnn_tf0.2_s3"], SEEDS, DEPOL, configs)
    with pytest.raises(SystemExit, match="1 hybrid runs"):
        col = f"acc_depol_{DEPOL[0]}"
        an.check_complete(runs.assign(**{col: runs[col].where(runs["run_id"] != "mnist_pre_angle_linear_n4_tf0.2_s0")}),
                          SEEDS, DEPOL, configs)
