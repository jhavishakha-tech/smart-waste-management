"""Reproducible experiments for the paper.  Usage:
   python experiments.py forecast | main | pareto | robust | figures | all     (add --quick for a tiny smoke test)
Results are appended to results/*.csv (resumable: finished runs are skipped)."""
import sys, os, time, itertools
import numpy as np, pandas as pd
import config as C, forecasting as F, simulator as S

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
QUICK = "--quick" in sys.argv
SIZES = [20, 50, 100, 200] if not QUICK else [20, 50]
SEEDS = list(range(10)) if not QUICK else [0, 1]
DAYS = 14 if not QUICK else 5
TL = {20: 0.2, 50: 0.3, 100: 0.5, 200: 0.9}


def run_grid(name, configs, fc):
    """configs: list of dict(n, policy, seed, [theta, threshold, noise, loss, tag]). Resumable."""
    path = os.path.join(RES, f"{name}{'_quick' if QUICK else ''}.csv")
    done = set()
    if os.path.exists(path):
        old = pd.read_csv(path)
        done = set(old["key"])
    t0 = time.time()
    for i, c in enumerate(configs):
        key = "|".join(f"{k}={c[k]}" for k in sorted(c))
        if key in done:
            continue
        r = S.run_simulation(c["n"], c["policy"], c["seed"], days=DAYS, forecaster=fc,
                             theta=c.get("theta", C.THETA), threshold=c.get("threshold", C.THRESHOLD),
                             noise=c.get("noise", C.SENSOR_NOISE_SD), loss=c.get("loss", C.PACKET_LOSS),
                             time_limit_s=TL[c["n"]])
        base = dict(theta=np.nan, threshold=np.nan, noise=C.SENSOR_NOISE_SD, loss=C.PACKET_LOSS, tag="main")
        row = {**base, **c, **r, "key": key}
        pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"[{name}] {i+1}/{len(configs)} {key}  ({time.time()-t0:.0f}s)", flush=True)


def exp_forecast():
    rows, (Xte, yte) = F.compare_models()
    pd.DataFrame(rows).to_csv(os.path.join(RES, "forecast_models.csv"), index=False)
    X, y = F.build_dataset(range(8)); fc = F.Forecaster("hgb").fit(X, y)
    import joblib; joblib.dump(fc, F.MODEL_PATH)
    np.savez(os.path.join(RES, "forecast_scatter.npz"), y=yte[:3000], p=fc.model.predict(Xte)[:3000])
    print(pd.DataFrame(rows).round(3))


def exp_main(fc):
    cfg = [dict(n=n, policy=p, seed=s) for n in SIZES for s in SEEDS for p in C.POLICIES]
    run_grid("main", cfg, fc)


def exp_pareto(fc):
    n = 100 if not QUICK else 50
    seeds = SEEDS[:5]
    cfg = []
    for s in seeds:
        cfg += [dict(n=n, policy="fixed_daily", seed=s, tag="fixed_daily"), dict(n=n, policy="fixed_alt", seed=s, tag="fixed_alt")]
        cfg += [dict(n=n, policy="threshold", seed=s, threshold=th, tag=f"threshold_{th}") for th in (0.5, 0.6, 0.7, 0.8)]
        cfg += [dict(n=n, policy="proposed", seed=s, theta=th, tag=f"proposed_{th}") for th in (0.7, 0.8, 0.9, 1.0)]
    run_grid("pareto", cfg, fc)


def exp_robust(fc):
    n = 100 if not QUICK else 50
    seeds = SEEDS[:5]
    cfg = []
    for s in seeds:
        for pol in ("threshold", "proposed"):
            for noise in (0.0, 0.02, 0.05, 0.10):
                cfg.append(dict(n=n, policy=pol, seed=s, noise=noise, loss=0.0, tag="noise"))
            for loss in (0.1, 0.2, 0.3):
                cfg.append(dict(n=n, policy=pol, seed=s, noise=0.02, loss=loss, tag="loss"))
    run_grid("robust", cfg, fc)


# ------------------------------------------------------------------ figures & tables
def figures():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 130})
    sfx = "_quick" if QUICK else ""
    col = dict(fixed_daily="#7f7f7f", fixed_alt="#bcbd22", threshold="#1f77b4", proposed="#d62728")

    # forecast
    p = os.path.join(RES, "forecast_models.csv")
    if os.path.exists(p):
        d = pd.read_csv(p); z = np.load(os.path.join(RES, "forecast_scatter.npz"))
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].bar(d.model, d.MAE, color=["#999", "#6baed6", "#fd8d3c", "#d62728"]); ax[0].set_ylabel("MAE (% of bin capacity)")
        ax[0].set_title("24 h fill-accumulation forecast"); ax[0].tick_params(axis="x", rotation=20)
        ax[1].scatter(z["y"] * 100, z["p"] * 100, s=3, alpha=.3); m = max(z["y"].max(), z["p"].max()) * 100
        ax[1].plot([0, m], [0, m], "k--", lw=1); ax[1].set_xlabel("Actual (%)"); ax[1].set_ylabel("Predicted (%)"); ax[1].set_title("Gradient Boosting")
        fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_forecast.png")); plt.close(fig)

    # main
    p = os.path.join(RES, f"main{sfx}.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        metrics = [("distance_km_per_day", "Distance (km/day)"), ("overflow_pct", "Overflow (% bin-hours)"),
                   ("avg_fill_at_collection", "Avg. fill at collection"), ("co2_kg", "CO$_2$ over period (kg)")]
        fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
        sizes = sorted(d.n.unique()); w = 0.2
        for ax, (m, lab) in zip(axs, metrics):
            for i, pol in enumerate(C.POLICIES):
                g = d[d.policy == pol].groupby("n")[m].agg(["mean", "std"]).reindex(sizes)
                ax.bar(np.arange(len(sizes)) + (i - 1.5) * w, g["mean"], w, yerr=g["std"], capsize=2,
                       color=col[pol], label=C.POLICY_LABEL[pol])
            ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes); ax.set_xlabel("Number of bins"); ax.set_ylabel(lab)
        axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_main.png")); plt.close(fig)
        # summary table mean ± std
        cols = ["distance_km_per_day", "trucks_per_day", "overflow_pct", "overflow_events_per_bin_week",
                "lost_waste_pct", "avg_fill_at_collection", "fuel_L", "co2_kg"]
        agg = d.groupby(["n", "policy"])[cols].agg(["mean", "std"])
        agg.to_csv(os.path.join(RES, "main_summary.csv"))
        # paired tests (Wilcoxon) proposed vs others
        rows = []
        for n in sizes:
            for other in ("fixed_daily", "fixed_alt", "threshold"):
                for m in ("distance_km_per_day", "overflow_pct"):
                    a = d[(d.n == n) & (d.policy == "proposed")].sort_values("seed")[m].values
                    b = d[(d.n == n) & (d.policy == other)].sort_values("seed")[m].values
                    if len(a) == len(b) and len(a) > 4 and np.any(a != b):
                        pv = stats.wilcoxon(a, b).pvalue
                    else:
                        pv = np.nan
                    rows.append(dict(n=n, vs=other, metric=m, proposed_mean=a.mean(), other_mean=b.mean(),
                                     change_pct=100 * (a.mean() - b.mean()) / max(b.mean(), 1e-9), p_wilcoxon=pv))
        pd.DataFrame(rows).to_csv(os.path.join(RES, "stat_tests.csv"), index=False)

    # pareto
    p = os.path.join(RES, f"pareto{sfx}.csv")
    if os.path.exists(p):
        d = pd.read_csv(p); g = d.groupby("tag")[["distance_km_per_day", "overflow_pct"]].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for tag, r in g.iterrows():
            fam = tag.split("_")[0] if not tag.startswith("fixed") else tag
            c = col.get(fam, "k")
            ax.errorbar(r[("distance_km_per_day", "mean")], r[("overflow_pct", "mean")],
                        xerr=r[("distance_km_per_day", "std")], yerr=r[("overflow_pct", "std")], fmt="o", color=c, capsize=2)
            ax.annotate(tag.replace("proposed_", "θ=").replace("threshold_", "τ="), (r[("distance_km_per_day", "mean")], r[("overflow_pct", "mean")]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set_xlabel("Distance (km/day)"); ax.set_ylabel("Overflow (% bin-hours)"); ax.set_title("Cost vs service-quality trade-off")
        fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_pareto.png")); plt.close(fig)
        g.to_csv(os.path.join(RES, "pareto_summary.csv"))

    # robustness
    p = os.path.join(RES, f"robust{sfx}.csv")
    if os.path.exists(p):
        d = pd.read_csv(p); fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, tag, xk, xl in [(axs[0], "noise", "noise", "Sensor noise SD"), (axs[1], "loss", "loss", "Packet-loss probability")]:
            for pol in ("threshold", "proposed"):
                g = d[(d.tag == tag) & (d.policy == pol)].groupby(xk)["overflow_pct"].agg(["mean", "std"])
                ax.errorbar(g.index, g["mean"], g["std"], marker="o", color=col[pol], label=C.POLICY_LABEL[pol], capsize=2)
            ax.set_xlabel(xl); ax.set_ylabel("Overflow (% bin-hours)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_robust.png")); plt.close(fig)
    print("figures/tables written to", RES)


if __name__ == "__main__":
    cmd = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = cmd[0] if cmd else "all"
    if cmd in ("forecast", "all"): exp_forecast()
    fc = F.load_or_train()
    if cmd in ("main", "all"): exp_main(fc)
    if cmd in ("pareto", "all"): exp_pareto(fc)
    if cmd in ("robust", "all"): exp_robust(fc)
    if cmd in ("figures", "all"): figures()
