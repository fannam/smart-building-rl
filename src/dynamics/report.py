"""Cross-fold aggregation, Markdown summary and diagnostic plots for Phase 2A."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dynamics.evaluate import summarize
from generalization import station_of

POOL_KEYS = ("zone_mae", "zone_mse", "zone_bias", "hws_err", "e_pred", "r_exact_phys", "r_deploy", "zone_mean_pred", "hws_pred")
META_KEYS = ("e_true", "r_true", "occ_transition", "comfort_violation", "heating_recovery", "recovery_window", "cold_step",
             "zone_mean_true", "hws_true", "oat_true", "zone_mean_obs_t0", "t0", "r_true_deploy")
MODEL_KEYS = ("persistence|persistence", "linear|persistence", "gru|persistence", "gru|oracle")
LABEL = {"persistence|persistence": "persistence", "linear|persistence": "linear AR", "gru|persistence": "GRU",
         "gru|oracle": "GRU (oracle OAT)"}


def load_fold(d: Path) -> dict:
    z = np.load(d / "eval_results.npz", allow_pickle=False)
    res: dict = {}
    for k in z.files:
        a, b = k.split("::")
        res.setdefault(a, {})[b] = z[k]
    return res


def pooled(folds: list[tuple[dict, dict]]):
    res = {"meta": {}}
    for k in MODEL_KEYS:
        res[k] = {kk: np.concatenate([r[k][kk] for _, r in folds]) for kk in POOL_KEYS}
    for kk in META_KEYS:
        res["meta"][kk] = np.concatenate([r["meta"][kk] for _, r in folds])
    res["meta"]["scenario"] = np.concatenate([r["meta"]["scenario"] for _, r in folds])
    res["meta"]["building"] = np.concatenate([r["meta"]["building"] for _, r in folds])
    res["meta"]["K"] = np.concatenate([r["meta"]["K"] for _, r in folds])
    res["meta"]["V"] = np.concatenate([r["meta"]["V"] for _, r in folds])
    return res


def f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def aggregate_and_plot(out: Path, cfg: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folds_cfg = json.loads(Path(cfg["folds_config"]).read_text())["folds"]
    folds = []
    for fo in folds_cfg:
        d = out / f"fold_{fo['fold']}_{fo['test_building']}"
        folds.append((json.loads((d / "metrics.json").read_text()), load_fold(d)))
    P = pooled(folds)
    m = P["meta"]
    H = cfg["evaluation"]["report_horizons"]
    weather = np.array([s.split("/")[1] for s in m["scenario"]])
    station = np.array([station_of(w) for w in weather])
    cold_scen = np.isin(weather, ["fairbanks_1", "fairbanks_2", "fairbanks_8"])
    agg = {"pooled": {k: summarize(P, k, H) for k in MODEL_KEYS},
           "by_station": {s: {k: summarize(P, k, H, station == s) for k in MODEL_KEYS} for s in sorted(set(station))},
           "by_K": {int(K): {k: summarize(P, k, H, m["K"] == K) for k in MODEL_KEYS} for K in sorted(set(m["K"].tolist()))},
           "severe_cold_scenarios": {w: {k: summarize(P, k, H, weather == w) for k in MODEL_KEYS}
                                     for w in ("fairbanks_1", "fairbanks_2", "fairbanks_8")},
           "not_severe_cold": {k: summarize(P, k, H, ~cold_scen) for k in MODEL_KEYS},
           "conditions": {c: {k: summarize(P, k, H, step_sel=m[c]) for k in MODEL_KEYS}
                          for c in ("occ_transition", "comfort_violation", "heating_recovery", "cold_step")},
           "per_fold": {fm["fold"]["test_building"]: {k: fm["models"][k]["overall"] for k in MODEL_KEYS} for fm, _ in folds}}
    (out / "metrics.json").write_text(json.dumps(agg, indent=1, default=float))

    # ---------------------------------------------------------------- markdown
    L = ["# Phase 2A — dynamics model results (generated)", "",
         "Zone temperature in °C error (= K), HWS in K, energy in kW, reward in reward units. "
         "Deployable outside air (held at o_t0) unless marked oracle. h = rollout step (h = 1 is teacher-forced).", ""]
    def table(title, getter, cols):
        L.extend([f"## {title}", "", "| fold / held-out | K | V | model | " + " | ".join(cols) + " |",
                  "|---|---|---|---|" + "---|" * len(cols)])
        for fm, _ in folds:
            b = fm["fold"]["test_building"]
            for k in MODEL_KEYS:
                o = fm["models"][k]["overall"]
                L.append(f"| {fm['fold']['fold']} {b} | {fm['meta']['K']} | {fm['meta']['V']} | {LABEL[k]} | "
                         + " | ".join(getter(o, c) for c in cols) + " |")
        L.append("")
    hs = [str(h) for h in H]
    table("Zone temperature MAE (°C) by horizon h", lambda o, c: f(o[c]["zone_temp_c"]["mae"]) if o.get(c) else "—", hs)
    table("Zone temperature bias (°C) by horizon h", lambda o, c: f(o[c]["zone_temp_c"]["bias"]) if o.get(c) else "—", hs)
    table("HWS / tank temperature MAE (K)", lambda o, c: f(o[c]["hws_k"]["mae"], 2) if o.get(c) else "—", hs)
    table("Total electricity MAE (kW) [relative MAE]", lambda o, c: (f"{o[c]['electricity_total_kw']['mae']:.1f} [{o[c]['electricity_total_kw']['rel_mae']:.2f}]") if o.get(c) else "—", hs)
    table("Gas burner MAE (kW) [relative MAE]", lambda o, c: (f"{o[c]['energy_kw']['gas_burner_w']['mae']:.1f} [{o[c]['energy_kw']['gas_burner_w']['rel_mae']:.2f}]") if o.get(c) else "—", hs)
    table("Reward MAE — deployable estimator", lambda o, c: f(o[c]["reward"]["deploy"]["mae"], 5) if o.get(c) else "—", hs)
    table("Reward MAE — exact-physics (privileged occupancy/calendar)", lambda o, c: f(o[c]["reward"]["exact_phys"]["mae"], 5) if o.get(c) else "—", hs)

    def pooled_table(title, groups):
        L.extend([f"## {title}", "", "| group | model | windows | zone MAE h1 | h12 | h36 | HWS MAE h12 | elec MAE kW h12 | gas MAE kW h12 | reward MAE (deploy) h12 |",
                  "|---|---|---|---|---|---|---|---|---|---|"])
        for g, d in groups.items():
            for k in MODEL_KEYS:
                o = d[k]
                if not o or not o.get(12):
                    continue
                L.append(f"| {g} | {LABEL[k]} | {o[12]['windows']} | {f(o[1]['zone_temp_c']['mae'])} | {f(o[12]['zone_temp_c']['mae'])} | "
                         f"{f(o[36]['zone_temp_c']['mae']) if o.get(36) else '—'} | {f(o[12]['hws_k']['mae'], 2)} | "
                         f"{o[12]['electricity_total_kw']['mae']:.1f} | {o[12]['energy_kw']['gas_burner_w']['mae']:.1f} | "
                         f"{f(o[12]['reward']['deploy']['mae'], 5)} |")
        L.append("")
    pooled_table("Pooled over all held-out buildings", {"all": agg["pooled"]})
    pooled_table("By weather station (pooled)", agg["by_station"])
    pooled_table("Severe-cold scenarios (each n = 1 scenario, 2 episodes)", {**agg["severe_cold_scenarios"], "not severe cold": agg["not_severe_cold"]})
    pooled_table("Operating conditions (step-level selection)", agg["conditions"])
    (out / "summary.md").write_text("\n".join(L))

    # ---------------------------------------------------------------- plots
    pd_ = out / "plots"
    pd_.mkdir(exist_ok=True)
    colors = {"persistence|persistence": "tab:gray", "linear|persistence": "tab:blue", "gru|persistence": "tab:red", "gru|oracle": "tab:orange"}
    hh = np.arange(1, P["gru|persistence"]["zone_mae"].shape[1] + 1)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for k in MODEL_KEYS:
        r = P[k]
        ax[0, 0].plot(hh, r["zone_mae"].mean(0), label=LABEL[k], color=colors[k])
        ax[0, 1].plot(hh, np.abs(r["hws_err"]).mean(0), color=colors[k])
        ax[1, 0].plot(hh, np.abs(r["e_pred"][..., :3].sum(-1) - m["e_true"][..., :3].sum(-1)).mean(0) / 1e3, color=colors[k])
        ax[1, 1].plot(hh, np.abs(r["e_pred"][..., 3] - m["e_true"][..., 3]).mean(0) / 1e3, color=colors[k])
    for a_, t in zip(ax.ravel(), ["zone temperature MAE (°C)", "HWS MAE (K)", "electricity MAE (kW)", "gas MAE (kW)"]):
        a_.set_title(t); a_.set_xlabel("rollout step h (5 min)"); a_.grid(alpha=.3)
    ax[0, 0].legend()
    fig.suptitle("Error vs horizon, pooled over the 6 held-out buildings")
    fig.tight_layout(); fig.savefig(pd_ / "error_vs_horizon.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    names = [fm["fold"]["test_building"].replace("building_train_", "b") for fm, _ in folds]
    x = np.arange(len(names))
    for i, k in enumerate(MODEL_KEYS[:3]):
        for j, h in enumerate((12, 36)):
            vals = [fm["models"][k]["overall"][str(h)]["zone_temp_c"]["mae"] if str(h) in fm["models"][k]["overall"] else
                    fm["models"][k]["overall"][h]["zone_temp_c"]["mae"] for fm, _ in folds]
            ax[j].bar(x + i * 0.27 - 0.27, vals, 0.27, label=LABEL[k], color=colors[k])
            ax[j].set_title(f"zone MAE (°C) at h={h} by held-out building"); ax[j].set_xticks(x, names)
    ax[0].legend(); fig.tight_layout(); fig.savefig(pd_ / "error_by_building.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    st = sorted(set(station))
    for i, k in enumerate(MODEL_KEYS[:3]):
        ax.bar(np.arange(len(st)) + i * 0.27 - 0.27, [agg["by_station"][s][k][12]["zone_temp_c"]["mae"] for s in st], 0.27, label=LABEL[k], color=colors[k])
    ax.set_xticks(np.arange(len(st)), st); ax.set_title("zone MAE (°C) at h=12 by weather station (pooled)"); ax.legend()
    fig.tight_layout(); fig.savefig(pd_ / "error_by_station.png", dpi=110); plt.close(fig)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a_, k in zip(ax, MODEL_KEYS[:3]):
        allp, allt, allb = [], [], []
        for fm, r in folds:
            allp.append(r[k]["zone_pred_h"].ravel()); allt.append(r[k]["zone_true_h"].ravel())
        p_, t_ = np.concatenate(allp) - 273.15, np.concatenate(allt) - 273.15
        ok = np.isfinite(p_) & np.isfinite(t_) & (t_ > -100)
        s = rng.choice(np.flatnonzero(ok), min(20000, ok.sum()), replace=False)
        a_.scatter(t_[s], p_[s], s=2, alpha=.3, color=colors[k]); lo, hi = np.percentile(t_[ok], [0.1, 99.9])
        a_.plot([lo, hi], [lo, hi], "k--", lw=1); a_.set_xlim(lo, hi); a_.set_ylim(lo, hi)
        a_.set_title(f"{LABEL[k]}: zone T at h=12 (°C)"); a_.set_xlabel("actual"); a_.set_ylabel("predicted")
    fig.tight_layout(); fig.savefig(pd_ / "zone_pred_vs_actual_h12.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a_, h in zip(ax, (1, 12, 36)):
        for k in ("linear|persistence", "gru|persistence"):
            b = P[k]["zone_bias"][:, h - 1]
            a_.hist(np.clip(b, -3, 3), bins=80, alpha=.5, label=LABEL[k], color=colors[k], density=True)
        a_.set_title(f"window-mean zone residual (°C), h={h}"); a_.legend()
    fig.tight_layout(); fig.savefig(pd_ / "zone_residual_hist.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a_, (c, nm) in zip(ax, (((slice(0, 3)), "electricity"), (3, "gas"))):
        tr = m["e_true"][:, 11, c].sum(-1) if nm == "electricity" else m["e_true"][:, 11, c]
        for k in ("linear|persistence", "gru|persistence"):
            pr = P[k]["e_pred"][:, 11, c].sum(-1) if nm == "electricity" else P[k]["e_pred"][:, 11, c]
            a_.scatter(tr / 1e3, pr / 1e3, s=3, alpha=.3, label=LABEL[k], color=colors[k])
        mx = np.percentile(tr / 1e3, 99.9); a_.plot([0, mx], [0, mx], "k--", lw=1)
        a_.set_title(f"{nm} at h=12 (kW)"); a_.set_xlabel("actual"); a_.set_ylabel("predicted"); a_.legend()
    fig.tight_layout(); fig.savefig(pd_ / "energy_pred_vs_actual_h12.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a_, v in zip(ax, ("deploy", "exact_phys")):
        for k in ("linear|persistence", "gru|persistence"):
            a_.scatter(m["r_true"][:, 11], P[k][f"r_{v}"][:, 11], s=3, alpha=.3, label=LABEL[k], color=colors[k])
        lo = np.percentile(m["r_true"][:, 11], 0.1); a_.plot([lo, 0], [lo, 0], "k--", lw=1)
        a_.set_title(f"reward at h=12: {v}"); a_.set_xlabel("true simulator reward"); a_.set_ylabel("predicted"); a_.legend()
    fig.tight_layout(); fig.savefig(pd_ / "reward_pred_vs_actual_h12.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a_, (cname, title) in zip(ax, (("heating_recovery", "heating recovery steps"), ("cold_step", "steps with OAT < -10 °C"))):
        for k in MODEL_KEYS[:3]:
            sel = m[cname]
            r = P[k]["zone_mae"]
            on = [r[sel[:, j], j].mean() if sel[:, j].any() else np.nan for j in range(r.shape[1])]
            off = [r[~sel[:, j], j].mean() for j in range(r.shape[1])]
            a_.plot(hh, on, color=colors[k], label=f"{LABEL[k]} ({title})")
            a_.plot(hh, off, color=colors[k], ls=":", label=f"{LABEL[k]} (other)")
        a_.set_title(f"zone MAE (°C): {title} vs other"); a_.set_xlabel("h"); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(pd_ / "error_recovery_and_cold.png", dpi=110); plt.close(fig)

    # time-series examples: coldest and a recovery window per cold fold
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for col, (fm, r) in enumerate([x for x in folds if x[0]["fold"]["test_building"] in ("building_train_1", "building_train_2", "building_train_8")][:3]):
        meta = r["meta"]
        i_cold = int(np.argmin(meta["oat_true"][:, 0]))
        rec = np.flatnonzero(meta["heating_recovery"][:, 6])
        i_rec = int(rec[np.argmax(meta["e_true"][rec, 6, 3])]) if rec.size else i_cold
        for row, i in enumerate((i_cold, i_rec)):
            a_ = axes[row, col]
            a_.plot(hh, meta["zone_mean_true"][i] - 273.15, "k", lw=2, label="actual (zone mean)")
            for k in MODEL_KEYS[:3]:
                a_.plot(hh, r[k]["zone_mean_pred"][i] - 273.15, color=colors[k], label=LABEL[k])
            a_.set_title(f"{fm['fold']['test_building']} {'coldest' if row == 0 else 'heating recovery'} window "
                         f"(OAT {meta['oat_true'][i, 0] - 273.15:.0f} °C)", fontsize=9)
            a_.set_xlabel("rollout step h"); a_.set_ylabel("°C")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(pd_ / "rollout_examples.png", dpi=110); plt.close(fig)
    return agg
