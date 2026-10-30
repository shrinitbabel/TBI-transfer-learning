"""
Revision 2 - Step 3: figures and tables from the outputs of steps 1-2.

  Figure 2  cumulative incidence of live discharge (death as competing event), per cohort
  Figure 4  2-D C-index plot with 95% intervals (replaces the 3-D bar charts)
  Figure 5  selected dynamic-AUC curves, mean with 95% band over repeated splits
  Supplemental: calibration plots, all 36 dynamic-AUC panels, results tables
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
ORDER = ["India", "Jordan", "Florida", "California"]
LABEL = {"India": "India", "Jordan": "Jordan", "Florida": "Florida (USA)", "California": "California (USA)"}
METH = [("baseline", "Baseline (target only)", "#1f77b4", "o"),
        ("standard", "Standard TL", "#ff7f0e", "s"),
        ("weighted", "Weighted TL", "#2ca02c", "^")]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 10, "savefig.dpi": 300})

# ---------------------------------------------------------------- Figure 2
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
cols = dict(zip(ORDER, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
for name in ORDER:
    p = os.path.join(OUT, "cohorts", f"{name.lower()}_r2.csv")
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    t = np.linspace(0, 90, 901)
    # complete follow-up (every patient ends in discharge or death) -> Aalen-Johansen = empirical proportion
    cif_dis = [((df.discharged == 1) & (df.los <= x)).mean() for x in t]
    cif_die = [((df.died == 1) & (df.los <= x)).mean() for x in t]
    ax[0].step(t, cif_dis, where="post", color=cols[name], lw=2, label=f"{LABEL[name]} (n={len(df)})")
    ax[1].step(t, cif_die, where="post", color=cols[name], lw=2, label=LABEL[name])
ax[0].set_title("A. Cumulative incidence of live discharge")
ax[1].set_title("B. Cumulative incidence of in-hospital death (competing event)")
for a in ax:
    a.set_xlabel("Days since admission"); a.set_xlim(0, 90); a.grid(alpha=0.3)
ax[0].set_ylabel("Cumulative incidence"); ax[0].set_ylim(0, 1)
ax[0].legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Figure2_cumulative_incidence.png")); plt.close(fig)

# ---------------------------------------------------------------- Figure 4
s = pd.read_csv(os.path.join(OUT, "results_summary_cr.csv"))
targets = [t for t in ORDER if t in s.target.unique()]
fig, axes = plt.subplots(len(targets), 3, figsize=(13, 3.1 * len(targets)), sharey="row")
for i, tgt in enumerate(targets):
    srcs = [x for x in ORDER if x != tgt and x in s.source.unique()]
    for jx, src in enumerate(srcs):
        a = axes[i, jx]
        g = s[(s.source == src) & (s.target == tgt)]
        for k, (m, lab, c, mk) in enumerate(METH):
            gm = g[g.method == m].sort_values("ratio")
            x = np.arange(3) + (k - 1) * 0.18
            a.errorbar(x, gm.cindex_mean, yerr=[gm.cindex_mean - gm.cindex_lo, gm.cindex_hi - gm.cindex_mean],
                       fmt=mk, color=c, capsize=3, ms=6, lw=1.4, label=lab)
        a.axhline(0.5, ls=":", color="gray", lw=1)
        a.set_xticks(range(3)); a.set_xticklabels(["5%", "10%", "20%"])
        a.set_title(f"{LABEL[src]} → {LABEL[tgt]}")
        a.grid(alpha=0.3, axis="y")
        if jx == 0:
            a.set_ylabel(f"Target: {LABEL[tgt]}\nC-index")
        if i == len(targets) - 1:
            a.set_xlabel("Target data used for fine-tuning")
lo_ = max(0.2, np.floor(s.cindex_lo.min() * 10) / 10)
for a in axes.ravel():
    a.set_ylim(lo_, 1.0)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(FIG, "Figure4_cindex_2D.png")); plt.close(fig)

# ---------------------------------------------------------------- dynamic AUC curves
cur = pd.read_csv(os.path.join(OUT, "auc_curves_cr.csv"))
# bands = interquartile range across repeated splits (typical split-to-split spread)
agg = cur.groupby(["source", "target", "ratio", "method", "day"])["auc"].agg(
    mean="mean", lo=lambda x: np.nanpercentile(x, 25) if x.notna().any() else np.nan,
    hi=lambda x: np.nanpercentile(x, 75) if x.notna().any() else np.nan).reset_index()


def auc_panel(a, src, tgt, ratio, panel=""):
    g = agg[(agg.source == src) & (agg.target == tgt) & np.isclose(agg.ratio, ratio)]
    for m, lab, c, mk in METH:
        gm = g[g.method == m]
        a.plot(gm.day, gm["mean"], color=c, marker=mk, ms=4, lw=1.8, label=lab)
        a.fill_between(gm.day, gm["lo"], gm["hi"], color=c, alpha=0.15)
    a.axhline(0.5, ls="--", color="gray", lw=1)
    a.set_ylim(0.3, 1.0); a.grid(alpha=0.3)
    a.set_title(f"{panel}{LABEL[src]} → {LABEL[tgt]}, {int(ratio * 100)}% fine-tuning")
    a.set_xlabel("Days since admission"); a.set_ylabel("Time-dependent AUC")


# largest, most consistent TL gains (A-C) and the clearest negative transfer (D)
SEL = [("India", "Jordan", 0.05), ("California", "Jordan", 0.05),
       ("India", "Jordan", 0.10), ("Florida", "California", 0.10)]
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for a, (src, tgt, r), lab in zip(axes.ravel(), SEL, "ABCD"):
    auc_panel(a, src, tgt, r, panel=f"{lab}. ")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(FIG, "Figure5_dynamic_auc.png")); plt.close(fig)

os.makedirs(os.path.join(FIG, "supplement_auc"), exist_ok=True)
for src in ORDER:
    for tgt in ORDER:
        if src == tgt or not ((agg.source == src) & (agg.target == tgt)).any():
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
        for a, r in zip(axes, [0.05, 0.10, 0.20]):
            auc_panel(a, src, tgt, r)
        axes[0].legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "supplement_auc", f"auc_{src}_to_{tgt}.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- calibration (supplement)
cal = pd.read_csv(os.path.join(OUT, "calibration_points_cr.csv.gz"))
HORIZON = {"India": 7, "Jordan": 7, "California": 7, "Florida": 30}   # as in 02_transfer_experiment.py
CAL_SEL = [("India", "Jordan"), ("California", "Jordan"), ("India", "California"),
           ("Jordan", "California"), ("California", "India"), ("India", "Florida")]
fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
for a, (src, tgt) in zip(axes.ravel(), CAL_SEL):
    g = cal[(cal.source == src) & (cal.target == tgt) & np.isclose(cal.ratio, 0.10)]
    for m, lab, c, mk in METH:
        gm = g[g.method == m].copy()
        nb = 5 if tgt == "Jordan" else 10
        gm["bin"] = pd.qcut(gm.pred.rank(method="first"), nb, labels=False)
        b = gm.groupby("bin").agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))
        se = np.sqrt(b.obs * (1 - b.obs) / b.n)
        a.errorbar(b.pred, b.obs, yerr=1.96 * se, fmt=mk + "-", color=c, ms=5, capsize=2, lw=1.2, label=lab)
    a.plot([0, 1], [0, 1], ls="--", color="gray")
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.grid(alpha=0.3)
    hzn = HORIZON[tgt]
    a.set_title(f"{LABEL[src]} → {LABEL[tgt]} (10%), day {hzn}")
    a.set_xlabel(f"Predicted probability of discharge by day {hzn}")
    a.set_ylabel("Observed proportion discharged")
axes[0, 0].legend(loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "SuppFigure_calibration.png")); plt.close(fig)

# ---------------------------------------------------------------- tables
comp = pd.read_csv(os.path.join(OUT, "comparisons_cr.csv"))


def stars(q):
    return "" if pd.isna(q) else ("***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "")


rows = []
for src in ORDER:
    for tgt in ORDER:
        if src == tgt:
            continue
        for r in [0.05, 0.10, 0.20]:
            g = s[(s.source == src) & (s.target == tgt) & np.isclose(s.ratio, r)].set_index("method")
            if g.empty:
                continue
            row = {"Source": src, "Target": tgt, "Fine-tune": f"{int(r * 100)}%"}
            for m, lab in [("baseline", "Baseline"), ("standard", "Standard TL"), ("weighted", "Weighted TL")]:
                txt = f"{g.loc[m, 'cindex_mean']:.3f} ({g.loc[m, 'cindex_lo']:.3f}-{g.loc[m, 'cindex_hi']:.3f})"
                if m != "baseline":
                    c = comp[(comp.source == src) & (comp.target == tgt) & np.isclose(comp.ratio, r) & (comp.method == m)].iloc[0]
                    txt += stars(c.q_fdr)
                row[lab] = txt
            for m, lab in [("baseline", "Base"), ("standard", "Std"), ("weighted", "Wtd")]:
                row[f"IBS {lab}"] = f"{g.loc[m, 'ibs_mean']:.3f}"
                row[f"MAE {lab}"] = f"{g.loc[m, 'mae_mean']:.1f}"
                row[f"Cal O/E {lab}"] = f"{g.loc[m, 'obs_h_mean']:.2f}/{g.loc[m, 'pred_h_mean']:.2f}"
            row["IBS null"] = f"{g['ibs_null'].iloc[0]:.3f}"
            row["MAE null"] = f"{g['mae_null'].iloc[0]:.1f}"
            rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_results_cr.csv"), index=False)

if os.path.exists(os.path.join(OUT, "results_summary_cs.csv")):
    cs = pd.read_csv(os.path.join(OUT, "results_summary_cs.csv"))
    cc = pd.read_csv(os.path.join(OUT, "comparisons_cs.csv"))
    rows = []
    for _, g in cs.groupby(["source", "target", "ratio"]):
        g = g.set_index("method"); src, tgt, r = g.source.iloc[0], g.target.iloc[0], g.ratio.iloc[0]
        row = {"Source": src, "Target": tgt, "Fine-tune": f"{int(r * 100)}%"}
        for m, lab in [("baseline", "Baseline"), ("standard", "Standard TL"), ("weighted", "Weighted TL")]:
            txt = f"{g.loc[m, 'cindex_mean']:.3f} ({g.loc[m, 'cindex_lo']:.3f}-{g.loc[m, 'cindex_hi']:.3f})"
            if m != "baseline":
                c = cc[(cc.source == src) & (cc.target == tgt) & np.isclose(cc.ratio, r) & (cc.method == m)].iloc[0]
                txt += stars(c.q_fdr)
            row[lab] = txt
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_results_cs.csv"), index=False)
print("figures and tables written to", OUT)

# ---------------------------------------------------------------- Figure 4 (final): paired change in C-index
# Baseline and TL are evaluated on the same splits, so the comparison is paired: plot the mean of the
# per-split difference (standard TL - baseline) with its 95% confidence interval (1.96 * SD / sqrt(n)).
raw = pd.read_csv(os.path.join(OUT, "results_raw_cr.csv"))
wide = raw.pivot_table(index=["source", "target", "ratio", "rep"], columns="method", values="cindex").reset_index()
SRC_COL = {"India": "#d62728", "Jordan": "#ff7f0e", "Florida": "#2ca02c", "California": "#1f77b4"}
paired = []
for (src, tgt, ra), g in wide.groupby(["source", "target", "ratio"]):
    for m in ["standard", "weighted"]:
        dlt = g[m] - g["baseline"]
        half = 1.96 * dlt.std() / np.sqrt(len(dlt))
        paired.append({"source": src, "target": tgt, "ratio": ra, "method": m, "delta": dlt.mean(),
                       "ci_lo": dlt.mean() - half, "ci_hi": dlt.mean() + half,
                       "wins": int((dlt > 0).sum()), "n": len(dlt)})
paired = pd.DataFrame(paired)
paired.to_csv(os.path.join(OUT, "paired_delta_cindex_cr.csv"), index=False)

T_ORDER = ["Jordan", "California", "India", "Florida"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), sharey=True)
for ax, tgt, lab in zip(axes, T_ORDER, "ABCD"):
    srcs = [x for x in ["India", "Jordan", "Florida", "California"] if x != tgt]
    for k, src in enumerate(srcs):
        g = paired[(paired.source == src) & (paired.target == tgt) & (paired.method == "standard")].sort_values("ratio")
        xs = np.arange(3) + (k - 1) * 0.15
        ax.errorbar(xs, g.delta, yerr=[g.delta - g.ci_lo, g.ci_hi - g.delta], fmt="o-", color=SRC_COL[src],
                    capsize=3, ms=6, lw=1.6, label=f"Source: {LABEL[src]}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["5%", "10%", "20%"])
    ax.set_title(f"{lab}. Target: {LABEL[tgt]}"); ax.set_xlabel("Target data used for fine-tuning")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9, loc="upper right")
axes[0].set_ylabel("Change in C-index vs. baseline\n(standard TL − target-only)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Figure4_delta_cindex.png"), dpi=300); plt.close(fig)
print("Figure 4 (paired change in C-index) written")
