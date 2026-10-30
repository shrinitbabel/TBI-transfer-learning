"""
Revision 2 - Step 4: descriptive statistics (Supplemental Table 2, medians/IQR) and the
cohort-flow diagram (Supplemental Figure 3). Vital-sign categories are summarized among
patients with an observed (non-missing, plausible) value.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
spec = importlib.util.spec_from_file_location("exp", os.path.join(HERE, "02_transfer_experiment.py"))
ORDER = ["Florida", "India", "California", "Jordan"]   # column order of Supplemental Table 2


def hr_c(a, v):
    lim = 160 if a < 1 else 150 if a < 3 else 140 if a < 6 else 120 if a < 12 else 100
    return v > lim


def rr_c(a, v):
    lim = 60 if a < 1 else 40 if a < 3 else 34 if a < 6 else 30 if a < 12 else 20
    return v > lim


def sbp_c(a, v):
    lim = 70 if a < 1 else (70 + 2 * a) if a < 10 else 90
    return v < lim


rows = {}
for name in ORDER:
    df = pd.read_csv(os.path.join(OUT, "cohorts", f"{name.lower()}_r2.csv"))
    r = {"N": f"{len(df):,}"}
    a = df.age.quantile([.25, .5, .75])
    fmt = "{:.1f}" if name == "Jordan" else "{:.0f}"
    r["Age, median (IQR), y"] = f"{fmt.format(a[.5])} ({fmt.format(a[.25])}-{fmt.format(a[.75])})"
    r["Male, %"] = f"{100 * df.sex.mean():.1f}%"

    def pct(flag, given, num, fn):
        if df[given].notna().any():
            s = df[given].dropna()
            return f"{100 * (s == flag).mean():.1f}%"
        m = df[["age", num]].dropna()
        return f"{100 * np.mean([fn(x, y) for x, y in zip(m.age, m[num])]):.1f}%"
    r["Tachycardia, %"] = pct("tachycardia", "hr_cat_given", "hr", hr_c)
    r["Tachypnea, %"] = pct("tachypnea", "rr_cat_given", "rr", rr_c)
    r["Hypotension (age-adjusted), %"] = pct("low", "sbp_cat_given", "sbp", sbp_c)
    g = df.gcs.dropna()
    r["GCS 3-8, %"] = f"{100 * (g <= 8).mean():.1f}%"
    r["GCS 9-12, %"] = f"{100 * ((g >= 9) & (g <= 12)).mean():.1f}%"
    r["GCS 13-15, %"] = f"{100 * (g >= 13).mean():.1f}%"
    q = df.los.quantile([.25, .5, .75])
    r["LOS, median (IQR), days"] = f"{q[.5]:.1f} ({q[.25]:.1f}-{q[.75]:.1f})"
    r["Discharged alive, n (%)"] = f"{df.discharged.sum():,} ({100 * df.discharged.mean():.1f}%)"
    r["In-hospital death, n (%)"] = f"{df.died.sum():,} ({100 * df.died.mean():.1f}%)"
    r["Discharged by day 15, %"] = f"{100 * ((df.discharged == 1) & (df.los <= 15)).mean():.1f}%"
    moi = df.MOI.value_counts(normalize=True) * 100
    r["MOI (top 3)"] = "; ".join(f"{k} {v:.1f}%" for k, v in moi.head(3).items())
    if name == "Florida":
        r["MOI (top 3)"] += f"; Animal/Environmental {moi.get('Animal / Environmental', 0):.1f}%"
    rows[name] = r
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT, "supp_table2_characteristics.csv"))
print(tab.to_string())

# ---------------------------------------------------------------- flow diagram
flow = pd.read_csv(os.path.join(OUT, "cohort_flow.csv"))
fig, axes = plt.subplots(1, 4, figsize=(17, 7.5))
for ax, name in zip(axes, ["India", "Jordan", "California", "Florida"]):
    f = flow[flow.cohort == name].reset_index(drop=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title({"Florida": "Florida (USA)", "California": "California (USA)"}.get(name, name),
                 fontsize=14, fontweight="bold")
    # main boxes = inclusion steps; consecutive exclusions are merged into one side box
    mains, excl, prev = [], {}, None
    for _, r in f.iterrows():
        if r.step.startswith("Excluded"):
            d = prev - r.n
            if d > 0:
                excl.setdefault(len(mains) - 1, []).append(f"{r.step.replace('Excluded: ', '')} (n={d:,})")
        else:
            mains.append(r.step.replace(">=", "≥") + f"\nn = {r.n:,}")
        prev = r.n
    ys = np.linspace(0.88, 0.10, len(mains))
    for k, (txt, y) in enumerate(zip(mains, ys)):
        wrapped = "\n".join(textwrap.fill(line, 30) for line in txt.split("\n"))
        ax.text(0.33, y, wrapped, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="#dceefb", ec="#1f77b4"))
        if k + 1 < len(mains):
            ax.annotate("", xy=(0.33, ys[k + 1] + 0.07), xytext=(0.33, y - 0.07),
                        arrowprops=dict(arrowstyle="->", color="black"))
        if k in excl and k + 1 < len(mains):
            ym = (y + ys[k + 1]) / 2
            body = "\n".join(textwrap.fill("• " + e, 24) for e in excl[k])
            ax.text(0.58, ym, "Excluded:\n" + body, ha="left", va="center", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.4", fc="#fdecea", ec="#d62728"))
            ax.annotate("", xy=(0.555, ym), xytext=(0.33, ym), arrowprops=dict(arrowstyle="->", color="gray"))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "SuppFigure_cohort_flow.png"), dpi=300)

# ---------------------------------------------------------------- Supplemental Figure 2 (GCS), observed values only
fig, ax = plt.subplots(figsize=(9, 5))
names = ["California", "Florida", "India", "Jordan"]
sev, mod, mild = [], [], []
for name in names:
    g = pd.read_csv(os.path.join(OUT, "cohorts", f"{name.lower()}_r2.csv")).gcs.dropna()
    sev.append(100 * (g <= 8).mean()); mod.append(100 * ((g >= 9) & (g <= 12)).mean()); mild.append(100 * (g >= 13).mean())
x = np.arange(len(names))
ax.bar(x, sev, color="#d73027", label="Severe (3-8)")
ax.bar(x, mod, bottom=sev, color="#fc8d59", label="Moderate (9-12)")
ax.bar(x, mild, bottom=np.add(sev, mod), color="#fec44f", label="Mild (13-15)")
ax.set_xticks(x); ax.set_xticklabels(["California (USA)", "Florida (USA)", "India", "Jordan"])
ax.set_ylabel("Proportion of patients with recorded GCS (%)"); ax.set_ylim(0, 100)
ax.set_title("GCS Severity Distribution by Cohort"); ax.legend(title="GCS category", loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "SuppFigure2_GCS.png"), dpi=300)
