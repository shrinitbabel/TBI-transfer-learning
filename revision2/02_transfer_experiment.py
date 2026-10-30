"""
Revision 2 - Step 2: few-shot transfer-learning experiment (corrected).

Derived from transfer_learning_experiment_fixed.ipynb. Changes (reviewer #1):

  Outcome / estimand (comment 2)
    * The event is LIVE DISCHARGE. In-hospital death is a COMPETING event, handled with the
      Fine-Gray subdistribution risk set: patients who die remain "at risk" of discharge
      for the whole follow-up (coded as censored at T_INF > any observed time). Because
      every patient is followed to discharge or death (no loss to follow-up), this coding
      gives the exact subdistribution risk set, and the GBSA Cox loss then models the
      subdistribution hazard of discharge; 1 - S(t) is the cumulative incidence (CIF)
      of live discharge by day t.
      NOTE: the earlier code passed the *death* indicator as the survival event, i.e. it
      modeled time to in-hospital death with discharge censored. This is corrected here.
    * Sensitivity analysis (--coding cs): cause-specific coding (deaths censored at death).

  Leakage (comment 3)
    * Imputation (IterativeImputer, Bayesian ridge) is fitted on predictors only (never LOS
      or vital status) and only on the training data of each split: the source cohort for
      pretraining, the few-shot target subset for fine-tuning / baseline. Vital-sign
      categories are derived after imputation, inside the split.
    * One-hot encoding uses fixed, pre-specified category lists (no data-dependent fit).

  Validation and fair comparison (comment 4)
    * Baseline capacity matched to TL: 150 boosting stages (TL = 100 source + 50 target).
    * 20 repeated random target splits (80% pool / 20% untouched test, stratified on death);
      few-shot subset drawn from the pool. ALL metrics (C-index, time-dependent AUC,
      Brier, calibration, MAE) are computed on the untouched test set only.
    * Exact few-shot sizes and event counts are logged for each split.

  Calibration / absolute accuracy (comment 6)
    * Integrated Brier score (IBS) and scaled IBS vs. a null (marginal CIF) model,
      calibration at a fixed horizon (observed vs predicted proportion discharged), and
      MAE (days) of the predicted median LOS among patients discharged alive.

Usage:  python 02_transfer_experiment.py [--coding cr|cs] [--reps 20]
"""
import os, copy, argparse, itertools, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.utils import compute_sample_weight
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--coding", default="cr", choices=["cr", "cs"])
ap.add_argument("--reps", type=int, default=20)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
TAG = args.coding
RATIOS = [0.05, 0.10, 0.20]
N_REP = args.reps
T_INF = 10_000.0          # "never discharged" time for deaths under the subdistribution coding
HORIZON = {"India": 7, "Jordan": 7, "California": 7, "Florida": 30}   # calibration horizon (days)
GBSA = dict(learning_rate=0.1, max_depth=3, subsample=1.0, loss="coxph")   # sksurv defaults, fixed a priori
N_PRE, N_FT, N_BASE = 100, 50, 150

NUM_IMPUTE = ["age", "sex", "gcs", "sbp", "hr", "rr"]
FALLBACK = {"age": 30.0, "sex": 1.0, "gcs": 15.0, "sbp": 120.0, "hr": 80.0, "rr": 16.0}
NUMERIC = ["age", "gcs"]
MOI_CATS = ["Animal / Environmental", "Assault / Blunt Force", "Bicycle Crash", "Burn / Fire / Blast",
            "Fall", "Gunshot Wound (GSW)", "Motor Vehicle Collision", "Motorcycle Crash",
            "Other / Unknown", "Pedestrian Struck"]
CATS = {"sex": ["0", "1"], "hr_cat": ["normal", "tachycardia"], "rr_cat": ["normal", "tachypnea"],
        "sbp_cat": ["low", "normal"], "MOI": MOI_CATS}
ENC = OneHotEncoder(categories=[CATS[k] for k in CATS], handle_unknown="ignore", sparse_output=False)
ENC.fit(pd.DataFrame({k: [v[0]] for k, v in CATS.items()}))   # fixed categories; nothing learned from data

registry = {}
for name in ["India", "Jordan", "Florida", "California"]:
    p = os.path.join(OUT, "cohorts", f"{name.lower()}_r2.csv")
    if os.path.exists(p):
        registry[name] = pd.read_csv(p).reset_index(drop=True)
print("Loaded:", {k: len(v) for k, v in registry.items()})


# ------------------------------------------------------------------ preprocessing
def hr_category(age, hr):
    if age < 1: return 'tachycardia' if hr > 160 else 'normal'
    if age < 3: return 'tachycardia' if hr > 150 else 'normal'
    if age < 6: return 'tachycardia' if hr > 140 else 'normal'
    if age < 12: return 'tachycardia' if hr > 120 else 'normal'
    return 'tachycardia' if hr > 100 else 'normal'


def rr_category(age, rr):
    if age < 1: return 'tachypnea' if rr > 60 else 'normal'
    if age < 3: return 'tachypnea' if rr > 40 else 'normal'
    if age < 6: return 'tachypnea' if rr > 34 else 'normal'
    if age < 12: return 'tachypnea' if rr > 30 else 'normal'
    return 'tachypnea' if rr > 20 else 'normal'


def sbp_category(age, sbp):
    if age < 1: return 'low' if sbp < 70 else 'normal'
    if age < 10: return 'low' if sbp < (70 + 2 * age) else 'normal'
    return 'low' if sbp < 90 else 'normal'


class Prep:
    """Predictor-only imputation fitted on training rows; categories derived after imputation."""
    def __init__(self, seed):
        self.seed = seed

    def fit(self, df):
        self.cols = [c for c in NUM_IMPUTE if df[c].notna().any()]
        self.empty = [c for c in NUM_IMPUTE if c not in self.cols]
        self.imp = IterativeImputer(random_state=self.seed, max_iter=10).fit(df[self.cols].to_numpy(float))
        return self

    def transform(self, df):
        d = df.copy()
        d[self.cols] = self.imp.transform(d[self.cols].to_numpy(float))
        for c in self.empty:
            d[c] = d[c].fillna(FALLBACK[c])
        d["sex"] = d["sex"].round().clip(0, 1)
        d["gcs"] = d["gcs"].round().clip(3, 15)
        d["age"] = d["age"].clip(lower=0)
        d["hr_cat"] = [g if isinstance(g, str) else hr_category(a, h) for g, a, h in zip(d.hr_cat_given, d.age, d.hr)]
        d["rr_cat"] = [g if isinstance(g, str) else rr_category(a, r) for g, a, r in zip(d.rr_cat_given, d.age, d.rr)]
        d["sbp_cat"] = [g if isinstance(g, str) else sbp_category(a, s) for g, a, s in zip(d.sbp_cat_given, d.age, d.sbp)]
        cat = pd.DataFrame({"sex": d["sex"].astype(int).astype(str), "hr_cat": d["hr_cat"],
                            "rr_cat": d["rr_cat"], "sbp_cat": d["sbp_cat"], "MOI": d["MOI"].astype(str)})
        return np.hstack([d[NUMERIC].to_numpy(float), ENC.transform(cat)])


def to_y(df, coding=TAG):
    dis = df["discharged"].to_numpy().astype(bool)
    los = df["los"].to_numpy(float)
    if coding == "cr":     # subdistribution: deaths never leave the risk set
        time_ = np.where(dis, los, T_INF)
    else:                  # cause-specific: deaths censored at time of death
        time_ = los
    return np.array(list(zip(dis, time_)), dtype=[("event", bool), ("time", float)])


def cindex(y, risk):
    return concordance_index_censored(y["event"], y["time"], risk)[0]


def cif_at(model, X, times):
    """Predicted cumulative incidence of live discharge (1 - S) at given times."""
    sf = model.predict_survival_function(X)
    lo_, hi_ = sf[0].domain   # Breslow step function is flat outside the training times
    times = np.clip(times, lo_, hi_)
    return 1.0 - np.vstack([f(times) for f in sf])


def pred_median_los(model, X, tmax):
    """Predicted median time to discharge among those discharged alive: first day on which
    the predicted CIF of discharge reaches half of its plateau (its value at tmax)."""
    grid = np.arange(1, tmax + 1, 0.5)
    c = cif_at(model, X, grid)
    hit = c >= 0.5 * c[:, [-1]]
    return grid[hit.argmax(axis=1)]


def null_cif(y_train, times):
    """Marginal CIF of discharge (empirical, all patients) from the training subset."""
    dis_t = y_train["time"][y_train["event"]]
    return np.array([(dis_t <= t).sum() / len(y_train) for t in times])


def eval_model(model, X_te, y_te, df_te, y_tr_fs, grid, h, tmax):
    risk = model.predict(X_te)
    out = {"cindex": cindex(y_te, risk)}
    # time grid restricted to the follow-up range of this test split
    ok = (grid >= y_te["time"].min()) & (grid < y_te["time"].max())
    g = grid[ok]
    auc_full = np.full(len(grid), np.nan)
    try:
        auc_t, mean_auc = cumulative_dynamic_auc(y_te, y_te, risk, g)
        auc_full[ok] = auc_t
    except Exception:
        mean_auc = np.nan
    out["iauc"] = mean_auc
    out["auc_curve"] = auc_full
    if TAG == "cr":
        surv = 1 - cif_at(model, X_te, g)
        out["ibs"] = integrated_brier_score(y_te, y_te, surv, g) if len(g) > 1 else np.nan
        cif_h = cif_at(model, X_te, np.array([h]))[:, 0]
        obs_h = ((df_te["discharged"] == 1) & (df_te["los"] <= h)).to_numpy()
        out["pred_h"] = cif_h.mean()
        out["obs_h"] = obs_h.mean()
        out["cal_points"] = (cif_h, obs_h)
        dis = df_te["discharged"].to_numpy() == 1
        out["mae"] = np.abs(pred_median_los(model, X_te[dis], tmax) - df_te["los"].to_numpy()[dis]).mean()
    return out


# ------------------------------------------------------------------ source pretraining (once per source)
source = {}
for name, df in registry.items():
    prep = Prep(42).fit(df)
    m = GradientBoostingSurvivalAnalysis(n_estimators=N_PRE, warm_start=True, random_state=42, **GBSA)
    m.fit(prep.transform(df), to_y(df))
    source[name] = m
    print("pretrained", name, flush=True)


# ------------------------------------------------------------------ main loop
def target_grid(df):
    t = df.loc[df.discharged == 1, "los"]
    return np.linspace(t.quantile(0.10), t.quantile(0.80), 20)


rows, curves, calib, sizes, fallbacks = [], [], [], [], 0
t0 = time.time()
for tgt, dft in registry.items():
    grid = target_grid(dft)
    h = HORIZON[tgt]
    for rep in range(N_REP):
        seed = 42 + rep
        pool, test = train_test_split(dft, test_size=0.2, random_state=seed, stratify=dft["died"])
        y_te = to_y(test)
        for ratio in RATIOS:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=ratio, random_state=seed)
            idx, _ = next(sss.split(pool, pool["died"]))
            fs = pool.iloc[idx]
            prep = Prep(seed).fit(fs)               # imputer: few-shot training rows only
            fallbacks += sum(dft[c].notna().any() for c in prep.empty)   # ignore predictors absent from the whole cohort
            X_fs, y_fs = prep.transform(fs), to_y(fs)
            X_te = prep.transform(test)
            tmax = float(fs.loc[fs.discharged == 1, "los"].max()) if fs.discharged.any() else float(grid[-1])
            sizes.append({"target": tgt, "rep": rep, "ratio": ratio, "n_fewshot": len(fs),
                          "fewshot_discharged": int(fs.discharged.sum()), "fewshot_deaths": int(fs.died.sum()),
                          "n_test": len(test), "test_discharged": int(test.discharged.sum()),
                          "test_deaths": int(test.died.sum())})

            mb = GradientBoostingSurvivalAnalysis(n_estimators=N_BASE, random_state=seed, **GBSA).fit(X_fs, y_fs)
            ev_b = eval_model(mb, X_te, y_te, test, y_fs, grid, h, tmax)
            null = {}
            if TAG == "cr":
                g = grid[(grid >= y_te["time"].min()) & (grid < y_te["time"].max())]
                nc = null_cif(y_fs, g)
                null["ibs_null"] = (integrated_brier_score(y_te, y_te, np.tile(1 - nc, (len(test), 1)), g)
                                    if len(g) > 1 else np.nan)
                dis = test["discharged"].to_numpy() == 1
                med = float(fs.loc[fs.discharged == 1, "los"].median()) if fs.discharged.any() else float(grid[-1])
                null["mae_null"] = np.abs(med - test["los"].to_numpy()[dis]).mean()

            for src in registry:
                if src == tgt:
                    continue
                ms = copy.deepcopy(source[src]); ms.set_params(n_estimators=N_PRE + N_FT)
                ms.fit(X_fs, y_fs)
                mw = copy.deepcopy(source[src]); mw.set_params(n_estimators=N_PRE + N_FT)
                sw = compute_sample_weight(class_weight={1: 5, 0: 1}, y=fs["died"].astype(int))
                mw.fit(X_fs, y_fs, sample_weight=sw)
                ev = {"baseline": ev_b,
                      "standard": eval_model(ms, X_te, y_te, test, y_fs, grid, h, tmax),
                      "weighted": eval_model(mw, X_te, y_te, test, y_fs, grid, h, tmax)}
                for meth, e in ev.items():
                    r = {"source": src, "target": tgt, "rep": rep, "ratio": ratio, "method": meth,
                         "cindex": e["cindex"], "iauc": e["iauc"]}
                    if TAG == "cr":
                        r.update({"ibs": e["ibs"], "ibs_null": null["ibs_null"],
                                  "scaled_ibs": 1 - e["ibs"] / null["ibs_null"],
                                  "pred_h": e["pred_h"], "obs_h": e["obs_h"], "horizon": h,
                                  "mae": e["mae"], "mae_null": null["mae_null"]})
                        p, o = e["cal_points"]
                        calib.append(pd.DataFrame({"source": src, "target": tgt, "rep": rep, "ratio": ratio,
                                                   "method": meth, "pred": p, "obs": o.astype(int)}))
                    rows.append(r)
                    curves.append(pd.DataFrame({"source": src, "target": tgt, "rep": rep, "ratio": ratio,
                                                "method": meth, "day": grid, "auc": e["auc_curve"]}))
        print(f"{tgt} rep {rep} done  ({time.time() - t0:.0f}s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, f"results_raw_{TAG}.csv"), index=False)
pd.concat(curves).to_csv(os.path.join(OUT, f"auc_curves_{TAG}.csv"), index=False)
if calib:
    pd.concat(calib).to_csv(os.path.join(OUT, f"calibration_points_{TAG}.csv.gz"), index=False)
sz = pd.DataFrame(sizes)
sz.to_csv(os.path.join(OUT, f"fewshot_sizes_{TAG}.csv"), index=False)
print("imputation fallbacks (predictor entirely missing in a few-shot subset):", fallbacks)


# ------------------------------------------------------------------ summaries
def lo(x): return np.nanpercentile(x, 2.5)
def hi(x): return np.nanpercentile(x, 97.5)


metrics = ["cindex", "iauc"] + (["ibs", "scaled_ibs", "mae", "pred_h", "obs_h"] if TAG == "cr" else [])
summ = res.groupby(["source", "target", "ratio", "method"])[metrics].agg(["mean", lo, hi])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.reset_index()
if TAG == "cr":
    nul = res.groupby(["target", "ratio"])[["ibs_null", "mae_null"]].mean().reset_index()
    summ = summ.merge(nul, on=["target", "ratio"])
summ.to_csv(os.path.join(OUT, f"results_summary_{TAG}.csv"), index=False)

# paired differences vs baseline + Wilcoxon, BH-FDR across all comparisons
wide = res.pivot_table(index=["source", "target", "ratio", "rep"], columns="method", values="cindex").reset_index()
comp = []
for (s, t, r), g in wide.groupby(["source", "target", "ratio"]):
    for meth in ["standard", "weighted"]:
        d = g[meth] - g["baseline"]
        try:
            p = wilcoxon(g[meth], g["baseline"]).pvalue
        except ValueError:
            p = np.nan
        comp.append({"source": s, "target": t, "ratio": r, "method": meth,
                     "delta_mean": d.mean(), "delta_lo": lo(d), "delta_hi": hi(d), "p": p})
    try:
        pw = wilcoxon(g["standard"], g["weighted"]).pvalue
    except ValueError:
        pw = np.nan
    comp.append({"source": s, "target": t, "ratio": r, "method": "standard_vs_weighted",
                 "delta_mean": (g["weighted"] - g["standard"]).mean(), "delta_lo": np.nan,
                 "delta_hi": np.nan, "p": pw})
comp = pd.DataFrame(comp)
mask = comp["method"] != "standard_vs_weighted"
pv = comp.loc[mask, "p"].to_numpy()
order = np.argsort(pv); m = len(pv)
q = np.empty(m); prev = 1.0
for rank, i in list(enumerate(order))[::-1]:
    prev = min(prev, pv[i] * m / (rank + 1)); q[i] = prev
comp.loc[mask, "q_fdr"] = q
comp.to_csv(os.path.join(OUT, f"comparisons_{TAG}.csv"), index=False)

szs = sz.groupby(["target", "ratio"]).agg(
    n_fewshot=("n_fewshot", "first"),
    fewshot_discharged_min=("fewshot_discharged", "min"), fewshot_discharged_max=("fewshot_discharged", "max"),
    fewshot_deaths_min=("fewshot_deaths", "min"), fewshot_deaths_max=("fewshot_deaths", "max"),
    n_test=("n_test", "first"), test_discharged=("test_discharged", "first"), test_deaths=("test_deaths", "first"))
szs.to_csv(os.path.join(OUT, f"fewshot_sizes_summary_{TAG}.csv"))
print(szs.to_string())
print(comp[comp.ratio == 0.05].round(4).to_string())
print(f"total time {time.time() - t0:.0f}s")
