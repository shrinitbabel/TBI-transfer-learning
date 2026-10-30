"""
Pooled TBI length-of-stay model for the web calculator (side project; not part of the paper).

Trains one Gradient Boosting Survival Analysis model on the pooled open-access cohorts
(India, Jordan, California; optionally Florida with --include-florida, ONLY if the IRB/data-use
terms allow a publicly released model derived from it). Outcome = time to live discharge with
in-hospital death as a competing event (Fine-Gray subdistribution coding, as in revision2/).

Outputs (los_calculator/):
  model.json        trees + baseline cumulative hazard, evaluated by index.html in the browser
  validation.json   5-fold cross-validated performance, overall and per cohort
"""
import os, json, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--include-florida", action="store_true")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
COH = os.path.join(HERE, "..", "revision2", "outputs", "cohorts")
NAMES = ["India", "Jordan", "California"] + (["Florida"] if args.include_florida else [])
T_INF = 10_000.0
GBSA = dict(n_estimators=150, learning_rate=0.1, max_depth=3, subsample=1.0, loss="coxph")
NUM_IMPUTE = ["age", "sex", "gcs", "sbp", "hr", "rr"]
MOI_CATS = ["Animal / Environmental", "Assault / Blunt Force", "Bicycle Crash", "Burn / Fire / Blast",
            "Fall", "Gunshot Wound (GSW)", "Motor Vehicle Collision", "Motorcycle Crash",
            "Other / Unknown", "Pedestrian Struck"]
FEATURES = (["age", "gcs", "male", "icu", "tachycardia", "tachypnea", "hypotension"] + [f"moi={m}" for m in MOI_CATS])

df = pd.concat([pd.read_csv(os.path.join(COH, f"{n.lower()}_r2.csv")).assign(cohort=n) for n in NAMES],
               ignore_index=True)
print({n: int((df.cohort == n).sum()) for n in NAMES})


def hr_tachy(a, v):
    return v > (160 if a < 1 else 150 if a < 3 else 140 if a < 6 else 120 if a < 12 else 100)


def rr_tachy(a, v):
    return v > (60 if a < 1 else 40 if a < 3 else 34 if a < 6 else 30 if a < 12 else 20)


def sbp_low(a, v):
    return v < (70 if a < 1 else (70 + 2 * a) if a < 10 else 90)


class Prep:
    """Predictor-only imputation fitted on training rows, then the same features the web page builds."""
    def fit(self, d):
        self.imp = IterativeImputer(random_state=42, max_iter=10).fit(d[NUM_IMPUTE].to_numpy(float))
        return self

    def transform(self, d):
        d = d.copy()
        d[NUM_IMPUTE] = self.imp.transform(d[NUM_IMPUTE].to_numpy(float))
        X = pd.DataFrame({"age": d.age.clip(lower=0), "gcs": d.gcs.round().clip(3, 15),
                          "male": d.sex.round().clip(0, 1),
                          "icu": d.icu.fillna(0).astype(float)})   # 9 patients with unknown ICU status -> not ICU
        given = lambda c, flag: d[c].astype(str).str.lower().eq(flag)
        X["tachycardia"] = np.where(d.hr_cat_given.notna(), given("hr_cat_given", "tachycardia"),
                                    [hr_tachy(a, v) for a, v in zip(d.age, d.hr)]).astype(float)
        X["tachypnea"] = np.where(d.rr_cat_given.notna(), given("rr_cat_given", "tachypnea"),
                                  [rr_tachy(a, v) for a, v in zip(d.age, d.rr)]).astype(float)
        X["hypotension"] = np.where(d.sbp_cat_given.notna(), given("sbp_cat_given", "low"),
                                    [sbp_low(a, v) for a, v in zip(d.age, d.sbp)]).astype(float)
        for m in MOI_CATS:
            X[f"moi={m}"] = (d.MOI == m).astype(float)
        return X[FEATURES].to_numpy(float)


def to_y(d):
    dis = d.discharged.to_numpy().astype(bool)
    return np.array(list(zip(dis, np.where(dis, d.los.to_numpy(float), T_INF))),
                    dtype=[("event", bool), ("time", float)])


def cif(model, X, times):
    sf = model.predict_survival_function(X)
    lo, hi = sf[0].domain
    t = np.clip(times, lo, hi)
    return 1 - np.vstack([f(t) for f in sf])


def median_los(model, X, tmax):
    grid = np.arange(1, tmax + 1, 0.5)
    c = cif(model, X, grid)
    return grid[(c >= 0.5 * c[:, [-1]]).argmax(axis=1)]


# ------------------------------------------------------------------ 5-fold cross-validation
strata = df.cohort + "_" + df.died.astype(str)
rows = []
for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, strata)):
    dtr, dte = df.iloc[tr], df.iloc[te]
    prep = Prep().fit(dtr)
    m = GradientBoostingSurvivalAnalysis(random_state=42, **GBSA).fit(prep.transform(dtr), to_y(dtr))
    tmax = float(dtr.loc[dtr.discharged == 1, "los"].quantile(0.99))
    for name in ["Pooled"] + NAMES:
        sub = dte if name == "Pooled" else dte[dte.cohort == name]
        X, y = prep.transform(sub), to_y(sub)
        disc = sub.loc[sub.discharged == 1, "los"]
        grid = np.linspace(disc.quantile(.1), disc.quantile(.8), 15)
        grid = grid[(grid >= y["time"].min()) & (grid < y["time"].max())]
        c7 = cif(m, X, np.array([7.0]))[:, 0]
        obs7 = ((sub.discharged == 1) & (sub.los <= 7)).to_numpy()
        isd = sub.discharged.to_numpy() == 1
        rows.append({"fold": fold, "cohort": name, "n": len(sub),
                     "cindex": concordance_index_censored(y["event"], y["time"], m.predict(X))[0],
                     "ibs": integrated_brier_score(y, y, 1 - cif(m, X, grid), grid),
                     "pred_day7": c7.mean(), "obs_day7": obs7.mean(),
                     "mae_days": np.abs(median_los(m, X[isd], tmax) - sub.los.to_numpy()[isd]).mean(),
                     "mae_median_reference": np.abs(dtr.loc[dtr.discharged == 1, "los"].median() - sub.los.to_numpy()[isd]).mean()})
cv = pd.DataFrame(rows).groupby("cohort").mean(numeric_only=True).drop(columns="fold").round(3)
print(cv.to_string())

# ------------------------------------------------------------------ final model on all pooled data
prep = Prep().fit(df)
X_all, y_all = prep.transform(df), to_y(df)
model = GradientBoostingSurvivalAnalysis(random_state=42, **GBSA).fit(X_all, y_all)

# export trees in a form the browser can evaluate: risk = init + lr * sum(scale_m * tree_m(x))
trees = []
scales = getattr(model, "_scale", np.ones(len(model.estimators_)))   # per-tree scale is 1 without dropout
for est, sc in zip(model.estimators_[:, 0], scales):
    t = est.tree_
    trees.append({"f": t.feature.tolist(), "t": [round(float(v), 6) for v in t.threshold],
                  "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                  "v": [round(float(v), 8) for v in t.value[:, 0, 0]], "s": float(sc)})
init = float(model._raw_predict_init(X_all[:1]).ravel()[0])
days = np.round(np.arange(0, 365.5, 0.5), 1)
H0 = model._baseline_model.cum_baseline_hazard_
lo, hi = H0.domain
H = H0(np.clip(days, lo, hi))


def py_risk(x):  # pure-python replica of the browser computation, used to verify the export
    s = init
    for tr in trees:
        n = 0
        while tr["l"][n] != -1:
            n = tr["l"][n] if x[tr["f"][n]] <= tr["t"][n] else tr["r"][n]
        s += model.learning_rate * tr["s"] * tr["v"][n]
    return s


chk = np.abs(np.array([py_risk(x) for x in X_all[:500]]) - model.predict(X_all[:500])).max()
print("max |browser-equivalent risk - sklearn risk| =", chk)
assert chk < 1e-4

json.dump({"features": FEATURES, "moi_categories": MOI_CATS, "learning_rate": model.learning_rate,
           "init": init, "trees": trees, "days": days.tolist(), "cum_baseline_hazard": [round(float(v), 8) for v in H],
           "trained_on": {n: int((df.cohort == n).sum()) for n in NAMES}},
          open(os.path.join(HERE, "model.json"), "w"), separators=(",", ":"))
json.dump({"cv_5fold": cv.reset_index().to_dict(orient="records"), "cohorts": NAMES},
          open(os.path.join(HERE, "validation.json"), "w"), indent=1)
print("model.json:", os.path.getsize(os.path.join(HERE, "model.json")) // 1024, "KB")
