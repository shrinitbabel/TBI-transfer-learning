"""
Final calculator model, chosen by benchmark_methods.py (best LOS error and calibrated 80% range;
mortality AUROC equal to the alternatives):

  * in-hospital death:  gradient-boosted classifier
  * length of stay among patients discharged alive: gradient-boosted quantile trees on log(LOS)
    (median, 10th and 90th percentiles) with split-conformal calibration of the 80% range (CQR)

Pooled cohorts: India, Jordan, California (Florida excluded: mortality AUROC ~0.5 in benchmark).
Outputs: model.json (trees for the browser) and validation.json (5-fold CV, pooled).
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
src = open(os.path.join(HERE, "train_pooled_model.py"), encoding="utf-8").read()
exec(src[:src.index("# ------------------------------------------------------------------ 5-fold cross-validation")])
GB = dict(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
LO, HI = 0.10, 0.90


def fit_all(d, seed=0):
    prep = Prep().fit(d)
    X = prep.transform(d)
    alive = (d.discharged == 1).to_numpy()
    ylog = np.log(d.los.to_numpy()[alive])
    clf = GradientBoostingClassifier(**GB).fit(X, d.died)
    q50 = GradientBoostingRegressor(loss="quantile", alpha=0.5, **GB).fit(X[alive], ylog)
    ia, ic = train_test_split(np.arange(alive.sum()), test_size=0.25, random_state=seed)
    Xa = X[alive]
    ql = GradientBoostingRegressor(loss="quantile", alpha=LO, **GB).fit(Xa[ia], ylog[ia])
    qh = GradientBoostingRegressor(loss="quantile", alpha=HI, **GB).fit(Xa[ia], ylog[ia])
    score = np.maximum(ql.predict(Xa[ic]) - ylog[ic], ylog[ic] - qh.predict(Xa[ic]))
    adj = float(np.quantile(score, min(1.0, (HI - LO) * (1 + 1 / len(ic)))))
    return prep, clf, q50, ql, qh, adj


def predict(m, d):
    prep, clf, q50, ql, qh, adj = m
    X = prep.transform(d)
    return (clf.predict_proba(X)[:, 1], np.exp(q50.predict(X)),
            np.exp(ql.predict(X) - adj), np.exp(qh.predict(X) + adj))


# ------------------------------------------------------------------ 5-fold CV (pooled)
strata = df.cohort + "_" + df.died.astype(str)
P = {k: np.zeros(len(df)) for k in ("p", "med", "lo", "hi", "ref")}
for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, strata)):
    dtr, dte = df.iloc[tr], df.iloc[te]
    p, med, lo, hi = predict(fit_all(dtr, fold), dte)
    P["p"][te], P["med"][te], P["lo"][te], P["hi"][te] = p, med, lo, hi
    P["ref"][te] = dtr.los[dtr.discharged == 1].median()
isd = (df.discharged == 1).to_numpy()
los = df.los.to_numpy()
# calibration of death risk by fifths of predicted risk
q = pd.qcut(P["p"], 5, labels=False)
calib = [{"pred": float(P["p"][q == k].mean()), "obs": float(df.died.to_numpy()[q == k].mean())} for k in range(5)]
validation = {
    "n_patients": int(len(df)), "cohorts": NAMES, "folds": 5,
    "mortality_auroc": float(roc_auc_score(df.died, P["p"])),
    "mortality_brier": float(brier_score_loss(df.died, P["p"])),
    "mortality_pred": float(P["p"].mean()), "mortality_obs": float(df.died.mean()),
    "mortality_calibration_fifths": calib,
    "los_mae": float(np.abs(P["med"][isd] - los[isd]).mean()),
    "los_mae_reference": float(np.abs(P["ref"][isd] - los[isd]).mean()),
    "range80_coverage": float(((los[isd] >= P["lo"][isd]) & (los[isd] <= P["hi"][isd])).mean()),
    "range80_width_median": float(np.median(P["hi"][isd] - P["lo"][isd])),
}
bench = pd.read_csv(os.path.join(HERE, "benchmark_results.csv"))
validation["benchmark_pooled"] = bench[bench.cohort == "Pooled"].drop(columns="cohort").round(3).to_dict(orient="records")
print(json.dumps({k: v for k, v in validation.items() if k != "benchmark_pooled"}, indent=1))

# ------------------------------------------------------------------ final model on all data + export
prep, clf, q50, ql, qh, adj = fit_all(df, 0)
X = prep.transform(df)


def export(model):
    trees = []
    for est in model.estimators_[:, 0]:
        t = est.tree_
        trees.append({"f": t.feature.tolist(), "t": [round(float(v), 6) for v in t.threshold],
                      "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                      "v": [round(float(v), 8) for v in t.value[:, 0, 0]]})
    return {"init": float(model._raw_predict_init(X[:1]).ravel()[0]), "lr": model.learning_rate, "trees": trees}


def raw(m, x):
    s = m["init"]
    for tr in m["trees"]:
        n = 0
        while tr["l"][n] != -1:
            n = tr["l"][n] if x[tr["f"][n]] <= tr["t"][n] else tr["r"][n]
        s += m["lr"] * tr["v"][n]
    return s


E = {"death": export(clf), "q50": export(q50), "q_lo": export(ql), "q_hi": export(qh)}
xs = X[:300]
checks = {"death": np.abs(1 / (1 + np.exp(-np.array([raw(E["death"], x) for x in xs]))) - clf.predict_proba(xs)[:, 1]).max(),
          "q50": np.abs(np.array([raw(E["q50"], x) for x in xs]) - q50.predict(xs)).max(),
          "q_lo": np.abs(np.array([raw(E["q_lo"], x) for x in xs]) - ql.predict(xs)).max(),
          "q_hi": np.abs(np.array([raw(E["q_hi"], x) for x in xs]) - qh.predict(xs)).max()}
print("browser replica max error:", checks)
assert max(checks.values()) < 1e-4

alive = df.discharged == 1
model = {"features": FEATURES, "moi_categories": MOI_CATS, "models": E, "conformal_adjust": adj,
         "range": [LO, HI], "trained_on": {n: int((df.cohort == n).sum()) for n in NAMES},
         "reference": {"death_rate": float(df.died.mean()), "los_median": float(df.los[alive].median()),
                       "los_p10": float(df.los[alive].quantile(LO)), "los_p90": float(df.los[alive].quantile(HI))}}
json.dump(model, open(os.path.join(HERE, "model.json"), "w"), separators=(",", ":"))
json.dump(validation, open(os.path.join(HERE, "validation.json"), "w"), indent=1)
print("model.json:", os.path.getsize(os.path.join(HERE, "model.json")) // 1024, "KB")
