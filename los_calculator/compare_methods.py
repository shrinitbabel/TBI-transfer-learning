"""
Compare modeling approaches for the calculator with the same 5-fold cross-validation:

  Mortality  : gradient-boosted classifier for in-hospital death (AUROC, Brier, calibration-in-the-large)
               vs. the survival model's "not discharged alive" probability.
  Length of stay among patients discharged alive:
    (a) survival model (Fine-Gray coding)  -> predicted median time to discharge
    (b) two-part model: quantile gradient boosting on log(LOS), trained on survivors only
        -> median and 25th/75th percentiles
  Reported per cohort so the decision about Florida can be made on evidence.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0], "--include-florida"]
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "train_pooled_model.py"), encoding="utf-8").read()
exec(src[:src.index("# ------------------------------------------------------------------ 5-fold cross-validation")])

GB = dict(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
strata = df.cohort + "_" + df.died.astype(str)
rows = []
for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, strata)):
    dtr, dte = df.iloc[tr], df.iloc[te]
    prep = Prep().fit(dtr)
    Xtr, Xte = prep.transform(dtr), prep.transform(dte)
    surv = GradientBoostingSurvivalAnalysis(random_state=42, **{k: v for k, v in GBSA.items()}).fit(Xtr, to_y(dtr))
    clf = GradientBoostingClassifier(**GB).fit(Xtr, dtr.died)
    alive = dtr.discharged == 1
    qr = {a: GradientBoostingRegressor(loss="quantile", alpha=a, **GB).fit(Xtr[alive.to_numpy()], np.log(dtr.los[alive]))
          for a in (0.25, 0.5, 0.75)}
    tmax = float(dtr.loc[alive, "los"].quantile(0.99))
    p_die_clf = clf.predict_proba(Xte)[:, 1]
    p_die_surv = 1 - cif(surv, Xte, np.array([tmax]))[:, 0]
    isd = dte.discharged.to_numpy() == 1
    med_surv = median_los(surv, Xte[isd], tmax)
    med_qr = np.exp(qr[0.5].predict(Xte[isd]))
    q25, q75 = np.exp(qr[0.25].predict(Xte[isd])), np.exp(qr[0.75].predict(Xte[isd]))
    los = dte.los.to_numpy()
    for name in ["Pooled"] + NAMES:
        m = np.ones(len(dte), bool) if name == "Pooled" else (dte.cohort == name).to_numpy()
        md = m[isd]
        y = dte.died.to_numpy()[m]
        rows.append({"fold": fold, "cohort": name, "deaths": int(y.sum()),
                     "auc_mort_classifier": roc_auc_score(y, p_die_clf[m]) if 0 < y.sum() < len(y) else np.nan,
                     "auc_mort_survival": roc_auc_score(y, p_die_surv[m]) if 0 < y.sum() < len(y) else np.nan,
                     "brier_mort_classifier": brier_score_loss(y, p_die_clf[m]),
                     "pred_mort": p_die_clf[m].mean(), "obs_mort": y.mean(),
                     "mae_los_survival": np.abs(med_surv[md] - los[isd][md]).mean(),
                     "mae_los_quantile": np.abs(med_qr[md] - los[isd][md]).mean(),
                     "iqr_coverage": ((los[isd][md] >= q25[md]) & (los[isd][md] <= q75[md])).mean(),
                     "mae_median_reference": np.abs(dtr.los[alive].median() - los[isd][md]).mean()})
res = pd.DataFrame(rows).groupby("cohort").mean(numeric_only=True).drop(columns="fold").round(3)
pd.set_option("display.width", 200)
print(res.to_string())
res.to_csv(os.path.join(HERE, "method_comparison.csv"))
