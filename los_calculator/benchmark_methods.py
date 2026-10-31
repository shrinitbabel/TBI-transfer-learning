"""
Benchmark five approaches for predicting in-hospital mortality and length of stay (LOS)
on the pooled India + Jordan + California cohorts (Florida excluded: mortality AUROC ~0.5),
using identical 5-fold cross-validation.

  1. survival   Gradient Boosting Survival Analysis, discharge with death as competing event
  2. twopart    gradient-boosted classifier (death) + gradient-boosted quantile trees on log LOS (survivors)
  3. conformal  twopart with conformalized quantile regression (CQR) for the LOS range
  4. ngboost    NGBoost: Bernoulli (death) + LogNormal distribution of LOS (survivors)
  5. deephit    DeepHit neural network (PyTorch), joint distribution over (cause, time)

Metrics (test folds): mortality AUROC, Brier, mean predicted vs observed death rate;
LOS among survivors: MAE of the predicted median, coverage and median width of the 80% range.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Bernoulli, LogNormal
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]                         # default: public cohorts only (no Florida)
src = open(os.path.join(HERE, "train_pooled_model.py"), encoding="utf-8").read()
exec(src[:src.index("# ------------------------------------------------------------------ 5-fold cross-validation")])
torch.manual_seed(0); np.random.seed(0)
GB = dict(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
LO, HI = 0.10, 0.90                               # 80% range


# ------------------------------------------------------------------ DeepHit (competing risks, no censoring)
class DeepHit:
    def __init__(self, n_bins=40, alpha=0.2, sigma=0.1, epochs=60, seed=0):
        self.K, self.alpha, self.sigma, self.epochs, self.seed = n_bins, alpha, sigma, epochs, seed

    def _bins(self, t):
        return np.clip(np.searchsorted(self.cuts, t, side="right"), 0, self.K - 1)

    def fit(self, X, los, died):
        torch.manual_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.cuts = np.unique(np.quantile(los, np.linspace(0, 1, self.K + 1)[1:-1]))
        self.K = len(self.cuts) + 1
        b = self._bins(los)
        self.rep = np.array([np.median(los[b == k]) if (b == k).any() else np.nan for k in range(self.K)])
        self.rep = pd.Series(self.rep).interpolate(limit_direction="both").to_numpy()
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        cause = torch.tensor(died.astype(int))           # 0 = discharge, 1 = death
        tb = torch.tensor(b)
        target = cause * self.K + tb
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                       torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                       torch.nn.Linear(64, 2 * self.K))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        n = len(Xt)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                logits = self.net(Xt[idx])
                nll = torch.nn.functional.cross_entropy(logits, target[idx])
                p = torch.softmax(logits, 1).view(-1, 2, self.K)
                cif = torch.cumsum(p, 2)                                # (B, 2, K)
                c, t = cause[idx], tb[idx]
                F_own = cif[torch.arange(len(idx)), c, t]               # F_c(T_i | x_i)
                F_other = cif[:, c, t].T                                # F_c(T_i | x_j): (B_i, B_j)
                comparable = (t.view(-1, 1) < t.view(1, -1)).float()    # T_i < T_j, event of cause c for i
                rank = (comparable * torch.exp(-(F_own.view(-1, 1) - F_other) / self.sigma)).sum() / comparable.sum().clamp(min=1)
                loss = nll + self.alpha * rank
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            p = torch.softmax(self.net(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)), 1).numpy().reshape(-1, 2, self.K)
        p_die = p[:, 1].sum(1)
        cond = p[:, 0] / p[:, 0].sum(1, keepdims=True)                  # LOS distribution given discharged alive
        cdf = np.cumsum(cond, 1)
        q = lambda a: self.rep[(cdf >= a).argmax(1)]
        return p_die, q(0.5), q(LO), q(HI)


def surv_quantiles(model, X, tmax):
    grid = np.arange(1, tmax + 1, 0.5)
    c = cif(model, X, grid)
    rel = c / c[:, [-1]]
    q = lambda a: grid[(rel >= a).argmax(1)]
    return 1 - c[:, -1], q(0.5), q(LO), q(HI)


# ------------------------------------------------------------------ cross-validation
strata = df.cohort + "_" + df.died.astype(str)
rows = []
for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, strata)):
    dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    prep = Prep().fit(dtr)
    Xtr, Xte = prep.transform(dtr), prep.transform(dte)
    alive = (dtr.discharged == 1).to_numpy()
    ylog = np.log(dtr.los.to_numpy()[alive])
    tmax = float(dtr.los[alive].quantile(0.99))
    preds = {}

    surv = GradientBoostingSurvivalAnalysis(random_state=42, **GBSA).fit(Xtr, to_y(dtr))
    preds["survival"] = surv_quantiles(surv, Xte, tmax)

    clf = GradientBoostingClassifier(**GB).fit(Xtr, dtr.died)
    qm = {a: GradientBoostingRegressor(loss="quantile", alpha=a, **GB).fit(Xtr[alive], ylog) for a in (LO, 0.5, HI)}
    pd_ = clf.predict_proba(Xte)[:, 1]
    preds["twopart"] = (pd_, *[np.exp(qm[a].predict(Xte)) for a in (0.5, LO, HI)])

    # CQR: refit the quantile models on 75% of survivors, conformalize on the remaining 25%
    ia, ic = train_test_split(np.arange(alive.sum()), test_size=0.25, random_state=fold)
    Xa = Xtr[alive]
    ql = GradientBoostingRegressor(loss="quantile", alpha=LO, **GB).fit(Xa[ia], ylog[ia])
    qh = GradientBoostingRegressor(loss="quantile", alpha=HI, **GB).fit(Xa[ia], ylog[ia])
    score = np.maximum(ql.predict(Xa[ic]) - ylog[ic], ylog[ic] - qh.predict(Xa[ic]))
    adj = np.quantile(score, min(1.0, (HI - LO) * (1 + 1 / len(ic))))   # finite-sample conformal correction
    preds["conformal"] = (pd_, np.exp(qm[0.5].predict(Xte)), np.exp(ql.predict(Xte) - adj), np.exp(qh.predict(Xte) + adj))

    nc = NGBClassifier(Dist=Bernoulli, n_estimators=300, verbose=False, random_state=42).fit(Xtr, dtr.died.to_numpy())
    nr = NGBRegressor(Dist=LogNormal, n_estimators=300, verbose=False, random_state=42).fit(Xtr[alive], dtr.los.to_numpy()[alive])
    dist = nr.pred_dist(Xte)
    preds["ngboost"] = (nc.predict_proba(Xte)[:, 1], dist.dist.median(), dist.dist.ppf(LO), dist.dist.ppf(HI))

    dh = DeepHit(seed=fold).fit(Xtr, dtr.los.to_numpy(), dtr.died.to_numpy())
    preds["deephit"] = dh.predict(Xte)

    isd = dte.discharged.to_numpy() == 1
    los, died = dte.los.to_numpy(), dte.died.to_numpy()
    for method, (p_die, med, lo, hi) in preds.items():
        for name in ["Pooled"] + NAMES:
            m = np.ones(len(dte), bool) if name == "Pooled" else (dte.cohort == name).to_numpy()
            s = m & isd
            rows.append({"fold": fold, "method": method, "cohort": name,
                         "mort_auroc": roc_auc_score(died[m], p_die[m]) if 0 < died[m].sum() < m.sum() else np.nan,
                         "mort_brier": brier_score_loss(died[m], np.clip(p_die[m], 0, 1)),
                         "mort_pred": p_die[m].mean(), "mort_obs": died[m].mean(),
                         "los_mae": np.abs(med[s] - los[s]).mean(),
                         "range80_coverage": ((los[s] >= lo[s]) & (los[s] <= hi[s])).mean(),
                         "range80_width": np.median(hi[s] - lo[s])})
    print("fold", fold, "done", flush=True)

res = pd.DataFrame(rows).groupby(["cohort", "method"]).mean(numeric_only=True).drop(columns="fold").round(3)
pd.set_option("display.width", 220)
print(res.to_string())
res.to_csv(os.path.join(HERE, "benchmark_results.csv"))
