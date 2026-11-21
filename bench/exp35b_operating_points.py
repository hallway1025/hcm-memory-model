#!/usr/bin/env python3
# exp35b_operating_points.py -- G-4 (calibrated thresholds) + K7-A level-A protocol on exp35 probe rows.
# Pre-registered: G-4 = calibrated risk-coverage not worse than fixed raw1 (+0.005 tol);
# K7-A = ECE<0.05 after calibration AND best dev-chosen operating point cost <= current fixed rule.
# Cost = c_err*risk*cov + c_abst*(1-cov), c_err=1.0, c_abst=0.25. Dev/eval = odd/even within tag.
import os, sys, json, math
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = json.load(open(os.environ.get("EXP35_ROWS_IN") or os.path.join(ROOT, "results", "exp35_probe_rows.json"), encoding="utf-8"))["rows"]
C_ERR, C_ABST = 1.0, 0.25
COVS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def ece(p, y, nb=10):
    p = np.asarray(p, float); y = np.asarray(y, float)
    e = 0.0
    for b in range(nb):
        lo, hi = b / nb, (b + 1) / nb
        m = (p >= lo) & (p < hi) if b < nb - 1 else (p >= lo) & (p <= hi)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def irls(X, y, ridge=1e-3, iters=25):
    X = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        g = X.T @ (y - p) - ridge * w
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        w = w + np.linalg.solve(H, g)
    return w

def curve(s_eval, y_eval, s_dev, covs):
    out = []
    for c in covs:
        th = float(np.quantile(s_dev, 1 - c))
        m = s_eval >= th
        cov = float(m.mean())
        risk = float(1 - y_eval[m].mean()) if m.sum() else float("nan")
        out.append(dict(target_cov=c, th=round(th, 4), cov=round(cov, 4), risk=None if math.isnan(risk) else round(risk, 4),
                        cost=round(cov * risk * C_ERR + (1 - cov) * C_ABST, 4) if not math.isnan(risk) else None))
    return out

dev, ev = [], []
for tag in sorted(set(r["tag"] for r in rows)):
    rs = [r for r in rows if r["tag"] == tag]
    for i, r in enumerate(rs):
        (dev if i % 2 == 0 else ev).append(r)
FE = ["raw1", "pert_mean", "pert_std", "cons"]
Xd = np.array([[r[f] for f in FE] for r in dev]); yd = np.array([r["label"] for r in dev], float)
Xe = np.array([[r[f] for f in FE] for r in ev]);  ye = np.array([r["label"] for r in ev], float)
w = irls(Xd, yd)
def sig(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
A = lambda X: np.hstack([np.ones((len(X), 1)), X])
p_dev = sig(A(Xd) @ w); p_ev = sig(A(Xe) @ w)
raw_dev = Xd[:, 0]; raw_ev = Xe[:, 0]
res = dict(n=len(rows), n_dev=len(dev), n_eval=len(ev), feats=FE,
           ece_raw_eval=ece(raw_ev, ye), ece_cal_eval=ece(p_ev, ye),
           ece_raw_dev=ece(raw_dev, yd), ece_cal_dev=ece(p_dev, yd),
           curve_fixed=curve(raw_ev, ye, raw_dev, COVS),
           curve_cal=curve(p_ev, ye, p_dev, COVS))
# current fixed rule raw1>=0.50
m = raw_ev >= 0.50
cov_c, risk_c = float(m.mean()), float(1 - ye[m].mean())
cost_cur = cov_c * risk_c * C_ERR + (1 - cov_c) * C_ABST
res["current"] = dict(rule="raw1>=0.50", cov=round(cov_c, 4), risk=round(risk_c, 4), cost=round(cost_cur, 4))
# best dev threshold on calibrated score
grid = np.quantile(p_dev, np.linspace(0.02, 0.98, 49))
best = None
for th in grid:
    mm = p_dev >= th
    if not mm.sum(): continue
    rr = 1 - yd[mm].mean(); cc = mm.mean()
    c = cc * rr * C_ERR + (1 - cc) * C_ABST
    if best is None or c < best["cost_dev"]:
        best = dict(th=round(float(th), 4), cost_dev=round(float(c), 4))
mm = p_ev >= best["th"]
cov_b, risk_b = float(mm.mean()), float(1 - ye[mm].mean())
cost_b = cov_b * risk_b * C_ERR + (1 - cov_b) * C_ABST
res["best_cal"] = dict(**best, cov=round(cov_b, 4), risk=round(risk_b, 4), cost=round(cost_b, 4))
res["g4"] = dict()
ok4, details = True, []
for cf, cc in zip(res["curve_fixed"], res["curve_cal"]):
    if cf["risk"] is None or cc["risk"] is None: continue
    d = cc["risk"] - cf["risk"]
    details.append(dict(cov=cf["target_cov"], fixed=cf["risk"], cal=cc["risk"], delta=round(d, 4)))
    if cc["target_cov"] <= 0.9 and d > 0.005: ok4 = False
res["g4"] = dict(passed=bool(ok4), details=details, tol=0.005)
res["k7a"] = dict(ece_below_005=bool(res["ece_cal_eval"] < 0.05),
                  cost_not_worse=bool(cost_b <= cost_cur + 1e-9),
                  cost_current=round(cost_cur, 4), cost_best=round(cost_b, 4))
res["verdict"] = dict(G4="PASS" if ok4 else "FAIL",
                      K7A="PASS" if (res["k7a"]["ece_below_005"] and res["k7a"]["cost_not_worse"]) else "FAIL")
json.dump(res, open(os.path.join(ROOT, "results", os.environ.get("EXP35B_OUT") or "exp35b_operating_points.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: res[k] for k in ["ece_raw_eval", "ece_cal_eval", "current", "best_cal", "g4", "k7a", "verdict"]}, indent=1))
