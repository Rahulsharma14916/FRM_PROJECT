#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA634: Financial Risk Management -- Assignment (Nov 2026)
Rolling 6-month formation / 3-month holding backtest, 2009-2022.

Portfolios : GMV, MV (tangency / max-Sharpe), EW, Active (CAPM alpha significance), NIFTY50
Constraints: long-only, fully invested (w_i >= 0, sum w_i = 1) for the four stock portfolios.

Group members : Rahul (2023EEB1238), Lakshya Vajpayee (2023CEB1037), Sargun Singh (2023CEB1046)
Submitted to  : Prof. Puneet Pasricha, Department of Mathematics, IIT Ropar

Usage
-----
    python Assignment_2023EEB1238_Codes.py [data_dir]

Reads  Stocks_data.csv and market_Factor_risk_Free.csv (from data_dir, the script folder, or cwd).
Writes (into ./outputs):
    returns_matrix_nx5.csv     n x 5 matrix of 3-month holding returns  [GMV, MV, EW, Active, NIFTY50]
    window_details.csv         window dates, #stocks, #alpha-significant stocks, fallback flag, ...
    weights_<portfolio>.csv    weights chosen at the end of each formation window
    performance_table.csv      performance statistics
    cumulative_returns.png     cumulative return plot (compounded 3M holding returns)
    diagnostics.png            effective number of holdings / # significant-alpha stocks
    results.json               everything the report builder needs

No look-ahead: every weight is estimated ONLY from formation-window data; holding-period
data is used only to realise the returns.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------------------------
FORMATION_MONTHS = 6          # length of formation window (months)
HOLDING_MONTHS = 3            # length of holding window (months); also the roll step
FIRST_FORMATION_START = pd.Timestamp("2009-01-01")
LAST_HOLDING_END = pd.Timestamp("2022-12-31")
ALPHA_LEVEL = 0.05            # two-sided test level for H0: alpha_i = 0
TRADING_DAYS = 252            # trading days per year
WINDOWS_PER_YEAR = 4          # 4 x 63 = 252 trading days -> one 3M holding window = 1/4 year
# The factor file states RF is a "decimal", but its values (mean 0.0245) are only sensible as
# PERCENT per day (0.0245 %/day ~ 6.2 % p.a.; as a decimal it would be 2.45 % per DAY).
# MF is explicitly in percent. Both are converted to decimals below.
RF_IN_PERCENT = True
PORTFOLIOS = ["GMV", "MV", "EW", "Active", "NIFTY50"]
OUT_DIR = "outputs"


def find_file(name, extra_dir=None):
    """Locate a data file in (user supplied dir, script dir, cwd, upload dir)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in [extra_dir, here, os.getcwd(), "/mnt/user-data/uploads"]:
        if d and os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    raise FileNotFoundError(name)


# ----------------------------------------------------------------------------------------
# 1. Data preparation  (Problem 1: common dates, simple returns, no look-ahead)
# ----------------------------------------------------------------------------------------
def load_data(data_dir=None):
    # --- prices: '#N/A N/A' strings are Bloomberg missing markers
    px = pd.read_csv(find_file("Stocks_data.csv", data_dir),
                     na_values=["#N/A N/A", "#N/A", "NA", ""])
    px["Dates"] = pd.to_datetime(px["Dates"], format="%d-%m-%Y")
    px = px.dropna(subset=["Dates"]).set_index("Dates").sort_index()
    px = px[~px.index.duplicated(keep="first")]

    bench_col = px.columns[-1]                       # last column = Nifty 50 close
    assert "NIFTY" in bench_col.upper(), "last column must be the Nifty 50 index"
    stocks = px.drop(columns=bench_col)

    # --- remove stocks with missing / partial observations in the analysis period
    n_missing = stocks.isna().sum()
    dropped = n_missing[n_missing > 0].index.tolist()
    stocks = stocks.drop(columns=dropped)
    bench = px[bench_col]

    # --- factor file (drop empty trailing rows)
    fac = pd.read_csv(find_file("market_Factor_risk_Free.csv", data_dir)).dropna()
    fac["Date"] = pd.to_datetime(fac["Date"], format="%d-%m-%Y")
    fac = fac.set_index("Date").sort_index()
    fac = fac[~fac.index.duplicated(keep="first")]
    fac["MF"] = fac["MF"] / 100.0                    # percent -> decimal
    if RF_IN_PERCENT:
        fac["RF"] = fac["RF"] / 100.0                # percent -> decimal

    # --- daily SIMPLE returns from CONSECUTIVE available prices (computed BEFORE merging
    #     with the factor file, so a return never spans a date dropped by the intersection)
    ret_all = pd.concat([stocks, bench], axis=1).pct_change(fill_method=None).iloc[1:]
    ret_all = ret_all.dropna(how="any")

    # --- keep only the intersection of dates (stocks, index, factors)
    common = ret_all.index.intersection(fac.index)
    ret_all = ret_all.loc[common]
    fac = fac.loc[common]
    R = ret_all[stocks.columns]
    B = ret_all[bench_col].rename("NIFTY50")
    return R, B, fac, dropped, len(px)


# ----------------------------------------------------------------------------------------
# 2. Portfolio construction (Problem 2)
# ----------------------------------------------------------------------------------------
SCALE = 1e4   # variance is ~1e-4 for daily data; scale so SLSQP tolerances are meaningful


def solve_gmv(Sigma):
    """(a) GMV: min w'Sigma w  s.t. 1'w = 1, w >= 0.   Solver: SciPy SLSQP (analytic gradient)."""
    N = Sigma.shape[0]
    S = Sigma * SCALE
    f = lambda w: w @ S @ w
    g = lambda w: 2.0 * S @ w
    res = optimize.minimize(f, np.full(N, 1.0 / N), jac=g, method="SLSQP",
                            bounds=[(0, None)] * N,
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                          "jac": lambda w: np.ones(N)}],
                            options={"maxiter": 1000, "ftol": 1e-14})
    w = clean_weights(res.x)
    return w, gmv_kkt_ok(w, Sigma)


def gmv_kkt_ok(w, Sigma, tol=1e-4):
    """Validate a GMV solution through the KKT conditions (independent of the optimizer's flag):
    gradient 2*Sigma*w must be equal (=lambda) on active assets and >= lambda on zero-weight assets."""
    g = 2.0 * (Sigma * SCALE) @ w
    pos = w > 0
    lam = g[pos].mean()
    scale = max(abs(lam), 1e-12)
    return bool(np.ptp(g[pos]) <= tol * scale and (g[~pos] >= lam - tol * scale).all())


def solve_tangency(mu, Sigma, rf_bar):
    """(b) Max daily Sharpe ratio  (w'mu - rf)/sqrt(w'Sigma w)  s.t. 1'w = 1, w >= 0.
    Solver: SciPy SLSQP (analytic gradient), multi-start (EW, GMV, best-mean vertex);
    the best converged solution is kept."""
    N = len(mu)
    ex = mu - rf_bar

    def neg_sharpe(w):
        v = max(w @ Sigma @ w, 1e-16)
        return -(w @ ex) / np.sqrt(v)

    def grad(w):
        v = max(w @ Sigma @ w, 1e-16)
        s = np.sqrt(v)
        return -(ex / s - (w @ ex) * (Sigma @ w) / (s ** 3))

    gmv_w, _ = solve_gmv(Sigma)
    vertex = np.zeros(N); vertex[np.argmax(ex)] = 1.0
    starts = [np.full(N, 1.0 / N), gmv_w, 0.5 * vertex + 0.5 * np.full(N, 1.0 / N)]
    cands = []
    for w0 in starts:
        res = optimize.minimize(neg_sharpe, w0, jac=grad, method="SLSQP",
                                bounds=[(0, None)] * N,
                                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                              "jac": lambda w: np.ones(N)}],
                                options={"maxiter": 1000, "ftol": 1e-14})
        w = clean_weights(res.x)
        cands.append((neg_sharpe(w), w))
    # independent candidate: convex reformulation (Cornuejols-Tutuncu) min y'Sy s.t. y'ex = 1, y >= 0,
    # w = y / sum(y); valid when some asset has positive excess mean.
    if ex.max() > 0:
        S = Sigma * SCALE
        res = optimize.minimize(lambda y: y @ S @ y, vertex / ex.max(), jac=lambda y: 2 * S @ y,
                                method="SLSQP", bounds=[(0, None)] * N,
                                constraints=[{"type": "eq", "fun": lambda y: y @ ex - 1.0,
                                              "jac": lambda y: ex}],
                                options={"maxiter": 1000, "ftol": 1e-14})
        if res.x.sum() > 0:
            w = clean_weights(res.x)
            cands.append((neg_sharpe(w), w))
    best_val, best_w = min(cands, key=lambda c: c[0])
    # validation: the best Sharpe must be reproduced by >= 2 independent runs (different starts / formulation)
    ok = sum(abs(c[0] - best_val) <= 1e-6 * max(1.0, abs(best_val)) for c in cands) >= 2
    return best_w, ok, -best_val


def clean_weights(w, tol=1e-8):
    """Remove numerical dust, enforce w>=0 and sum(w)=1 exactly."""
    w = np.where(w < tol, 0.0, w)
    return w / w.sum()


def capm_alpha_test(Rf_win, MF_win, RF_win):
    """(d) For every stock run (R_i - RF) = a_i + b_i MF + e on the formation window (OLS) and
    return alpha, beta, two-sided p-value of H0: a_i = 0 (t-test, n-2 d.o.f.).
    Closed-form OLS for all stocks at once (equivalent to statsmodels OLS)."""
    y = Rf_win.values - RF_win.values[:, None]            # T x N excess returns
    x = MF_win.values                                      # T
    T = len(x)
    xm, ym = x.mean(), y.mean(axis=0)
    sxx = ((x - xm) ** 2).sum()
    beta = ((x - xm)[:, None] * (y - ym)).sum(axis=0) / sxx
    alpha = ym - beta * xm
    resid = y - alpha - np.outer(x, beta)
    s2 = (resid ** 2).sum(axis=0) / (T - 2)
    se_alpha = np.sqrt(s2 * (1.0 / T + xm ** 2 / sxx))
    tval = alpha / se_alpha
    pval = 2.0 * stats.t.sf(np.abs(tval), df=T - 2)
    return alpha, beta, pval


# ----------------------------------------------------------------------------------------
# 3. Holding-period returns (Problem 3)
# ----------------------------------------------------------------------------------------
def buy_and_hold(w0, R_hold):
    """R_p,hold = sum_i w_i^(0) * [ prod_d (1 + R_i,d) - 1 ]   (no rebalancing)."""
    asset_cum = np.prod(1.0 + R_hold.values, axis=0) - 1.0
    return float(w0 @ asset_cum)


# ----------------------------------------------------------------------------------------
# 4. Rolling backtest (Problems 4-5)
# ----------------------------------------------------------------------------------------
def run_backtest(R, B, fac):
    tickers = R.columns.tolist()
    rows, W = [], {p: [] for p in ["GMV", "MV", "EW", "Active"]}
    k = 0
    while True:
        f_start = FIRST_FORMATION_START + pd.DateOffset(months=HOLDING_MONTHS * k)
        f_end_excl = f_start + pd.DateOffset(months=FORMATION_MONTHS)       # first day after formation
        h_end_excl = f_end_excl + pd.DateOffset(months=HOLDING_MONTHS)
        if h_end_excl - pd.Timedelta(days=1) > LAST_HOLDING_END:
            break
        fm = (R.index >= f_start) & (R.index < f_end_excl)                  # formation mask
        hm = (R.index >= f_end_excl) & (R.index < h_end_excl)               # holding mask
        Rf_, Rh_ = R.loc[fm], R.loc[hm]
        Bh_ = B.loc[hm]
        MF_f, RF_f = fac.loc[fm, "MF"], fac.loc[fm, "RF"]

        # stocks "available" in the window: all columns remaining after the global filter
        N = len(tickers)
        mu = Rf_.mean().values                       # sample mean of daily simple returns
        Sigma = np.cov(Rf_.values, rowvar=False, ddof=1)
        rf_bar = RF_f.mean()                         # arithmetic mean of daily RF in formation window

        w_gmv, ok1 = solve_gmv(Sigma)
        w_mv, ok2, sh_d = solve_tangency(mu, Sigma, rf_bar)
        w_ew = np.full(N, 1.0 / N)

        alpha, beta, pval = capm_alpha_test(Rf_, MF_f, RF_f)
        A = (alpha > 0) & (pval < ALPHA_LEVEL)
        fallback = not A.any()
        w_act = np.zeros(N) if fallback else A.astype(float) / A.sum()

        # --- realised holding-period returns (buy & hold, weights fixed at formation end)
        r_gmv, r_mv, r_ew = (buy_and_hold(w, Rh_) for w in (w_gmv, w_mv, w_ew))
        r_nifty = float(np.prod(1.0 + Bh_.values) - 1.0)                    # compounded index returns
        r_act = r_nifty if fallback else buy_and_hold(w_act, Rh_)
        rf_hold = float(np.prod(1.0 + fac.loc[hm, "RF"].values) - 1.0)      # compounded RF over holding window

        rows.append(dict(window=k + 1,
                         formation_start=f_start.date(), formation_end=(f_end_excl - pd.Timedelta(days=1)).date(),
                         holding_start=f_end_excl.date(), holding_end=(h_end_excl - pd.Timedelta(days=1)).date(),
                         n_form_days=int(fm.sum()), n_hold_days=int(hm.sum()), N_stocks=N,
                         n_alpha_sig=int(A.sum()), active_fallback=bool(fallback),
                         gmv_n_pos=int((w_gmv > 0).sum()), mv_n_pos=int((w_mv > 0).sum()),
                         gmv_eff_n=1 / np.sum(w_gmv ** 2), mv_eff_n=1 / np.sum(w_mv ** 2),
                         mv_daily_sharpe_form=sh_d, solver_ok=bool(ok1 and ok2),
                         GMV=r_gmv, MV=r_mv, EW=r_ew, Active=r_act, NIFTY50=r_nifty, RF_hold=rf_hold))
        for nm, w in zip(["GMV", "MV", "EW", "Active"], [w_gmv, w_mv, w_ew, w_act]):
            W[nm].append(w)
        k += 1

    det = pd.DataFrame(rows)
    wdf = {nm: pd.DataFrame(np.array(v), columns=tickers,
                            index=pd.to_datetime(det["formation_end"])) for nm, v in W.items()}
    return det, wdf


# ----------------------------------------------------------------------------------------
# 5. Performance reporting (Problem 6)
# ----------------------------------------------------------------------------------------
def performance_table(det):
    """Annualisation (stated in the report):
       * window return r (simple, 3M)  -> log return  l = ln(1+r)
       * 4 windows (=4 x 63 = 252 trading days) per year, so
            annual mean(log) = 4 * mean(l),  annual vol = sqrt(4) * std(l)   (windows treated as iid)
       * Risk-free convention: daily RF is compounded over each holding window -> log rf l_f; the
            Sharpe numerator is the window-by-window excess log return (l - l_f).
       * Sharpe = 4*mean(l - l_f) / (2*std(l - l_f));   IR = 4*mean(l - l_B) / (2*std(l - l_B)),
         where l_B is the Nifty 50 log return in the same window (tracking error from window differences).
    """
    L = np.log1p(det[PORTFOLIOS].astype(float))
    LF = np.log1p(det["RF_hold"].astype(float))
    out = {}
    for p in PORTFOLIOS:
        r, l = det[p].astype(float), L[p]
        ex = l - LF
        act = l - L["NIFTY50"]
        cum = (1 + r).cumprod()
        dd = (cum / np.maximum.accumulate(np.r_[1.0, cum.values])[1:] - 1).min()
        ir = np.nan if p == "NIFTY50" else (WINDOWS_PER_YEAR * act.mean()) / (np.sqrt(WINDOWS_PER_YEAR) * act.std(ddof=1))
        out[p] = {
            "Mean 3M return (%)": 100 * r.mean(),
            "Std 3M return (%)": 100 * r.std(ddof=1),
            "Ann. return (%) [exp(4*mean log)-1]": 100 * (np.exp(WINDOWS_PER_YEAR * l.mean()) - 1),
            "Ann. volatility (%) [2*std log]": 100 * np.sqrt(WINDOWS_PER_YEAR) * l.std(ddof=1),
            "Sharpe ratio (ann.)": (WINDOWS_PER_YEAR * ex.mean()) / (np.sqrt(WINDOWS_PER_YEAR) * ex.std(ddof=1)),
            "Information ratio (ann.)": ir,
            "Cumulative return (%)": 100 * (cum.iloc[-1] - 1),
            "CAGR (%)": 100 * (cum.iloc[-1] ** (WINDOWS_PER_YEAR / len(r)) - 1),
            "Max drawdown (%) [window-end]": 100 * dd,
            "Hit rate vs Nifty (%)": np.nan if p == "NIFTY50" else 100 * (r > det["NIFTY50"]).mean(),
        }
    return pd.DataFrame(out)


def make_plots(det):
    dates = pd.to_datetime(det["holding_end"])
    cum = (1 + det[PORTFOLIOS].astype(float)).cumprod()
    start = pd.Timestamp(det["holding_start"].iloc[0]) - pd.Timedelta(days=1)
    cum = pd.concat([pd.DataFrame([[1.0] * 5], columns=PORTFOLIOS, index=[start]), cum.set_axis(dates)])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sty = {"GMV": ("#1f77b4", "-"), "MV": ("#d62728", "-"), "EW": ("#2ca02c", "-"),
           "Active": ("#ff7f0e", "-"), "NIFTY50": ("k", "--")}
    for p in PORTFOLIOS:
        ax.plot(cum.index, cum[p], label=p, color=sty[p][0], ls=sty[p][1], lw=1.8)
    ax.set_yscale("log"); ax.set_ylabel("Growth of 1 (log scale)")
    ax.set_title("Cumulative returns (compounded 3-month holding returns), Jul 2009 - Dec 2022")
    ax.grid(alpha=.3); ax.legend(ncol=5, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/cumulative_returns.png", dpi=170); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].plot(dates, det["gmv_eff_n"], label="GMV", color="#1f77b4")
    axs[0].plot(dates, det["mv_eff_n"], label="MV", color="#d62728")
    axs[0].set_title("Effective number of holdings (1/sum w^2)"); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].bar(dates, det["n_alpha_sig"], width=60, color="#ff7f0e")
    axs[1].set_title("# stocks with significant positive alpha (Active)"); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/diagnostics.png", dpi=170); plt.close(fig)


# ----------------------------------------------------------------------------------------
def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    os.makedirs(OUT_DIR, exist_ok=True)
    R, B, fac, dropped, n_raw = load_data(data_dir)
    print(f"Raw price rows: {n_raw} | common-date daily returns: {len(R)} "
          f"({R.index[0].date()} -> {R.index[-1].date()})")
    print(f"Dropped stocks (missing/partial data): {dropped}")
    print(f"Stocks retained: {R.shape[1]}")

    det, wdf = run_backtest(R, B, fac)
    n = len(det)
    print(f"Rolling windows n = {n}")
    assert n == 54, "expected 54 rolling windows"
    assert det["solver_ok"].all(), "optimizer failure in some window"
    for nm, w in wdf.items():
        # Active rows are all-zero in fallback windows (100 % Nifty 50 index instead)
        rows_ok = ~det["active_fallback"].values if nm == "Active" else np.ones(len(w), bool)
        assert np.allclose(w.sum(axis=1)[rows_ok], 1) and (w.values >= -1e-12).all()

    # --- deliverables
    Rmat = det[PORTFOLIOS].copy()
    Rmat.index = pd.RangeIndex(1, n + 1, name="window")
    Rmat.to_csv(f"{OUT_DIR}/returns_matrix_nx5.csv", float_format="%.8f")
    det.to_csv(f"{OUT_DIR}/window_details.csv", index=False, float_format="%.8f")
    for nm, w in wdf.items():
        w.index.name = "formation_end"
        w.to_csv(f"{OUT_DIR}/weights_{nm}.csv", float_format="%.8f")
    perf = performance_table(det)
    perf.to_csv(f"{OUT_DIR}/performance_table.csv", float_format="%.4f")
    make_plots(det)

    with open(f"{OUT_DIR}/results.json", "w") as fh:
        json.dump({"dropped": dropped, "n_stocks": R.shape[1], "n_days": len(R),
                   "first_date": str(R.index[0].date()), "last_date": str(R.index[-1].date()),
                   "n_windows": n, "n_fallback": int(det["active_fallback"].sum()),
                   "tickers": R.columns.tolist()}, fh, indent=1)
    pd.set_option("display.width", 200)
    print(perf.round(3).T)
    print("Active fallback windows:", int(det["active_fallback"].sum()), "of", n)


if __name__ == "__main__":
    main()
