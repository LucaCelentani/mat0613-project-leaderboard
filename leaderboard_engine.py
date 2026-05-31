"""
leaderboard_engine.py
=====================================================================
Scoring engine for the Hydro Dispatch Optimization competition.

Given, for the backtest window (June 1-10, 2026 in production; a held-out
historical window in the demo):

  * the ACTUAL quarter-hourly values of the three market variables, and
  * each team's SUBMISSION = a dispatch schedule q(t) + a forecast of the
    three variables,

this module computes, per team:

  1. MONEY  -- total realised profit (EUR) from the dispatch decisions,
               using the revenue model in the project brief, after
               applying the binary daily unavailability draw.
  2. CONSTRAINTS -- whether the dispatch is feasible (per-quarter-hour
               release cap, reservoir balance, end-of-window minimum).
               Infeasible submissions are flagged and disqualified from
               the money ranking (but still scored on accuracy).
  3. ACCURACY -- MAE of the forecast vs. actuals, for each of the three
               target variables.

The revenue model (brief sec. 4.1), per quarter-hour t:

    C_prod(t) = q(t) / 0.25            (MW of producing capacity)
    C_idle(t) = CAPACITY_MW - C_prod   (MW of idle capacity)

    R(t) = q(t) * DA(t)                         # energy sales
         + C_idle(t) * AvailIncrease(t)         # paid to keep idle capacity ready to ramp UP
         + C_prod(t) * AvailReduce(t)           # paid to keep producing capacity ready to ramp DOWN

Unavailability (brief sec. 4.2): each day is independently unavailable
with prob. 5%. On an unavailable day the plant cannot run; if the team
had offered availability, the penalty equals that day's earnings, so the
day nets to 0 either way. Teams that do not offer availability simply
forgo the availability income on every day.
=====================================================================
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
CAPACITY_MW       = 1000.0          # installed capacity
QH_HOURS          = 0.25            # quarter-hour duration
MAX_RELEASE_QH    = CAPACITY_MW * QH_HOURS   # 250 MWh per quarter-hour
W0                = 500_000.0       # reservoir at window start (MWh)
W_MIN_END         = 400_000.0       # reservoir floor at window end (MWh)
TOTAL_BUDGET      = W0 - W_MIN_END  # 100,000 MWh dispatchable over the window
UNAVAIL_PROB      = 0.05            # daily unavailability probability
QH_PER_DAY        = 96

TARGETS = [
    "Day-Ahead Energy Price (EUR/MWh)",
    "Availability Price to Reduce Production (EUR/MW/qh)",     # AvailReduce -> paid on producing capacity
    "Availability Price to Increase Production (EUR/MW/qh)",   # AvailIncrease -> paid on idle capacity
]
COL_DA     = TARGETS[0]
COL_REDUCE = TARGETS[1]
COL_INCR   = TARGETS[2]


# ============================================================= revenue model
def quarter_hour_revenue(q, da, avail_reduce, avail_increase, offered_availability=True):
    """Vectorised EUR revenue for an array of quarter-hours (before unavailability)."""
    q = np.asarray(q, float)
    c_prod = q / QH_HOURS
    c_idle = CAPACITY_MW - c_prod
    energy = q * np.asarray(da, float)
    if offered_availability:
        avail = c_idle * np.asarray(avail_increase, float) + c_prod * np.asarray(avail_reduce, float)
    else:
        avail = np.zeros_like(q)
    return energy, avail


def score_dispatch(q, actual_df, unavailable_days, offered_availability=True):
    """
    Score a full-window dispatch schedule against actual prices.

    Parameters
    ----------
    q : array, length = n_days * 96   -- dispatched energy per quarter-hour (MWh)
    actual_df : DataFrame with COL_DA, COL_REDUCE, COL_INCR (same length as q)
    unavailable_days : set[int]       -- day indices (0-based) drawn as unavailable
    offered_availability : bool       -- whether the team participated in availability markets

    Returns dict with realised profit, per-day breakdown, and constraint report.
    """
    q = np.asarray(q, float)
    n = len(q)
    n_days = n // QH_PER_DAY

    da  = actual_df[COL_DA].to_numpy(float)
    red = actual_df[COL_REDUCE].to_numpy(float)
    inc = actual_df[COL_INCR].to_numpy(float)

    energy, avail = quarter_hour_revenue(q, da, red, inc, offered_availability)

    profit_by_day, energy_by_day, avail_by_day = [], [], []
    for d in range(n_days):
        sl = slice(d * QH_PER_DAY, (d + 1) * QH_PER_DAY)
        e_day = float(energy[sl].sum())
        a_day = float(avail[sl].sum())
        if d in unavailable_days:
            realised = 0.0          # plant down: energy lost; if offered, penalty zeroes the day
        else:
            realised = e_day + a_day
        profit_by_day.append(realised)
        energy_by_day.append(e_day)
        avail_by_day.append(a_day)

    # ---- constraint validation -----------------------------------------
    total_dispatch = float(q.sum())
    max_qh         = float(q.max()) if n else 0.0
    min_qh         = float(q.min()) if n else 0.0
    violations = []
    if max_qh > MAX_RELEASE_QH + 1e-6:
        violations.append(f"release cap exceeded: max {max_qh:,.1f} > {MAX_RELEASE_QH:.0f} MWh/qh")
    if min_qh < -1e-6:
        violations.append(f"negative dispatch: min {min_qh:,.1f} MWh")
    if total_dispatch > TOTAL_BUDGET + 1e-3:
        violations.append(
            f"end-of-window reservoir < {W_MIN_END:,.0f}: dispatched "
            f"{total_dispatch:,.0f} > {TOTAL_BUDGET:,.0f} MWh"
        )

    w_end = W0 - total_dispatch
    return {
        "profit_total": float(sum(profit_by_day)),
        "profit_by_day": profit_by_day,
        "energy_by_day": energy_by_day,
        "avail_by_day": avail_by_day,
        "total_dispatch": total_dispatch,
        "max_qh": max_qh,
        "reservoir_end": w_end,
        "valid": len(violations) == 0,
        "violations": violations,
        "offered_availability": offered_availability,
    }


def score_forecast(forecast_df, actual_df):
    """Mean Absolute Error of the forecast for each target variable."""
    out = {}
    for col in TARGETS:
        f = forecast_df[col].to_numpy(float)
        a = actual_df[col].to_numpy(float)
        out[col] = float(np.mean(np.abs(f - a)))
    return out


# ====================================================== dispatch optimizer
def optimize_dispatch(price_df, offered_availability=True,
                      total_budget=TOTAL_BUDGET, n_days=None):
    """
    Greedy revenue-maximising dispatch under a per-day water budget
    (brief sec. 6.2, Alternative 1).

    The marginal value of producing one extra MWh in quarter-hour t is
        m(t) = DA(t) + 4 * (AvailReduce(t) - AvailIncrease(t))   [if availability offered]
        m(t) = DA(t)                                             [otherwise]
    (the factor 4 = 1 MWh / 0.25 h, the MW added to producing capacity).
    Within each day we fill the highest positive-marginal quarter-hours
    first, up to 250 MWh each, until the daily budget is spent.

    Returns an array of q(t) (MWh) of length n_days * 96.
    """
    da  = price_df[COL_DA].to_numpy(float)
    red = price_df[COL_REDUCE].to_numpy(float)
    inc = price_df[COL_INCR].to_numpy(float)
    if offered_availability:
        marginal = da + 4.0 * (red - inc)
    else:
        marginal = da.copy()

    n = len(da)
    if n_days is None:
        n_days = n // QH_PER_DAY
    q = np.zeros(n)

    w_remaining = W0
    for d in range(n_days):
        days_left = n_days - d
        daily_budget = max(0.0, (w_remaining - W_MIN_END) / days_left)
        sl = slice(d * QH_PER_DAY, (d + 1) * QH_PER_DAY)
        m = marginal[sl]
        order = np.argsort(-m)            # most valuable quarter-hours first
        budget = daily_budget
        qd = np.zeros(QH_PER_DAY)
        for idx in order:
            if budget <= 0 or m[idx] <= 0:
                break
            take = min(MAX_RELEASE_QH, budget)
            qd[idx] = take
            budget -= take
        q[sl] = qd
        w_remaining -= qd.sum()
    return q


# ============================================================= leaderboard
def _rank(values, ascending):
    """Dense-ish competition ranks (1 = best). ascending=True => lower is better."""
    order = np.argsort(values if ascending else [-v for v in values])
    ranks = [0] * len(values)
    for pos, i in enumerate(order):
        ranks[i] = pos + 1
    return ranks


def build_payload(teams, actual_df, day_dates, unavailable_days,
                  hindsight_profit, window_label, last_updated,
                  hindsight_by_day=None):
    """Assemble the JSON payload consumed by the HTML leaderboard."""
    # money ranking over compliant teams only
    compliant = [t for t in teams if t["score"]["valid"]]
    comp_profit = [t["score"]["profit_total"] for t in compliant]
    comp_ranks = _rank(comp_profit, ascending=False)
    for t, r in zip(compliant, comp_ranks):
        t["rank_profit"] = r
    for t in teams:
        if not t["score"]["valid"]:
            t["rank_profit"] = None

    # accuracy ranks per target (lower MAE = better), over all teams
    mae_ranks = {}
    for col in TARGETS:
        vals = [t["mae"][col] for t in teams]
        mae_ranks[col] = _rank(vals, ascending=True)

    teams_out = []
    for i, t in enumerate(teams):
        s = t["score"]
        teams_out.append({
            "id": t["id"],
            "name": t["name"],
            "members": t.get("members", ""),
            "strategy": t.get("strategy", ""),
            "offered_availability": s["offered_availability"],
            "profit_total": s["profit_total"],
            "profit_by_day": s["profit_by_day"],
            "pct_of_hindsight": (100.0 * s["profit_total"] / hindsight_profit
                                 if hindsight_profit else 0.0),
            "total_dispatch": s["total_dispatch"],
            "max_qh": s["max_qh"],
            "reservoir_end": s["reservoir_end"],
            "valid": s["valid"],
            "violations": s["violations"],
            "rank_profit": t["rank_profit"],
            "mae": {col: t["mae"][col] for col in TARGETS},
            "mae_rank": {col: mae_ranks[col][i] for col in TARGETS},
        })

    return {
        "meta": {
            "title": "Hydro Dispatch Optimization - Live Leaderboard",
            "subtitle": "1,000 MW Reservoir Plant \u00b7 Spain \u00b7 AY 2025\u201326",
            "window_label": window_label,
            "days": day_dates,
            "n_days": len(day_dates),
            "unavailable_days": sorted(unavailable_days),
            "last_updated": last_updated,
            "hindsight_profit": hindsight_profit,
            "hindsight_by_day": hindsight_by_day or [],
            "plant": {
                "capacity_mw": CAPACITY_MW,
                "w0": W0, "w_min_end": W_MIN_END,
                "total_budget": TOTAL_BUDGET,
                "max_release_qh": MAX_RELEASE_QH,
                "unavail_prob": UNAVAIL_PROB,
            },
        },
        "targets": TARGETS,
        "teams": teams_out,
    }


def build_leaderboard(payload, template_path, output_path="leaderboard.html"):
    """Inject the payload into the HTML template and write the static site."""
    html = Path(template_path).read_text(encoding="utf-8")
    html = html.replace("__PAYLOAD__", json.dumps(payload, default=str))
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
