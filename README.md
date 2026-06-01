# Hydro Dispatch Optimization - Leaderboard (MAT0613)

Leaderboard for the hydro dispatch competition (1-10 June 2026). Teams are ranked on the
two project dimensions:

- **Profit**: realised euros from their dispatch decisions, with constraint checking.
- **Forecast accuracy**: mean absolute error on the three market variables (day-ahead
  price and the two availability prices), measured against the realised values.

The site is a single static `index.html` served by GitHub Pages.

## Scoring

Per quarter-hour, revenue is

```
R(t) = q(t) * DA(t) + C_idle(t) * AvailIncrease(t) + C_prod(t) * AvailReduce(t)
```

with `C_prod = q / 0.25` MW and `C_idle = 1000 - C_prod`, summed over the window.

Each day has a 5% chance of an unavailability event (drawn once, the same for every team).
On a day drawn unavailable the plant cannot run, so that day's revenue is zero: the penalty
cancels both the energy and the availability income for that day. These days are flagged on
the leaderboard chart.

A schedule is feasible when every quarter-hour is within `0-250 MWh` and the total released
over the window stays under `100,000 MWh` (reservoir floor of `400,000 MWh`). Infeasible
schedules are dropped from the profit ranking but still scored on forecast accuracy. The
project brief has the full model.

## Files

| File | Role |
|---|---|
| `index.html` | the leaderboard page (this is what is published) |
| `leaderboard_template.html` | page template with the data placeholder |
| `leaderboard_engine.py` | scoring code: revenue model, constraints, MAE |
| `requirements.txt` | Python packages to run the engine locally (`pandas`, `numpy`) |
| `.github/workflows/deploy.yml` | publishes `index.html` to GitHub Pages |
| `.nojekyll` | tells Pages to skip Jekyll processing |

## Visualization

Results for day D+1 are updated during day D after 18:00 (might take more time) and can be visualized in:
 - ['https://github.com/LucaCelentani/mat0613-project-leaderboard'](https://lucacelentani.github.io/mat0613-project-leaderboard/)
