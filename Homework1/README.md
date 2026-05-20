# Nowcasting State-Level Unemployment Shocks with Google Trends

Can Google search data help nowcast unemployment shocks in US states before official BLS data is released? This project builds a state-level panel (50 states + DC, Jan 2010 – Dec 2024) merging BLS LAUS unemployment rates with Google Trends search indices, DOL initial claims, and BLS payroll data. We use a rolling LightGBM classifier to evaluate whether real-time search behavior adds predictive value over a strong macro baseline (M3), then translate the improvement into dollar terms via an asymmetric cost model for state Rapid Response Team activation.

## Setup

Install all dependencies from `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

This creates a `.venv/` in the project directory and installs every pinned package. No other steps needed.

## Run

Open the notebooks in order:

```bash
uv run jupyter lab
```

Then work through `notebooks/01` through `07` in sequence. All scripts and one-off commands should be prefixed with `uv run` so they use the project environment:

```bash
uv run python src/data/download_bls.py
uv run python -m pytest
```

## Structure

```
data/raw/          # Downloaded source files (gitignored)
data/interim/      # Cleaned per-source (gitignored)
data/processed/    # Final merged panel
notebooks/         # Analysis notebooks 01-07
src/               # Reusable Python modules (data, features, models, evaluation)
output/            # Figures and tables
report/            # Final report
reference/         # Ben's notebook and other reference material
```

## Notes

- `uv.lock` and `pyproject.toml` are committed and define the exact environment.
- `data/raw/` and `data/interim/` are gitignored -- re-run notebooks 01-03 to regenerate.
- Google Trends collection (notebook 02) is rate-limited; start early and expect ~1 min/state.
