# Claude Code Prompts — Step by Step

This file contains prompts to drive the project forward using Claude Code. Run them **in order**. Each prompt is self-contained: paste it as the message to Claude Code, let it work, review results, then move to the next.

Before starting, make sure:
- You have `PROJECT_OUTLINE.md` in the project root
- You have Ben's notebook in the project (place it under `reference/bens_notebook.ipynb`)
- You have [`uv`](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux, or see uv docs for Windows)
- You're using Python 3.11+ (uv will manage the interpreter for you)

**Convention:** All Python execution goes through `uv run`. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Never use `pip install` directly — use `uv add <package>` instead.

---

## PROMPT 0 — Bootstrap the repository

```
You are helping me with a forecasting project for my BSE course. Read PROJECT_OUTLINE.md carefully — it describes the full project. Your job right now is only Phase 0: set up the project skeleton.

We use `uv` as the package and environment manager. All Python execution goes through `uv run`. Do NOT use pip directly. Do NOT create requirements.txt — dependencies live in pyproject.toml.

Tasks:
1. Initialize the uv project in the current directory:
   - Run `uv init --python 3.11` (creates pyproject.toml and .python-version)
   - This also creates a basic .gitignore and src layout — keep what's useful, remove what isn't
2. Add project dependencies with uv:
   - `uv add pandas numpy requests pytrends pandas-datareader lightgbm scikit-learn matplotlib seaborn jupyter openpyxl scipy statsmodels pyarrow`
   - `uv add --dev ipykernel ruff` (dev tools for notebooks and linting)
3. Verify the environment works by running `uv run python -c "import pandas, lightgbm, pytrends; print('ok')"`
4. Create the directory structure described in section 12 of PROJECT_OUTLINE.md (data/, notebooks/, src/, output/, report/, reference/ — with the subdirectories listed)
5. Create empty notebook stubs at notebooks/01 through 07 with just a title cell and a "TODO" cell. They should already work with `uv run jupyter lab` because we added jupyter and ipykernel.
6. Update .gitignore to include: .venv/, data/raw/, data/interim/, *.parquet (if large), .ipynb_checkpoints/, __pycache__/, output/ (but keep output/.gitkeep). Leave uv.lock and pyproject.toml tracked.
7. Create a README.md with:
   - Project title and one-paragraph description
   - "Setup" section showing `uv sync` to install everything
   - "Run" section showing `uv run jupyter lab` to open notebooks
   - Note that all commands should be prefixed with `uv run`

Do NOT start writing data-loading code yet. Just the skeleton. After you're done:
- Show me the output of `uv tree` (or `uv pip list`) to confirm packages are installed
- List the directory structure you created
- Confirm `uv.lock` exists and is tracked
```

---

## PROMPT 1 — Download and clean BLS LAUS (Notebook 01)

```
We're working on notebook 01_data_bls_laus.ipynb. Goal: download monthly state-level unemployment rates from BLS LAUS, clean them, and save a tidy panel to data/interim/laus_panel.parquet.

Read PROJECT_OUTLINE.md section 2.1 for the data spec.

Instructions:
1. The data source is BLS LAUS flat files. The simplest approach: download the seasonally adjusted state series from https://download.bls.gov/pub/time.series/la/la.data.3.AllStatesS (you can also use la.data.30.AllStatesS depending on what BLS currently exposes — verify the URL first by listing https://download.bls.gov/pub/time.series/la/). You'll also need la.series for series metadata and la.area for area names. BLS requires a User-Agent header; use something like {'User-Agent': 'academic-research email@example.com'}.
2. Filter to:
   - 50 states + DC (area_type = 'A')
   - Unemployment rate (measure code = '03')
   - Seasonally adjusted series
   - Period Jan 2010 – Dec 2024
3. Output a tidy panel: columns [state_code, state_name, year, month, date, unemployment_rate]
4. Sanity check: print panel shape (should be ~9,180 rows), print the first/last 5 rows, print a quick summary by state (mean, min, max).
5. Save to data/interim/laus_panel.parquet
6. Also produce a quick EDA chart: a small-multiples plot of unemployment rate over time for 6 representative states (CA, TX, NY, FL, MI, ND). Save to output/figures/01_laus_eda.png

Important:
- Do not commit raw downloaded files larger than 50MB to git
- Cache downloads in data/raw/bls_laus/ — if the file is already there, don't re-download
- Be defensive with the BLS file format (it's tab-separated with weird whitespace)

After implementation, show me the panel shape, the EDA chart, and stop. Don't proceed to other notebooks.
```

---

## PROMPT 2 — Build the shock outcome variable using the Sahm Rule (still Notebook 01)

```
Continue in notebook 01. We now build the onset variable using a state-level adaptation of the Sahm Rule, the canonical recession indicator published by FRED (SAHMCURRENT). This is robust to monthly noise, residual seasonality, and slow drifts in the natural rate — all weaknesses of a naive monthly-diff threshold.

Read PROJECT_OUTLINE.md section 3.1 carefully — the full algorithm is there.

The math, for each state i and month t:
- MA3_{i,t}   = mean of unemployment_rate over months {t, t-1, t-2}
- min12_{i,t} = min of MA3 over the PRIOR 12 months: {t-1, t-2, ..., t-12}   # no look-ahead
- sahm_gap_{i,t} = MA3_{i,t} - min12_{i,t}
- shock_{i,t} = 1 if (sahm_gap_{i,t} >= threshold) AND (sahm_gap_{i,t-1} < threshold)
              = 0 otherwise

The first-crossing condition is critical — it gives us ONSET (the moment the rule fires), not PREVALENCE (still elevated). This matches the decision-maker framing in section 3.3 of the outline.

Tasks:

1. Load data/interim/laus_panel.parquet from Prompt 1.

2. Implement a function `build_sahm_shock(df, threshold=0.5, ma_window=3, lookback=12)` that operates state-by-state. Use pandas .groupby('state_code') and .rolling(). Critical detail: when computing min12, you MUST shift by 1 before rolling to exclude the current month from its own prior minimum. Example:
   group['ma3'] = group['unemployment_rate'].rolling(3).mean()
   group['min12'] = group['ma3'].shift(1).rolling(12).min()
   group['sahm_gap'] = group['ma3'] - group['min12']
   group['above'] = (group['sahm_gap'] >= threshold).astype(int)
   group['above_prev'] = group['above'].shift(1).fillna(0)
   group['shock'] = ((group['above'] == 1) & (group['above_prev'] == 0)).astype(int)

3. Build three shock variables for robustness:
   - shock_sahm_05 (threshold=0.5, the canonical Sahm)
   - shock_sahm_03 (threshold=0.3, earlier-warning variant)
   - shock_sahm_07 (threshold=0.7, stricter variant)
   Also keep the continuous sahm_gap column as a feature for later.

4. Validation checks — print all of these:
   a. Total shock count and base rate for each threshold
   b. Shocks per state (full table for the 0.5 variant — sorted by count)
   c. Shocks per year (highlight 2020)
   d. Distribution of months_since_last_shock — should always be >= 1 by construction; most should be many years apart given Sahm is conservative
   e. Sanity check: compare the date of FIRST shock in 2020 across states. Most should fire around March-May 2020.
   f. Compare our national-level Sahm gap (aggregating state unemployment with labor force weights, or just using BLS national series if simpler) to FRED's SAHMCURRENT for a spot-check. Spirit: the two series should track closely.

5. EDA charts:
   - Bar chart of shock counts per year (threshold = 0.5). Save to output/figures/01_sahm_shocks_per_year.png
   - Pick 4 representative states (CA, TX, MI, ND). For each, plot unemployment_rate over time with vertical red lines at shock dates. 2x2 grid. Save to output/figures/01_sahm_shocks_examples.png
   - Histogram of sahm_gap values (excluding NaN). Mark the three thresholds with vertical lines. Save to output/figures/01_sahm_gap_distribution.png

6. Save the enriched panel to data/interim/laus_panel_with_shocks.parquet, columns: [state_code, state_name, year, month, date, unemployment_rate, ma3, sahm_gap, shock_sahm_03, shock_sahm_05, shock_sahm_07]

Stop after this. Show me:
- The validation outputs (total counts, per-state, per-year)
- The 4-state examples chart (this is the key sanity check — the shocks should LOOK like real onsets)
- Confirmation that the national-aggregated series tracks FRED's SAHMCURRENT

If the shock count for threshold=0.5 is below 50 or above 400, stop and flag it — something may be off.
```

---

## PROMPT 3a — Write the Google Trends collection script (script, not notebook)

```
Goal: Write a standalone Python script src/collect_google_trends.py that I will run myself from the terminal. Do NOT run it from within Claude Code — Google Trends has aggressive rate limiting and the collection takes 1-3 hours; it needs to run in my own terminal where I can monitor it and restart if needed.

Read PROJECT_OUTLINE.md section 2.2.

Spec for the script:

1. Inputs (configurable as constants at the top of the script):
   - TERMS = ['unemployment', 'file for unemployment', 'unemployment benefits', 'jobs hiring', 'layoffs', 'resume']
   - STATES = list of 50 state postal codes + 'DC' (51 total)
   - TIMEFRAME = '2010-01-01 2024-12-31'
   - SLEEP_BASE = 60 (seconds between successful calls)
   - SLEEP_ON_ERROR_BASE = 120 (initial backoff after a failure)
   - MAX_RETRIES = 3
   - CACHE_DIR = 'data/raw/google_trends'

2. Behavior:
   - Build the full list of (state_code, term) pairs
   - For each pair, the cache filename is data/raw/google_trends/{state_code}__{term_slug}.csv where term_slug replaces spaces with underscores
   - SKIP any pair whose CSV already exists and is non-empty (this makes the script restartable — I can kill it and re-run)
   - For each pair to fetch:
     - Use pytrends with geo=f'US-{state_code}', timeframe=TIMEFRAME, kw_list=[term]
     - Fetch weekly interest_over_time
     - Save raw output to CSV with columns: week_start (date), term, interest, state_code
     - Sleep SLEEP_BASE seconds before the next call
   - Error handling: wrap each fetch in try/except. On failure:
     - Print clearly which pair failed and the exception
     - Wait SLEEP_ON_ERROR_BASE * (2 ** retry_count) seconds (exponential backoff)
     - Retry up to MAX_RETRIES times
     - If still failing, append the pair to data/raw/google_trends/_FAILED.txt and continue to the next pair (so one bad term doesn't stop the whole run)
   - Logging: print one line per attempt with timestamp, e.g.:
     [2025-05-20 14:32:01] [142/306] CA__unemployment_benefits ... OK (cached size: 783 rows)
     [2025-05-20 14:33:05] [143/306] CA__layoffs ... FAILED (429 Too Many Requests), retry 1/3 in 120s

3. Output of the script:
   - Per-pair CSV files in data/raw/google_trends/
   - A summary printed at the end: how many succeeded, how many failed, expected total
   - Any failures listed in _FAILED.txt for easy re-runs

4. Make the script idempotent and restartable. If I Ctrl+C it midway and re-run, it should pick up exactly where it left off (because already-cached pairs are skipped).

5. Add a one-line CLI:
   - `uv run python src/collect_google_trends.py` to start the full run
   - `uv run python src/collect_google_trends.py --dry-run` to just print what would be fetched without calling the API

6. After writing the script, ALSO print to me:
   - Total number of (state, term) pairs (should be 51 × 6 = 306)
   - Estimated runtime at 60s/call: ~5 hours worst case if nothing is cached. Realistically more like 2-3 hours with successful caching and occasional retries
   - The exact command I should run

DO NOT run the script. Just write it, show me the file, and tell me the command. I'll run it locally and come back when collection is done.
```

---

## PROMPT 3b — Validate the collected Trends data and build the panel (Notebook 02)

```
I've now run src/collect_google_trends.py locally and have the raw CSVs in data/raw/google_trends/. Time to validate them and build the panel.

We're working on notebook 02_data_google_trends.ipynb. Goal: check what was actually collected, deal with any gaps, and build data/interim/google_trends_panel.parquet.

Tasks:

1. Inventory check:
   - Count CSV files in data/raw/google_trends/ — expected: 306 (51 states × 6 terms)
   - List any (state, term) pairs that are MISSING (compare actual files vs the expected full list)
   - Check data/raw/google_trends/_FAILED.txt if it exists and print its contents
   - For each CSV, check the row count (should be ~780 weekly observations for 2010-2024) and date range
   - Flag any CSV with fewer than 600 rows or with date range not covering 2010-01 to 2024-12 — these may have hit Google Trends quota issues and need to be re-fetched

2. If any pairs are missing or look incomplete:
   - List them clearly
   - Suggest running src/collect_google_trends.py again (it will only re-fetch the missing ones thanks to the cache)
   - STOP and ask me to re-run before continuing, unless the missing pairs are <2% of total (in which case proceed and just drop those state-term combinations from the panel)

3. Once the inventory is clean, load all CSVs and stack into a long panel: [state_code, term, week_start, interest]

4. Aggregate weekly -> monthly: monthly_interest = mean over weeks whose week_start falls within that month. Handle edge cases (week spanning two months — assign to the month containing the Tuesday, or whatever rule pytrends uses; document the choice).

5. Validation on the merged panel:
   - Shape check: should be roughly 51 states × 6 terms × 180 months = ~55,000 rows (long format)
   - Print head and tail
   - Per state, per term: count of non-null monthly values
   - Distribution of interest values per term (some terms like 'resume' may have very low signal in small states — important to know)

6. Save to data/interim/google_trends_panel.parquet

7. EDA charts:
   - Time series of 'unemployment' search interest for 6 representative states (CA, TX, NY, FL, MI, ND). 2x3 grid. Save to output/figures/02_trends_eda_unemployment.png
   - Time series of 'layoffs' for the same 6 states (sanity check that the COVID spike is visible). Save to output/figures/02_trends_eda_layoffs.png
   - Heatmap: average interest level per (state, term), normalized within term. Save to output/figures/02_trends_state_term_heatmap.png

Stop after this. Show me:
- Inventory result (how many pairs collected, any gaps)
- Final panel shape
- The three EDA charts
- Any state-term combinations with concerning data quality (low coverage, all-zeros, etc.)
```

---

## PROMPT 4 — Download macro + leading indicators (Notebook 03)

```
We're working on notebook 03_data_macro_leading.ipynb. Goal: collect FRED macro controls, DOL ETA-539 state weekly claims, and BLS CES state payrolls.

Read PROJECT_OUTLINE.md sections 2.3 and 2.4.

Tasks:

PART A — FRED (use pandas_datareader.data.DataReader; if you find you need fredapi instead, install it with `uv add fredapi` first):
- UNRATE (national unemployment rate)
- ICSA (national initial claims)
- VIXCLS (VIX)
- SP500 (S&P 500 — daily, aggregate to monthly returns)
- T10Y2Y (yield curve slope)
- Period: 2010-01-01 to 2024-12-31
- Save to data/interim/fred_macro.parquet, monthly frequency

PART B — DOL ETA-539 (state weekly claims):
- URL: https://oui.doleta.gov/unemploy/csv/ar539.csv
- Download, parse the 'st' (state), 'rptdate' (week ending date), and 'c3' (initial claims) columns
- Aggregate weekly -> monthly (sum of weekly initial claims falling in the month)
- Save to data/interim/state_initial_claims.parquet

PART C — BLS CES state payrolls:
- URL pattern: similar to LAUS, use https://download.bls.gov/pub/time.series/sm/ files
- Need: total nonfarm payrolls, seasonally adjusted, by state, monthly
- Save to data/interim/state_payrolls.parquet

For each part:
- Cache downloads in data/raw/{fred,dol_eta539,bls_ces}/
- Print shape and head of each output
- EDA: one chart per part showing the data over time for a few states (or for the national series in part A). Save to output/figures/03_macro_eda.png

Stop after all three parts are done. Show me the shapes and the chart.
```

---

## PROMPT 5 — Merge and feature engineering (Notebook 04)

```
We're working on notebook 04_merge_and_features.ipynb. Goal: merge all sources into a single state-month panel and build all features described in PROJECT_OUTLINE.md section 4.

Tasks:

1. Load:
   - data/interim/laus_panel_with_shocks.parquet (outcome + raw unemployment rate)
   - data/interim/google_trends_panel.parquet (text)
   - data/interim/fred_macro.parquet (national controls)
   - data/interim/state_initial_claims.parquet (state leading indicator)
   - data/interim/state_payrolls.parquet (state leading indicator)

2. Merge on (state_code, year, month). The base panel is LAUS (51 states × 180 months).

3. Build features (refer to PROJECT_OUTLINE.md section 4 for the full list):

   A. Text features (per state, z-score within state):
      - Level of each of the 6 terms
      - 1-month change
      - 3-month rolling std
      - Year-over-year ratio
      - PCA first component across the 6 z-scored levels (fit PCA on TRAINING DATA ONLY — leave a placeholder note saying this needs to be done inside the rolling forecast loop, not here; for now, store the raw inputs and we'll PCA later)

   B. Lags of unemployment:
      - 1, 2, 3, 6, 12-month lags
      - 1-month and 3-month change
      - 3-month rolling std

   C. Since-variables (using shock_sahm_05 as the target shock):
      - months_since_last_sahm_shock (cap at 36 to avoid extreme values; default 36 if no prior shock in the panel)
      - cumulative_sahm_shocks_in_last_24m
      - sahm_gap_lag1 — the continuous Sahm gap itself, lagged by 1 month (informative continuous feature alongside the binary derivatives)

   D. Macro lags (lag 1):
      - All FRED series
      - log(state initial claims)
      - state payrolls % change YoY

   E. Fixed effects:
      - state_code (will be one-hot encoded inside training)
      - month_of_year (1-12)

4. Critical: any feature that uses information from the future (e.g., rolling stats) must be computed using only past data. Use shift() before rolling. Document this carefully.

5. After all features are built, drop rows with NaN (the first 12 months per state will be dropped due to lags).

6. Save to data/processed/panel_with_features.parquet

7. Print:
   - Final panel shape
   - Feature list grouped by category
   - Correlation matrix of the text features (heatmap, save to output/figures/04_text_features_corr.png)
   - Number of shocks remaining after dropping NaN rows

Stop after this. Show me the shape, the feature list, and the heatmap.
```

---

## PROMPT 6 — Rolling forecast (Notebook 05)

```
We're working on notebook 05_rolling_forecast.ipynb. Goal: implement the rolling forecast for models M0, M2, M3, M4 as defined in PROJECT_OUTLINE.md section 5.

CRITICAL: Look at reference/bens_notebook.ipynb first. We must adapt his rolling forecast code, not write our own from scratch. The professor explicitly said this in the assignment. Stay faithful to his structure.

Tasks:

1. Load data/processed/panel_with_features.parquet

2. Define the four feature sets clearly at the top of the notebook (use variables, not magic strings):
   - features_M0 = [unemployment_rate_lag1, state dummies, month-of-year dummies]
   - features_M2 = features_M0 + [AR lags 2-12, change, rolling std, FRED macro lags]
   - features_M3 = features_M2 + [state claims lag, state payrolls lag]
   - features_M4 = features_M3 + [Google Trends features + PCA component]

3. Rolling forecast loop:
   - For each month t from 2014-01 to 2024-12:
     - Train set: all rows with date < t
     - Test set: all rows with date == t (51 states)
     - For each model M in [M0, M2, M3, M4]:
       - If features include PCA: fit PCA on training set only, transform train and test
       - Fit LightGBMClassifier with class_weight='balanced', default hyperparameters
       - Predict probability on test set
       - Store (date, state_code, model_name, true_label, predicted_prob)
   - Use shock_sahm_05 as the target (the canonical 0.5 pp Sahm threshold). The Sahm Rule is conservative — expect very few positives (~1-2% base rate, heavily concentrated in 2020). The class_weight='balanced' setting is essential.

4. Output: data/processed/predictions.parquet with columns [date, state_code, model, y_true, y_pred_proba]

5. This will take a while (~132 months × 4 models = 528 fits). Print progress every 12 months.

6. Quick sanity check at the end: print AUC-ROC AND AUC-PR per model on the full out-of-sample pool — for a rare-event target, AUC-PR is the more honest metric. Don't make plots yet — that's the next notebook.

Stop after predictions are saved and AUCs are printed.
```

---

## PROMPT 7 — Evaluation (Notebook 06)

```
We're working on notebook 06_evaluation.ipynb. Goal: produce all evaluation outputs described in PROJECT_OUTLINE.md section 6.

Tasks:

1. Load data/processed/predictions.parquet.

2. Pooled ROC curves:
   - Plot ROC for M0, M2, M3, M4 on one chart with AUC in the legend
   - Save to output/figures/06_roc_pooled.png

3. Pooled PR curves:
   - Plot PR for M0, M2, M3, M4 on one chart with AUC-PR in the legend
   - Add the no-skill baseline (horizontal line at empirical base rate)
   - Save to output/figures/06_pr_pooled.png

4. Headline table — output/tables/06_headline_metrics.csv:
   - Columns: model, AUC-ROC, AUC-PR, base_rate
   - One row per model

5. Time-varying performance:
   - Split predictions into: 2014-2019, 2020, 2021-2024
   - For each sub-period, compute AUC-ROC and AUC-PR for each model
   - Make a grouped bar chart (sub-period on x, AUC on y, model as color)
   - Save to output/figures/06_subperiod_performance.png
   - Save the table to output/tables/06_subperiod_metrics.csv

6. Where does text help most? Test:
   - For each shock in the test set, compute "national comovement" = number of other states with a shock in the same month
   - Split shocks into quartiles by comovement
   - Compare M4 vs M3 recall (at a fixed threshold, e.g. the threshold giving precision = 0.3 on the full pool) across quartiles
   - Hypothesis: M4 - M3 is larger for idiosyncratic (low comovement) shocks
   - Make a simple table and a bar chart. Save to output/tables/ and output/figures/

7. Error analysis:
   - At a fixed threshold from M4, list the top 10 false negatives (worst missed shocks) and top 10 false positives (highest predicted prob with no shock)
   - Save to output/tables/06_error_analysis.csv

Stop after all outputs are produced. Show me all charts and tables.
```

---

## PROMPT 8 — Cost model (Notebook 07)

```
We're working on notebook 07_cost_model.ipynb. Goal: implement the cost model and find optimal cutoffs as described in PROJECT_OUTLINE.md section 7.

Tasks:

1. Load data/processed/predictions.parquet.

2. Define the cost function:
   - C_act = 2_000_000  (cost of activating Rapid Response Team)
   - B_help = 10_000_000  (benefit if shock + intervention)
   - C_social = 8_000_000  (cost if shock + no intervention)
   - total_cost(y_true, y_pred_at_cutoff):
     - TP cost: -B_help + C_act  → net loss prevented = B_help - C_act
     - FP cost: C_act
     - FN cost: C_social
     - TN cost: 0
     - We report TOTAL COST (lower is better)

3. For each model M0, M2, M3, M4:
   - For τ in np.linspace(0.01, 0.99, 50):
     - Compute total_cost on the full out-of-sample pool
   - Find τ* (cutoff minimizing cost)
   - Record cost at τ*

4. Output:
   - Plot total cost vs cutoff for all 4 models. Mark τ* for each. Save to output/figures/07_cost_curves.png
   - Table output/tables/07_optimal_cutoffs.csv with columns [model, tau_star, cost_at_tau_star, savings_vs_no_action]
   - Key result: value of Google Trends = cost(M3, τ*_M3) - cost(M4, τ*_M4). Print clearly.

5. Sensitivity analysis:
   - Re-run with C_act × 0.5, × 2; and C_social × 0.5, × 2
   - For each scenario, recompute τ* for M3 and M4, and the savings of M4 over M3
   - Output table to output/tables/07_sensitivity.csv

Stop after all outputs. Show me the cost curves and the sensitivity table.
```

---

## PROMPT 9 — Report draft

```
You have all the analysis done. Now draft the final report.

Read PROJECT_OUTLINE.md for structure. The report has 6 main sections (Q1-Q6) plus a bonus section.

Tasks:
1. Create report/report.md
2. Structure: Introduction (research question, motivation, light literature anchor) → Data → Onset definition + decision-makers → Feature engineering → Rolling forecast → Evaluation (ROC, PR, sub-periods, error analysis) → Cost model → Conclusion (honest limitations, what we learned)
3. Embed key figures inline (reference paths from output/figures/)
4. Embed key tables (or summarize them in markdown tables)
5. Tone: academic but readable. Be honest about limitations — if M4 - M3 is small, say so and frame it as a meaningful negative-ish result.
6. Length: aim for ~3000-4000 words. Not too long. The professor said brevity matters.
7. End with a short "Connection to course concepts" subsection — checklist-style — confirming we addressed onset, since-variables, rolling forecast, time-varying performance, forecast-decision link.

After the draft, list anything you couldn't write because you needed input from us (e.g., interpretations of unexpected results).
```

---

## PROMPT 10 — Final polish

```
Final pass. Tasks:

1. Re-run all notebooks top-to-bottom from a clean kernel (`uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb --inplace`) — confirm they all execute without errors. If any fails, fix it.
2. Check all figures are saved correctly and referenced in the report.
3. Run `uv sync` from scratch (delete .venv first if needed) to confirm reproducibility from pyproject.toml + uv.lock alone.
4. Update README.md with:
   - Final results summary (AUC-ROC and AUC-PR for each model, dollar savings of Google Trends)
   - Setup: `uv sync`
   - Reproduce: `uv run jupyter lab` then run notebooks 01–07 in order, then build report
   - Note any non-Python steps (e.g., pandoc for PDF)
5. Check the report renders cleanly (no broken markdown, no broken image links).
6. Convert report.md to PDF if possible (use `uv run pandoc` if pandoc-equivalent is needed, or install via `uv add` if a Python alternative like `pypandoc` works) — save as report/report.pdf.
7. Final reproducibility check:
   - `uv.lock` and `pyproject.toml` are tracked
   - `.venv/` and `data/raw/` are NOT tracked
   - List any files larger than 50MB that shouldn't be in the submission

Show me a final summary: what's in the submission, what the headline numbers are, and anything that still needs human review.
```

---

## Tips for using these prompts

- **Run them in order.** Each builds on the previous.
- **After each prompt, read what Claude Code produced** before moving on. Don't fire-and-forget.
- **If something looks wrong** (weird shape, suspicious numbers), stop and investigate before continuing. Cascading errors are painful here.
- **Prompts 3a and 3b are split intentionally.** Prompt 3a tells Claude Code to *write* a Google Trends collection script — I then run it myself in my own terminal (it takes 2-5 hours due to rate limiting). Prompt 3b runs after collection and validates the output. Start Prompt 3a as early as possible so the collection script is running in the background while I work on other prompts.
- **Don't skip the EDA charts.** They catch data problems early.
- **Keep `PROJECT_OUTLINE.md` open** when reviewing Claude Code's work — it's the source of truth.
- **When in doubt, ask Claude Code to "explain what you just did and why."** Better to slow down than to ship broken analysis.

### Suggested execution order (with parallelism)

1. Prompt 0 (bootstrap) — ~5 min
2. Prompt 3a (write Trends script) — ~5 min
3. **Start the Trends script running in a separate terminal:** `uv run python src/collect_google_trends.py` — leave running for 2-5 hours
4. While Trends collects, do Prompt 1 (BLS LAUS), Prompt 2 (Sahm shocks), Prompt 4 (FRED + claims + payrolls)
5. When Trends finishes, do Prompt 3b (validate)
6. Then Prompt 5 (merge + features), 6 (rolling forecast), 7 (evaluation), 8 (cost model), 9 (report), 10 (polish)

---

## If Claude Code gets stuck

Common issues and what to ask:

- **BLS download fails**: "The BLS download isn't working. Check the URL pattern at https://download.bls.gov/pub/time.series/la/ and find the right file for state-level monthly unemployment rate, seasonally adjusted."
- **pytrends rate-limited (during local script run)**: This is normal. The script's retry logic should handle it. If a specific (state, term) pair keeps failing after 3 retries, it'll be logged in `data/raw/google_trends/_FAILED.txt`. After the main run, you can re-run the script (it skips already-cached pairs) — sometimes Google's rate limiter relaxes after a few hours. If many pairs fail repeatedly, ask Claude Code to "increase SLEEP_BASE to 90 in src/collect_google_trends.py and update the backoff strategy".
- **LightGBM warnings**: "Suppress LightGBM verbose output (set verbose=-1). Keep only progress prints from our own loop."
- **PCA leakage**: "Confirm PCA is fit only on training data inside the rolling loop, never on the full panel. Show me the code that does this."
- **Numbers look wrong**: "Stop and audit the most recent notebook output. Print the panel head, the shock counts, and re-verify against PROJECT_OUTLINE.md section 3.1."
- **Missing package**: "We're missing a package. Add it with `uv add <package>` so it's tracked in pyproject.toml and uv.lock — do NOT use pip directly."
- **Jupyter doesn't pick up the env**: "Make sure you're launching with `uv run jupyter lab`, and that the kernel selected is the project's .venv (`uv run python -c 'import sys; print(sys.executable)'` should point inside .venv)."
- **`uv sync` fails after pulling changes**: "Check that pyproject.toml and uv.lock are both present and consistent. If lock is stale, regenerate with `uv lock`."
