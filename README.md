# ViT + XAI on 90 × 65 technical-indicator images (NIFTY 50)

Buy / Hold / Sell prediction from next-day return (threshold 0.01). A Vision Transformer is
trained on images with 90 indicators as rows and 65 trading days as columns, then explained with
Integrated Gradients, row occlusion and Chefer et al. (2021) relevance.
Base paper: Gezici & Sefer, *Deep Transformer-Based Asset Price and Direction Prediction*, IEEE Access 2024.
The phases follow `ViT_XAI_90x65_Workflow.pdf`.

## Setup

```bash
pip install -r requirements.txt     # TA-Lib >= 0.6 wheels bundle the C library on most platforms
pip install -e .                    # optional; scripts also work without it
python -m pytest -q
```

On GPU, use `notebooks/colab_run.ipynb`.

## Pipeline

| Phase | Command | Main outputs |
|---|---|---|
| 1 Data | `python scripts/01_download.py` | `data/raw`, `data/interim`, `data/universe.csv`, `data/calendar.csv`, `data/data_quality_report.csv`, `data/big_moves.csv` |
| 2 Indicators | `python scripts/02_indicators.py` | `data/processed/indicators/*.parquet`, `indicator_list.csv` |
| 3 Labels, folds | `python scripts/03_labels_folds.py` | `labels.parquet`, `folds.yaml`, `class_distribution.csv` |
| 4 Images | `python scripts/04_check_images.py` | shape/NaN checks, `results/<run>/figures/sample_images_fold0.png` |
| 5 Training | `python scripts/05_train.py [--fold 0 1]` | `results/<run>/fold_XX/seed_S/{checkpoint.pt, history.csv, predictions.parquet}` |
| 6 Evaluation | `python scripts/06_evaluate.py` | `results/<run>/evaluation/metrics_*.csv`, confusion matrices |
| 7 XAI | `python scripts/07_xai.py [--fold ..] [--aggregate-only]` | `results/<run>/xai/seed_S/fold_XX/attributions_<method>.npz`, `summary/` rankings, recency, agreement, figures |
| 8 Backtest | `python scripts/08_backtest.py` | `results/<run>/backtest/backtest_per_ticker_seedS.csv`, `backtest_summary_seedS.csv`, `equity_seedS.parquet` |
| 9 Rule baseline | `python scripts/09_rule_baseline.py --set run_id=rule_...` | no-model predictions (close = lowest / highest of the last 6 days) on the ViT's test samples; evaluate with 06 and 08 |

Every script accepts overrides such as `--set train.max_epochs=1 data.max_tickers=3 run_id=test`.
Smoke run (CPU, a few minutes):

```bash
S="paths.data_dir=data_smoke run_id=smoke data.max_tickers=3 train.max_epochs=2 model.embed_dim=48 model.depth=2 model.num_heads=4 xai.samples_per_class=8"
python scripts/01_download.py --set $S   # then 02 ... 07 with the same --set; use --fold 0 for 05 and 07
```

## Design decisions

- **Trading calendar**: a weekday counts as a session if most stocks traded with volume *or* the ^NSEI close
  moved. Yahoo's index series misses real sessions (e.g. several Jan 1 sessions), and on some real sessions every
  stock row is a zero-volume placeholder (e.g. 2025-03-18). Weekend special sessions (Muhurat, Budget Saturday)
  are merged into the next session.
- **Cleaning**: a single missing session becomes a flat bar at the previous close. A placeholder row has zero volume
  and O=H=L=C equal to the previous close. Both are kept so indicators stay continuous, but they are flagged
  `synthetic`. Longer gaps are dropped. A zero-volume row with real prices keeps its prices, and its volume is
  forward-filled.
- **Labels**: a label is kept only if it is a one-session return between two real price rows. Labels that touch a
  synthetic day or span a dropped gap are removed. `data/big_moves.csv` lists every daily move above 20% for
  manual review, because Yahoo does not adjust for demergers (e.g. ADANIENT 2015-06-03, −38.7%).
- **Image**: shape (1, 90, 65). Rows follow the frozen order in `configs/indicators.yaml`, grouped by category
  (overlap 18, momentum 38, volume 6, volatility 8, price transform 4, statistics 16). Columns run from day t−64 to t,
  oldest on the left. The PDF appendix only adds up to 87 indicators, so **MOM 5, CMO 28 and ADOSC 5/20** were added.
- **Relative form**: price-level indicators are divided by the close (e.g. SMA20/Close − 1, ATR/Close, VAR/Close²).
  OBV and AD are differenced and divided by 20-day average volume. This lets indicators be standardized across stocks
  with very different price levels. BETA_20 is the stock's beta against ^NSEI. TA-Lib's `BETA(a, b)` regresses `b`
  on `a`, so the index is passed first.
- **Standardization**: per indicator, fitted on each fold's training dates pooled across stocks, clipped to ±5, and saved
  as `norm_stats.npz`.
- **Folds**: 5-year training window (the last 6 months are validation for early stopping), then 1 test year, rolling
  forward 1 year. Test years run from 2013 to 2025. A sample is used only if its label day is in the same segment
  (1-day embargo).
- **Model**: timm `VisionTransformer`, img (90, 65), patch (10, 13) → 45 tokens, embed 192, depth 6, heads 6.
  AdamW with cosine schedule, cross-entropy, early stopping on validation loss. `model.drop_rate` is applied after
  the position embedding, attention projection and MLP, and before the head. In timm, `drop_rate` alone would only
  cover the head.
- **Metrics**: computed only from `predictions.parquet`. The code asserts `accuracy == trace(CM)/sum(CM)` and reports
  majority-class accuracy alongside.
- **XAI**, all returning (N, 90, 65) maps:
  - `integrated_gradients`: Captum, with an all-zero baseline (= training mean).
  - `row_occlusion`: each indicator row is set to 0, and the score is the drop in predicted-class probability.
  - `chefer`: LRP port of hila-chefer/Transformer-Explainability (`src/vitxai/models/vit_lrp.py`, MIT). It loads
    the timm weights directly, and a test checks its logits match timm's.
  - Maps are converted to absolute values, normalized to sum 1 and averaged per predicted class. Row sums rank the
    indicators, column sums give the recency profile, and Spearman correlation measures agreement between methods.
  - `signed_<class>` columns keep the sign: a negative value means the indicator pushed *against* the class on
    average. Use them before calling an indicator a driver of Buy or Sell.
  - `xai.target` selects the class whose score is explained: `predicted` (default) or `actual`.
  - **Chefer resolution**: Chefer scores are per patch, and each patch covers 10 indicator rows, so all 10 rows get
    the same score. Tied rows share a rank. Integrated Gradients and row occlusion rank single indicators.
- **Universe**: current NIFTY 50 constituents with full 2008–2025 history. This introduces survivorship bias, which
  should be stated as a limitation. Verify the ticker list in `configs/base.yaml` against NSE before the final run.

## Base-paper replication (`configs/paper_etf`)

Same scripts, different config folder: add `--config-dir configs/paper_etf` to every command (01 to 08).

| | Paper (Gezici & Sefer 2024) | This config |
|---|---|---|
| Data | 9 ETFs (XLF, XLU, QQQ, SPY, XLP, EWZ, EWH, XLY, XLE), 2002 to 2022, one joint model | same, from Yahoo; `data_paper_etf/` |
| Image | 65 indicators (Table 2) x 65 days, rows in Table 2 category order | same function list and order |
| Labels | theta 0.01 (main), 0.0038 (balanced) | both; separate label files in one data folder |
| Folds | 5 training years, test each year 2007 to 2021 | same; last 6 months of training used for early stopping |
| Backtest | $10,000, all-in long on Buy, exit on Sell, repeats ignored, $1 per trade | `08_backtest.py`, trades at the signal day's close, compared with buy and hold |

Not stated in the paper, so filled in here:
- **Indicator periods**: 14 days for period-based indicators (the paper says one to three weeks). MA uses 7 days, SMA 10 days,
  and FASTK/FASTD a 5-day %K, because with 14 days they would be exact copies of BBANDS middle and SLOWK.
- **Relative form**: price-level indicators are divided by the close, as in the main setup. Raw price levels would make
  standardization across ETFs with different prices meaningless.
- **Standardization**: fitted on training dates only. The paper does not say, and fitting on all dates would leak test data.
- **BETA / CORREL** against the S&P 500 (^GSPC). **ViT**: the main model settings with a 13 x 13 patch (25 tokens).

Balanced run (reuses the downloaded data and indicators, so run 03 onwards):
```bash
T="labels.theta=0.0038 labels.file=labels_theta0038.parquet run_id=paper_vit_theta0038"
python scripts/03_labels_folds.py --config-dir configs/paper_etf --set $T   # then 04 ... 08 with the same --set
```
Peak/valley labels (CNN-TA rule: day t is Buy if its close is the lowest of days t-5..t+5, Sell if the highest):
```bash
V="labels.method=peak_valley labels.window=11 labels.file=labels_pv11.parquet run_id=paper_vit_pv11"
```
The paper's text describes the threshold rule, but its confusion matrices (1,094 Hold / 75 Buy / 75 Sell) do not fit it:
at theta 0.01 the ETFs are 41-81% Hold over 2007-2021, while the 11-day peak/valley rule gives about 88% / 6% / 6% on
every ETF and every test year. The label uses 5 future closes, so folds embargo 5 days (`label_end`). A no-model rule
(Buy if today's close is the lowest of the last 6 days, Sell if the highest) already gets macro-F1 0.51 on these labels;
report it next to the ViT.

Compare runs on macro-F1, per-class recall and backtest against buy and hold, not accuracy alone. At theta 0.01, predicting
Hold every day already gives high accuracy. A model that never trades also avoids every crash, so check how many
trades it made before calling it profitable.

## Extending

| To add | Where |
|---|---|
| Indicator / subset (e.g. base-paper 65) | new row in `configs/indicators.yaml` / `subsets:` + `active_subset`, and set `model.img_size[0]` |
| Different window (e.g. 90 days) | `image.window`, `model.img_size`, `model.patch_size` |
| Model (CNN-TA, LSTM, ...) | `src/vitxai/models/<name>.py` + `MODEL_REGISTRY` + `configs/model_<name>.yaml` (`model_config`) |
| Seeds | `train.seeds: [0, 1, 2]` (outputs already go to `seed_S/`) |
| Class weights | `train.class_weights: true` |
| XAI method (rollout, Grad-CAM, ...) | `src/vitxai/xai/<name>.py` with `explain(x, target) -> (N, H, W)` + `XAI_REGISTRY` + `xai.methods` |
| Top-k retraining, statistics | read the existing `predictions.parquet` and `attributions_*.npz`; no retraining needed |

## Layout

```
configs/      base.yaml  indicators.yaml  model_vit.yaml  train.yaml  xai.yaml  paper_etf/
src/vitxai/   config  seed  data/  features/  labels/  images/  models/  train/  eval/  xai/  viz/
scripts/      01_download ... 09_rule_baseline
tests/        indicators (causality, count, scale-free), labels/folds (leakage), dataset, metrics, xai, backtest, paper config
notebooks/    colab_run.ipynb
```
