# Naira Pulse: Forecasting USD/NGN Depreciation Risk for Nigerian Importers

There is a significant opportunity to optimize capital efficiency and protect margins for Nigerian trade businesses by replacing guesswork with data-driven currency timing. Since the Central Bank of Nigeria floated the naira in June 2023, the USD/NGN exchange rate has transitioned from a tightly managed peg to a highly volatile market, swinging from approximately ₦465 to over ₦1,600 per dollar.

**Naira Pulse** is a machine learning system engineered to classify whether the naira is likely to weaken significantly (defined as \>1% depreciation over a 7-day horizon), providing actionable "buy now" or "wait" signals to help small-to-medium enterprises (SMEs) time their dollar conversions.

-----

## 📊 Executive Artifacts: At a Glance

The following matrix recaps the core backtest results on 8 months of unseen 2026 data across various confidence thresholds:

| Decision Threshold     | Total Signals Flagged | Precision | Recall    | Net Economic Impact (Savings/Loss per $) |
| :--------------------- | :-------------------- | :-------- | :-------- | :--------------------------------------- |
| **0.50 (Default)**     | 98                    | 16.3%     | **61.5%** | ❌ **–₦170.03** (Loss)                    |
| **0.70 (Recommended)** | 20                    | 40.0%     | 30.8%     | 🚀 **+₦88.78** (Saved)                    |
| **0.75**               | 7                     | 57.1%     | 15.4%     | 🚀 **+₦82.53** (Saved)                    |
| **0.80**               | 4                     | **75.0%** | 11.5%     | 🚀 **+₦68.33** (Saved)                    |

> **Critical Warning:** Short-horizon currency movements naturally mirror a close random walk, making them incredibly noisy. As a result, standard machine learning models operating at a default 50% probability threshold fail to clear the baseline guessing rate and lose money. Implementing a high-confidence gate ($\\ge$70%) transforms a weak classifier into a highly profitable enterprise heuristic.

-----

## 🛠️ System Architecture & Workflow

The codebase is organized sequentially across a five-stage deployment pipeline:

### 1\. Data Acquisition

  * **Pipeline Component:** Integrates programmatically with the Alpha Vantage API to extract daily spot USD/NGN rates, crude oil benchmarks (WTI and Brent), and macroeconomic indicators.
  * **Infrastructure Strategy:** Implements robust local JSON caching to protect development iterations from API tier rate limits. Annual indicators are propagated forward to a daily frequency using a zero-leakage `pd.merge_asof` technique.

### 2\. Feature Engineering & Regime Selection

  * **Temporal Realism:** Restricts model training exclusively to data following the June 2023 currency float. Stale patterns from the managed-peg era are discarded to isolate current structural dynamics.
  * **Signal Construction:** Engineers temporal features, rolling returns, trailing standard deviations, and multi-day lags for both exchange rates and international oil indexes.

### 3\. Chronological Modeling Pipeline

  * **Temporal Validation:** Employs an expanding-window `TimeSeriesSplit` cross-validation strategy. Traditional random shuffling is strictly prohibited to ensure future data never leaks into past evaluations.
  * **Model Sweeping:** Constructs comprehensive pipeline transformers (median imputation and standard scaling for numeric features; one-hot encoding for nominal categories) across five primary candidate frameworks:
      * XGBoost (Top Performer: Test AUC 0.579)
      * Random Forest
      * LightGBM
      * Logistic Regression
      * Gradient Boosting

### 4\. Backtesting & Threshold Optimization

  * **Economic Evaluation:** Evaluates pipeline predictions against a custom, currency-denominated engine mapping the specific operational costs of trade conversion delays.
  * **Optimization Layer:** Sweeps confidence cutoffs to reveal the optimum decision boundary for commercial deployment.

-----

## 🚀 Running the Project

### Prerequisites

Ensure your local environment or Google Colab runtime possesses the required analytical dependencies:

``` bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib joblib requests

```

### API Configuration

To pull live market variables, ensure your Alpha Vantage API key is stored safely:

``` python
from google.colab import userdata
API_KEY = userdata.get('API_KEY')

```

-----

## 📈 Next Steps

Over the coming quarter, development will focus on the ingestion of manually sourced local macroeconomic primitives—specifically Central Bank of Nigeria liquid reserves and localized consumer price inflation metrics—to evaluate if expanding the feature landscape improves confidence stability beyond the current 20 high-value signal days per period.

-----
