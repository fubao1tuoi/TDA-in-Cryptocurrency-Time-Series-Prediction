# Topological Structure Analysis of Cryptocurrency Time Series and Its Application in Market Volatility Prediction

Undergraduate Thesis - Major in Data Science in Economics and Business (DSEB).

---

## 📖 1. Overview

This project focuses on researching and developing a **Hybrid Framework** combining **Topological Data Analysis (TDA)** and advanced machine learning architectures (**XGBoost & LSTM**) to optimize multi-horizon volatility forecasting for the cryptocurrency market (Bitcoin).

By applying **Takens' Delay Embedding Theorem** from nonlinear dynamical systems theory, the 1D time series of log return (price), volume, and funding rate is reconstructed into a trajectory in a multidimensional phase space ($d=4$). From this point cloud, the **Persistent Homology** algorithm is deployed to extract invariant geometric features (topological invariants), including connected components ($H_0$) and fundamental loops ($H_1$). These topological features (quantified via *Topological Entropy* and *Wasserstein Amplitude*) act as a sophisticated geometric regularizer, helping the system early detect topological fragility states and structural reversal points of the market—which traditional technical indicators often miss.

### 👥 Author & Research Information

* **Student:** Phung Gia Bao (Student ID: 11220814)
* **Class:** Data Science in Economics and Business 64B (DSEB64B)
* **Faculty:** Mathematical Economics - National Economics University (NEU)
* **Supervisor:** PhD. Nguyen Manh Toan
* **Completion Date:** May 2026

---

## 📊 2. Data Pipeline & Sources

The dataset is collected at a **Daily** frequency, extending from **August 17, 2017, to March 31, 2026**, covering multiple eras of extreme volatility, explosive growth phases, and deep recessions (liquidation cascades) of the Crypto market.

### Data Partitioning:

* **Development Set (Train/Validation):** From `17/08/2017` to `31/12/2024`. This dataset is structured via the **Expanding Window Walk-Forward Validation** method for the XGBoost model to strictly prevent data leakage, and a static `90/10` split combined with Early Stopping for the LSTM network.
* **Hold-out Test Set:** From `01/01/2025` to `31/03/2026`. A completely "blind" environment, strictly sealed, used only to evaluate the final out-of-sample generalization capability of the models.

### Integrated Data Sources:

1. **Market Microstructure Data (Binance Spot API):** Standard OHLCV price series supplemented with *Quote Asset Volume* (Total transaction value in USDT) and *Number of Trades*.
2. **Derivatives Market (Binance Futures API):** *Funding Rate* aggregated daily via mean-aggregation, serving as a proxy for leverage levels and long/short position imbalance.
3. **Crowd Sentiment (Alternative.me):** Normalized *Crypto Fear and Greed Index (FnG)*.
4. **Exogenous Attention (Google Trends API):** Search volume score for the keyword `"Bitcoin"`, applying a strict *forward-filling* technique to preserve temporal causality without introducing look-ahead bias.

---

## 🧠 3. Technical Architecture & Approach

### 3.1. TDA Pipeline

The entire process is multi-thread parallelized using the `giotto-tda` library:

* **Multivariate Phase Space Embedding:** Embedding the target matrix consisting of log-returns, funding rate, and quote asset volume with embedding dimension $d=4$ and time delay $\tau=1$.
* **Sliding Window:** Fragmenting the phase space trajectory into local point clouds over a 30-day period ($w=30$).
* **Vietoris-Rips Complex Filtration:** Expanding the proximity radius $\epsilon$ around data points to form high-order simplices ($k$-simplices).
* **Topological Vectorization:** Extracting Persistence Diagrams into two scalar metrics:
* **Wasserstein Amplitude:** Represents the geometric magnitude and robustness of market cycles (especially in the $H_1$ dimension).
* **Topological Entropy:** Measures the complexity and fragmentation level of the order book structure.


* **Topological Kinematics:** Calculating the rate of change (Deltas and ROC) of $H_1$ amplitude and $H_0$ entropy over 7, 14, and 30-day milestones to create nonlinear interaction variables with actual price trends.

### 3.2. Forecasting Model Design

#### 🌲 Extreme Gradient Boosting (XGBoost)

* Applying the histogram-based algorithm (`tree_method='hist'`) to accelerate computation on the flattened data space unrolled over time.
* Strict constraints to prevent noise: `max_depth=5`, `min_child_weight=3`, subsample and colsample_bytree ratios fixed at **0.8**.
* Hyperparameter optimization via Bayesian analysis using **Optuna (Tree-structured Parzen Estimator - TPE)** running on a walk-forward cross-validation infrastructure.

#### 🐋 Long Short-Term Memory (LSTM) Network

* Using a **Stacked LSTM** structure processing 3D data tensors of shape `(Batch Size, Sequence Length, Feature Dimension)`.
* Applying the **Variational Dropout** mechanism (maintaining a single Bernoulli mask across all recurrent time steps) to preserve long-term memory continuity and avoid memory vanishing.
* Integrating a custom asymmetric loss function, **CryptoLoss**:

$$\mathcal{L}_{Crypto} = \mathcal{L}_{Huber}(y, \hat{y}) + \lambda_d \cdot \mathbb{E}[ReLU(-y \cdot \hat{y})] + \lambda_v \cdot |\sigma(y) - \sigma(\hat{y})|$$



Where:
* $\mathcal{L}_{Huber}$ protects the model from outliers.
* Directional penalty $\lambda_d$: Heavily penalizes wrong predictions of the log-return sign (wrong trend direction).
* Volatility-preserving penalty $\lambda_v$: Eliminates the "flatline" phenomenon (a lazy model predicting a flat mean of 0), forcing the network's output to have an oscillation amplitude matching market reality.



---

## 📈 4. Out-of-Sample Experimental Results (2025 - 2026 Period)

Below is a summary table of the evaluation results on the completely independent hold-out Test set from January 1, 2025, to March 31, 2026:

| Forecasting Horizon | Model Architecture | Finance-Only RMSE | Hybrid Finance + TDA RMSE | Diebold-Mariano Stat | Statistical Significance ($p$-value) |
| --- | --- | --- | --- | --- | --- |
| **1 Day** | XGBoost | **0.0247** | 0.0257 | -1.8292 | $p = 0.0674$ (Baseline Preferred)* |
| **1 Day** | LSTM | 0.0402 | **0.0323** | **5.3639** | $p < 0.0001$ (Hybrid Superior)*** |
| **7 Days** | XGBoost | 0.0719 | **0.0687** | 0.5753 | $p = 0.5651$ (Insignificant) |
| **7 Days** | LSTM | 0.1163 | **0.0916** | **1.9903** | $p = 0.0466$ (Hybrid Superior)** |
| **14 Days** | XGBoost | **0.0900** | 0.0966 | -1.1629 | $p = 0.2449$ (Insignificant) |
| **14 Days** | LSTM | 0.1354 | **0.1057** | 1.5277 | $p = 0.1266$ (Insignificant) |

### 🔍 Experimental Conclusions:

1. **Architectural Divergence:** Local space partitioning algorithms like XGBoost perform best at short-term horizons when using raw tabular features; embedding continuous topological structures easily introduces noise into the decision trees. Conversely, recurrent sequential memory networks like **LSTM show perfect compatibility with TDA features**.
2. **The Sweet Spot (7 Days):** The 7-day timeframe witnesses a performance explosion of the Hybrid LSTM model, reducing RMSE by **21.2%**. A DM test achieving statistical significance below the **5%** mark proves that topological information truly provides superior forecasting alpha, helping the model completely eliminate "trend hallucination" phases (falsely predicting an upward trend) during black swan crashes (e.g., the massive crash in February 2026).
3. **Macro Limit (14 Days):** At the extended 14-day horizon, although TDA helps effectively restructure the oscillation amplitude for the LSTM (reducing RMSE from 0.1354 to 0.1057), the extreme randomness and unstructured shocks cause the models' confidence intervals to overlap, pushing the $p$-value to an out-of-sample statistically insignificant level.

---

## 📂 5. Project Directory Structure

```text
├── DB/                             # Directory to store forecast result csv files for DB test (Diebold-Mariano) execution
├── src/                            # Directory containing the core source code of the pipeline system
│   ├── augmentation.py             # Data Augmentation Module (adds Gaussian noise to prevent overfitting)
│   ├── config.py                   # System parameter configuration, hyperparameters, and directory paths
│   ├── data_utils.py               # Utilities for processing, cleaning data, and preparing Dataloaders
│   ├── dl_model.py                 # Deep Learning architecture definition (Stacked LSTM, CryptoLoss)
│   ├── fe.py                       # Feature Engineering
│   ├── ml_model.py                 # Machine Learning architecture definition (XGBoost)
│   └── train_utils.py              # Training support utilities (Early Stopping), optimization, and evaluation
├── BTC.ipynb                       # Independent analysis notebook or local test run for the BTC dataset
├── DB test.ipynb                   # Notebook executing the DB statistical test (Diebold-Mariano test)
├── btc_dataset.csv                 # Input Bitcoin dataset
├── btc_full.csv                    # Full Bitcoin dataset
├── main_all_features.ipynb         # Notebook running the entire pipeline for the Hybrid model (Finance + TDA)
├── main_finance_features.ipynb     # Notebook running the pipeline for the Baseline model (Finance-Only)
├── requirements.txt                # List of required library dependencies for the project environment
└── README.md                       # This project documentation

```
