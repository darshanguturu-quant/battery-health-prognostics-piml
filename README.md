# Battery Health Monitoring & Prognostics (PIML)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NASA PCoE](https://img.shields.io/badge/Dataset-NASA%20PCoE-0B3D91.svg)](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip)
[![PIML](https://img.shields.io/badge/Architecture-Physics--Informed%20ML-brightgreen.svg)]()

> **An end-to-end Physics-Informed Machine Learning (PIML) pipeline for lithium-ion battery State of Health (SOH) estimation and Remaining Useful Life (RUL) prognostics with Bayesian uncertainty quantification.**

---

## 📌 Executive Summary

Accurate estimation of **State of Health (SOH)** and prognostication of **Remaining Useful Life (RUL)** in lithium-ion batteries are critical for electric vehicles (EVs), grid energy storage, and mission-critical aerospace systems. Pure data-driven models frequently fail out-of-distribution or violate physical laws by predicting non-physical capacity recoveries, while purely empirical electrochemical models often lack adaptability across variable operational profiles.

This repository implements a **hybrid Physics-Informed Machine Learning (PIML)** framework that integrates empirical Solid Electrolyte Interphase (SEI) growth kinetics directly into neural sequence models and comparative regression baselines. Tested on the **NASA Prognostics Center of Excellence (PCoE)** battery dataset and a calibrated synthetic fleet, the system provides reliable SOH tracking, rigorous uncertainty bounds, and a robust 4-tier RUL fallback engine.

---

## 🚀 Key Features

- **Physics-Informed LSTM (PI-LSTM)**: Custom PyTorch recurrent neural network enforcing a soft monotonicity penalty on irreversible capacity fade during training.
- **Epistemic Uncertainty Quantification**: Monte Carlo (MC) Dropout inference producing calibrated $\pm 2\sigma$ predictive uncertainty envelopes.
- **Empirical SEI Fade Law**: Parametric non-linear least squares fitting ($a, b, c$) capturing short-term exponential SEI passivating growth and long-term linear active material loss.
- **Robust 4-Method RUL Estimation**: A failure-proof RUL calculator preventing `N/A` outcomes using:
  1. *Direct EOL Crossing* ($SOH < 80\%$)
  2. *Global Trajectory Extrapolation*
  3. *Tail-Gradient Projection*
  4. *Analytical Physics Law Fallback*
- **Comparative Multi-Model Benchmark**: Side-by-side evaluation across **Analytical Physics**, **Ridge Regression**, **Support Vector Regression (SVR)**, **Random Forest**, and **PI-LSTM**.
- **Automated Data Pipeline**: Seamless retrieval of NASA PCoE `.mat` files with autonomous fallback to a physics-calibrated synthetic degradation simulator with regeneration events.
- **Comprehensive Visual Suite**: Generates 8 publication-ready diagnostic charts and automated JSON metrics.

---

## 📐 Mathematical & Physical Formulations

### 1. SEI Growth & Capacity Fade Law
The degradation kinetics governing long-term capacity fade are parameterized as:

$$\text{SOH}(n) = 1 - a \cdot \left(1 - e^{-b \cdot n}\right) - c \cdot n$$

Where:
- $n$: Operating cycle index
- $a$: Dimensionless scale parameter representing early-life SEI film formation magnitude
- $b$: SEI growth rate constant ($\text{cycle}^{-1}$)
- $c$: Linear aging coefficient accounting for structural lithium inventory loss and particle fracture

### 2. Physical Impedance & Discharge Dynamics
Internal resistance $R(n)$ and effective discharge duration $t_{\text{discharge}}(n)$ are modeled as coupled functions of SOH:

$$R(n) = R_0 \cdot \left[1 + 1.8 \cdot (1 - \text{SOH}(n))\right] + \epsilon_R, \quad \epsilon_R \sim \mathcal{N}(0, \sigma_R^2)$$

$$t_{\text{discharge}}(n) = 3600 \cdot \left(\frac{Q(n)}{I_{\text{rated}}}\right) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_t^2)$$

### 3. Physics-Informed Monotonicity Penalty
To eliminate non-physical capacity gains during continuous discharge cycling, a directional penalty is incorporated into the training loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(y, \hat{y}) + \lambda_{\text{phys}} \cdot \frac{1}{B - 1} \sum_{i=1}^{B-1} \left( \max\left(0, \Delta \hat{y}_{i, i+1} - \delta_{\text{regen}}\right) \right)^2$$

Where:
- $\Delta \hat{y}_{i, i+1} = \hat{y}_{i+1} - \hat{y}_i$
- $\delta_{\text{regen}} = 0.01$ allows minor physical relaxation recoveries
- $\lambda_{\text{phys}} = 0.1$ controls the strength of the physics regularization

### 4. Epistemic Uncertainty Quantification via MC Dropout
Using Monte Carlo Dropout over $M = 30$ stochastic forward passes:

$$\mu(x) = \frac{1}{M}\sum_{m=1}^{M} f_{\hat{W}_m}(x)$$

$$\sigma^2(x) = \frac{1}{M}\sum_{m=1}^{M} \left[f_{\hat{W}_m}(x) - \mu(x)\right]^2$$

The $95\%$ credible interval is bounded by $\mu(x) \pm 2\sigma(x)$.

---

## 🏗️ Architecture & Pipeline Flow

```mermaid
flowchart TD
    A[Data Ingestion<br>NASA PCoE .mat or Synthetic Fleet] --> B[Feature Engineering & Temporal Split<br>60% Train | 20% Val | 20% Test]
    B --> C1[Physics Fade Law Fitting<br>Curve Fit & Brentq]
    B --> C2[Statistical Baselines<br>Ridge, SVR, Random Forest]
    B --> C3[Sliding Window Generation<br>Seq Len = 20 Cycles]
    C3 --> D[Physics-Informed LSTM Training<br>Loss = MSE + Monotonicity Penalty]
    
    C1 --> E[RUL Estimation Engine<br>4-Tier Fallback Resolver]
    C2 --> E
    D --> E
    
    D --> F[MC Dropout Inference<br>30 Stochastic Passes]
    F --> G[Uncertainty Envelope<br>±2σ Credible Interval]
    
    E --> H[Consolidated Evaluation & Analytics]
    G --> H
    H --> I[8 Diagnostic Plots & metrics.json]
```

---

## 📊 Comparative Model Leaderboard

| Model Architecture | Type | Regularization / Constraints | Uncertainty Bounds | SOH MAE | RUL Capability |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Physics Fade Law** | Parametric Analytical | SEI Kinetic Bounds | Analytical Extrapolation | Baseline | Brentq Root Finding |
| **Ridge Regression** | Linear ($L_2$) | $L_2$ Weight Penalty ($\alpha=1.0$) | N/A | Low | Tail Extrapolation |
| **Support Vector Regressor** | Non-linear Kernel | RBF Kernel ($C=10, \epsilon=0.005$) | N/A | Competitive | Tail Extrapolation |
| **Random Forest Regressor** | Tree Ensemble | 200 Estimators, Max Depth 10 | Tree Variance | Low | Stepwise Projection |
| **Physics-Informed LSTM** | Deep Recurrent Neural Net | Monotonicity Regularization ($\lambda=0.1$) | **Yes (MC Dropout $\pm 2\sigma$)** | **Optimal** | **4-Method Fallback** |

---

## 📈 Generated Artifacts & Diagnostics

Running the pipeline automatically outputs the following publication-quality artifacts:

| Artifact | File Name | Description |
| :--- | :--- | :--- |
| **Exploratory Data Analysis** | `01_eda.png` | Capacity fade profiles, Pearson correlation heat-rankings, and resistance growth dynamics. |
| **Physics Fade Fit** | `02_physics_fit.png` | Non-linear regression of SEI kinetics with extracted parameters ($a, b, c$) per battery. |
| **Test Set Forecasting** | `03_test_predictions.png` | Ground-truth SOH vs. Ridge, SVR, RF, Physics, and PI-LSTM with $\pm 2\sigma$ uncertainty bands. |
| **Learning Dynamics** | `04_training_curve.png` | Evolution of training and validation loss under the composite physics penalty. |
| **Benchmark Leaderboard** | `05_model_comparison.png` | Comparative bar chart ranking all candidate models by Test Mean Absolute Error (MAE). |
| **Feature Attribution** | `06_feature_importance.png` | Gini importance highlighting contributions of physical priors vs. electrical measurements. |
| **RUL Diagnostics** | `07_rul_comparison.png` | True RUL vs. Analytical Physics RUL vs. PI-LSTM predicted RUL across test batteries. |
| **Executive Dashboard** | `08_dashboard.png` | Consolidated multi-panel diagnostic report combining degradation, error, and prognostics. |
| **Metrics Serialization** | `metrics.json` | Exact numerical values for MAE, RMSE, and cycle errors across all test partitions. |
| **Inference Samples** | `example_cases.json` | 3 end-to-end serialized cycle test cases detailing inputs, predictions, and uncertainty. |

---

## 📂 Repository Structure

```
battery-health-prognostics-piml/
│
├── battery_health_prognostics.ipynb   # Main interactive notebook (reproducible workflow)
├── battery_no_gradio (1).ipynb        # Original source notebook
├── requirements.txt                   # Production dependencies
├── .gitignore                         # Python, Jupyter, and dataset ignore rules
├── LICENSE                            # MIT Open Source License
└── README.md                          # Detailed documentation
```

---

## 🛠️ Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/darshanguturu-quant/battery-health-prognostics-piml.git
cd battery-health-prognostics-piml
```

### 2. Create Virtual Environment
```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline

#### In Jupyter Notebook / Lab:
```bash
jupyter notebook battery_health_prognostics.ipynb
```
Select **"Run All Cells"** from the menu.

#### In Google Colab:
1. Open [Google Colab](https://colab.research.google.com/).
2. Select the **GitHub** tab and enter: `https://github.com/darshanguturu-quant/battery-health-prognostics-piml`
3. Click on `battery_health_prognostics.ipynb`.
4. Run all cells sequentially. The notebook will automatically download the NASA dataset (or initialize the synthetic fleet) and package all results into `battery_results.zip`.

---

## ⚙️ Configuration Parameters

Key pipeline parameters are centralized in **Cell 2** and can be customized:

```python
EOL_THRESHOLD = 0.80   # SOH below 80% marks End of Life
TRAIN_FRAC    = 0.60   # 60% historical cycles used for model fitting
VAL_FRAC      = 0.20   # 20% validation split (remaining 20% for future testing)

SEQ_LEN       = 20     # Lookback window for temporal LSTM sequences
HIDDEN_SIZE   = 64     # LSTM recurrent hidden units
NUM_LAYERS    = 2      # Stacked recurrent layers
DROPOUT       = 0.20   # Dropout probability for regularized training & MC sampling
LR            = 1e-3   # Adam optimizer learning rate
EPOCHS        = 100    # Maximum epochs with early stopping patience
PHYS_LAMBDA   = 0.10   # Regularization weight for monotonicity constraint
MC_SAMPLES    = 30     # Number of stochastic forward passes for uncertainty bounds
```

---

## 🔬 Dataset Acknowledgments

- **NASA Prognostics Center of Excellence (PCoE)**: B. Saha and K. Goebel (2007). *"Battery Data Set"*, NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Dharshan Guturu**  
GitHub: [@darshanguturu-quant](https://github.com/darshanguturu-quant)
