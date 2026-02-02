# Clinical Survival Analysis for Primary Biliary Cirrhosis (PBC)

A deep learning project implementing advanced RNN architectures (LSTM and GRU-D) for survival analysis, predicting patient outcomes in Primary Biliary Cirrhosis using clinical data with missing values and irregular time intervals.

## Project Overview

This project develops a survival analysis model using Long Short-Term Memory (LSTM) neural networks to predict time-to-event outcomes for patients with Primary Biliary Cirrhosis. The model processes longitudinal clinical data to estimate patient risk over time.

## Objectives

- Build deep learning models for survival analysis on clinical data
- Predict time-to-event outcomes (death or transplant) for PBC patients
- Handle censored data and missing values (62.55% missingness)
- Address irregular time intervals in clinical visits
- Compare LSTM vs GRU-D architectures for temporal clinical data
- Evaluate model performance using concordance index (C-index)

## Dataset

The dataset contains clinical records from PBC patients with the following features:

### Patient Demographics
- **age**: Patient age in years
- **sex**: Gender (0=female, 1=male)
- **drug**: Treatment (1=D-penicillamine, 0=placebo)

### Clinical Measurements
- **serBilir**: Serum bilirubin concentration (mg/dl)
- **serChol**: Cholesterol concentration (mg/dl)
- **albumin**: Albumin concentration (mg/dl)
- **alkaline**: Alkaline phosphatase (U/liter)
- **SGOT**: Serum glutamic oxaloacetic transaminase (U/ml)
- **platelet**: Platelet count per cubic mm (÷1000)
- **prothrombin**: Prothrombin time (seconds)
- **total_protein**: Total protein concentration (mg/dl)

### Clinical Indicators
- **ascite**: Ascites presence (0/1)
- **hepatomegaly**: Liver enlargement (0/1)
- **spiders**: Spider angiomas (0/1)
- **edema**: Edema severity (0=none/mild, 1=moderate, 2=severe)
- **histologic**: Histological disease stage (1-4)

### Target Variables
- **tte**: Time to event (weeks from enrollment to event)
- **times**: Time to visit/measurement (weeks)
- **label**: Event tys

### 1. LSTM Risk Estimator (Baseline)

The baseline model implements a standard LSTM architecture with pre-imputation of missing values:

```python
LSTM_risk_estimator(
    input_size: int,           # Number of clinical features (15)
    hidden_size: int,          # LSTM hidden state dimension (64)
    num_layers: int,           # Number of LSTM layers (1)
    number_time_discrete: int  # Discrete time steps for risk estimation (745)
)
```

**Architecture Components:**
1. **LSTM Layer**: Processes sequential patient data to capture temporal patterns
2. **Fully Connected Layer**: Maps hidden states to risk probabilities
3. **Softmax Layer**: Converts outputs to probability distribution over discrete time steps

**Limitation:** Requires pre-imputation of missing values, losing informative missingness patterns.

### 2. GRU-D (Advanced Architecture)

GRU-D (Gated Recurrent Unit with Decay) explicitly models missing values and irregular time intervals:

```python
GRUD_risk_estimator(
    input_size: int,           # Number of clinical features (15)
    hidden_size: int,          # GRU hidden state dimension (64)
    num_layers: int,           # Number of GRU layers (1)
    number_time_discrete: int, # Discrete time steps (745)
    dropout: float             # Dropout rate (0.2)
)
```

**Key Features:**
1. **Temporal Decay Mechanism**: Models information decay over time using learnable parameters
2. **Missing Value Masking**: Explicit binary mask indicating which values are observed
3. **Time Delta Encoding**: Captures time elapsed since last observation
4. **Augmented Input**: [values, mask, time_delta] - triple input representation
5. **No Pre-imputation**: Learns optimal handling of missing data end-to-end

**Mathematical Foundation:**
- DeComparative Performance

| Metric | LSTM (Baseline) | GRU-D | Improvement |
|--------|-----------------|-------|-------------|
| **Test C-index** | 0.5912 | **0.8329** | **+40.9%** ✅ |
| **Train C-index** | 0.9769 | 0.9864 | +1.0% |
| **Generalization Gap** | 0.3856 | **0.1535** | **-60.2%** ✅ |
| **Parameters** | ~50k | 70k | +40% |
| **Training Time/Epoch** | 15s | 17s | +13% |
| **Pre-processing** | Imputation required | None | N/A |

### Key Findings

🎯 **GRU-D Achieves Superior Performance:**
- **Test C-index: 0.8329** - Excellent risk stratification (83% correct pairwise rankings)
- **Reduces overfitting by 60%** - Much better generalization to unseen patients
- **Handles 62.55% missing data** - Without pre-imputation

### Training Progress Comparison

**LSTM:**
- Initial C-index (Epoch 1): 0.3136 → Final: 0.9769
- Severe overfitting: Train-test gap of 0.3856
- Test C-index: 0.5912 (barely better than random)

**GRU-D:**
- Initial C-index (Epoch 1): 0.5934 → Final: 0.9864
- Controlled overfitting: Train-test gap of 0.1535
- Test C-index: 0.8329 (clinically excellent)

### Clinical Impact with censoring
2. **Time-Dependent C-index**: Manual implementation for model evaluation
3. **GRU-D Implementation**: Full implementation of temporal decay mechanisms
4. **Missing Value Handling**: Explicit masking and time delta encoding
5. **Censored Data Handling**: Proper treatment of censored observations (0), death (1), transplant (2)
6. **Comparative Analysis**: Systematic evaluation of LSTM vs GRU-D


## References

1. **Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018).** *Recurrent neural networks for multivariate time series with missing values.* Scientific Reports, 8(1), 6085.


