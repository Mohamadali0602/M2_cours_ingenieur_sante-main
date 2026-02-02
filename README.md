# Clinical Survival Analysis for Primary Biliary Cirrhosis (PBC)

A deep learning project implementing LSTM-based survival analysis for predicting patient outcomes in Primary Biliary Cirrhosis using clinical data.

## Project Overview

This project develops a survival analysis model using Long Short-Term Memory (LSTM) neural networks to predict time-to-event outcomes for patients with Primary Biliary Cirrhosis. The model processes longitudinal clinical data to estimate patient risk over time.

## Objectives

- Build a deep learning model for survival analysis on clinical data
- Predict time-to-event outcomes (death or transplant) for PBC patients
- Handle censored data typical in survival analysis
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
- **label**: Event type (0=censored, 1=death, 2=transplant)

## Model Architecture

### LSTM Risk Estimator

The model implements a custom LSTM-based architecture for survival analysis:

```python
LSTM_risk_estimator(
    input_size: int,           # Number of clinical features
    hidden_size: int,          # LSTM hidden state dimension
    num_layers: int,           # Number of LSTM layers
    number_time_discrete: int  # Discrete time steps for risk estimation
)
```

**Architecture Components:**
1. **LSTM Layer**: Processes sequential patient data to capture temporal patterns
2. **Fully Connected Layer**: Maps hidden states to risk probabilities
3. **Softmax Layer**: Converts outputs to probability distribution over discrete time steps

**Loss Function:** Negative Log-Likelihood for Survival Data (NLLSurv)

## Implementation Details

### Data Preprocessing
- Data cleaning and handling missing values
- Feature normalization and standardization
- Creation of temporal sequences for LSTM input
- Stratified train-test split to maintain event distribution

### Training Configuration
- **Epochs**: 100
- **Loss Function**: Custom NLLSurv (Negative Log-Likelihood for Survival)
- **Evaluation Metric**: Time-dependent Concordance Index (C-index)
- **Data Split**: Stratified to preserve event distribution

### Evaluation Metric

**Concordance Index (C-index):** Measures the model's ability to correctly rank patients by risk. A C-index of 0.5 indicates random predictions, while 1.0 indicates perfect ranking.

## Results

### Final Model Performance

| Metric | Value |
|--------|-------|
| **Final Training C-index** | 0.9769 |
| **Test C-index** | 0.5912 |
| **Final Training Loss** | 0.3928 |
| **Mean Loss (100 epochs)** | 0.5432 |
| **Number of Epochs** | 100 |

### Training Progress

The model showed strong learning on the training set:
- Initial C-index (Epoch 1): 0.3136
- Final C-index (Epoch 100): 0.9769
- The training loss decreased consistently from 1.3918 to 0.3928

### Analysis

 **Strengths:**
- Excellent performance on training data (C-index: 0.9769)
- Consistent loss reduction throughout training
- Successful implementation of custom survival analysis loss function

 **Observations:**
- Gap between training (0.9769) and test (0.5912) C-index suggests overfitting
- Test C-index of 0.5912 indicates moderate predictive ability on unseen data

## Technical Highlights

1. **Custom Loss Function**: Implemented NLLSurv specifically for survival analysis
2. **Time-Dependent C-index**: Manual implementation for model evaluation
3. **Sequential Data Processing**: LSTM architecture to capture temporal patterns
4. **Censored Data Handling**: Proper treatment of censored observations in survival data

## 👤 Author

Mohamad Ali  
Master 2 - AI and Language Engineering in Health Sciences

**Note:** This project demonstrates the application of deep learning techniques to clinical survival analysis, a crucial area in medical research and patient care management.


