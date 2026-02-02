import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch


def build_grud_tensor(df, id_col, time_col, tte_col, event_col, 
                      feature_continuous_cols, features_binary_cols):
    """
    Build tensors for GRU-D: features, masks, and time deltas.
    
    This function preserves missing values and tracks temporal information,
    which is essential for the GRU-D architecture.
    
    Args:
        df (pd.DataFrame): DataFrame with longitudinal data (with original missing values)
        id_col (str): Patient ID column name
        time_col (str): Visit time column name (e.g., 'times' in weeks)
        tte_col (str): Time to event column name
        event_col (str): Event indicator column name
        feature_continuous_cols (list): List of continuous feature names
        features_binary_cols (list): List of binary feature names
    
    Returns:
        X (np.ndarray): Feature tensor (n_patients, max_visits, n_features)
        M (np.ndarray): Mask tensor (n_patients, max_visits, n_features) - 1=observed, 0=missing
        T (np.ndarray): Time delta tensor (n_patients, max_visits, n_features) - time since last observation
        Y (np.ndarray): Target tensor (n_patients, 2) - [time_to_event, event_indicator]
        feature_means (np.ndarray): Empirical means for each feature (for decay mechanism)
    """
    
    features_cols = feature_continuous_cols + features_binary_cols
    
    # Compute empirical means BEFORE any imputation
    # These will be used by the decay mechanism in GRU-D
    feature_means = df[features_cols].mean().values
    
    print("Building GRU-D tensors:")
    print(f"  - Computing feature means for decay mechanism...")
    print(f"  - Feature means: {dict(zip(features_cols, np.round(feature_means, 3)))}")
    
    # Create scaled copy - normalize continuous features
    # IMPORTANT: Keep NaN as NaN for now
    df_scaled = df.copy()
    
    # Convert all feature columns to float to avoid dtype issues
    for col in features_cols:
        df_scaled[col] = df_scaled[col].astype(float)
    
    # Normalize continuous features (only on observed values)
    for col in feature_continuous_cols:
        mask = df_scaled[col].notna()
        if mask.any():
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(df_scaled.loc[mask, [col]]).flatten()
            df_scaled.loc[mask, col] = scaled_values
    
    # Binary features remain unchanged (but already converted to float)
    
    # Group by patient
    grouped = df_scaled.groupby(id_col)
    unique_ids = df[id_col].unique()
    n_samples = len(unique_ids)
    n_features = len(features_cols)
    max_len = grouped.size().max()
    
    print(f"  - Dataset shape:")
    print(f"    * {n_samples} patients")
    print(f"    * {max_len} max visits per patient")
    print(f"    * {n_features} features")
    
    # Initialize tensors
    X = np.zeros((n_samples, max_len, n_features))
    M = np.zeros((n_samples, max_len, n_features))  # Mask: 1=observed, 0=missing
    T = np.zeros((n_samples, max_len, n_features))  # Time delta since last observation
    Y = np.zeros((n_samples, 2))  # [time_to_event, event_indicator]
    
    # Process each patient
    for i, patient_id in enumerate(unique_ids):
        patient_data = grouped.get_group(patient_id).sort_values(time_col)
        
        # Track last observation time for each feature independently
        last_obs_time = {feat: 0.0 for feat in features_cols}
        
        seq_len = len(patient_data)
        
        for t in range(seq_len):
            visit_time = patient_data.iloc[t][time_col]
            
            for f, feat in enumerate(features_cols):
                value = patient_data.iloc[t][feat]
                
                # Check if value is missing
                is_missing = pd.isna(value)
                
                if not is_missing:
                    # Observed value
                    X[i, t, f] = value
                    M[i, t, f] = 1.0
                    
                    # Time since last observation of THIS specific feature
                    T[i, t, f] = visit_time - last_obs_time[feat]
                    
                    # Update last observation time for this feature
                    last_obs_time[feat] = visit_time
                else:
                    # Missing value
                    X[i, t, f] = 0.0  # Will be imputed by GRU-D's decay mechanism
                    M[i, t, f] = 0.0
                    
                    # Time since last observation of this feature
                    T[i, t, f] = visit_time - last_obs_time[feat]
        
        # Target: time to event and event indicator
        Y[i, 0] = patient_data[tte_col].max()
        Y[i, 1] = patient_data[event_col].max()
    
    # Compute statistics
    missing_rate = 1.0 - M.mean()
    print(f"  - Overall missing rate: {missing_rate:.2%}")
    print(f"  - Time delta statistics:")
    print(f"    * Mean: {T[T > 0].mean():.2f} weeks")
    print(f"    * Max: {T.max():.2f} weeks")
    print(f"  - Event distribution:")
    print(f"    * Events (death/transplant): {Y[:, 1].sum():.0f}")
    print(f"    * Censored: {(1 - Y[:, 1]).sum():.0f}")
    
    return X, M, T, Y, feature_means


def split_grud_tensors_stratified(X, M, T, Y, test_size=0.2, random_state=42):
    """
    Split GRU-D tensors into train/test with stratification by event type.
    
    Args:
        X (np.ndarray): Feature tensor (n_patients, max_visits, n_features)
        M (np.ndarray): Mask tensor (n_patients, max_visits, n_features)
        T (np.ndarray): Time delta tensor (n_patients, max_visits, n_features)
        Y (np.ndarray): Target tensor (n_patients, 2)
        test_size (float): Proportion of test set (default: 0.2)
        random_state (int): Random seed for reproducibility
    
    Returns:
        X_train, X_test, M_train, M_test, T_train, T_test, Y_train, Y_test (torch.FloatTensor)
    """
    
    # Stratify by event type (column 1 of Y)
    event_labels = Y[:, 1]
    
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=event_labels
    )
    
    print(f"\nSplitting data:")
    print(f"  - Training set: {len(train_idx)} patients")
    print(f"  - Test set: {len(test_idx)} patients")
    print(f"  - Train events: {Y[train_idx, 1].sum():.0f}")
    print(f"  - Test events: {Y[test_idx, 1].sum():.0f}")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X[train_idx])
    X_test = torch.FloatTensor(X[test_idx])
    
    M_train = torch.FloatTensor(M[train_idx])
    M_test = torch.FloatTensor(M[test_idx])
    
    T_train = torch.FloatTensor(T[train_idx])
    T_test = torch.FloatTensor(T[test_idx])
    
    Y_train = torch.FloatTensor(Y[train_idx])
    Y_test = torch.FloatTensor(Y[test_idx])
    
    return X_train, X_test, M_train, M_test, T_train, T_test, Y_train, Y_test
