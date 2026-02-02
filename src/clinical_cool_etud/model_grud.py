import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_D_risk_estimator(nn.Module):
    """
    GRU-D (Gated Recurrent Unit with Decay) for clinical survival analysis.
    
    This model handles irregular time series with missing values by:
    1. Temporal decay mechanism - values decay toward empirical means based on time gaps
    2. Missing value masking - explicitly tracks observed vs. missing values
    3. Time gap encoding - uses actual time intervals between measurements
    
    Based on: Che et al. (2018) - Recurrent Neural Networks for Multivariate 
    Time Series with Missing Values, Scientific Reports, Nature.
    
    Args:
        input_size (int): Number of clinical features
        hidden_size (int): Hidden state dimension
        number_time_discrete (int): Number of discrete time points for risk estimation
        feature_means (np.ndarray or None): Empirical means of features for decay target
        dropout (float): Dropout rate for regularization (default: 0.0)
    """
    
    def __init__(self, input_size, hidden_size, number_time_discrete, 
                 feature_means=None, dropout=0.0):
        super(GRU_D_risk_estimator, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.number_time_discrete = number_time_discrete
        
        # Store feature means for decay mechanism
        # If not provided, default to zeros
        if feature_means is not None:
            self.register_buffer('feature_means', torch.FloatTensor(feature_means))
        else:
            self.register_buffer('feature_means', torch.zeros(input_size))
        
        # === Decay Parameters ===
        # For input features: controls how fast values decay toward empirical mean
        self.W_gamma = nn.Linear(input_size, input_size)
        self.b_gamma = nn.Parameter(torch.zeros(input_size))
        
        # For hidden state: controls how fast hidden state decays over time
        self.W_gamma_h = nn.Linear(1, hidden_size)
        self.b_gamma_h = nn.Parameter(torch.zeros(hidden_size))
        
        # === Augmented Input ===
        # Input is concatenation of: [decayed_features, mask, time_delta]
        augmented_input_size = input_size * 3
        
        # === GRU Gates ===
        # Reset gate: controls how much of previous hidden state to forget
        self.W_r = nn.Linear(augmented_input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        
        # Update gate: controls how much to update hidden state
        self.W_z = nn.Linear(augmented_input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)
        
        # Candidate hidden state
        self.W_h = nn.Linear(augmented_input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)
        
        # === Output Layer ===
        self.fc = nn.Linear(hidden_size, number_time_discrete)
        self.softmax = nn.Softmax(dim=1)
        
        # === Regularization ===
        self.dropout = nn.Dropout(dropout)
    
    def compute_decay(self, delta, W, b):
        """
        Compute temporal decay factor.
        
        Decay follows: gamma = exp(-max(0, W*delta + b))
        - When time gap is small: gamma ≈ 1 (little decay)
        - When time gap is large: gamma ≈ 0 (much decay)
        
        Args:
            delta (torch.Tensor): Time gaps (batch, seq_len, features) or (batch, seq_len, 1)
            W (nn.Linear): Weight matrix
            b (torch.Parameter): Bias vector
        
        Returns:
            gamma (torch.Tensor): Decay factors in [0, 1]
        """
        decay_input = W(delta) + b
        gamma = torch.exp(-torch.clamp(decay_input, min=0.0))
        return gamma
    
    def forward(self, x, mask, time_delta):
        """
        Forward pass through GRU-D.
        
        Args:
            x (torch.Tensor): Input features (batch, seq_len, input_size)
                              For missing values, can be zeros or last observation
            mask (torch.Tensor): Binary mask (batch, seq_len, input_size)
                                 1 = observed, 0 = missing
            time_delta (torch.Tensor): Time since last observation (batch, seq_len, input_size)
                                       In weeks or normalized time units
        
        Returns:
            risk_probs (torch.Tensor): Risk probabilities (batch, number_time_discrete)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        
        # Initialize last observed values (start with empirical means)
        x_last_obs = self.feature_means.unsqueeze(0).expand(batch_size, -1).clone()
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t, :]  # Current values (batch, input_size)
            m_t = mask[:, t, :]  # Current mask (batch, input_size)
            delta_t = time_delta[:, t, :]  # Time gaps (batch, input_size)
            
            # === INPUT DECAY ===
            # Compute decay factor for each feature
            gamma_x = self.compute_decay(delta_t, self.W_gamma, self.b_gamma)
            
            # Decay previous values toward empirical mean
            # If time gap is small: x_decayed ≈ x_last_obs
            # If time gap is large: x_decayed ≈ feature_means
            x_decayed = gamma_x * x_last_obs + (1 - gamma_x) * self.feature_means
            
            # Use observed values where available, decayed values where missing
            x_hat = m_t * x_t + (1 - m_t) * x_decayed
            
            # Update last observed values
            # If observed: use current value
            # If missing: keep previous last observed value
            x_last_obs = m_t * x_t + (1 - m_t) * x_last_obs
            
            # === HIDDEN STATE DECAY ===
            # Compute average time gap for hidden state decay
            delta_h = delta_t.mean(dim=1, keepdim=True)  # (batch, 1)
            gamma_h = self.compute_decay(delta_h, self.W_gamma_h, self.b_gamma_h)
            h_decayed = gamma_h * h
            
            # === AUGMENTED INPUT ===
            # Concatenate: [decayed_input, mask, time_delta]
            # This gives the model explicit information about:
            # - What values are (after decay)
            # - Which values were observed vs. missing
            # - How long since last observation
            x_aug = torch.cat([x_hat, m_t, delta_t], dim=1)
            
            # === GRU UPDATE ===
            # Reset gate: how much of previous hidden state to use
            r = torch.sigmoid(self.W_r(x_aug) + self.U_r(h_decayed))
            
            # Update gate: how much to update hidden state
            z = torch.sigmoid(self.W_z(x_aug) + self.U_z(h_decayed))
            
            # Candidate hidden state
            h_tilde = torch.tanh(self.W_h(x_aug) + self.U_h(r * h_decayed))
            
            # New hidden state (weighted combination of previous and candidate)
            h = (1 - z) * h_decayed + z * h_tilde
        
        # === OUTPUT LAYER ===
        # Apply dropout for regularization
        h = self.dropout(h)
        
        # Map to risk probabilities for each discrete time point
        risk_logits = self.fc(h)
        risk_probs = self.softmax(risk_logits)
        
        return risk_probs
