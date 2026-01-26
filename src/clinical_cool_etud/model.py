import torch.nn as nn

class LSTM_risk_estimator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, number_time_discrete):
        """
        Initialize the LSTM risk estimator model.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The number of features in the hidden state of the LSTM.
            num_layers (int): The number of recurrent layers.
            number_time_discrete (int): The number of discrete time steps for risk estimation.
        """

        super(LSTM_risk_estimator, self).__init__()

        # storage of hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.number_time_discrete = number_time_discrete 


        # Define the layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # tensor of shape (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, number_time_discrete) # tensor of shape (batch_size, number_time_discrete)
        self.softmax = nn.Softmax(dim=1) # tensor of shape (batch_size, number_time_discrete)

    def forward(self, x):
        """
        Forward pass through the LSTM risk estimator model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Probability distribution over discrete time steps with shape (batch_size, number_time_discrete).
        """

        # Forward propagate LSTM + FC layer
        hidden_states_lstm, _ = self.lstm(x)  # tensor of shape (batch_size, seq_length, hidden_size)``
        last_hidden_states_lstm = hidden_states_lstm[:, -1, :] # tensor of shape (batch_size, hidden_size)

        risk_estimator = self.fc(last_hidden_states_lstm)  # tensor of shape (batch_size, number_time_discrete)
        probabilities_risk_estimator = self.softmax(risk_estimator)  # tensor of shape (batch_size, number_time_discrete)
        return probabilities_risk_estimator