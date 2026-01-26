from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

from clinical_cool_etud.NLLsurv import NLLSurvLoss
from clinical_cool_etud.config import DATA_DIR
from clinical_cool_etud.model import LSTM_risk_estimator
from clinical_cool_etud.prepa_data_model import build_lstm_tensor, split_tensors_stratified
#from clinical_cool_etud.sksurv_format import to_sksurv_format


def manual_concordance_index(y_true, risk_matrix):
    """
    Calculate the time-dependent C-index for survival analysis.
    
    Args:
        y_true: Array of shape (n_samples, 2) where:
                - y_true[:, 0] = time to event
                - y_true[:, 1] = event indicator (1 = event occurred, 0 = censored)
        risk_matrix: Array of shape (n_samples, n_time_points) with cumulative risk scores
    
    Returns:
        c_index: Float value between 0 and 1 (higher is better)
    """
    times = y_true[:, 0]
    events = y_true[:, 1]
    concordant_pairs = 0
    total_comparable_pairs = 0
    n = len(times)
    
    for i in range(n):
        if events[i] == 1:  # Patient i died
            t_i = int(times[i])
            # Ensure we don't go out of bounds in the risk matrix
            t_idx = min(t_i, risk_matrix.shape[1] - 1)
            
            for j in range(n):
                if times[j] > times[i]:  # j survived longer than i
                    total_comparable_pairs += 1
                    # Compare risks at time t_i
                    risk_i = risk_matrix[i, t_idx]
                    risk_j = risk_matrix[j, t_idx]
                    
                    if risk_i > risk_j:
                        concordant_pairs += 1
                    elif risk_i == risk_j:
                        concordant_pairs += 0.5
    
    return concordant_pairs / total_comparable_pairs if total_comparable_pairs > 0 else 0


def main():

    # Charger les données

    data_pbc = pd.read_csv(DATA_DIR / "clinical_data_pbc_cleaned.csv")

    list_features_continuous = ["age", "edema", "serBilir", "serChol", "albumin", "alkaline", "SGOT", "platelets", "prothrombin", "histologic"]
    list_features_binary = ["drug", "sex", "ascites", "hepatomegaly", "spiders"]

    time_to_event_column = "tte"
    event_column = "label"

    number_features = len(list_features_continuous) + len(list_features_binary)

    # Construction des tenseurs
    X_tensor, y_tensor, all_ids = build_lstm_tensor(
        data_pbc,  # Ton dataframe longitudinal complet (pas le baseline !)
        id_col='id',
        tte_col=time_to_event_column,
        event_col=event_column,
        feature_continuous_cols=list_features_continuous,
        features_binary_cols=list_features_binary,
    )

    # Split et datasets

    X_train, X_test, Y_train, Y_test = split_tensors_stratified(X_tensor, y_tensor)

    # Tensordataset : necessaire pour utiliser le dataloader (création des batchs)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    # dataloader : pour créer des batchs

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # définition du model 
    MAX_TIME_HORIZON = int(Y_train[:,0].max()) +1

    model = LSTM_risk_estimator(input_size=number_features, hidden_size = 64, num_layers =1, number_time_discrete=MAX_TIME_HORIZON)
    criterion = NLLSurvLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    number_epochs = 100

    history_loss = []
    history_train_cindex = []

    model.train()

    for epoch in range(number_epochs):
        epoch_loss = 0.0

        for X_batch, Y_batch in train_loader:
            # Always reset gradients before each step
            optimizer.zero_grad()
            
            # Model outputs risk probabilities for each time point
            risk_death_predict = model(X_batch)
            
            # Calculate the negative log likelihood loss
            loss = criterion(risk_death_predict, Y_batch)
            
            # Backpropagation to calculate gradients
            loss.backward()
            
            # Optimize weights
            optimizer.step()
            
            # Add batch loss to epoch loss
            epoch_loss += loss.item()

        # Calculate mean loss over the entire epoch
        epoch_loss /= train_dataset.__len__()
        history_loss.append(epoch_loss)
        
        # Calculate time-dependent C-index on training set every epoch
        model.eval()
        with torch.no_grad():
            train_probs = model(X_train)
            # Calculate cumulative risk: Matrix [n_patients, n_time_points]
            train_risk_cumulative = torch.cumsum(train_probs, dim=1).numpy()
            train_cindex = manual_concordance_index(Y_train.numpy(), train_risk_cumulative)
            history_train_cindex.append(train_cindex)
        model.train()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{number_epochs}], Loss: {history_loss[-1]:.4f}, C-index: {train_cindex:.4f}")

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        # Test set
        test_probs = model(X_test)
        test_risk_cumulative = torch.cumsum(test_probs, dim=1).numpy()
        test_cindex = manual_concordance_index(Y_test.numpy(), test_risk_cumulative)
        
        # Train set (final)
        train_probs = model(X_train)
        train_risk_cumulative = torch.cumsum(train_probs, dim=1).numpy()
        final_train_cindex = manual_concordance_index(Y_train.numpy(), train_risk_cumulative)
    
    print(f"\nFinal Training C-index (time-dependent): {final_train_cindex:.4f}")
    print(f"Test C-index (time-dependent): {test_cindex:.4f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history_loss)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    
    ax2.plot(history_train_cindex)
    ax2.set_title("Training C-index")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("C-index")
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = DATA_DIR.parent / "training_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Save training history to CSV
    history_df = pd.DataFrame({
        'epoch': range(1, number_epochs + 1),
        'loss': history_loss,
        'train_cindex': history_train_cindex
    })
    history_csv_path = DATA_DIR.parent / "training_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to: {history_csv_path}")
    
    # Save results summary
    results_summary = {
        'final_train_cindex': final_train_cindex,
        'test_cindex': test_cindex,
        'mean_loss': np.mean(history_loss),
        'final_loss': history_loss[-1],
        'number_epochs': number_epochs
    }
    summary_df = pd.DataFrame([results_summary])
    summary_csv_path = DATA_DIR.parent / "results_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Results summary saved to: {summary_csv_path}")