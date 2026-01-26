from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

from clinical_cool_etud.NLLsurv import NLLSurvLoss
from clinical_cool_etud.config import DATA_DIR
from clinical_cool_etud.model import LSTM_risk_estimator
from clinical_cool_etud.prepa_data_model import build_lstm_tensor, split_tensors_stratified
#from clinical_cool_etud.sksurv_format import to_sksurv_format


def concordance_index(y_true, risk_scores):
    """
    Calculate the C-index (Concordance Index) for survival analysis.
    
    Args:
        y_true: Tensor of shape (n_samples, 2) where:
                - y_true[:, 0] = time to event
                - y_true[:, 1] = event indicator (1 = event occurred, 0 = censored)
        risk_scores: Tensor of shape (n_samples,) with predicted risk scores
    
    Returns:
        c_index: Float value between 0 and 1 (higher is better)
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(risk_scores):
        risk_scores = risk_scores.detach().cpu().numpy()
    
    times = y_true[:, 0]
    events = y_true[:, 1]
    
    n = len(times)
    concordant = 0
    permissible = 0
    
    # For each pair of samples
    for i in range(n):
        for j in range(i + 1, n):
            # Only consider pairs where at least one has an event
            # Case 1: i has event and occurs before j
            if events[i] == 1 and times[i] < times[j]:
                permissible += 1
                # Check if risk scores agree (higher risk should have lower survival time)
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
            
            # Case 2: j has event and occurs before i
            elif events[j] == 1 and times[j] < times[i]:
                permissible += 1
                # Check if risk scores agree
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
    
    if permissible == 0:
        return 0.5  # Random performance if no valid pairs
    
    return concordant / permissible


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
            optimizer.zero_grad()
            risk_death_predict = model(X_batch)
            loss = criterion(risk_death_predict, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        history_loss.append(epoch_loss / len(train_loader))
        
        # Calculate C-index on training set every epoch
        model.eval()
        with torch.no_grad():
            train_risk_scores = model(X_train)
            # Sum the risk scores over time to get overall risk
            train_risk_total = train_risk_scores.sum(dim=1)
            train_cindex = concordance_index(Y_train, train_risk_total)
            history_train_cindex.append(train_cindex)
        model.train()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{number_epochs}], Loss: {history_loss[-1]:.4f}, C-index: {train_cindex:.4f}")

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_risk_scores = model(X_test)
        test_risk_total = test_risk_scores.sum(dim=1)
        test_cindex = concordance_index(Y_test, test_risk_total)
        
        train_risk_scores = model(X_train)
        train_risk_total = train_risk_scores.sum(dim=1)
        final_train_cindex = concordance_index(Y_train, train_risk_total)
    
    print(f"\nFinal Training C-index: {final_train_cindex:.4f}")
    print(f"Test C-index: {test_cindex:.4f}")

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
    
    plt.show()
    
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
        'final_loss': history_loss[-1],
        'number_epochs': number_epochs
    }
    summary_df = pd.DataFrame([results_summary])
    summary_csv_path = DATA_DIR.parent / "results_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Results summary saved to: {summary_csv_path}")