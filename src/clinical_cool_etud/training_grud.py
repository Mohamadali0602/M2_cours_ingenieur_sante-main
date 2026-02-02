import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

from clinical_cool_etud.NLLsurv import NLLSurvLoss
from clinical_cool_etud.config import DATA_DIR
from clinical_cool_etud.model_grud import GRU_D_risk_estimator
from clinical_cool_etud.prepa_data_grud import build_grud_tensor, split_grud_tensors_stratified
from clinical_cool_etud.training import manual_concordance_index  # Reuse C-index function


def main():
    """
    Train GRU-D model for survival analysis with irregular time series and missing values.
    
    This script implements the complete training pipeline for GRU-D, which handles:
    - Irregular time intervals between measurements
    - Missing values (modeled explicitly, not pre-imputed)
    - Temporal decay of feature values and hidden states
    """
    
    print("="*70)
    print("GRU-D SURVIVAL ANALYSIS - Training Script")
    print("="*70)
    
    # === LOAD DATA ===
    # IMPORTANT: Use the ORIGINAL data with missing values, NOT the cleaned version!
    # GRU-D needs to see which values are missing to model them properly
    print("\n[1/6] Loading data...")
    data_pbc = pd.read_csv(DATA_DIR / "clinical_data_pbc.csv")
    
    print(f"  - Loaded {len(data_pbc)} rows from {len(data_pbc['id'].unique())} patients")
    
    # === FEATURE DEFINITION ===
    list_features_continuous = [
        "age", "edema", "serBilir", "serChol", "albumin", 
        "alkaline", "SGOT", "platelets", "prothrombin", "histologic"
    ]
    list_features_binary = [
        "drug", "sex", "ascites", "hepatomegaly", "spiders"
    ]
    
    time_col = "times"  # Visit time in weeks
    time_to_event_column = "tte"
    event_column = "label"
    
    number_features = len(list_features_continuous) + len(list_features_binary)
    
    print(f"  - {number_features} features total:")
    print(f"    * {len(list_features_continuous)} continuous")
    print(f"    * {len(list_features_binary)} binary")
    
    # === BUILD GRU-D TENSORS ===
    print("\n[2/6] Building GRU-D tensors...")
    X_tensor, M_tensor, T_tensor, y_tensor, feature_means = build_grud_tensor(
        data_pbc,
        id_col='id',
        time_col=time_col,
        tte_col=time_to_event_column,
        event_col=event_column,
        feature_continuous_cols=list_features_continuous,
        features_binary_cols=list_features_binary,
    )
    
    # === SPLIT DATA ===
    print("\n[3/6] Splitting data into train/test sets...")
    X_train, X_test, M_train, M_test, T_train, T_test, Y_train, Y_test = \
        split_grud_tensors_stratified(X_tensor, M_tensor, T_tensor, y_tensor)
    
    # === CREATE DATALOADERS ===
    print("\n[4/6] Creating data loaders...")
    train_dataset = torch.utils.data.TensorDataset(X_train, M_train, T_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"  - Batch size: 32")
    print(f"  - Number of batches: {len(train_loader)}")
    
    # === INITIALIZE MODEL ===
    print("\n[5/6] Initializing GRU-D model...")
    MAX_TIME_HORIZON = int(Y_train[:, 0].max()) + 1
    
    model = GRU_D_risk_estimator(
        input_size=number_features,
        hidden_size=64,
        number_time_discrete=MAX_TIME_HORIZON,
        feature_means=feature_means,
        dropout=0.2  # Regularization to reduce overfitting
    )
    
    print(f"  - Input size: {number_features}")
    print(f"  - Hidden size: 64")
    print(f"  - Output size (time horizon): {MAX_TIME_HORIZON}")
    print(f"  - Dropout: 0.2")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # === TRAINING SETUP ===
    criterion = NLLSurvLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    number_epochs = 100
    
    history_loss = []
    history_train_cindex = []
    
    print(f"  - Loss function: Negative Log-Likelihood for Survival")
    print(f"  - Optimizer: Adam (lr=0.001, weight_decay=1e-5)")
    print(f"  - Epochs: {number_epochs}")
    
    # === TRAINING LOOP ===
    print("\n[6/6] Training GRU-D model...")
    print("-" * 70)
    
    model.train()
    
    for epoch in range(number_epochs):
        epoch_loss = 0.0
        
        for X_batch, M_batch, T_batch, Y_batch in train_loader:
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass with mask and time delta
            risk_death_predict = model(X_batch, M_batch, T_batch)
            
            # Calculate loss
            loss = criterion(risk_death_predict, Y_batch)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
        
        # Average loss over all samples
        epoch_loss /= len(train_dataset)
        history_loss.append(epoch_loss)
        
        # === EVALUATION ===
        # Calculate C-index on training set
        model.eval()
        with torch.no_grad():
            train_probs = model(X_train, M_train, T_train)
            train_risk_cumulative = torch.cumsum(train_probs, dim=1).numpy()
            train_cindex = manual_concordance_index(Y_train.numpy(), train_risk_cumulative)
            history_train_cindex.append(train_cindex)
        model.train()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{number_epochs}] | "
                  f"Loss: {history_loss[-1]:.4f} | "
                  f"Train C-index: {train_cindex:.4f}")
    
    print("-" * 70)
    
    # === FINAL EVALUATION ===
    print("\nFinal Evaluation:")
    print("-" * 70)
    
    model.eval()
    with torch.no_grad():
        # Test set
        test_probs = model(X_test, M_test, T_test)
        test_risk_cumulative = torch.cumsum(test_probs, dim=1).numpy()
        test_cindex = manual_concordance_index(Y_test.numpy(), test_risk_cumulative)
        
        # Train set (final)
        train_probs = model(X_train, M_train, T_train)
        train_risk_cumulative = torch.cumsum(train_probs, dim=1).numpy()
        final_train_cindex = manual_concordance_index(Y_train.numpy(), train_risk_cumulative)
    
    print(f"Final Training C-index: {final_train_cindex:.4f}")
    print(f"Test C-index:           {test_cindex:.4f}")
    print(f"Generalization Gap:     {abs(final_train_cindex - test_cindex):.4f}")
    print("-" * 70)
    
    # === SAVE RESULTS ===
    print("\nSaving results...")
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, number_epochs + 1),
        'loss': history_loss,
        'train_cindex': history_train_cindex
    })
    history_csv_path = DATA_DIR.parent / "training_history_grud.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"  - Training history: {history_csv_path}")
    
    # Save results summary
    results_summary = {
        'model': 'GRU-D',
        'final_train_cindex': final_train_cindex,
        'test_cindex': test_cindex,
        'generalization_gap': abs(final_train_cindex - test_cindex),
        'mean_loss': np.mean(history_loss),
        'final_loss': history_loss[-1],
        'number_epochs': number_epochs,
        'hidden_size': 64,
        'dropout': 0.2
    }
    summary_df = pd.DataFrame([results_summary])
    summary_csv_path = DATA_DIR.parent / "results_summary_grud.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  - Results summary: {summary_csv_path}")
    
    # === PLOTTING ===
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history_loss, linewidth=2, color='#2E86AB')
    axes[0].set_title("GRU-D Training Loss", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, number_epochs])
    
    # C-index plot
    axes[1].plot(history_train_cindex, linewidth=2, color='#A23B72', label='Training')
    axes[1].axhline(y=test_cindex, color='#F18F01', linestyle='--', 
                    linewidth=2, label=f'Test (final): {test_cindex:.3f}')
    axes[1].set_title("GRU-D Training C-index", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("C-index", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, number_epochs])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    plot_path = DATA_DIR.parent / "training_results_grud.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  - Training plots: {plot_path}")
    
    # === MODEL COMPARISON ===
    print("\n" + "="*70)
    print("COMPARISON WITH BASELINE LSTM")
    print("="*70)
    
    # Try to load LSTM results for comparison
    try:
        lstm_results = pd.read_csv(DATA_DIR.parent / "results_summary.csv")
        lstm_test_cindex = lstm_results['test_cindex'].values[0]
        lstm_train_cindex = lstm_results['final_train_cindex'].values[0]
        lstm_gap = abs(lstm_train_cindex - lstm_test_cindex)
        
        print("\nModel Performance Comparison:")
        print("-" * 70)
        print(f"{'Metric':<30} {'LSTM':>12} {'GRU-D':>12} {'Change':>12}")
        print("-" * 70)
        print(f"{'Test C-index':<30} {lstm_test_cindex:>12.4f} {test_cindex:>12.4f} "
              f"{((test_cindex - lstm_test_cindex) / lstm_test_cindex * 100):>+11.1f}%")
        print(f"{'Train C-index':<30} {lstm_train_cindex:>12.4f} {final_train_cindex:>12.4f} "
              f"{((final_train_cindex - lstm_train_cindex) / lstm_train_cindex * 100):>+11.1f}%")
        print(f"{'Generalization Gap':<30} {lstm_gap:>12.4f} "
              f"{abs(final_train_cindex - test_cindex):>12.4f} "
              f"{((abs(final_train_cindex - test_cindex) - lstm_gap) / lstm_gap * 100):>+11.1f}%")
        print("-" * 70)
        
        if test_cindex > lstm_test_cindex:
            improvement = ((test_cindex - lstm_test_cindex) / lstm_test_cindex * 100)
            print(f"\n✓ GRU-D achieved {improvement:.1f}% improvement in test C-index!")
        else:
            print(f"\n✗ GRU-D did not improve test C-index (may need hyperparameter tuning)")
        
        if abs(final_train_cindex - test_cindex) < lstm_gap:
            gap_reduction = ((lstm_gap - abs(final_train_cindex - test_cindex)) / lstm_gap * 100)
            print(f"✓ GRU-D reduced generalization gap by {gap_reduction:.1f}%!")
        
    except FileNotFoundError:
        print("\nNote: Could not find LSTM results for comparison.")
        print("Run the baseline LSTM training first to enable comparison.")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()
