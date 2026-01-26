import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

def build_lstm_tensor(df, id_col, tte_col, event_col, feature_continuous_cols, features_binary_cols):
    """
    Transforme un DataFrame longitudinal en un Tenseur 3D pour LSTM.
    Gère le Padding (remplissage par des zéros) automatiquement.
    Renvoi le label [temps d'évènement et l'évènement]
    
    Parameters:
        df (pd.DataFrame): Le DataFrame contenant les données longitudinales.
        id_col (str): Le nom de la colonne identifiant chaque patient.
        tte_col (str): Le nom de la colonne contenant le temps de survie.
        event_col (str): Le nom de la colonne contenant l'événement.
        feature_continuous_cols (list of str): La liste des noms des colonnes features continus à normaliser.
        feature_binary_cols (list of str): La liste des noms des colonnes features binaires.
        max_len (int, optional): La longueur maximale des séquences. Si None, on prendra la plus grande séquence du DataFrame.

    Returns:
        X (np.ndarray): Le tenseur 3D des features (patients x visites x features).
        y (np.ndarray): Le vecteur des labels [temps d'évènement et l'évènement].
        unique_ids (np.ndarray): Les identifiants uniques des patients.
    """

    # On précise les features non binaires pour les normaliser
    
    features_cols = feature_continuous_cols + features_binary_cols

    # Normalisation des colonnes continues et catégorielles
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_continuous_cols] = scaler.fit_transform(df[feature_continuous_cols])
    df_scaled[features_binary_cols] = df[features_binary_cols]  # Conserver les binaires tels quels

    # On va créer le tenseur dont la dimension est (n_samples, max_len, n_features)
    # On determine alors max_len (nombre de visites maximales)
    # Puis on rempli le tenseur avec les valeurs des variables des patients observés dans nos données
    
    # Groupement par Patient
    grouped = df_scaled.groupby(id_col)
    unique_ids = df[id_col].unique()
    n_samples = len(unique_ids)
    n_features = len(features_cols)
    
    # Détermination de la longueur maximale 
    max_len = grouped.size().max()
        
    print(f"   - Dimensions cibles : ({n_samples} patients, {max_len} visites max, {n_features} features)")
    
    # Initialisation des tenseurs (Remplis de Zéros pour le Padding)
    X = np.zeros((n_samples, max_len, n_features))
    y = np.zeros((n_samples,2))  # Label contenant le temps d'évènement et l'évènement
    
    # Dictionnaire pour mapper ID -> Index dans le tenseur (pour le debug)
    id_to_idx = {}
    
    # Remplissage du Tenseur
    for i, patient_id in enumerate(unique_ids):
        # Récupération des données du patient
        patient_data = grouped.get_group(patient_id)
        
        # Extraction des features et du label
        # On prend les features sous forme de matrice (Visites x Features)
        seq_data = patient_data[features_cols].values
        
        # Longueur réelle de la séquence du patient
        seq_len = len(seq_data)
        
        # On remplit le tenseur X (Padding à la fin : les zéros restent à la fin)
        # On ne remplit que jusqu'à seq_len, le reste reste à 0
        X[i, :seq_len, :] = seq_data
        
        # On récupère le temps d'évènement et l'évènement
        y[i,0] = patient_data[tte_col].max()
        y[i,1] = patient_data[event_col].max()
        
        id_to_idx[patient_id] = i
    
    return X, y, unique_ids






def split_tensors_stratified(X, Y, test_size=0.2, random_state=42):
    """
    Split des données préparées (X, Y) en train set et test set en gardant l'équilibre des événements.
    
    Args:
        X (np.array): Input 3D (Batch, Visits, Features)
        Y (np.array): Target 2D (Batch, [Time, Event])
        
    Returns:
        Tenseurs PyTorch : X_train, X_test, y_train, y_test
    """
    
    # Extraction de la colonne 'Event' pour la stratification (Y[:, 1])
    events = Y[:, 1]
    
    # Création des indices (0 à N-1)
    indices = np.arange(X.shape[0])
    
    # Split des indices en stratifiant sur 'events'
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=events, # Le point crucial
        random_state=random_state
    )
    
    # Découpage des arrays
    X_train_np, X_test_np = X[train_idx], X[test_idx]
    Y_train_np, Y_test_np = Y[train_idx], Y[test_idx]
    
    # Conversion immédiate en Tenseurs PyTorch (Float)
    # Les réseaux de neurones veulent du Float32
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test_np, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32)
    Y_test_tensor  = torch.tensor(Y_test_np, dtype=torch.float32)
    
    print(f"   - Train : {len(train_idx)} patients (Event rate: {Y_train_np[:,1].mean():.1%})")
    print(f"   - Test  : {len(test_idx)} patients (Event rate: {Y_test_np[:,1].mean():.1%})")
    print(f"   - Output Shapes : X_train={X_train_tensor.shape}, Y_train={Y_train_tensor.shape}")
    
    return X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor