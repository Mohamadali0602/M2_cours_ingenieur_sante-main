import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv('data/clinical_data_pbc.csv')

print("="*60)
print("DONNÉES ORIGINALES")
print("="*60)
print(f"Dimensions: {df.shape}")
print(f"Colonnes: {list(df.columns)}\n")

# 1. Supprimer la colonne date_diag
print("1. Suppression de la colonne 'date_diag'...")
if 'date_diag' in df.columns:
    df = df.drop('date_diag', axis=1)
    print("   ✓ Colonne 'date_diag' supprimée\n")

# 2. Supprimer la colonne total_protein
print("2. Suppression de la colonne 'total_protein'...")
if 'total_protein' in df.columns:
    df = df.drop('total_protein', axis=1)
    print("   ✓ Colonne 'total_protein' supprimée\n")

# 3. Corriger la valeur de serBilir pour le patient 104 (diviser par 100)
print("3. Correction de serBilir pour le patient 104...")
if 'serBilir' in df.columns:
    # Trouver les lignes du patient 104
    patient_104_mask = df['id'] == 104
    if patient_104_mask.any():
        print(f"   Avant correction: {df.loc[patient_104_mask, 'serBilir'].values}")
        df.loc[patient_104_mask, 'serBilir'] = df.loc[patient_104_mask, 'serBilir'] / 100
        print(f"   Après correction: {df.loc[patient_104_mask, 'serBilir'].values}")
        print("   ✓ Valeur de serBilir corrigée pour le patient 104\n")

# 4. Imputer les variables continues manquantes par la médiane
print("4. Imputation des variables continues par la médiane...")
# Identifier les variables continues (numériques)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclure les colonnes qui ne sont pas vraiment continues (id, label, drug, sex, etc.)
categorical_numeric = ['id', 'label', 'drug', 'sex', 'ascites', 'hepatomegaly', 
                       'spiders', 'edema', 'histologic']
continuous_cols = [col for col in numeric_cols if col not in categorical_numeric]

print(f"   Variables continues identifiées: {continuous_cols}")
for col in continuous_cols:
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"   ✓ {col}: {missing_count} valeurs manquantes imputées avec la médiane {median_value:.2f}")
print()

# 5. Imputer les variables catégorielles manquantes par le mode
print("5. Imputation des variables catégorielles par le mode...")
categorical_cols = [col for col in df.columns if col in categorical_numeric]

for col in categorical_cols:
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        mode_value = df[col].mode()[0] if not df[col].mode().empty else df[col].value_counts().index[0]
        df[col] = df[col].fillna(mode_value)
        print(f"   ✓ {col}: {missing_count} valeurs manquantes imputées avec le mode {mode_value}")
print()

# 6. Transformer les patients transplantés en patients censurés
# Status 1 devient 0, status 2 devient 1
print("6. Transformation des statuts (transplantés → censurés)...")
if 'label' in df.columns:
    status_counts_before = df['label'].value_counts().sort_index()
    print(f"   Avant transformation:\n{status_counts_before}")
    
    # Créer une copie pour la transformation
    df['label'] = df['label'].replace({1: 0, 2: 1})
    
    status_counts_after = df['label'].value_counts().sort_index()
    print(f"   Après transformation:\n{status_counts_after}")
    print("   ✓ Status 1 → 0 (censuré), Status 2 → 1 (événement)\n")

# Afficher le résumé final
print("="*60)
print("DONNÉES NETTOYÉES")
print("="*60)
print(f"Dimensions: {df.shape}")
print(f"Colonnes: {list(df.columns)}")
print(f"\nValeurs manquantes par colonne:")
print(df.isna().sum())

# Sauvegarder les données nettoyées
output_path = 'data/clinical_data_pbc_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Données nettoyées sauvegardées dans: {output_path}")
