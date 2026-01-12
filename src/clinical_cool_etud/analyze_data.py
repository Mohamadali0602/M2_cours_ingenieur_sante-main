import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('data/clinical_data_pbc.csv')

# Print data types
print("="*60)
print("VARIABLE TYPES")
print("="*60)
for column in df.columns:
    dtype = df[column].dtype
    # Count non-null values
    non_null = df[column].notna().sum()
    total = len(df)
    print(f"{column:20s} | Type: {str(dtype):10s} | Non-null: {non_null}/{total}")

print("\n" + "="*60)
print("DATA OVERVIEW")
print("="*60)
print(df.info())

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)
print(df.describe())

# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\n\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Non-numeric columns ({len(non_numeric_cols)}): {non_numeric_cols}")

# Create boxplots for numeric variables
if numeric_cols:
    # Calculate grid dimensions
    n_numeric = len(numeric_cols)
    n_cols = 4
    n_rows = (n_numeric + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_numeric > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        # Remove NaN values for boxplot
        data_clean = df[col].dropna()
        
        ax.boxplot(data_clean, vert=True)
        ax.set_title(f'{col}\n(Type: {df[col].dtype})', fontsize=10, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add some statistics
        ax.text(0.02, 0.98, f'n={len(data_clean)}\nMean={data_clean.mean():.2f}\nStd={data_clean.std():.2f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide empty subplots
    for idx in range(n_numeric, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('boxplots_numeric_variables.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Boxplots for {n_numeric} numeric variables saved to 'boxplots_numeric_variables.png'")
    plt.show()

# For categorical/non-numeric variables, show value distributions
if non_numeric_cols:
    print("\n" + "="*60)
    print("NON-NUMERIC VARIABLES - VALUE DISTRIBUTIONS")
    print("="*60)
    for col in non_numeric_cols:
        print(f"\n{col} (Type: {df[col].dtype}):")
        print(df[col].value_counts())
        print(f"Unique values: {df[col].nunique()}")

# Create boxplot for time to event (tte) by label
print("\n" + "="*60)
print("TIME TO EVENT BY LABEL")
print("="*60)

# Get unique labels
labels = sorted(df['label'].unique())
print(f"Unique labels: {labels}")

# Prepare data for boxplot
data_by_label = [df[df['label'] == label]['tte'].dropna() for label in labels]

# Create the boxplot
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(data_by_label, labels=[f'Label {int(label)}' for label in labels], 
                patch_artist=True, showmeans=True)

# Customize colors
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors[:len(labels)]):
    patch.set_facecolor(color)

ax.set_xlabel('Label', fontsize=12, fontweight='bold')
ax.set_ylabel('Time to Event (tte)', fontsize=12, fontweight='bold')
ax.set_title('Time to Event Distribution by Label', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add statistics for each label
for i, label in enumerate(labels):
    data = data_by_label[i]
    print(f"\nLabel {int(label)}:")
    print(f"  Count: {len(data)}")
    print(f"  Mean: {data.mean():.2f}")
    print(f"  Median: {data.median():.2f}")
    print(f"  Std: {data.std():.2f}")
    print(f"  Min: {data.min():.2f}")
    print(f"  Max: {data.max():.2f}")

plt.tight_layout()
plt.savefig('boxplot_tte_by_label.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Boxplot of time to event by label saved to 'boxplot_tte_by_label.png'")
plt.show()
