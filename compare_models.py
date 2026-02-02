import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
lstm_results = pd.read_csv('results_summary.csv')
grud_results = pd.read_csv('results_summary_grud.csv')

# Extract metrics
metrics = {
    'LSTM': {
        'train': lstm_results['final_train_cindex'].values[0],
        'test': lstm_results['test_cindex'].values[0]
    },
    'GRU-D': {
        'train': grud_results['final_train_cindex'].values[0],
        'test': grud_results['test_cindex'].values[0]
    }
}

# Create comparison plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: C-index Comparison
x = np.arange(2)
width = 0.35

train_scores = [metrics['LSTM']['train'], metrics['GRU-D']['train']]
test_scores = [metrics['LSTM']['test'], metrics['GRU-D']['test']]

bars1 = axes[0].bar(x - width/2, train_scores, width, label='Training', color='#2E86AB', alpha=0.8)
bars2 = axes[0].bar(x + width/2, test_scores, width, label='Test', color='#F18F01', alpha=0.8)

axes[0].set_ylabel('C-index', fontsize=14, fontweight='bold')
axes[0].set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['LSTM', 'GRU-D'], fontsize=13)
axes[0].legend(fontsize=12)
axes[0].set_ylim([0, 1.0])
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Generalization Gap
gaps = [
    abs(metrics['LSTM']['train'] - metrics['LSTM']['test']),
    abs(metrics['GRU-D']['train'] - metrics['GRU-D']['test'])
]

bars = axes[1].bar(['LSTM', 'GRU-D'], gaps, color=['#A23B72', '#06A77D'], alpha=0.8, width=0.6)
axes[1].set_ylabel('Generalization Gap', fontsize=14, fontweight='bold')
axes[1].set_title('Overfitting Comparison', fontsize=16, fontweight='bold')
axes[1].set_ylim([0, max(gaps) * 1.2])
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotation
improvement = ((gaps[0] - gaps[1]) / gaps[0]) * 100
axes[1].text(0.5, max(gaps) * 1.1, f'↓ {improvement:.1f}% reduction',
            ha='center', fontsize=12, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Plot 3: Test C-index Improvement
test_improvement = ((metrics['GRU-D']['test'] - metrics['LSTM']['test']) / metrics['LSTM']['test']) * 100

bars = axes[2].barh(['LSTM', 'GRU-D'], 
                    [metrics['LSTM']['test'], metrics['GRU-D']['test']],
                    color=['#E63946', '#06A77D'], alpha=0.8, height=0.6)

axes[2].set_xlabel('Test C-index', fontsize=14, fontweight='bold')
axes[2].set_title('Test Performance', fontsize=16, fontweight='bold')
axes[2].set_xlim([0, 1.0])
axes[2].grid(True, alpha=0.3, axis='x')

# Add value labels
for bar in bars:
    width = bar.get_width()
    axes[2].text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                ha='left', va='center', fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Add improvement annotation
axes[2].text(0.5, 0.5, f'↑ {test_improvement:.1f}% improvement',
            ha='center', fontsize=12, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
            transform=axes[2].transAxes)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved: model_comparison.png")

# Print summary
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
print(f"\nTest C-index:")
print(f"  LSTM:  {metrics['LSTM']['test']:.4f}")
print(f"  GRU-D: {metrics['GRU-D']['test']:.4f}")
print(f"  → Improvement: +{test_improvement:.1f}%")

print(f"\nGeneralization Gap:")
print(f"  LSTM:  {gaps[0]:.4f}")
print(f"  GRU-D: {gaps[1]:.4f}")
print(f"  → Reduction: -{improvement:.1f}%")

print("\n" + "="*70)

plt.show()
