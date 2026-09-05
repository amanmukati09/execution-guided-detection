"""
Generate publication-quality figures for Paper 3.
Creates 4 figures:
1. Per-strategy recall comparison (bar chart)
2. Detector performance comparison
3. Trace difference visualization
4. Improvement over static analysis
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'figure.figsize': (6, 4),
})


def plot_per_strategy_comparison():
    """Figure 1: Bar chart comparing static vs execution-guided recall."""
    strategies = ['Boundary\nInversion', 'Import\nAliasing', 'Variable\nShadowing', 'Dead\nCode', 'Comment\nPlanting']
    static_recall = [0.000, 0.070, 1.000, 1.000, 1.000]
    pairwise_recall = [0.875, 0.133, 0.969, 0.980, 0.969]
    import_aware_recall = [0.725, 0.582, 0.959, 0.990, 0.929]

    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width, static_recall, width, label='Static (Paper 1)', color='#E74C3C', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, pairwise_recall, width, label='Pairwise Execution', color='#2ECC71', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, import_aware_recall, width, label='Import-Aware', color='#3498DB', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Perturbation Strategy', fontweight='bold')
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_title('Per-Strategy Detection Recall: Static vs Execution-Guided', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    output_path = FIGURES_DIR / "per_strategy_comparison.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_detector_comparison():
    """Figure 2: Detector performance comparison."""
    detectors = ['Standalone\n(GB)', 'Pairwise\n(GB)', 'Import-Aware\n(GB)', 'Hybrid\n(XGBoost)']
    accuracy = [0.787, 0.898, 0.902, 0.904]
    precision = [0.822, 0.918, 0.922, 0.928]
    recall = [0.944, 0.962, 0.961, 0.957]
    f1 = [0.879, 0.939, 0.941, 0.942]

    x = np.arange(len(detectors))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#A23B72', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#F18F01', edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x + 1.5*width, f1, width, label='F1', color='#27AE60', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Detection Approach', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Detection Performance Across Approaches', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(detectors)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = FIGURES_DIR / "detector_comparison.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_improvement_over_static():
    """Figure 3: Improvement over static analysis."""
    strategies = ['Boundary\nInversion', 'Import\nAliasing']
    static_recall = [0.000, 0.070]
    exec_recall = [0.875, 0.582]

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 4))
    bars1 = ax.bar(x - width/2, static_recall, width, label='Static Analysis', color='#E74C3C', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, exec_recall, width, label='Execution-Guided', color='#2ECC71', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Perturbation Type', fontweight='bold')
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_title('Improvement on Semantic Perturbations', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left')

    # Add annotation arrows
    ax.annotate('', xy=(x[0]+width/2, 0.875), xytext=(x[0]-width/2, 0.000),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=(x[1]+width/2, 0.582), xytext=(x[1]-width/2, 0.070),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Add improvement labels
    ax.text(x[0], 0.45, '+87.5%', ha='center', fontsize=11, fontweight='bold', color='green')
    ax.text(x[1], 0.33, '+51.2%', ha='center', fontsize=11, fontweight='bold', color='green')

    plt.tight_layout()
    output_path = FIGURES_DIR / "improvement_over_static.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_statistical_validation():
    """Figure 4: Statistical test results visualization."""
    tests = ['McNemar\'s\nTest', 'Wilcoxon\nSigned-Rank']
    p_values = [0.0466, 0.0074]
    alpha_005 = [0.05, 0.05]
    alpha_001 = [0.01, 0.01]

    x = np.arange(len(tests))
    width = 0.3

    fig, ax = plt.subplots(figsize=(5, 4))
    bars1 = ax.bar(x - width/2, p_values, width, label='p-value', color='#3498DB', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, alpha_005, width, label='α = 0.05', color='#F39C12', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width*1.5, alpha_001, width, label='α = 0.01', color='#E74C3C', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Statistical Test', fontweight='bold')
    ax.set_ylabel('p-value', fontweight='bold')
    ax.set_title('Statistical Validation', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tests)
    ax.legend(loc='upper right')

    # Add significance markers
    ax.text(0, 0.025, '✓', ha='center', fontsize=14, fontweight='bold', color='green')
    ax.text(1, 0.003, '✓✓', ha='center', fontsize=14, fontweight='bold', color='green')

    plt.tight_layout()
    output_path = FIGURES_DIR / "statistical_validation.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Generating figures for Paper 3...")
    plot_per_strategy_comparison()
    plot_detector_comparison()
    plot_improvement_over_static()
    plot_statistical_validation()
    print("\nAll figures generated successfully!")
