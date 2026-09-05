# Execution-Guided Detection of Semantic Code Perturbations

**Paper 3 in the AI Code Security research series**

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of **execution-guided detection** — a novel framework that detects adversarial perturbations in AI-generated code by observing code *behavior* rather than code *structure*.

Our prior work established that static analysis (AST features) and embeddings (CodeBERT) fail to detect semantic perturbations. This work solves that gap.

## Key Finding

**Boundary inversion detection improved from 0% → 87.5% recall** by using execution traces instead of static analysis.

| Perturbation Type | Paper 1 (Static) | Paper 3 (Execution) | Improvement |
|-------------------|------------------|---------------------|-------------|
| Boundary inversion | 0.000 | **0.875** | +87.5% |
| Import aliasing | 0.070 | **0.582** | +51.2% |
| Variable shadowing | 1.000 | 0.959 | -4.1% |
| Dead code | 1.000 | 0.990 | -1.0% |
| Comment planting | 1.000 | 0.929 | -7.1% |

## System Architecture

```
┌─────────────────────────────────────────────────┐
│           EXECUTION-GUIDED DETECTOR             │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. Property Generator (Python)                │
│     → Auto-generates diverse test inputs       │
│                                                 │
│  2. Sandbox Executor (C)                       │
│     → Runs code in isolated process            │
│     → Captures outputs, exit codes, timing     │
│                                                 │
│  3. Trace Extractor (Python)                   │
│     → Builds execution trace vectors           │
│                                                 │
│  4. Trace Comparator (Python)                  │
│     → Computes differences between traces      │
│                                                 │
│  5. Detector (Gradient Boosting)               │
│     → Classifies clean vs perturbed            │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Results

| Detector | Accuracy | Precision | Recall | F1 |
|----------|----------|-----------|--------|-----|
| Standalone (GB) | 0.787 | 0.822 | 0.944 | 0.879 |
| Pairwise (GB) | 0.898 | 0.918 | 0.962 | 0.939 |
| Import-Aware (GB) | 0.902 | 0.922 | 0.961 | 0.941 |
| **Hybrid (XGBoost)** | **0.904** | **0.928** | **0.957** | **0.942** |

## Statistical Validation

| Test | Statistic | p-value | Significant? |
|------|-----------|---------|--------------|
| McNemar's test | χ² = 113.00 | 0.0466 | ✅ (α=0.05) |
| Wilcoxon signed-rank | 476.00 | 0.0074 | ✅ (α=0.01) |

## Repository Structure

```
execution-guided-detection/
├── src/
│   ├── property_generator.py    # Test input generation
│   ├── sandbox_executor.c       # C sandbox (fork + exec + timeout)
│   ├── sandbox_bridge.py        # Python ↔ C bridge
│   ├── trace_extractor.py       # Execution trace extraction
│   ├── trace_comparator.py      # Trace comparison features
│   ├── detector.py              # Standalone detector
│   ├── pairwise_detector.py     # Pairwise comparison detector
│   ├── improved_pairwise.py     # Import-aware detector
│   ├── hybrid_detector.py       # Combined AST + execution
│   ├── statistical_tests.py     # McNemar + Wilcoxon tests
│   └── extract_all_traces.py    # Full dataset trace extraction
├── data/
│   ├── clean/                   # 100 clean Python samples
│   └── perturbed/               # 450 perturbed samples
├── results/
│   ├── traces/                  # Extracted execution traces
│   └── metrics/                 # Detection performance
├── paper/
│   ├── main.tex                 # 8-page paper
│   └── figures/                 # Publication-quality figures
└── requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.10+
- GCC (for C sandbox)
- pip

### Installation

```bash
git clone https://github.com/amanmukati09/execution-guided-detection.git
cd execution-guided-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Compile the Sandbox

```bash
gcc -O2 -Wall -o src/sandbox_executor src/sandbox_executor.c
```

### Run Experiments

```bash
# Extract execution traces from all 550 samples
python src/extract_all_traces.py

# Run standalone detector
python src/detector.py

# Run pairwise detector
python src/pairwise_detector.py

# Run import-aware detector
python src/improved_pairwise.py

# Run statistical validation
python src/statistical_tests.py
```

## Related Work

This is Paper 3 in a research series:

| Paper | Title | DOI |
|-------|-------|-----|
| 1 | Detecting Adversarial Perturbations in AI-Generated Code | [10.5281/zenodo.22349074](https://doi.org/10.5281/zenodo.22349074) |
| 2 | Adversarial Code Review | [10.5281/zenodo.22357507](https://doi.org/10.5281/zenodo.22357507) |
| 3 | Execution-Guided Detection (this work) | Coming soon |

## Citation

```bibtex
@misc{mukati2026execution,
  author = {Mukati, Aman},
  title = {Execution-Guided Detection of Semantic Code Perturbations},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **GitHub**: [@amanmukati09](https://github.com/amanmukati09)
- **Email**: amanmukati2002@gmail.com
