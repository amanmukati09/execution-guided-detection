"""
Pairwise Trace Comparison Detector
Compares execution traces of code against its clean counterpart.
This is the key insight: semantic perturbations manifest as DIFFERENCES
in execution traces, not as absolute trace values.
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))

from trace_extractor import TraceExtractor
from detector import get_strategy

TRACES_FILE = Path(__file__).parent.parent / "results" / "traces" / "all_traces.json"


def compute_trace_difference(clean_trace: dict, perturbed_trace: dict) -> list:
    """Compute numerical features representing the DIFFERENCE between two traces."""
    if clean_trace is None or perturbed_trace is None:
        return [0.0] * 10

    features = []

    # Success rate difference
    features.append(abs(clean_trace.get("success_rate", 0) - perturbed_trace.get("success_rate", 0)))

    # Output diversity difference
    features.append(abs(clean_trace.get("output_diversity", 0) - perturbed_trace.get("output_diversity", 0)))

    # Signature difference (binary: same or different)
    sig_clean = str(clean_trace.get("output_signature", ""))
    sig_pert = str(perturbed_trace.get("output_signature", ""))
    features.append(1.0 if sig_clean != sig_pert else 0.0)

    # Exit code pattern difference
    clean_codes = clean_trace.get("exit_codes", [])
    pert_codes = perturbed_trace.get("exit_codes", [])
    if clean_codes and pert_codes:
        diff_codes = sum(1 for c, p in zip(clean_codes, pert_codes) if c != p)
        features.append(diff_codes / len(clean_codes))
    else:
        features.append(0.0)

    # Execution time difference
    clean_times = clean_trace.get("execution_times", [])
    pert_times = perturbed_trace.get("execution_times", [])
    if clean_times and pert_times:
        avg_clean = sum(clean_times) / len(clean_times)
        avg_pert = sum(pert_times) / len(pert_times)
        features.append(abs(avg_clean - avg_pert))
    else:
        features.append(0.0)

    # Output difference ratio
    clean_outputs = clean_trace.get("outputs", [])
    pert_outputs = perturbed_trace.get("outputs", [])
    if clean_outputs and pert_outputs:
        diff_count = sum(1 for c, p in zip(clean_outputs, pert_outputs) if c != p)
        features.append(diff_count / len(clean_outputs))
    else:
        features.append(0.0)

    # Number of unique outputs in clean
    if clean_outputs:
        features.append(len(set(clean_outputs)) / len(clean_outputs))
    else:
        features.append(0.0)

    # Number of unique outputs in perturbed
    if pert_outputs:
        features.append(len(set(pert_outputs)) / len(pert_outputs))
    else:
        features.append(0.0)

    # Error rate difference
    clean_errors = sum(1 for c in clean_trace.get("exit_codes", []) if c != 0)
    pert_errors = sum(1 for c in perturbed_trace.get("exit_codes", []) if c != 0)
    if clean_trace.get("exit_codes"):
        features.append(abs(clean_errors - pert_errors) / len(clean_trace["exit_codes"]))
    else:
        features.append(0.0)

    while len(features) < 10:
       features.append(0.0)

    return features[:10]


def build_pairwise_dataset():
    """Build dataset of clean-vs-perturbed trace comparisons."""
    with open(TRACES_FILE) as f:
        data = json.load(f)

    clean_traces = data["clean"]
    perturbed_traces = data["perturbed"]

    X = []
    y = []
    sample_ids = []

    # For each perturbed sample, find its clean counterpart
    for pert_id, pert_trace in perturbed_traces.items():
        # Extract base sample ID (e.g., sample_01 from sample_01_boundary_invert)
        parts = pert_id.split("_")
        base_id = parts[0] + "_" + parts[1]  # sample_01

        clean_trace = clean_traces.get(base_id)
        if clean_trace is not None:
            # Positive example: clean vs perturbed (label=1)
            features = compute_trace_difference(clean_trace, pert_trace)
            X.append(features)
            y.append(1)
            sample_ids.append(pert_id)

    # Create negative examples: clean vs clean (different test runs)
    # Use different clean samples as "different" pairs
    clean_items = list(clean_traces.items())
    for i in range(len(clean_items) - 1):
        clean_id1, clean_trace1 = clean_items[i]
        clean_id2, clean_trace2 = clean_items[i + 1]
        features = compute_trace_difference(clean_trace1, clean_trace2)
        X.append(features)
        y.append(0)  # Not a perturbation (both clean)
        sample_ids.append(f"{clean_id1}_vs_{clean_id2}")

    return np.array(X), np.array(y), sample_ids


def main():
    print("=== Pairwise Trace Comparison Detector ===\n")

    X, y, sample_ids = build_pairwise_dataset()
    print(f"Dataset: {X.shape[0]} comparisons, {X.shape[1]} features")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    # Train and evaluate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    detectors = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, random_state=42
        ),
    }

    print("=== Cross-Validation Results ===\n")
    print(f"{'Detector':<25} {'Accuracy':>8} {'Precision':>9} {'Recall':>8} {'F1':>8}")
    print("-" * 60)

    for name, clf in detectors.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()
        prec = cross_val_score(pipe, X, y, cv=cv, scoring="precision").mean()
        rec = cross_val_score(pipe, X, y, cv=cv, scoring="recall").mean()
        f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean()

        print(f"{name:<25} {acc:>8.3f} {prec:>9.3f} {rec:>8.3f} {f1:>8.3f}")

    # Per-strategy analysis
    print(f"\n=== Per-Strategy Detection (Pairwise) ===\n")
    print(f"{'Strategy':<25} {'N':>4} {'Recall':>8}")
    print("-" * 40)

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    strategies = ["boundary_invert", "variable_shadow", "import_alias", "dead_code", "comment_plant"]

    for held_out in strategies:
        test_mask = np.array([get_strategy(sid) == held_out for sid in sample_ids])
        train_mask = ~test_mask

        if len(np.unique(y[train_mask])) < 2:
            continue

        clf.fit(X[train_mask], y[train_mask])
        preds = clf.predict(X[test_mask])
        true = y[test_mask]

        if sum(test_mask) > 0:
            recall = recall_score(true, preds, zero_division=0)
            n = sum(test_mask)
            print(f"{held_out:<25} {n:>4} {recall:>8.3f}")


if __name__ == "__main__":
    main()
