"""
Execution-Guided Detector
Trains a classifier on execution traces to detect perturbed code.
This is the core experiment of Paper 3.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TRACES_FILE = Path(__file__).parent.parent / "results" / "traces" / "all_traces.json"


def extract_trace_features(trace: dict) -> list:
    """Convert a trace dict into a numerical feature vector. Always returns exactly 12 features."""
    if trace is None:
        return [0.0] * 12

    features = []
    # Core metrics
    features.append(float(trace.get("success_rate", 0.0)))
    features.append(float(trace.get("output_diversity", 0.0)))
    features.append(float(trace.get("avg_execution_time", 0.0)))
    features.append(float(trace.get("num_test_cases", 0)))

    # Output signature as numeric
    sig = trace.get("output_signature", "0")
    try:
        features.append(float(sig) / 10**10)
    except (ValueError, TypeError):
        features.append(0.0)

    # Exit code statistics
    exit_codes = trace.get("exit_codes", [])
    if exit_codes:
        features.append(float(sum(1 for c in exit_codes if c != 0) / len(exit_codes)))
        features.append(float(sum(1 for c in exit_codes if c == 0) / len(exit_codes)))
    else:
        features.append(0.0)
        features.append(0.0)

    # Execution time statistics
    times = trace.get("execution_times", [])
    if times:
        features.append(float(min(times)))
        features.append(float(max(times)))
        features.append(float(sum(times) / len(times)))
    else:
        features.append(0.0)
        features.append(0.0)
        features.append(0.0)

    # Output pattern analysis
    outputs = trace.get("outputs", [])
    if outputs:
        unique_outputs = len(set(outputs))
        total_outputs = len(outputs)
        features.append(float(unique_outputs) / float(total_outputs))
    else:
        features.append(0.0)

    # Safety: ensure exactly 12 features
    while len(features) < 12:
        features.append(0.0)
    return features[:12]

def build_dataset():
    """Load traces and build feature matrix + labels."""
    with open(TRACES_FILE) as f:
        data = json.load(f)

    X = []
    y = []
    sample_ids = []

    # Clean samples → label 0
    for sample_id, trace in data["clean"].items():
        X.append(extract_trace_features(trace))
        y.append(0)
        sample_ids.append(sample_id)

    # Perturbed samples → label 1
    for sample_id, trace in data["perturbed"].items():
        X.append(extract_trace_features(trace))
        y.append(1)
        sample_ids.append(sample_id)

    return np.array(X), np.array(y), sample_ids


def get_strategy(sample_id: str) -> str:
    """Extract perturbation strategy from sample ID."""
    strategies = ["boundary_invert", "variable_shadow", "import_alias", "dead_code", "comment_plant"]
    for s in strategies:
        if s in sample_id:
            return s
    return "clean"


def evaluate_detector():
    """Train and evaluate multiple detectors on execution traces."""
    print("=== Execution-Guided Detector Evaluation ===\n")

    X, y, sample_ids = build_dataset()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    # Handle class imbalance
    scale = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1

    detectors = {
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, scale_pos_weight=scale, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=== Cross-Validation Results ===\n")
    print(f"{'Detector':<25} {'Accuracy':>8} {'Precision':>9} {'Recall':>8} {'F1':>8}")
    print("-" * 60)

    best_detector = None
    best_f1 = 0.0

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

        if f1 > best_f1:
            best_f1 = f1
            best_detector = name

    print(f"\nBest detector: {best_detector} (F1={best_f1:.3f})")

    # Per-strategy analysis using best detector
    print(f"\n=== Per-Strategy Analysis (Best Detector) ===\n")
    print(f"{'Strategy':<25} {'N':>4} {'Recall':>8}")
    print("-" * 40)

    strategies = ["boundary_invert", "variable_shadow", "import_alias", "dead_code", "comment_plant"]

    # Train on all but one strategy
    for held_out in strategies:
        test_mask = np.array([get_strategy(sid) == held_out for sid in sample_ids])
        train_mask = np.array([get_strategy(sid) != held_out and get_strategy(sid) != "clean" for sid in sample_ids])
        clean_mask = np.array([get_strategy(sid) == "clean" for sid in sample_ids])
        train_mask = train_mask | clean_mask

        if len(np.unique(y[train_mask])) < 2:
            print(f"{held_out:<25} {'—':>4} {'SKIP':>8}")
            continue

        clf = XGBClassifier(
            n_estimators=100, max_depth=4,
            scale_pos_weight=len(y[train_mask][y[train_mask]==0]) / max(len(y[train_mask][y[train_mask]==1]), 1),
            random_state=42
        )
        clf.fit(X[train_mask], y[train_mask])

        preds = clf.predict(X[test_mask])
        true = y[test_mask]

        if sum(test_mask) > 0:
            recall = recall_score(true, preds, zero_division=0)
            n = sum(test_mask)
            print(f"{held_out:<25} {n:>4} {recall:>8.3f}")

    return best_detector, best_f1


if __name__ == "__main__":
    best_name, best_f1 = evaluate_detector()
    print(f"\n=== Results ===")
    print(f"Best detector: {best_name}")
    print(f"Best F1 score: {best_f1:.3f}")
