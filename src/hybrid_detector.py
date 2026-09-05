"""
Hybrid Detector
Combines static AST features (Paper 1) with execution traces (Paper 3)
to build the ultimate perturbation detector.
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
sys.path.insert(0, str(Path(__file__).parent.parent / "adversarial-code-perturbations" / "src"))

from detector import build_dataset as build_trace_dataset
from detector import extract_trace_features, get_strategy

# Paper 1 feature extractor
from feature_extractor import extract_features as extract_ast_features

DATA_ROOT = Path(__file__).parent.parent


def build_hybrid_dataset():
    """Build combined feature matrix: AST + execution traces."""
    X_trace, y_trace, sample_ids = build_trace_dataset()

    # Load code samples for AST features
    clean_dir = DATA_ROOT / "data" / "clean"
    perturbed_dir = DATA_ROOT / "data" / "perturbed"

    ast_features = []
    for sid in sample_ids:
        # Determine file path
        if get_strategy(sid) == "clean":
            filepath = clean_dir / f"{sid}.py"
        else:
            filepath = perturbed_dir / f"{sid}.py"

        if filepath.exists():
            code = filepath.read_text(encoding="utf-8")
            feats = extract_ast_features(code)
            ast_features.append(list(feats.values()))
        else:
            ast_features.append([0.0] * 17)  # Default 17 features

    X_ast = np.array(ast_features)

    # Combine
    X_combined = np.hstack([X_trace, X_ast])

    return X_combined, y_trace, sample_ids


def main():
    print("=== Hybrid Detector: Static + Execution Features ===\n")

    X, y, sample_ids = build_hybrid_dataset()
    print(f"Combined features: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    scale = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1

    detectors = {
        "Gradient Boosting (Hybrid)": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ),
        "XGBoost (Hybrid)": XGBClassifier(
            n_estimators=100, max_depth=4, scale_pos_weight=scale, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=== Cross-Validation Results ===\n")
    print(f"{'Detector':<30} {'Accuracy':>8} {'Precision':>9} {'Recall':>8} {'F1':>8}")
    print("-" * 65)

    for name, clf in detectors.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()
        prec = cross_val_score(pipe, X, y, cv=cv, scoring="precision").mean()
        rec = cross_val_score(pipe, X, y, cv=cv, scoring="recall").mean()
        f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean()

        print(f"{name:<30} {acc:>8.3f} {prec:>9.3f} {rec:>8.3f} {f1:>8.3f}")

    # Per-strategy analysis with best hybrid
    print(f"\n=== Per-Strategy Analysis (Hybrid) ===\n")
    print(f"{'Strategy':<25} {'N':>4} {'Recall':>8}")
    print("-" * 40)

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    strategies = ["boundary_invert", "variable_shadow", "import_alias", "dead_code", "comment_plant"]

    for held_out in strategies:
        test_mask = np.array([get_strategy(sid) == held_out for sid in sample_ids])
        train_mask = np.array([get_strategy(sid) != held_out and get_strategy(sid) != "clean" for sid in sample_ids])
        clean_mask = np.array([get_strategy(sid) == "clean" for sid in sample_ids])
        train_mask = train_mask | clean_mask

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
