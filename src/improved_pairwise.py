"""
Improved Pairwise Detector
Adds import-awareness to execution traces to better detect import aliasing.
"""

import json
import sys
import re
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from detector import get_strategy

TRACES_FILE = Path(__file__).parent.parent / "results" / "traces" / "all_traces.json"
DATA_ROOT = Path(__file__).parent.parent


def extract_import_features(code: str) -> list:
    """Extract features about imports from code."""
    features = []

    # Count imports
    import_lines = re.findall(r"^\s*(import|from)\s+", code, re.MULTILINE)
    features.append(float(len(import_lines)))

    # Detect aliased imports
    aliased = re.findall(r"import\s+\w+\s+as\s+\w+", code)
    from_aliased = re.findall(r"from\s+\w+\s+import\s+\w+\s+as\s+\w+", code)
    features.append(float(len(aliased) + len(from_aliased)))

    # Detect suspicious aliases (md5 as sha256, etc.)
    suspicious = re.findall(r"import\s+(\w+)\s+as\s+(\w+)", code)
    features.append(1.0 if suspicious else 0.0)

    return features


def compute_enhanced_difference(clean_code: str, pert_code: str, clean_trace: dict, pert_trace: dict) -> list:
    """Compute trace differences + import awareness features."""
    features = []

    # Original trace difference features (from pairwise_detector)
    if clean_trace is None or pert_trace is None:
        features.extend([0.0] * 10)
    else:
        # Success rate difference
        features.append(abs(clean_trace.get("success_rate", 0) - pert_trace.get("success_rate", 0)))

        # Output diversity difference
        features.append(abs(clean_trace.get("output_diversity", 0) - pert_trace.get("output_diversity", 0)))

        # Signature difference
        sig_clean = str(clean_trace.get("output_signature", ""))
        sig_pert = str(pert_trace.get("output_signature", ""))
        features.append(1.0 if sig_clean != sig_pert else 0.0)

        # Exit code pattern difference
        clean_codes = clean_trace.get("exit_codes", [])
        pert_codes = pert_trace.get("exit_codes", [])
        if clean_codes and pert_codes:
            diff_codes = sum(1 for c, p in zip(clean_codes, pert_codes) if c != p)
            features.append(diff_codes / len(clean_codes))
        else:
            features.append(0.0)

        # Execution time difference
        clean_times = clean_trace.get("execution_times", [])
        pert_times = pert_trace.get("execution_times", [])
        if clean_times and pert_times:
            avg_clean = sum(clean_times) / len(clean_times)
            avg_pert = sum(pert_times) / len(pert_times)
            features.append(abs(avg_clean - avg_pert))
        else:
            features.append(0.0)

        # Output difference ratio
        clean_outputs = clean_trace.get("outputs", [])
        pert_outputs = pert_trace.get("outputs", [])
        if clean_outputs and pert_outputs:
            diff_count = sum(1 for c, p in zip(clean_outputs, pert_outputs) if c != p)
            features.append(diff_count / len(clean_outputs))
        else:
            features.append(0.0)

        # Unique outputs clean
        if clean_outputs:
            features.append(len(set(clean_outputs)) / len(clean_outputs))
        else:
            features.append(0.0)

        # Unique outputs perturbed
        if pert_outputs:
            features.append(len(set(pert_outputs)) / len(pert_outputs))
        else:
            features.append(0.0)

        # Error rate difference
        clean_errors = sum(1 for c in clean_trace.get("exit_codes", []) if c != 0)
        pert_errors = sum(1 for c in pert_trace.get("exit_codes", []) if c != 0)
        if clean_trace.get("exit_codes"):
            features.append(abs(clean_errors - pert_errors) / len(clean_trace["exit_codes"]))
        else:
            features.append(0.0)

    # Import awareness features
    clean_imports = extract_import_features(clean_code)
    pert_imports = extract_import_features(pert_code)
    features.extend(clean_imports)
    features.extend(pert_imports)
    features.append(abs(clean_imports[1] - pert_imports[1]))  # Aliased import count diff
    features.append(pert_imports[2])  # Suspicious alias present
    # Safety: ensure consistent length
    while len(features) < 16:
        features.append(0.0)
    return features[:16]

def build_enhanced_dataset():
    """Build dataset with enhanced features."""
    with open(TRACES_FILE) as f:
        data = json.load(f)

    clean_traces = data["clean"]
    perturbed_traces = data["perturbed"]

    clean_dir = DATA_ROOT / "data" / "clean"
    perturbed_dir = DATA_ROOT / "data" / "perturbed"

    X = []
    y = []
    sample_ids = []

    for pert_id, pert_trace in perturbed_traces.items():
        parts = pert_id.split("_")
        base_id = parts[0] + "_" + parts[1]

        clean_trace = clean_traces.get(base_id)
        if clean_trace is None:
            continue

        # Load code files
        clean_file = clean_dir / f"{base_id}.py"
        pert_file = perturbed_dir / f"{pert_id}.py"

        if clean_file.exists() and pert_file.exists():
            clean_code = clean_file.read_text(encoding="utf-8")
            pert_code = pert_file.read_text(encoding="utf-8")

            features = compute_enhanced_difference(clean_code, pert_code, clean_trace, pert_trace)
            X.append(features)
            y.append(1)
            sample_ids.append(pert_id)

    # Create negative examples: clean vs different clean
    clean_items = list(clean_traces.items())
    for i in range(len(clean_items) - 1):
        clean_id1, clean_trace1 = clean_items[i]
        clean_id2, clean_trace2 = clean_items[i + 1]

        clean_file1 = clean_dir / f"{clean_id1}.py"
        clean_file2 = clean_dir / f"{clean_id2}.py"

        if clean_file1.exists() and clean_file2.exists():
            clean_code1 = clean_file1.read_text(encoding="utf-8")
            clean_code2 = clean_file2.read_text(encoding="utf-8")

            features = compute_enhanced_difference(clean_code1, clean_code2, clean_trace1, clean_trace2)
            X.append(features)
            y.append(0)
            sample_ids.append(f"{clean_id1}_vs_{clean_id2}")

    return np.array(X), np.array(y), sample_ids


def main():
    print("=== Improved Pairwise Detector (Import-Aware) ===\n")

    X, y, sample_ids = build_enhanced_dataset()
    print(f"Dataset: {X.shape[0]} comparisons, {X.shape[1]} features")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    detectors = {
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, random_state=42),
    }

    print("=== Cross-Validation Results ===\n")
    print(f"{'Detector':<25} {'Accuracy':>8} {'Precision':>9} {'Recall':>8} {'F1':>8}")
    print("-" * 60)

    for name, clf in detectors.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()
        prec = cross_val_score(pipe, X, y, cv=cv, scoring="precision").mean()
        rec = cross_val_score(pipe, X, y, cv=cv, scoring="recall").mean()
        f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean()

        print(f"{name:<25} {acc:>8.3f} {prec:>9.3f} {rec:>8.3f} {f1:>8.3f}")

    # Per-strategy
    print(f"\n=== Per-Strategy Detection (Import-Aware) ===\n")
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
