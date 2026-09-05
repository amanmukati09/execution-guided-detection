"""
Extract All Traces
Processes all 550 samples (100 clean + 450 perturbed) through the
execution trace pipeline and saves results.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from trace_extractor import TraceExtractor

DATA_ROOT = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "traces"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_TEST_CASES = 15  # Balance between signal and speed
TIMEOUT_SEC = 3


def extract_for_sample(filepath: Path, extractor: TraceExtractor) -> Dict[str, Any]:
    """Extract trace for a single code sample."""
    code = filepath.read_text(encoding="utf-8")

    # Extract function name from filename (if it has one)
    # sample_01.py → sample_01
    # sample_01_boundary_invert.py → sample_01_boundary_invert
    stem = filepath.stem

    try:
        trace = extractor.extract_trace(code)
        return {
            "sample_id": stem,
            "trace": trace,
            "status": "success",
        }
    except Exception as e:
        return {
            "sample_id": stem,
            "trace": None,
            "status": "error",
            "error": str(e),
        }


def main():
    print("=== Full Dataset Trace Extraction ===\n")

    extractor = TraceExtractor(num_test_cases=NUM_TEST_CASES, timeout_sec=TIMEOUT_SEC)

    # Process clean samples
    clean_dir = DATA_ROOT / "clean"
    perturbed_dir = DATA_ROOT / "perturbed"

    clean_files = sorted(clean_dir.glob("*.py"))
    perturbed_files = sorted(perturbed_dir.glob("*.py"))

    print(f"Clean samples: {len(clean_files)}")
    print(f"Perturbed samples: {len(perturbed_files)}")
    print(f"Total: {len(clean_files) + len(perturbed_files)}")
    print(f"Test cases per sample: {NUM_TEST_CASES}")
    print(f"Timeout: {TIMEOUT_SEC}s\n")

    # Estimate time
    est_time_per_sample = 0.05 * NUM_TEST_CASES  # ~50ms per test case
    total_est = est_time_per_sample * (len(clean_files) + len(perturbed_files))
    print(f"Estimated time: {total_est:.1f} minutes\n")

    all_results = {
        "clean": {},
        "perturbed": {},
        "metadata": {
            "num_test_cases": NUM_TEST_CASES,
            "timeout_sec": TIMEOUT_SEC,
            "total_samples": len(clean_files) + len(perturbed_files),
        },
    }

    start_time = time.time()
    processed = 0
    errors = 0

    # Process clean samples
    print("Processing clean samples...")
    for i, filepath in enumerate(clean_files, 1):
        result = extract_for_sample(filepath, extractor)
        all_results["clean"][result["sample_id"]] = result["trace"]

        if result["status"] == "error":
            errors += 1

        processed += 1
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = (len(clean_files) + len(perturbed_files) - processed) / rate
            print(f"  [{processed}/{len(clean_files) + len(perturbed_files)}] "
                  f"Elapsed: {elapsed:.0f}s, ETA: {remaining:.0f}s")

    # Process perturbed samples
    print("\nProcessing perturbed samples...")
    for i, filepath in enumerate(perturbed_files, 1):
        result = extract_for_sample(filepath, extractor)
        all_results["perturbed"][result["sample_id"]] = result["trace"]

        if result["status"] == "error":
            errors += 1

        processed += 1
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = (len(clean_files) + len(perturbed_files) - processed) / rate
            print(f"  [{processed}/{len(clean_files) + len(perturbed_files)}] "
                  f"Elapsed: {elapsed:.0f}s, ETA: {remaining:.0f}s")

    total_time = time.time() - start_time

    # Save results
    output_file = RESULTS_DIR / "all_traces.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Extraction Complete ===")
    print(f"Total processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
