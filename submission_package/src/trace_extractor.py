"""
Trace Extractor
Builds execution trace vectors from code behavior.
A trace captures what code DOES, not what it looks like.
"""

import json
import time
from typing import Any, Dict, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from sandbox_bridge import SandboxExecutor
from property_generator import PropertyGenerator


class TraceExtractor:
    """Extracts execution traces from Python functions."""

    def __init__(self, num_test_cases: int = 20, timeout_sec: int = 3):
        self.sandbox = SandboxExecutor(timeout_sec=timeout_sec)
        self.generator = PropertyGenerator(num_cases=num_test_cases)
        self.num_test_cases = num_test_cases

    def extract_trace(self, function_code: str, function_name: str = None) -> Dict[str, Any]:
        """
        Extract execution trace from a function by running it on diverse inputs.
        Returns a trace dict with:
        - outputs: list of outputs for each test input
        - success_rates: fraction of inputs that executed successfully
        - exit_codes: list of exit codes
        - timing: execution time
        - branch_signatures: hash of output patterns
        """
        # Generate test inputs
        inputs = self._generate_inputs(function_code, function_name)

        outputs = []
        exit_codes = []
        success_count = 0
        execution_times = []

        for test_input in inputs:
            start_time = time.time()
            result = self.sandbox.execute_with_inputs(function_code, test_input)
            elapsed = time.time() - start_time

            exit_codes.append(result["exit_code"])
            execution_times.append(elapsed)

            # Parse the output
            try:
                output_data = json.loads(result["stdout"].strip())
                if output_data.get("success"):
                    outputs.append(output_data.get("result"))
                    success_count += 1
                else:
                    outputs.append(f"ERROR: {output_data.get('error', 'Unknown')}")
            except (json.JSONDecodeError, AttributeError):
                outputs.append(f"RAW: {result['stdout'].strip()[:100]}")

        # Build trace vector
        trace = {
            "outputs": outputs,
            "success_rate": success_count / len(inputs) if inputs else 0.0,
            "exit_codes": exit_codes,
            "execution_times": execution_times,
            "num_test_cases": len(inputs),
            "output_signature": self._compute_output_signature(outputs),
            "output_diversity": len(set(outputs)) / len(outputs) if outputs else 0.0,
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0.0,
        }

        return trace

    def _generate_inputs(self, function_code: str, function_name: str = None) -> List[Dict[str, Any]]:
        """Generate test inputs for the function."""
        # Try to execute the function definition to get type hints
        namespace = {}
        try:
            exec(function_code, namespace)
        except Exception:
            pass

        # Find the function
        target_func = None
        if function_name:
            target_func = namespace.get(function_name)
        else:
            # Find first function in namespace
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    target_func = obj
                    break

        if target_func and hasattr(target_func, "__annotations__"):
            return self.generator.generate_for_function(target_func)
        else:
            # Fallback: generate generic inputs
            return self.generator.generate_for_function(lambda x: x)

    def _compute_output_signature(self, outputs: List[str]) -> str:
        """Compute a hash-like signature of the output pattern."""
        # Use a simple deterministic hash
        combined = "|".join(str(o) for o in outputs)
        return str(hash(combined) % 10**10)

    def compare_traces(self, clean_trace: Dict[str, Any], perturbed_trace: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare two execution traces and return difference metrics.
        """
        clean_outputs = clean_trace["outputs"]
        pert_outputs = perturbed_trace["outputs"]

        # Count outputs that differ
        diff_count = sum(1 for c, p in zip(clean_outputs, pert_outputs) if c != p)

        # Success rate difference
        success_diff = abs(clean_trace["success_rate"] - perturbed_trace["success_rate"])

        # Output signature difference
        sig_diff = 1.0 if clean_trace["output_signature"] != perturbed_trace["output_signature"] else 0.0

        # Output diversity difference
        div_diff = abs(clean_trace["output_diversity"] - perturbed_trace["output_diversity"])

        return {
            "output_diff_count": diff_count,
            "output_diff_ratio": diff_count / len(clean_outputs) if clean_outputs else 0.0,
            "success_rate_diff": success_diff,
            "signature_diff": sig_diff,
            "diversity_diff": div_diff,
            "avg_time_diff": abs(clean_trace["avg_execution_time"] - perturbed_trace["avg_execution_time"]),
        }


if __name__ == "__main__":
    extractor = TraceExtractor(num_test_cases=10)

    print("=== Test 1: Extract trace from clean function ===")
    clean_code = '''
def is_even(n: int) -> bool:
    """Check if number is even."""
    return n % 2 == 0
'''
    clean_trace = extractor.extract_trace(clean_code, "is_even")
    print(f"Success rate: {clean_trace['success_rate']}")
    print(f"Output signature: {clean_trace['output_signature']}")
    print(f"Output diversity: {clean_trace['output_diversity']:.2f}")
    print(f"Avg execution time: {clean_trace['avg_execution_time']:.4f}s")

    print("\n=== Test 2: Extract trace from perturbed function ===")
    perturbed_code = '''
def is_even(n: int) -> bool:
    """Check if number is even."""
    return n % 2 == 1  # BUG: flipped condition
'''
    pert_trace = extractor.extract_trace(perturbed_code, "is_even")
    print(f"Success rate: {pert_trace['success_rate']}")
    print(f"Output signature: {pert_trace['output_signature']}")
    print(f"Output diversity: {pert_trace['output_diversity']:.2f}")

    print("\n=== Test 3: Compare traces ===")
    diff = extractor.compare_traces(clean_trace, pert_trace)
    print(f"Output diff ratio: {diff['output_diff_ratio']:.2f}")
    print(f"Signature diff: {diff['signature_diff']}")
    print(f"Success rate diff: {diff['success_rate_diff']:.2f}")
