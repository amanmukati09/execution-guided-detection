"""
Sandbox Bridge
Python interface to the C sandbox executor.
Provides safe execution of Python code snippets with timeout and output capture.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

SANDBOX_BINARY = Path(__file__).parent / "sandbox_executor"


class SandboxExecutor:
    """Executes Python code in the C sandbox and returns structured results."""

    def __init__(self, binary_path: Path = SANDBOX_BINARY, timeout_sec: int = 5):
        self.binary_path = binary_path
        self.timeout_sec = timeout_sec

        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"Sandbox binary not found: {self.binary_path}\n"
                f"Compile it with: gcc -O2 -Wall -o src/sandbox_executor src/sandbox_executor.c"
            )

    def execute(self, code: str, timeout_sec: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute Python code in the sandbox.
        Returns dict with keys: exit_code, timed_out, stdout, stderr, stdout_len, stderr_len
        """
        timeout = timeout_sec or self.timeout_sec

        try:
            result = subprocess.run(
                [str(self.binary_path), code, str(timeout)],
                capture_output=True,
                text=True,
                timeout=timeout + 2,  # Extra buffer for C sandbox overhead
            )

            # Parse JSON output from C sandbox
            if result.stdout.strip():
                return json.loads(result.stdout)
            else:
                return {
                    "exit_code": -99,
                    "timed_out": True,
                    "stdout": "",
                    "stderr": result.stderr,
                    "stdout_len": 0,
                    "stderr_len": len(result.stderr),
                }

        except subprocess.TimeoutExpired:
            return {
                "exit_code": -98,
                "timed_out": True,
                "stdout": "",
                "stderr": "Sandbox timeout expired",
                "stdout_len": 0,
                "stderr_len": 25,
            }
        except Exception as e:
            return {
                "exit_code": -97,
                "timed_out": False,
                "stdout": "",
                "stderr": str(e),
                "stdout_len": 0,
                "stderr_len": len(str(e)),
            }

    def execute_with_inputs(self, function_code: str, test_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a function with specific test inputs.
        Generates a complete Python script that calls the function and prints the result.
        """
        # Build the complete script
        input_str = ", ".join(f"{k}={repr(v)}" for k, v in test_inputs.items())

        script = f"""
{function_code}

# Execute the function with test inputs
import json
try:
    result = None
    # Find the function name
    import re
    match = re.search(r'def\\s+(\\w+)\\s*\\(', '''{function_code}''')
    if match:
        func_name = match.group(1)
        func = locals()[func_name]
        result = func({input_str})
    print(json.dumps({{"result": repr(result), "success": True}}))
except Exception as e:
    print(json.dumps({{"result": None, "success": False, "error": str(e)}}))
"""
        return self.execute(script)

    def batch_execute(self, code_samples: list) -> list:
        """Execute multiple code samples and return results."""
        results = []
        for i, code in enumerate(code_samples):
            result = self.execute(code)
            result["sample_index"] = i
            results.append(result)
        return results


if __name__ == "__main__":
    # Test the bridge
    sandbox = SandboxExecutor()

    print("=== Test 1: Simple execution ===")
    result = sandbox.execute("print('Hello from sandbox bridge')")
    print(f"Exit code: {result['exit_code']}")
    print(f"Stdout: {result['stdout'].strip()}")
    print(f"Timed out: {result['timed_out']}")

    print("\n=== Test 2: Execute function with inputs ===")
    func_code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
    result = sandbox.execute_with_inputs(func_code, {"a": 5, "b": 10})
    print(f"Exit code: {result['exit_code']}")
    print(f"Stdout: {result['stdout'].strip()}")

    print("\n=== Test 3: Timeout handling ===")
    result = sandbox.execute("import time; time.sleep(10)", timeout_sec=1)
    print(f"Exit code: {result['exit_code']}")
    print(f"Timed out: {result['timed_out']}")

    print("\n=== Test 4: Error handling ===")
    result = sandbox.execute("raise ValueError('Test error')")
    print(f"Exit code: {result['exit_code']}")
    print(f"Stderr: {result['stderr'].strip()[:100]}")
