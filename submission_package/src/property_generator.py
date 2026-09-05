"""
Property-Based Test Input Generator
Auto-generates diverse test inputs for Python functions based on type hints.
Supports int, float, str, list, dict, bool, and Optional types.
"""

import ast
import random
import string
from typing import Any, Dict, List, Tuple

# Set random seed for reproducibility
random.seed(42)

# Constants
MAX_LIST_LENGTH = 10
MAX_STRING_LENGTH = 50
MAX_INT_VALUE = 10**6
BOUNDARY_VALUES = [-1, 0, 1, 2, 10, 100, MAX_INT_VALUE, -MAX_INT_VALUE]
FLOAT_BOUNDARIES = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 10**6]


class PropertyGenerator:
    """Generates test inputs for Python functions based on type hints."""

    def __init__(self, num_cases: int = 20, seed: int = 42):
        self.num_cases = num_cases
        random.seed(seed)

    def generate_for_function(self, func) -> List[Dict[str, Any]]:
        """Generate test input dictionaries for a function's parameters."""
        # Parse type hints from function signature
        hints = func.__annotations__
        param_names = [p for p in func.__code__.co_varnames[:func.__code__.co_argcount]]

        test_cases = []
        for _ in range(self.num_cases):
            inputs = {}
            for param in param_names:
                type_hint = hints.get(param, str)  # Default to str if no hint
                inputs[param] = self._generate_value(type_hint)
            test_cases.append(inputs)
        return test_cases

    def _generate_value(self, type_hint: Any) -> Any:
        """Generate a value matching the given type hint."""
        # Handle Optional[X] -> None or X
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is not None:
            if type_hint.__origin__ in (list, List):
                inner = type_hint.__args__[0] if type_hint.__args__ else int
                return self._generate_list(inner)
            elif type_hint.__origin__ in (dict, Dict):
                key_type = type_hint.__args__[0] if type_hint.__args__ else str
                val_type = type_hint.__args__[1] if type_hint.__args__ else int
                return self._generate_dict(key_type, val_type)
            elif type_hint.__origin__ in (tuple, Tuple):
                return tuple(self._generate_value(t) for t in type_hint.__args__)

        # Handle Optional
        if hasattr(type_hint, "__union__") or (hasattr(type_hint, "__origin__") and str(type_hint).startswith("typing.Optional")):
            if random.random() < 0.2:
                return None
            # Get the non-None type
            if hasattr(type_hint, "__args__"):
                non_none_types = [t for t in type_hint.__args__ if t is not type(None)]
                if non_none_types:
                    return self._generate_value(non_none_types[0])

        # Basic types
        if type_hint is int or type_hint == int:
            return self._generate_int()
        elif type_hint is float or type_hint == float:
            return self._generate_float()
        elif type_hint is str or type_hint == str:
            return self._generate_string()
        elif type_hint is bool or type_hint == bool:
            return random.choice([True, False])
        elif type_hint is None or type_hint == type(None):
            return None

        # Fallback: generate based on type name
        type_name = str(type_hint).lower()
        if "int" in type_name:
            return self._generate_int()
        elif "float" in type_name:
            return self._generate_float()
        elif "str" in type_name:
            return self._generate_string()
        elif "bool" in type_name:
            return random.choice([True, False])
        elif "list" in type_name:
            return self._generate_list(int)
        elif "dict" in type_name:
            return self._generate_dict(str, int)
        else:
            # Default to a simple integer
            return self._generate_int()

    def _generate_int(self) -> int:
        """Generate an integer, biased toward boundary values."""
        if random.random() < 0.3:
            return random.choice(BOUNDARY_VALUES)
        return random.randint(-1000, 1000)

    def _generate_float(self) -> float:
        """Generate a float, biased toward boundaries."""
        if random.random() < 0.3:
            return random.choice(FLOAT_BOUNDARIES)
        return random.uniform(-100.0, 100.0)

    def _generate_string(self) -> str:
        """Generate a random string of varying length and content."""
        length = random.randint(0, MAX_STRING_LENGTH)
        if random.random() < 0.2:
            # Include special characters and digits
            chars = string.ascii_letters + string.digits + string.punctuation
        else:
            chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    def _generate_list(self, inner_type: Any, max_len: int = MAX_LIST_LENGTH) -> list:
        """Generate a random list."""
        length = random.randint(0, max_len)
        return [self._generate_value(inner_type) for _ in range(length)]

    def _generate_dict(self, key_type: Any, val_type: Any, max_entries: int = 5) -> dict:
        """Generate a random dictionary."""
        num_entries = random.randint(0, max_entries)
        result = {}
        for _ in range(num_entries):
            key = self._generate_value(key_type)
            val = self._generate_value(val_type)
            try:
                result[key] = val
            except (TypeError, ValueError):
                result[str(key)] = val
        return result


def extract_function_signature(code: str, function_name: str = None) -> Tuple[str, List[str], Dict[str, Any]]:
    """Extract function name, parameters, and type hints from source code."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if function_name is None or node.name == function_name:
                params = [a.arg for a in node.args.args]
                hints = {}
                for arg in node.args.args:
                    if arg.annotation is not None:
                        try:
                            hints[arg.arg] = ast.literal_eval(arg.annotation)
                        except (ValueError, SyntaxError):
                            hints[arg.arg] = str
                return node.name, params, hints
    return None, [], {}


if __name__ == "__main__":
    # Test the generator
    def test_func(a: int, b: str, c: list[int]) -> bool:
        return a > 0 and len(b) > 2 and len(c) > 0

    gen = PropertyGenerator(num_cases=5)
    cases = gen.generate_for_function(test_func)

    print("=== Generated Test Cases ===")
    for i, case in enumerate(cases):
        print(f"Case {i+1}: {case}")

    # Test signature extraction
    code = '''
def example(x: int, y: str) -> list:
    return [x, y]
'''
    name, params, hints = extract_function_signature(code)
    print(f"\n=== Extracted Signature ===")
    print(f"Function: {name}")
    print(f"Params: {params}")
    print(f"Hints: {hints}")
