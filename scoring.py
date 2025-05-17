
import subprocess
import json
import os
import tempfile
import numpy as np
from config import default_config

def calculate_score(code, unit_tests, language="python"):
    """
    Calculate a score for the given code based on unit tests.
    
    Args:
        code (str): The code to evaluate
        unit_tests (str): The unit tests to run against the code
        language (str): The programming language of the code (python or typescript)
        
    Returns:
        dict: A dictionary containing the test results and score
    """
    if language.lower() == "python":
        return calculate_python_score(code, unit_tests)
    elif language.lower() == "typescript":
        return calculate_typescript_score(code, unit_tests)
    else:
        return {"error": f"Unsupported language: {language}", "score": 0}

def calculate_python_score(code, unit_tests):
    """Execute Python code and unit tests in a Docker container."""
    try:
        # Create a JSON payload with the code and unit tests
        payload = json.dumps({"code": code, "unit_tests": unit_tests})

        # Execute the code in a separate process using a Docker container
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",  # Automatically remove the container after it exits
                "-i",  # Pass input to the container
                "python-sandbox",  # The name of the Docker image
                "python",
                "-c",
                "import sys; import json; payload = json.load(sys.stdin); exec(payload['code']); exec(payload['unit_tests'])"
            ],
            input=payload,  # Pass the payload as a string
            capture_output=True,  # Capture the output and error streams
            text=True,  # Decode the output as text
            timeout=10  # Set a timeout to prevent infinite loops
        )

        # Parse the results
        return parse_python_results(result)
    except subprocess.TimeoutExpired:
        return {"error": "Execution timed out", "score": 0}
    except Exception as e:
        return {"error": f"An error occurred: {e}", "score": 0}

def calculate_typescript_score(code, unit_tests):
    """Execute TypeScript code and unit tests in a Docker container."""
    try:
        # Create a JSON payload with the code and unit tests
        payload = json.dumps({"code": code, "tests": unit_tests})

        # Execute the code in a Docker container
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",  # Automatically remove the container after it exits
                "-i",  # Pass input to the container
                "typescript-sandbox",  # The name of the Docker image
            ],
            input=payload,  # Pass the payload as a string
            capture_output=True,  # Capture the output and error streams
            text=True,  # Decode the output as text
            timeout=10  # Set a timeout to prevent infinite loops
        )

        # Parse the results
        return parse_typescript_results(result)
    except subprocess.TimeoutExpired:
        return {"error": "Execution timed out", "score": 0}
    except Exception as e:
        return {"error": f"An error occurred: {e}", "score": 0}

def parse_python_results(result):
    """Parse the results from the Python execution."""
    if result.returncode == 0:
        # The code executed successfully
        try:
            # Try to parse the output as JSON
            output = json.loads(result.stdout)
            return {
                "passed": output.get("passed", 0),
                "failed": output.get("failed", 0),
                "total": output.get("total", 0),
                "score": calculate_test_score(output.get("passed", 0), output.get("total", 0)),
                "details": output.get("details", []),
                "raw_output": result.stdout
            }
        except json.JSONDecodeError:
            # If the output is not JSON, return the raw output
            return {
                "error": "Failed to parse test results",
                "raw_output": result.stdout,
                "score": 0
            }
    else:
        # The code raised an exception
        return {
            "error": result.stderr,
            "raw_output": result.stderr,
            "score": 0
        }

def parse_typescript_results(result):
    """Parse the results from the TypeScript execution."""
    if result.returncode == 0:
        try:
            # Try to parse the output as JSON
            output = json.loads(result.stdout)
            
            if "error" in output:
                return {
                    "error": output.get("message", "Unknown error"),
                    "raw_output": result.stdout,
                    "score": 0
                }
            
            # Count passed and failed tests
            results = output.get("results", [])
            passed = sum(1 for r in results if r.get("passed", False))
            total = len(results)
            
            return {
                "passed": passed,
                "failed": total - passed,
                "total": total,
                "score": calculate_test_score(passed, total),
                "details": results,
                "raw_output": result.stdout
            }
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse test results",
                "raw_output": result.stdout,
                "score": 0
            }
    else:
        return {
            "error": result.stderr,
            "raw_output": result.stderr,
            "score": 0
        }

def calculate_test_score(passed, total):
    """Calculate a score based on the number of passed tests."""
    if total == 0:
        return 0
    return (passed / total) * 100

def analyze_code_quality(code, language="python"):
    """
    Analyze the quality of the given code.
    
    Args:
        code (str): The code to analyze
        language (str): The programming language of the code (python or typescript)
        
    Returns:
        dict: A dictionary containing the quality metrics and score
    """
    if language.lower() == "python":
        return analyze_python_quality(code)
    elif language.lower() == "typescript":
        return analyze_typescript_quality(code)
    else:
        return {"error": f"Unsupported language: {language}", "score": 0}

def analyze_python_quality(code):
    """Analyze the quality of Python code using pylint."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(code.encode())
            temp_file = f.name
        
        result = subprocess.run(
            ["pylint", "--output-format=json", temp_file],
            capture_output=True,
            text=True
        )
        
        os.unlink(temp_file)
        
        try:
            # Parse the pylint output
            pylint_output = json.loads(result.stdout)
            
            # Calculate a score based on the pylint output
            if not pylint_output:
                return {"score": 100, "issues": []}
            
            # Count issues by type
            issue_counts = {"convention": 0, "refactor": 0, "warning": 0, "error": 0}
            for issue in pylint_output:
                issue_type = issue.get("type", "")
                if issue_type in issue_counts:
                    issue_counts[issue_type] += 1
            
            # Calculate a weighted score
            # Errors are most severe, followed by warnings, refactors, and conventions
            weights = {"error": 10, "warning": 3, "refactor": 1, "convention": 0.5}
            penalty = sum(count * weights[issue_type] for issue_type, count in issue_counts.items())
            
            # Cap the penalty at 100 to ensure the score is between 0 and 100
            penalty = min(penalty, 100)
            score = 100 - penalty
            
            return {
                "score": max(0, score),
                "issues": pylint_output,
                "issue_counts": issue_counts
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse pylint output", "score": 0}
    except Exception as e:
        return {"error": f"An error occurred: {e}", "score": 0}

def analyze_typescript_quality(code):
    """Analyze the quality of TypeScript code using ESLint."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
            f.write(code.encode())
            temp_file = f.name
        
        # Check if ESLint is available
        try:
            subprocess.run(["eslint", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # ESLint not available, return a basic score
            os.unlink(temp_file)
            return {"score": 50, "issues": [], "note": "ESLint not available for detailed analysis"}
        
        # Run ESLint
        result = subprocess.run(
            ["eslint", "--format=json", temp_file],
            capture_output=True,
            text=True
        )
        
        os.unlink(temp_file)
        
        try:
            # Parse the ESLint output
            eslint_output = json.loads(result.stdout)
            
            if not eslint_output or not eslint_output[0].get("messages"):
                return {"score": 100, "issues": []}
            
            # Count issues by severity
            messages = eslint_output[0].get("messages", [])
            issue_counts = {1: 0, 2: 0}  # 1: warning, 2: error
            
            for message in messages:
                severity = message.get("severity", 0)
                if severity in issue_counts:
                    issue_counts[severity] += 1
            
            # Calculate a weighted score
            # Errors (2) are more severe than warnings (1)
            weights = {1: 1, 2: 5}
            penalty = sum(count * weights[severity] for severity, count in issue_counts.items())
            
            # Cap the penalty at 100 to ensure the score is between 0 and 100
            penalty = min(penalty, 100)
            score = 100 - penalty
            
            return {
                "score": max(0, score),
                "issues": messages,
                "issue_counts": {
                    "warning": issue_counts[1],
                    "error": issue_counts[2]
                }
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse ESLint output", "score": 0}
    except Exception as e:
        return {"error": f"An error occurred: {e}", "score": 0}

def evaluate_code(code, unit_tests, language="python", weights=None):
    """
    Evaluate code based on test results and code quality.
    
    Args:
        code (str): The code to evaluate
        unit_tests (str): The unit tests to run against the code
        language (str): The programming language of the code (python or typescript)
        weights (dict): A dictionary containing the weights for each metric
            - test_weight: Weight for test results (default from configuration)
            - quality_weight: Weight for code quality (default from configuration)
            
    Returns:
        dict: A dictionary containing the evaluation results and final score
    """
    # Get configuration
    config = default_config
    
    # Set default weights if not provided
    if weights is None:
        weights = {
            "test_weight": config.simulation.test_weight,
            "quality_weight": config.simulation.quality_weight
        }
    
    # Calculate test score
    test_results = calculate_score(code, unit_tests, language)
    test_score = test_results.get("score", 0)
    
    # Analyze code quality
    quality_results = analyze_code_quality(code, language)
    quality_score = quality_results.get("score", 0)
    
    # Calculate final score
    final_score = (
        weights["test_weight"] * test_score +
        weights["quality_weight"] * quality_score
    )
    
    return {
        "test_results": test_results,
        "quality_results": quality_results,
        "test_score": test_score,
        "quality_score": quality_score,
        "final_score": final_score,
        "weights": weights,
        "language": language
    }

if __name__ == "__main__":
    # Example Python code
    python_code = """
def add(a, b):
    return a + b
"""
    python_tests = """
import unittest

class TestAddFunction(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
        
if __name__ == '__main__':
    unittest.main()
"""
    
    # Example TypeScript code
    typescript_code = """
export function add(a: number, b: number): number {
    return a + b;
}
"""
    typescript_tests = """
import { expect } from 'chai';
import { add } from './solution';

function testAdd() {
    expect(add(1, 2)).to.equal(3);
}
"""
    
    # Evaluate Python code
    python_results = evaluate_code(python_code, python_tests, "python")
    print("Python Evaluation:")
    print(f"Test Score: {python_results['test_score']}")
    print(f"Quality Score: {python_results['quality_score']}")
    print(f"Final Score: {python_results['final_score']}")
    
    # Evaluate TypeScript code
    typescript_results = evaluate_code(typescript_code, typescript_tests, "typescript")
    print("\nTypeScript Evaluation:")
    print(f"Test Score: {typescript_results['test_score']}")
    print(f"Quality Score: {typescript_results['quality_score']}")
    print(f"Final Score: {typescript_results['final_score']}")
