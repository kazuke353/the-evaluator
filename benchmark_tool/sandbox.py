
import subprocess
import json

def execute_python_code(code, unit_tests):
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
                "sandbox.py" # Execute sandbox.py inside the container
            ],
            input=payload.encode("utf-8"),  # Encode the payload as bytes
            capture_output=True,  # Capture the output and error streams
            text=True,  # Decode the output as text
            timeout=10  # Set a timeout to prevent infinite loops
        )

        # Check the return code
        if result.returncode == 0:
            # The code executed successfully
            return result.stdout
        else:
            # The code raised an exception
            return result.stderr
    except subprocess.TimeoutExpired:
        return "Execution timed out"
    except Exception as e:
        return f"An error occurred: {e}"

def execute_typescript_code(code, unit_tests):
    """
    Execute TypeScript code in a Docker container and return the results.
    
    Args:
        code (str): The TypeScript code to execute
        unit_tests (str): The unit tests to run against the code
        
    Returns:
        dict: The results of the execution
    """
    import json
    import subprocess
    import tempfile
    import os
    
    # Create a temporary directory for Docker volume mounting
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create input data for the TypeScript sandbox
        input_data = {
            "code": code,
            "tests": unit_tests
        }
        
        # Write input data to a file
        input_file = os.path.join(temp_dir, "input.json")
        with open(input_file, "w") as f:
            json.dump(input_data, f)
        
        # Run the Docker container
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{input_file}:/app/input.json",
                    "--network=none",  # Disable network access
                    "--memory=512m",   # Limit memory
                    "--cpus=1",        # Limit CPU
                    "typescript-sandbox",
                    "cat", "/app/input.json", "|", "node", "ts-sandbox.js"
                ],
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )
            
            # Parse the output
            if result.returncode == 0 and result.stdout:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {
                        "error": True,
                        "message": "Failed to parse output",
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
            else:
                return {
                    "error": True,
                    "message": "Execution failed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
        except subprocess.TimeoutExpired:
            return {
                "error": True,
                "message": "Execution timed out"
            }
        except Exception as e:
            return {
                "error": True,
                "message": str(e)
            }

if __name__ == "__main__":
    import sys

    # Read the JSON payload from stdin
    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    unit_tests = payload["unit_tests"]

    # Execute the code and unit tests
    try:
        # Execute the user's code
        exec(code)

        # Execute the unit tests
        import unittest
        suite = unittest.TestSuite()
        exec(unit_tests)
        unittest.TextTestRunner().run(suite)
    except Exception as e:
        print(f"Error during execution: {e}")