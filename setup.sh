#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Ensure that pipelinespipefail: return the exit status of the last command in the pipe that failed
set -o pipefail

# --- Dependency Checks ---
echo "Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is required but not installed."
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed."
    exit 1
fi

if ! command -v netstat &> /dev/null && ! command -v ss &> /dev/null; then
    echo "Error: netstat or ss command is required for checking ports but not found."
    exit 1
fi
echo "All initial dependencies found."

# --- Virtual Environment Setup ---
echo "Setting up Python virtual environment..."
python3 -m venv venv || {
    echo "Error: Failed to create virtual environment. Make sure python3-venv is installed."
    echo "On Debian/Ubuntu, you can install it with: sudo apt install python3-venv"
    exit 1
}

# Activate the virtual environment
# shellcheck disable=SC1091
source venv/bin/activate || {
    echo "Error: Failed to activate virtual environment."
    exit 1
}

if ! command -v pip &> /dev/null; then
    echo "Error: pip not found in virtual environment."
    exit 1
fi

echo "Installing Python dependencies in virtual environment..."
pip install fastapi uvicorn httpx numpy matplotlib deap requests pydantic streamlit || {
    echo "Error: Failed to install Python dependencies."
    deactivate || true # Attempt to deactivate
    exit 1
}

# --- Directory Setup ---
mkdir -p visualizations results

# --- Docker Image Builds ---
DOCKER_CONTEXT_DIR="benchmark_tool" # Define once, use multiple times

if [ ! -d "$DOCKER_CONTEXT_DIR" ]; then
    echo "Error: Docker context directory '$DOCKER_CONTEXT_DIR' not found."
    echo "Please ensure the '$DOCKER_CONTEXT_DIR' directory exists and contains the Dockerfiles."
    deactivate || true
    exit 1
fi
if [ ! -f "$DOCKER_CONTEXT_DIR/Dockerfile" ]; then
    echo "Error: Python sandbox Dockerfile '$DOCKER_CONTEXT_DIR/Dockerfile' not found."
    deactivate || true
    exit 1
fi
if [ ! -f "$DOCKER_CONTEXT_DIR/typescript.Dockerfile" ]; then
    echo "Error: TypeScript sandbox Dockerfile '$DOCKER_CONTEXT_DIR/typescript.Dockerfile' not found."
    deactivate || true
    exit 1
fi


echo "Building Python sandbox Docker image..."
docker build -t python-sandbox -f "$DOCKER_CONTEXT_DIR/Dockerfile" "$DOCKER_CONTEXT_DIR/" || {
    echo "Error: Failed to build Python sandbox Docker image."
    deactivate || true
    exit 1
}

echo "Building TypeScript sandbox Docker image..."
docker build -t typescript-sandbox -f "$DOCKER_CONTEXT_DIR/typescript.Dockerfile" "$DOCKER_CONTEXT_DIR/" || {
    echo "Error: Failed to build TypeScript sandbox Docker image."
    deactivate || true
    exit 1
}

# --- Model Proxy Check ---
MODEL_PROXY_URL="http://localhost:8080/v1/models"
echo "Checking if model proxy is running at $MODEL_PROXY_URL..."
if ! curl -fsS "$MODEL_PROXY_URL" > /dev/null 2>&1; then
    echo "Warning: Model proxy doesn't seem to be running or responding correctly at $MODEL_PROXY_URL."
    echo "Please make sure the model proxy is running before using the benchmark tool."
fi

# --- Port Finding Function ---
find_available_port() {
    local start_port=$1
    local max_port=$2
    local current_port=$start_port
    while true; do
        if command -v netstat &> /dev/null; then
            if ! netstat -tuln | grep -q ":$current_port "; then
                echo "$current_port"
                return 0
            fi
        elif command -v ss &> /dev/null; then
             if ! ss -tuln | grep -q ":$current_port "; then
                echo "$current_port"
                return 0
            fi
        else
            # Fallback if somehow netstat/ss check failed earlier but one is missing now
            echo "Error: netstat or ss command vanished. Cannot find port." >&2
            return 1
        fi

        current_port=$((current_port + 1))
        if [ "$current_port" -gt "$max_port" ]; then
            echo "Error: Could not find an available port between $start_port and $max_port." >&2
            return 1
        fi
    done
}

# --- FastAPI Server Setup ---
FASTAPI_PORT=$(find_available_port 3400 3500)
if [ -z "$FASTAPI_PORT" ]; then
    deactivate || true
    exit 1
fi

echo "Starting benchmark tool (FastAPI) on port $FASTAPI_PORT..."
# Use the Python from the virtual environment to start the server
PORT="$FASTAPI_PORT" "$PWD/venv/bin/uvicorn" main:app --host 0.0.0.0 --port "$FASTAPI_PORT" &
SERVER_PID=$!

# --- Wait for FastAPI Server ---
wait_for_server() {
    local pid=$1
    local url=$2
    local service_name=$3
    local attempts=0
    local max_attempts=30 # 30 attempts * 1 second = 30 seconds timeout

    echo "Waiting for $service_name to start and respond at $url..."
    while true; do
        attempts=$((attempts + 1))
        if [ $attempts -gt $max_attempts ]; then
            echo "Error: $service_name failed to start or respond at $url within $max_attempts seconds."
            if kill -0 "$pid" 2>/dev/null; then
                echo "$service_name process (PID: $pid) is running but not responding. Check logs."
            else
                echo "$service_name process (PID: $pid) failed to start or has crashed."
            fi
            return 1 # Failure
        fi

        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Error: $service_name process (PID: $pid) exited prematurely."
            return 1 # Failure
        fi

        if curl -fsS "$url" > /dev/null 2>&1; then
            echo "$service_name started and responded successfully."
            return 0 # Success
        fi
        sleep 1
        echo -n "." # Progress indicator
    done
    echo # Newline after progress dots
}

if ! wait_for_server "$SERVER_PID" "http://localhost:$FASTAPI_PORT" "FastAPI server"; then
    # Cleanup is handled by trap, but explicit deactivate and exit if server fails to start
    kill "$SERVER_PID" 2>/dev/null || true
    deactivate || true
    exit 1
fi

# --- Streamlit Setup ---
STREAMLIT_PORT=$(find_available_port 8501 8600)
if [ -z "$STREAMLIT_PORT" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    deactivate || true
    exit 1
fi

echo "Starting Streamlit interface on port $STREAMLIT_PORT..."
export API_URL="http://localhost:$FASTAPI_PORT"
export API_PORT="$FASTAPI_PORT"
export API_HOST="localhost" # Keep if streamlit app needs it explicitly
# STREAMLIT_SERVER_PORT and STREAMLIT_SERVER_HEADLESS are passed as CLI args

"$PWD/venv/bin/streamlit" run streamlit_app.py --server.port="$STREAMLIT_PORT" --server.headless=true &
STREAMLIT_PID=$!

if ! wait_for_server "$STREAMLIT_PID" "http://localhost:$STREAMLIT_PORT" "Streamlit interface"; then
    kill "$SERVER_PID" 2>/dev/null || true
    kill "$STREAMLIT_PID" 2>/dev/null || true
    deactivate || true
    exit 1
fi

echo "Benchmark tool is now fully operational."
echo "- FastAPI server running at http://localhost:$FASTAPI_PORT (PID: $SERVER_PID)"
echo "- Streamlit interface running at http://localhost:$STREAMLIT_PORT (PID: $STREAMLIT_PID)"

# --- Open Browser ---
STREAMLIT_URL="http://localhost:$STREAMLIT_PORT"
echo "Attempting to open Streamlit interface in your browser: $STREAMLIT_URL"
if command -v plandex &> /dev/null && plandex browser "$STREAMLIT_URL"; then
    echo "Plandex browser opened."
elif command -v xdg-open &> /dev/null; then
    xdg-open "$STREAMLIT_URL"
elif command -v open &> /dev/null; then # macOS
    open "$STREAMLIT_URL"
elif command -v start &> /dev/null; then # Windows (less likely in this script's context but for completeness)
    start "$STREAMLIT_URL"
else
    echo "Could not automatically open browser. Please manually navigate to $STREAMLIT_URL"
fi


# --- Cleanup Function ---
cleanup() {
    echo # Newline for cleaner exit messages
    echo "Cleaning up..."

    # Stop the FastAPI server
    if [ -n "${SERVER_PID-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping benchmark tool server (PID: $SERVER_PID)..."
        kill "$SERVER_PID"
        for _ in {1..5}; do # Wait up to 5 seconds
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi; sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server did not stop gracefully, forcing termination..."
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
    fi

    # Stop the Streamlit server
    if [ -n "${STREAMLIT_PID-}" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo "Stopping Streamlit interface (PID: $STREAMLIT_PID)..."
        kill "$STREAMLIT_PID"
        for _ in {1..5}; do # Wait up to 5 seconds
            if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then break; fi; sleep 1
        done
        if kill -0 "$STREAMLIT_PID" 2>/dev/null; then
            echo "Streamlit did not stop gracefully, forcing termination..."
            kill -9 "$STREAMLIT_PID" 2>/dev/null || true
        fi
    fi

    echo "Deactivating virtual environment..."
    deactivate 2>/dev/null || true # Suppress errors if not active or deactivate fails

    echo "Cleanup complete."
}

# Set up trap to ensure cleanup on exit, interrupt, or termination signals
trap cleanup EXIT INT TERM

# --- Wait for Servers to Exit ---
# The script will remain here until both processes exit or the script is interrupted.
# The trap will handle cleanup.
echo "Benchmark tool setup complete. Monitoring background processes."
echo "Press Ctrl+C to stop the tool."

# Wait for PIDs. If any process exits, `wait` will return, and the script will exit, triggering the cleanup.
# If running with `set -e`, if any of the `wait`ed PIDs exit with non-zero, the script would exit.
# However, `wait` itself on a specific PID returns that PID's exit status.
# To wait for all and exit if any exits:
if ! wait "$SERVER_PID" "$STREAMLIT_PID"; then
    echo "One or more background processes exited with an error."
    # Cleanup will be called by the EXIT trap.
    # If you need specific error code from wait:
    # server_exit_status=$?
    # Not straightforward to get individual exit statuses when waiting on multiple PIDs this way
    # and have `set -e` behave predictably for *which* process failed.
    # The EXIT trap is the main mechanism for ensuring cleanup.
fi


echo "All benchmark tool processes have finished."
# The EXIT trap will handle final deactivation.