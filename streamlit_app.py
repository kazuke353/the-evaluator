
import streamlit as st
import requests
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("streamlit_app")

# Set page configuration
st.set_page_config(
    page_title="Benchmark Tool",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API URL with better fallback logic
# First try environment variable, then try to detect the port from the FastAPI server
API_PORT = os.environ.get("API_PORT", "3400")  # Default to 3400 if not specified
API_HOST = os.environ.get("API_HOST", "localhost")
API_URL = os.environ.get("API_URL", f"http://{API_HOST}:{API_PORT}")

# Log the API URL being used
logger.info(f"Using API URL: {API_URL}")

# Function to test API connectivity
def test_api_connectivity():
    """Test connectivity to the API and adjust URL if needed."""
    global API_URL
    try:
        # Try the current API URL with a longer timeout
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"Successfully connected to API at {API_URL}")
            return True
    except requests.exceptions.Timeout:
        logger.warning(f"Connection to API at {API_URL} timed out. Trying again with longer timeout...")
        try:
            # Try again with an even longer timeout
            response = requests.get(f"{API_URL}/health", timeout=10)
            if response.status_code == 200:
                logger.info(f"Successfully connected to API at {API_URL} with longer timeout")
                return True
        except Exception as e:
            logger.warning(f"Second attempt to connect to API at {API_URL} failed: {e}")
    except Exception as e:
        logger.warning(f"Could not connect to API at {API_URL}: {e}")
    
    # If we couldn't connect, try the FastAPI port first before trying other common ports
    # This ensures we prioritize the FastAPI server over other services
    if API_PORT != "3400":
        test_url = f"http://{API_HOST}:3400"
        try:
            logger.info(f"Trying FastAPI default port: {test_url}")
            response = requests.get(f"{test_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info(f"Successfully connected to API at {test_url}")
                API_URL = test_url
                return True
        except Exception:
            pass
    
    # If we still couldn't connect, try some common ports
    common_ports = ["3400", "8000", "8080", "5000"]
    for port in common_ports:
        if port == API_PORT:  # Skip the port we already tried
            continue
        
        test_url = f"http://{API_HOST}:{port}"
        try:
            logger.info(f"Trying alternative API URL: {test_url}")
            response = requests.get(f"{test_url}/health", timeout=5)
            if response.status_code == 200:
                # Verify this is actually our FastAPI server by checking for a known endpoint
                try:
                    problems_response = requests.get(f"{test_url}/problems", timeout=5)
                    if problems_response.status_code == 200:
                        logger.info(f"Successfully connected to FastAPI server at {test_url}")
                        API_URL = test_url
                        return True
                    else:
                        logger.warning(f"Found a server at {test_url} but it doesn't appear to be our FastAPI server")
                except Exception:
                    logger.warning(f"Found a server at {test_url} but couldn't verify it's our FastAPI server")
        except Exception:
            pass
    
    logger.warning("Could not connect to API on any common port. Using default URL.")
    return False

# Load Streamlit configuration from the API
def load_streamlit_config():
    """Load Streamlit configuration from the API."""
    try:
        response = requests.get(f"{API_URL}/streamlit/config", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to load Streamlit configuration: {response.status_code}")
            return {}
    except Exception as e:
        logger.warning(f"Error loading Streamlit configuration: {e}")
        return {}

# Apply Streamlit configuration
def apply_streamlit_config(config):
    """Apply Streamlit configuration settings."""
    global API_URL
    
    if not config:
        return
    
    # Apply theme
    if "theme" in config:
        theme = config["theme"]
        if theme == "dark":
            st.set_page_config(
                page_title="Benchmark Tool",
                page_icon="🧪",
                layout="wide" if config.get("wide_mode", True) else "centered",
                initial_sidebar_state="expanded",
                menu_items={
                    "Get Help": "https://github.com/yourusername/benchmark-tool",
                    "Report a bug": "https://github.com/yourusername/benchmark-tool/issues",
                    "About": "# Benchmark Tool\nA tool for benchmarking language models on coding problems."
                }
            )
            st.markdown("""
            <style>
            :root {
                --background-color: #0e1117;
                --text-color: #fafafa;
                --widget-background-color: #262730;
                --widget-border-color: #4d4d4d;
            }
            </style>
            """, unsafe_allow_html=True)
    
    # Update API URL if provided in the configuration
    if "api_url" in config and config.get("show_api_url", True):
        new_api_url = config["api_url"]
        logger.info(f"Updating API URL from configuration: {new_api_url}")
        API_URL = new_api_url
        # Test the new API URL
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"Successfully connected to API at {API_URL}")
            else:
                logger.warning(f"API at {API_URL} returned status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to API at {API_URL}: {e}")

# Try to connect to the API
test_api_connectivity()

# Load and apply Streamlit configuration
streamlit_config = load_streamlit_config()
apply_streamlit_config(streamlit_config)

# Cache functions for better performance
@st.cache_data(ttl=60)
def fetch_problems():
    """Fetch problems from the API."""
    try:
        logger.info(f"Fetching problems from {API_URL}/problems")
        response = requests.get(f"{API_URL}/problems", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Failed to fetch problems: {response.status_code}"
            logger.error(error_msg)
            st.error(error_msg)
            # If we get a connection error, try to reconnect to the API
            if response.status_code in [404, 502, 503, 504]:
                if test_api_connectivity():
                    st.info(f"Reconnected to API at {API_URL}. Please refresh the page.")
            return []
    except requests.exceptions.Timeout:
        error_msg = f"Connection timeout fetching problems. Retrying with longer timeout..."
        logger.warning(error_msg)
        try:
            # Retry with longer timeout
            response = requests.get(f"{API_URL}/problems", timeout=20)
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Failed to fetch problems after retry: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return []
        except Exception as e:
            error_msg = f"Error fetching problems after retry: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return []
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error fetching problems: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        # Try to reconnect to the API
        if test_api_connectivity():
            st.info(f"Reconnected to API at {API_URL}. Please refresh the page.")
        return []
    except Exception as e:
        error_msg = f"Error fetching problems: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return []

@st.cache_data(ttl=60)
def fetch_models():
    """Fetch models from the API."""
    try:
        logger.info(f"Fetching models from {API_URL}/models")
        response = requests.get(f"{API_URL}/models", timeout=10)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            error_msg = f"Failed to fetch models: {response.status_code}"
            logger.error(error_msg)
            st.error(error_msg)
            # If we get a connection error, try to reconnect to the API
            if response.status_code in [404, 502, 503, 504]:
                if test_api_connectivity():
                    st.info(f"Reconnected to API at {API_URL}. Please refresh the page.")
            return []
    except requests.exceptions.Timeout:
        error_msg = f"Connection timeout fetching models. Retrying with longer timeout..."
        logger.warning(error_msg)
        try:
            # Retry with longer timeout
            response = requests.get(f"{API_URL}/models", timeout=20)
            if response.status_code == 200:
                return response.json().get("data", [])
            else:
                error_msg = f"Failed to fetch models after retry: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return []
        except Exception as e:
            error_msg = f"Error fetching models after retry: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return []
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error fetching models: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        # Try to reconnect to the API
        if test_api_connectivity():
            st.info(f"Reconnected to API at {API_URL}. Please refresh the page.")
        return []
    except Exception as e:
        error_msg = f"Error fetching models: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return []

@st.cache_data(ttl=60)
def fetch_config():
    """Fetch configuration from the API."""
    try:
        response = requests.get(f"{API_URL}/config", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Failed to fetch configuration: {response.status_code}"
            logger.error(error_msg)
            st.error(error_msg)
            if test_api_connectivity():
                st.info(f"Reconnected to API at {API_URL}. Please refresh the page.")
            return {}
    except requests.exceptions.Timeout:
        error_msg = f"Connection timeout fetching configuration. Retrying with longer timeout..."
        logger.warning(error_msg)
        try:
            # Retry with longer timeout
            response = requests.get(f"{API_URL}/config", timeout=20)
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Failed to fetch configuration after retry: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return {}
        except Exception as e:
            error_msg = f"Error fetching configuration after retry: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return {}
    except Exception as e:
        error_msg = f"Error fetching configuration: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        if test_api_connectivity():
            st.info(f"Reconnected to API at {API_URL}. Please refresh the page.")
        return {}

def fetch_simulation_status(simulation_id):
    """Fetch the status of a simulation."""
    try:
        response = requests.get(f"{API_URL}/simulations/{simulation_id}", timeout=30) # Increased timeout
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Failed to fetch simulation status: {response.status_code}"
            logger.error(error_msg)
            st.error(error_msg)
            if test_api_connectivity():
                st.info(f"Reconnected to API at {API_URL}. Please try again.")
            return None
    except requests.exceptions.Timeout:
        error_msg = f"Connection timeout fetching simulation status. Retrying with longer timeout..."
        logger.warning(error_msg)
        try:
            # Retry with longer timeout
            response = requests.get(f"{API_URL}/simulations/{simulation_id}", timeout=60) # Increased timeout
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Failed to fetch simulation status after retry: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return None
        except Exception as e:
            error_msg = f"Error fetching simulation status after retry: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
    except Exception as e:
        error_msg = f"Error fetching simulation status: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        if test_api_connectivity():
            st.info(f"Reconnected to API at {API_URL}. Please try again.")
        return None
    
def fetch_problem_results(problem_name):
    """Fetch results for a specific problem."""
    try:
        response = requests.get(f"{API_URL}/results/problem/{problem_name}", timeout=30) # Increased timeout
        if response.status_code == 200:
            data = response.json()
            # Check if the response has the expected structure
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            elif isinstance(data, list):
                # Handle case where API returns a list directly
                return data
            else:
                st.warning(f"Unexpected response format for problem results: {data}")
                return []
        else:
            error_msg = f"Failed to fetch problem results: {response.status_code}"
            logger.error(error_msg)
            st.error(error_msg)
            if test_api_connectivity():
                st.info(f"Reconnected to API at {API_URL}. Please try again.")
            return []
    except requests.exceptions.Timeout:
        error_msg = f"Connection timeout fetching problem results. Retrying with longer timeout..."
        logger.warning(error_msg)
        try:
            # Retry with longer timeout
            response = requests.get(f"{API_URL}/results/problem/{problem_name}", timeout=60) # Increased timeout
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "results" in data:
                    return data["results"]
                elif isinstance(data, list):
                    return data
                else:
                    st.warning(f"Unexpected response format for problem results: {data}")
                    return []
            else:
                error_msg = f"Failed to fetch problem results after retry: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return []
        except Exception as e:
            error_msg = f"Error fetching problem results after retry: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return []
    except Exception as e:
        error_msg = f"Error fetching problem results: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        if test_api_connectivity():
            st.info(f"Reconnected to API at {API_URL}. Please try again.")
        return []

def fetch_generation_stats(problem_name):
    """Fetch generation statistics for a problem."""
    try:
        response = requests.get(f"{API_URL}/results/problem/{problem_name}/generations", timeout=30) # Increased timeout
        if response.status_code == 200:
            data = response.json()
            # Check if the response has the expected structure
            if isinstance(data, dict) and "generations" in data:
                return data["generations"]
            elif isinstance(data, list):
                # Handle case where API returns a list directly
                return data
            else:
                st.warning(f"Unexpected response format for generation statistics: {data}")
                return []
        else:
            error_msg = f"Failed to fetch generation statistics: {response.status_code}"
            logger.error(error_msg)
            st.error(error_msg)
            if test_api_connectivity():
                st.info(f"Reconnected to API at {API_URL}. Please try again.")
            return []
    except requests.exceptions.Timeout:
        error_msg = f"Connection timeout fetching generation statistics. Retrying with longer timeout..."
        logger.warning(error_msg)
        try:
            # Retry with longer timeout
            response = requests.get(f"{API_URL}/results/problem/{problem_name}/generations", timeout=60) # Increased timeout
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "generations" in data:
                    return data["generations"]
                elif isinstance(data, list):
                    return data
                else:
                    st.warning(f"Unexpected response format for generation statistics: {data}")
                    return []
            else:
                error_msg = f"Failed to fetch generation statistics after retry: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return []
        except Exception as e:
            error_msg = f"Error fetching generation statistics after retry: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return []
    except Exception as e:
        error_msg = f"Error fetching generation statistics: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        if test_api_connectivity():
            st.info(f"Reconnected to API at {API_URL}. Please try again.")
        return None

def start_simulation(problem_name, model_id, population_size, num_generations):
    """Start a new simulation."""
    try:
        payload = {
            "problem_name": problem_name,
            "model_id": model_id,
            "population_size": population_size,
            "num_generations": num_generations
        }
        response = requests.post(f"{API_URL}/simulations/start", json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Failed to start simulation: {response.status_code}"
            logger.error(error_msg)
            st.error(error_msg)
            if test_api_connectivity():
                st.info(f"Reconnected to API at {API_URL}. Please try again.")
            return None
    except requests.exceptions.Timeout:
        error_msg = f"Connection timeout starting simulation. Retrying with longer timeout..."
        logger.warning(error_msg)
        try:
            # Retry with longer timeout
            response = requests.post(f"{API_URL}/simulations/start", json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Failed to start simulation after retry: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return None
        except Exception as e:
            error_msg = f"Error starting simulation after retry: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
    except Exception as e:
        error_msg = f"Error starting simulation: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        if test_api_connectivity():
            st.info(f"Reconnected to API at {API_URL}. Please try again.")
        return None

def update_config(config_data):
    """Update the configuration."""
    try:
        response = requests.post(f"{API_URL}/config/update", json=config_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Failed to update configuration: {response.status_code}"
            logger.error(error_msg)
            st.error(error_msg)
            if test_api_connectivity():
                st.info(f"Reconnected to API at {API_URL}. Please try again.")
            return None
    except requests.exceptions.Timeout:
        error_msg = f"Connection timeout updating configuration. Retrying with longer timeout..."
        logger.warning(error_msg)
        try:
            # Retry with longer timeout
            response = requests.post(f"{API_URL}/config/update", json=config_data, timeout=20)
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Failed to update configuration after retry: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return None
        except Exception as e:
            error_msg = f"Error updating configuration after retry: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
    except Exception as e:
        error_msg = f"Error updating configuration: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        if test_api_connectivity():
            st.info(f"Reconnected to API at {API_URL}. Please try again.")
        return None

def reset_config():
    """Reset the configuration to default values."""
    try:
        response = requests.post(f"{API_URL}/config/reset", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Failed to reset configuration: {response.status_code}"
            logger.error(error_msg)
            st.error(error_msg)
            if test_api_connectivity():
                st.info(f"Reconnected to API at {API_URL}. Please try again.")
            return None
    except requests.exceptions.Timeout:
        error_msg = f"Connection timeout resetting configuration. Retrying with longer timeout..."
        logger.warning(error_msg)
        try:
            # Retry with longer timeout
            response = requests.post(f"{API_URL}/config/reset", timeout=20)
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Failed to reset configuration after retry: {response.status_code}"
                logger.error(error_msg)
                st.error(error_msg)
                return None
        except Exception as e:
            error_msg = f"Error resetting configuration after retry: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
    except Exception as e:
        error_msg = f"Error resetting configuration: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        if test_api_connectivity():
            st.info(f"Reconnected to API at {API_URL}. Please try again.")
        return None

def display_home_page():
    """Display the home page."""
    global API_URL
    st.title("🧪 Benchmark Tool")
    
    st.markdown("""
    Welcome to the Benchmark Tool! This tool allows you to benchmark different language models
    on coding problems using genetic algorithms to optimize prompts and parameters.
    
    ### Features
    
    - **Run Simulations**: Test language models on coding problems
    - **View Results**: Analyze performance across models and problems
    - **Configure**: Customize genetic algorithm parameters
    
    ### Getting Started
    
    1. Navigate to the **Run Simulation** page to start a new benchmark
    2. Select a problem and model to test
    3. Configure the genetic algorithm parameters
    4. Start the simulation and monitor progress
    5. View detailed results in the **Results** page
    
    ### System Status
    """)
    
    # Check system status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("API Server: Online")
            else:
                st.error(f"API Server: Error (Status {response.status_code})")
                st.info(f"Current API URL: {API_URL}")
                if st.button("Retry API Connection"):
                    with st.spinner("Attempting to reconnect..."):
                        if test_api_connectivity():
                            st.success(f"Reconnected to API at {API_URL}")
                            st.experimental_rerun()
                        else:
                            st.error("Could not connect to API. Please check the server.")
                            # Provide more detailed troubleshooting information
                            st.info("Troubleshooting tips:")
                            st.info("1. Check if the FastAPI server is running")
                            st.info("2. Verify the API port is correct")
                            st.info("3. Check for any firewall or network issues")
                            # Add a manual URL input option
                            new_url = st.text_input("Enter API URL manually:", value=API_URL)
                            if st.button("Use this URL"):
                                API_URL = new_url
                                st.experimental_rerun()
        except Exception as e:
            st.error("API Server: Offline")
            st.info(f"Current API URL: {API_URL}")
            st.warning(f"Error details: {str(e)}")
            if st.button("Retry API Connection"):
                with st.spinner("Attempting to reconnect..."):
                    if test_api_connectivity():
                        st.success(f"Reconnected to API at {API_URL}")
                        st.experimental_rerun()
                    else:
                        st.error("Could not connect to API. Please check the server.")
                        # Provide more detailed troubleshooting information
                        st.info("Troubleshooting tips:")
                        st.info("1. Check if the FastAPI server is running")
                        st.info("2. Verify the API port is correct")
                        st.info("3. Check for any firewall or network issues")
                        # Add a manual URL input option
                        new_url = st.text_input("Enter API URL manually:", value=API_URL)
                        if st.button("Use this URL"):
                            API_URL = new_url
                            st.experimental_rerun()
    
    with col2:
        problems = fetch_problems()
        if problems:
            st.success(f"Problems: {len(problems)} available")
        else:
            st.warning("Problems: None available")
    
    with col3:
        models = fetch_models()
        if models:
            st.success(f"Models: {len(models)} available")
        else:
            st.warning("Models: None available")

def display_run_simulation_page():
    """Display the run simulation page."""
    st.title("🚀 Run Simulation")
    
    # Fetch problems and models
    problems = fetch_problems()
    models = fetch_models()
    
    if not problems or not models:
        st.warning("Cannot run simulations without problems and models.")
        return
    
    # Create a form for simulation parameters
    with st.form("simulation_form"):
        st.subheader("Simulation Parameters")
        
        # Problem selection
        problem_names = [p["name"] for p in problems]
        problem_name = st.selectbox("Select Problem", problem_names)
        
        # Model selection
        model_ids = [m["id"] for m in models]
        model_id = st.selectbox("Select Model", model_ids)
        
        # Genetic algorithm parameters
        col1, col2 = st.columns(2)
        with col1:
            population_size = st.slider("Population Size", min_value=10, max_value=100, value=30, step=5)
        with col2:
            num_generations = st.slider("Number of Generations", min_value=1, max_value=20, value=5, step=1)
        
        # Submit button
        submitted = st.form_submit_button("Start Simulation")
        
        if submitted:
            with st.spinner("Starting simulation..."):
                result = start_simulation(problem_name, model_id, population_size, num_generations)
                if result:
                    # The backend returns "run_id", not "simulation_id" for this endpoint
                    st.session_state.run_id = result.get("run_id") 
                    st.success(f"Simulation started with ID: {st.session_state.run_id}")
    
    # Display active simulation if one is running
    if hasattr(st.session_state, "run_id") and st.session_state.run_id is not None:
        st.subheader("Active Simulation")
        
        # Create a placeholder for the simulation status that will be updated
        status_placeholder = st.empty()

        # Auto-refresh loop
        while True:
            status = fetch_simulation_status(st.session_state.run_id)
            
            with status_placeholder.container():
                if status:
                    st.write(f"Simulation ID: {st.session_state.run_id}")
                    st.write(f"Problem: {status.get('problem_name', 'N/A')}")
                    st.write(f"Model: {status.get('model_id', 'N/A')}")
                    
                    current_status = status.get('status', 'Unknown')
                    st.write(f"Status: {current_status}")
                    
                    progress_val = status.get('progress', 0.0)
                    # Ensure progress is float for formatting and st.progress
                    try:
                        progress_val = float(progress_val)
                    except (ValueError, TypeError):
                        progress_val = 0.0
                    
                    st.write(f"Progress: {progress_val:.1f}%")
                    st.progress(progress_val / 100)
                    
                    current_gen = status.get('current_generation', 0)
                    total_gens = status.get('total_generations', 0)
                    if total_gens > 0 : # Show generation count if available and meaningful
                         st.write(f"Generation: {current_gen+1 if current_status != 'completed' else total_gens+1} / {total_gens+1}")


                    if current_status == 'completed':
                        st.success("Simulation completed successfully!")
                        if st.button("View Results", key=f"view_results_{st.session_state.run_id}"):
                            st.session_state.selected_problem = status.get('problem_name')
                            st.session_state.page = "results"
                            st.rerun() # Rerun to navigate to results page
                        break # Exit refresh loop
                    elif current_status == 'failed':
                        st.error(f"Simulation failed: {status.get('result', {}).get('error', 'Unknown error')}")
                        break # Exit refresh loop
                    elif current_status == 'starting':
                        st.info("Simulation is starting...")
                    elif current_status == 'running':
                        st.info("Simulation is running...")
                    else: # Pending or unknown
                        st.info(f"Simulation status: {current_status}")


                else:
                    st.warning("Could not fetch simulation status. Retrying...")
            
            if status and (status.get('status') == 'completed' or status.get('status') == 'failed'):
                break

            # Wait for a few seconds before refreshing, unless completed or failed
            # Use a shorter sleep time for better responsiveness if needed
            time.sleep(5) # Refresh every 5 seconds

def display_results_page():
    """Display the results page."""
    st.title("📊 Results")
    
    # Fetch problems
    problems = fetch_problems()
    if not problems:
        st.warning("No problems available.")
        return
    
    # Problem selection
    problem_names = [p["name"] for p in problems]
    selected_problem = st.selectbox(
        "Select Problem", 
        problem_names,
        index=problem_names.index(st.session_state.selected_problem) if hasattr(st.session_state, "selected_problem") and st.session_state.selected_problem in problem_names else 0
    )
    
    # Store the selected problem in session state
    st.session_state.selected_problem = selected_problem
    
    # Fetch results for the selected problem
    results = fetch_problem_results(selected_problem)
    generation_stats = fetch_generation_stats(selected_problem)
    
    if not results:
        st.info(f"No results available for problem: {selected_problem}")
        return
    
    # Ensure results is a list of dictionaries
    if not isinstance(results, list):
        st.error(f"Invalid results format: {results}")
        return
    
    # Display results overview
    st.subheader("Results Overview")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Performance", "Best Solution", "All Results"])
    
    with tab1:
        if generation_stats and isinstance(generation_stats, list) and len(generation_stats) > 0:
            # Ensure all required fields are present
            required_fields = ['generation', 'avg_score', 'max_score', 'min_score']
            if all(field in generation_stats[0] for field in required_fields):
                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(generation_stats)
                
                # Plot generation progression
                st.subheader("Score Progression Across Generations")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.plot(df['generation'], df['avg_score'], 'o-', label='Average Score', color='blue')
                ax.plot(df['generation'], df['max_score'], 'о-', label='Max Score', color='green')
                ax.plot(df['generation'], df['min_score'], 'o-', label='Min Score', color='red')
                
                ax.fill_between(df['generation'], df['min_score'], df['max_score'], alpha=0.2, color='gray')
                
                ax.set_xlabel('Generation')
                ax.set_ylabel('Score')
                ax.set_title(f'Score Progression for {selected_problem}')
                
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                st.pyplot(fig)
                
                # Display statistics table
                st.subheader("Generation Statistics")
                st.dataframe(df[['generation', 'avg_score', 'max_score', 'min_score']])
            else:
                st.warning("Generation statistics are missing required fields.")
        else:
            st.info("No generation statistics available.")

    with tab2:
        # Find the best result
        if results and isinstance(results, list) and len(results) > 0:
            # Ensure all items are dictionaries with the required fields
            if all(isinstance(r, dict) for r in results):
                try:
                    best_result = max(results, key=lambda x: x.get('final_score', 0))
                    
                    st.subheader("Best Solution")
                    
                    # Display best result details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Model:** {best_result.get('model_id', 'N/A')}")
                        final_score = best_result.get('final_score')
                        st.write(f"**Final Score:** {final_score:.2f}" if final_score is not None else "**Final Score:** N/A")
                        test_score = best_result.get('test_score')
                        st.write(f"**Test Score:** {test_score:.2f}" if test_score is not None else "**Test Score:** N/A")
                        quality_score = best_result.get('quality_score')
                        st.write(f"**Quality Score:** {quality_score:.2f}" if quality_score is not None else "**Quality Score:** N/A")
                    with col2:
                        st.write(f"**Generation:** {best_result.get('generation', 'N/A')}")
                        temperature = best_result.get('temperature')
                        st.write(f"**Temperature:** {temperature:.2f}" if temperature is not None else "**Temperature:** N/A")
                        st.write(f"**Max Tokens:** {best_result.get('max_tokens', 'N/A')}")
                        top_p = best_result.get('top_p')
                        st.write(f"**Top P:** {top_p:.2f}" if top_p is not None else "**Top P:** N/A")
                    
                    # Display the generated code
                    st.subheader("Generated Code")
                    st.code(best_result.get('code', ''), language=best_result.get('language', 'python'))
                    
                    # Display the prompt used
                    with st.expander("Show Prompt"):
                        st.write("**System Prompt:**")
                        st.text(best_result.get('system_prompt', ''))
                        st.write("**Prompt Template:**")
                        st.text(best_result.get('prompt_template', ''))
                except Exception as e:
                    st.error(f"Error processing best result: {e}")
            else:
                st.warning("Results data is not in the expected format.")
        else:
            st.info("No results available.")

    with tab3:
        # Display all results in a table
        if results and isinstance(results, list) and len(results) > 0:
            # Ensure all items are dictionaries with the required fields
            if all(isinstance(r, dict) for r in results):
                try:
                    # Convert to DataFrame for easier display
                    df = pd.DataFrame([
                        {
                            'Generation': r.get('generation', 'N/A'),
                            'Model': r.get('model_id', 'N/A'),
                            'Final Score': r.get('final_score', 0),
                            'Test Score': r.get('test_score', 0),
                            'Quality Score': r.get('quality_score', 0),
                            'Temperature': r.get('temperature', 0),
                            'Max Tokens': r.get('max_tokens', 'N/A'),
                            'Top P': r.get('top_p', 0)
                        } for r in results
                    ])
                    
                    # Sort by final score (descending)
                    df = df.sort_values('Final Score', ascending=False)
                    
                    st.dataframe(df)
                    
                    # Add a download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"{selected_problem}_results.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error processing results table: {e}")
            else:
                st.warning("Results data is not in the expected format.")
        else:
            st.info("No results available.")

def display_config_page():
    """Display the configuration page."""
    st.title("⚙️ Configuration")
    
    # Fetch current configuration
    config = fetch_config()
    if not config:
        st.warning("Could not fetch configuration.")
        return
    
    # Create tabs for different configuration sections
    tab1, tab2, tab3 = st.tabs(["Genetic Algorithm", "Simulation", "API"])
    
    with tab1:
        st.subheader("Genetic Algorithm Configuration")
        
        # Create a form for genetic algorithm configuration
        with st.form("ga_config_form"):
            # Population parameters
            st.write("**Population Parameters**")
            col1, col2 = st.columns(2)
            with col1:
                population_size = st.number_input(
                    "Population Size",
                    min_value=10,
                    max_value=100,
                    value=config.get("genetic_algorithm", {}).get("population_size", 30)
                )
            with col2:
                num_generations = st.number_input(
                    "Number of Generations",
                    min_value=1,
                    max_value=50,
                    value=config.get("genetic_algorithm", {}).get("num_generations", 10)
                )
            
            # Selection parameters
            st.write("**Selection Parameters**")
            col1, col2 = st.columns(2)
            with col1:
                tournament_size = st.number_input(
                    "Tournament Size",
                    min_value=2,
                    max_value=10,
                    value=config.get("genetic_algorithm", {}).get("tournament_size", 3)
                )
            with col2:
                elite_size = st.number_input(
                    "Elite Size",
                    min_value=0,
                    max_value=10,
                    value=config.get("genetic_algorithm", {}).get("elite_size", 1)
                )
            
            # Genetic operators
            st.write("**Genetic Operators**")
            col1, col2 = st.columns(2)
            with col1:
                crossover_prob = st.slider(
                    "Crossover Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.get("genetic_algorithm", {}).get("crossover_prob", 0.7),
                    step=0.05
                )
            with col2:
                mutation_prob = st.slider(
                    "Mutation Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.get("genetic_algorithm", {}).get("mutation_prob", 0.2),
                    step=0.05
                )
            
            # Submit button
            submitted = st.form_submit_button("Update Genetic Algorithm Configuration")
            
            if submitted:
                # Prepare the updated configuration
                updated_config = config.copy()
                if "genetic_algorithm" not in updated_config:
                    updated_config["genetic_algorithm"] = {}
                
                updated_config["genetic_algorithm"]["population_size"] = population_size
                updated_config["genetic_algorithm"]["num_generations"] = num_generations
                updated_config["genetic_algorithm"]["tournament_size"] = tournament_size
                updated_config["genetic_algorithm"]["elite_size"] = elite_size
                updated_config["genetic_algorithm"]["crossover_prob"] = crossover_prob
                updated_config["genetic_algorithm"]["mutation_prob"] = mutation_prob
                
                # Update the configuration
                result = update_config(updated_config)
                if result:
                    st.success("Genetic algorithm configuration updated successfully!")
                    # Clear the cache to fetch the updated configuration
                    fetch_config.clear()
    with tab2:
        st.subheader("Simulation Configuration")
        
        # Create a form for simulation configuration
        with st.form("sim_config_form"):
            # Simulation parameters
            st.write("**Simulation Parameters**")
            col1, col2 = st.columns(2)
            with col1:
                num_workers = st.number_input(
                    "Number of Workers",
                    min_value=1,
                    max_value=10,
                    value=config.get("simulation", {}).get("num_workers", 4)
                )
            with col2:
                timeout = st.number_input(
                    "Timeout (seconds)",
                    min_value=10,
                    max_value=300,
                    value=config.get("simulation", {}).get("timeout", 60)
                )
            
            # Submit button
            submitted = st.form_submit_button("Update Simulation Configuration")
            
            if submitted:
                # Prepare the updated configuration
                updated_config = config.copy()
                if "simulation" not in updated_config:
                    updated_config["simulation"] = {}
                
                updated_config["simulation"]["num_workers"] = num_workers
                updated_config["simulation"]["timeout"] = timeout
                
                # Update the configuration
                result = update_config(updated_config)
                if result:
                    st.success("Simulation configuration updated successfully!")
                    # Clear the cache to fetch the updated configuration
                    fetch_config.clear()
    with tab3:
        st.subheader("API Configuration")
        
        # Create a form for API configuration
        with st.form("api_config_form"):
            # API parameters
            st.write("**API Parameters**")
            col1, col2 = st.columns(2)
            with col1:
                debug = st.checkbox(
                    "Debug Mode",
                    value=config.get("api", {}).get("debug", False)
                )
            with col2:
                model_proxy_url = st.text_input(
                    "Model Proxy URL",
                    value=config.get("api", {}).get("model_proxy_url", "http://localhost:8080")
                )
            
            # Submit button
            submitted = st.form_submit_button("Update API Configuration")
            
            if submitted:
                # Prepare the updated configuration
                updated_config = config.copy()
                if "api" not in updated_config:
                    updated_config["api"] = {}
                
                updated_config["api"]["debug"] = debug
                updated_config["api"]["model_proxy_url"] = model_proxy_url
                
                # Update the configuration
                result = update_config(updated_config)
                if result:
                    st.success("API configuration updated successfully!")
                    # Clear the cache to fetch the updated configuration
                    fetch_config.clear()
    # Add a button to reset the configuration
    if st.button("Reset Configuration to Default Values"):
        result = reset_config()
        if result:
            st.success("Configuration reset to default values!")
            # Clear the cache to fetch the updated configuration
            fetch_config.clear()

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state for page navigation
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Navigation buttons
    if st.sidebar.button("🏠 Home"):
        st.session_state.page = "home"
    if st.sidebar.button("🚀 Run Simulation"):
        st.session_state.page = "run_simulation"
    if st.sidebar.button("📊 Results"):
        st.session_state.page = "results"
    if st.sidebar.button("⚙️ Configuration"):
        st.session_state.page = "config"
    
    # Display the selected page
    if st.session_state.page == "home":
        display_home_page()
    elif st.session_state.page == "run_simulation":
        display_run_simulation_page()
    elif st.session_state.page == "results":
        display_results_page()
    elif st.session_state.page == "config":
        display_config_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This is a benchmark tool for testing language models on coding problems. "
        "It uses genetic algorithms to optimize prompts and parameters."
    )
    st.sidebar.text("Version 1.0.0")

if __name__ == "__main__":
    main()
