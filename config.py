
import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config")

@dataclass
class GeneticAlgorithmConfig:
    """Configuration for the genetic algorithm."""
    population_size: int = 50
    num_generations: int = 10
    elite_size: int = 5
    crossover_probability: float = 0.5
    mutation_probability: float = 0.2
    tournament_size: int = 3
    
    # Model parameters
    available_models: List[str] = field(default_factory=list)
    
    # Prompt parameters
    prompt_templates: List[str] = field(default_factory=lambda: [
        "default",
        "detailed",
        "step_by_step"
    ])
    
    system_prompts: List[str] = field(default_factory=lambda: [
        "You are a helpful coding assistant. Write clean, efficient code that solves the given problem.",
        "You are an expert programmer. Write optimized code with detailed comments.",
        "You are a coding tutor. Write clear, well-structured code that is easy to understand."
    ])
    
    # Parameter ranges
    temperature_range: tuple = (0.1, 1.0)
    max_tokens_range: tuple = (100, 2000)
    top_p_range: tuple = (0.1, 1.0)

@dataclass
class SimulationConfig:
    """Configuration for simulations."""
    num_workers: int = 4 # Changed default to 1
    timeout: int = 30
    max_retries: int = 3
    
    # Docker configuration
    docker_memory_limit: str = "512m"
    docker_cpu_limit: str = "1"
    docker_network: str = "none"
    
    # Execution configuration
    python_sandbox_image: str = "python-sandbox"
    typescript_sandbox_image: str = "typescript-sandbox"
    
    # Scoring configuration
    test_weight: float = 0.7
    quality_weight: float = 0.3

@dataclass
class APIConfig:
    """Configuration for the API."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_proxy_url: str = "http://localhost:8080/v1"
    debug: bool = False

@dataclass
class StreamlitConfig:
    """Configuration for the Streamlit interface."""
    host: str = "0.0.0.0"
    port: int = 8501
    theme: str = "light"
    wide_mode: bool = True
    enable_cache: bool = True
    show_api_url: bool = True
    api_url: str = "http://localhost:8000"

@dataclass
class Config:
    """Main configuration class."""
    genetic_algorithm: GeneticAlgorithmConfig = field(default_factory=GeneticAlgorithmConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    
    # Database configuration
    db_path: str = "results.db"
    
    # Visualization configuration
    visualization_dir: str = "visualizations"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return asdict(self)
    
    def save(self, filepath: str) -> None:
        """Save the configuration to a file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load the configuration from a file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Create a new Config object
            config = cls()
            
            # Update the genetic algorithm configuration
            if "genetic_algorithm" in config_dict:
                ga_config = config_dict["genetic_algorithm"]
                for key, value in ga_config.items():
                    if hasattr(config.genetic_algorithm, key):
                        setattr(config.genetic_algorithm, key, value)
            
            # Update the simulation configuration
            if "simulation" in config_dict:
                sim_config = config_dict["simulation"]
                for key, value in sim_config.items():
                    if hasattr(config.simulation, key):
                        setattr(config.simulation, key, value)
            
            # Update the API configuration
            if "api" in config_dict:
                api_config = config_dict["api"]
                for key, value in api_config.items():
                    if hasattr(config.api, key):
                        setattr(config.api, key, value)
            
            # Update the Streamlit configuration
            if "streamlit" in config_dict:
                st_config_dict = config_dict["streamlit"]
                if isinstance(st_config_dict, dict): # Ensure it's a dict before iterating
                    for key, value in st_config_dict.items():
                        if hasattr(config.streamlit, key):
                            setattr(config.streamlit, key, value)
                else:
                    logger.warning(f"Streamlit configuration in {filepath} is not a dictionary, skipping.")

            # Update other top-level configuration fields (e.g., db_path, visualization_dir)
            for key, value in config_dict.items():
                if key not in ["genetic_algorithm", "simulation", "api", "streamlit"] and hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Configuration loaded from {filepath}")
            return config
        except FileNotFoundError:
            logger.warning(f"Configuration file {filepath} not found, using default configuration")
            return cls()
        except json.JSONDecodeError:
            logger.error(f"Error parsing configuration file {filepath}, using default configuration")
            return cls()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}, using default configuration")
            return cls()

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the configuration.
    
    Args:
        config_path: The path to the configuration file, or None to use the default
            configuration file path.
            
    Returns:
        The configuration.
    """
    # Use the default configuration file path if none is provided
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.json")

    config: Config
    # Check if the configuration file exists
    if os.path.exists(config_path):
        # Load the configuration from the file
        config = Config.load(config_path)
    else:
        # Create a default configuration
        logger.warning(f"Configuration file {config_path} not found, creating default configuration.")
        config = Config()
        # Attempt to save this default config if it was created, so it exists for next time
        try:
            config.save(config_path)
        except Exception as e:
            logger.error(f"Could not save initial default config to {config_path}: {e}")

    # Ensure API host is connectable for client-side calls
    if config.api.host == "0.0.0.0":
        config.api.host = "localhost"
        logger.info(f"Defaulted API host to 'localhost' for client connections, was '0.0.0.0'.")

    # Update API configuration from environment variables
    env_api_port_str = os.environ.get("API_PORT") or os.environ.get("PORT")
    logger.info(f"Environment API_PORT: {os.environ.get('API_PORT')}, PORT: {os.environ.get('PORT')}")

    if env_api_port_str:
        try:
            api_port_from_env = int(env_api_port_str)
            if config.api.port != api_port_from_env:
                logger.info(f"Updating API port from environment. Old: {config.api.port}, New: {api_port_from_env}")
                config.api.port = api_port_from_env
            else:
                logger.info(f"API port from environment ({api_port_from_env}) matches current config.api.port ({config.api.port}).")
            
            # Update Streamlit API URL based on the now-set config.api.port and upcoming config.api.host
            # This needs to be done carefully after host is also determined from env
        except ValueError:
            logger.warning(f"Invalid API_PORT/PORT environment variable: '{env_api_port_str}'. Using current config.api.port: {config.api.port}")
    else:
        logger.info(f"No API_PORT or PORT environment variable found. Using current config.api.port: {config.api.port}")

    env_api_host_str = os.environ.get("API_HOST")
    logger.info(f"Environment API_HOST: {env_api_host_str}")
    if env_api_host_str:
        if config.api.host != env_api_host_str:
            logger.info(f"Updating API host from environment. Old: {config.api.host}, New: {env_api_host_str}")
            config.api.host = env_api_host_str
        else:
            logger.info(f"API host from environment ({env_api_host_str}) matches current config.api.host ({config.api.host}).")
    else:
        logger.info(f"No API_HOST environment variable found. Using current config.api.host: {config.api.host}")
    
    # Final check for api.host for client use, especially if it came from env as 0.0.0.0
    if config.api.host == "0.0.0.0":
        config.api.host = "localhost"
        logger.info(f"Final check: Changed API host from '0.0.0.0' to 'localhost'.")

    # Update Streamlit API URL based on final API host and port
    # Ensure config.api.port is an int for f-string
    try:
        current_api_port = int(config.api.port)
        current_api_host = config.api.host
        new_streamlit_api_url = f"http://{current_api_host}:{current_api_port}"
        if config.streamlit.api_url != new_streamlit_api_url:
            logger.info(f"Updating Streamlit API URL. Old: {config.streamlit.api_url}, New: {new_streamlit_api_url}")
            config.streamlit.api_url = new_streamlit_api_url
        else:
            logger.info(f"Streamlit API URL ({new_streamlit_api_url}) is already correctly set.")
    except ValueError: # Should not happen if api_port is correctly an int
         logger.error(f"Cannot form Streamlit API URL, config.api.port is not an int: {config.api.port}")


    # Update model proxy URL from environment variables
    env_model_proxy_url = os.environ.get("MODEL_PROXY_URL")
    if env_model_proxy_url:
        if config.api.model_proxy_url != env_model_proxy_url:
            logger.info(f"Updating model proxy URL from environment. Old: {config.api.model_proxy_url}, New: {env_model_proxy_url}")
            config.api.model_proxy_url = env_model_proxy_url
        else:
            logger.info(f"Model proxy URL from environment ({env_model_proxy_url}) matches current config.api.model_proxy_url.")
    else:
        logger.info(f"No MODEL_PROXY_URL environment variable found. Using current config.api.model_proxy_url: {config.api.model_proxy_url}")

    # If config file did not exist initially, try to fetch available models for the default config
    if not os.path.exists(config_path): # This check is a bit redundant if we saved above, but good for logic flow
        try:
            import requests
            response = requests.get("http://localhost:8080/v1/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                config.genetic_algorithm.available_models = [model["id"] for model in models]
        except Exception as e:
            logger.warning(f"Error fetching available models: {e}")
        
        # Save the default configuration
        config.save(config_path)
        # This return was specific to the "if not os.path.exists" block.
        # The main return should be at the end of the function.
        # return config # Removed from here

    # If the config file existed, 'config' is already loaded and processed.
    # If it didn't exist, 'config' was created, processed, and saved.
    # Now, ensure all final logging and the actual return happens.
    logger.info(f"Final effective config for API: host='{config.api.host}', port={config.api.port}")
    logger.info(f"Final effective config for Streamlit API URL: {config.streamlit.api_url}")
    return config


def get_streamlit_config_from_env() -> StreamlitConfig:
    """
    Get the Streamlit configuration from environment variables.
    
    This function allows for better integration between FastAPI and Streamlit
    by using environment variables to configure the Streamlit interface.
    
    Returns:
        The Streamlit configuration.
    """
    config = StreamlitConfig()
    
    # Get the host from the environment
    if "STREAMLIT_HOST" in os.environ:
        config.host = os.environ["STREAMLIT_HOST"]
    
    # Get the port from the environment
    if "STREAMLIT_PORT" in os.environ:
        try:
            config.port = int(os.environ["STREAMLIT_PORT"])
        except ValueError:
            logger.warning(f"Invalid STREAMLIT_PORT: {os.environ['STREAMLIT_PORT']}")
    
    # Get the theme from the environment
    if "STREAMLIT_THEME" in os.environ:
        config.theme = os.environ["STREAMLIT_THEME"]
    
    # Get the wide mode from the environment
    if "STREAMLIT_WIDE_MODE" in os.environ:
        config.wide_mode = os.environ["STREAMLIT_WIDE_MODE"].lower() == "true"
    
    # Get the enable cache from the environment
    if "STREAMLIT_ENABLE_CACHE" in os.environ:
        config.enable_cache = os.environ["STREAMLIT_ENABLE_CACHE"].lower() == "true"
    
    # Get the show API URL from the environment
    if "STREAMLIT_SHOW_API_URL" in os.environ:
        config.show_api_url = os.environ["STREAMLIT_SHOW_API_URL"].lower() == "true"
    
    # Get the API URL from the environment with improved priority and fallback logic
    if "API_URL" in os.environ:
        config.api_url = os.environ["API_URL"]
    elif "API_HOST" in os.environ and "API_PORT" in os.environ:
        # If both API_HOST and API_PORT are set, construct the API URL
        try:
            api_port = int(os.environ["API_PORT"])
            api_host = os.environ["API_HOST"]
            config.api_url = f"http://{api_host}:{api_port}"
        except ValueError:
            logger.warning(f"Invalid API_PORT: {os.environ['API_PORT']}")
    elif "API_PORT" in os.environ:
        # If only API_PORT is set, construct the API URL with localhost
        try:
            api_port = int(os.environ["API_PORT"])
            config.api_url = f"http://localhost:{api_port}"
        except ValueError:
            logger.warning(f"Invalid API_PORT: {os.environ['API_PORT']}")
    
    logger.info(f"Streamlit configuration from environment: API URL = {config.api_url}")
    
    return config

# Default configuration instance
default_config = get_config()

if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(f"Genetic Algorithm Configuration:")
    print(f"  Population Size: {config.genetic_algorithm.population_size}")
    print(f"  Number of Generations: {config.genetic_algorithm.num_generations}")
    print(f"  Elite Size: {config.genetic_algorithm.elite_size}")
    print(f"  Crossover Probability: {config.genetic_algorithm.crossover_probability}")
    print(f"  Mutation Probability: {config.genetic_algorithm.mutation_probability}")
    
    print(f"\nSimulation Configuration:")
    print(f"  Number of Workers: {config.simulation.num_workers}")
    print(f"  Timeout: {config.simulation.timeout} seconds")
    print(f"  Test Weight: {config.simulation.test_weight}")
    print(f"  Quality Weight: {config.simulation.quality_weight}")
    
    print(f"\nAPI Configuration:")
    print(f"  Host: {config.api.host}")
    print(f"  Port: {config.api.port}")
    print(f"  Model Proxy URL: {config.api.model_proxy_url}")
    
    print(f"\nStreamlit Configuration:")
    print(f"  Host: {config.streamlit.host}")
    print(f"  Port: {config.streamlit.port}")
    print(f"  API URL: {config.streamlit.api_url}")
    print(f"  Theme: {config.streamlit.theme}")
    print(f"  Wide Mode: {config.streamlit.wide_mode}")
    
    # Save the configuration to a file
    config.save("config.json")
    print(f"\nConfiguration saved to config.json")
