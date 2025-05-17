from fastapi import FastAPI
from typing import List, Dict, Optional, Any
import importlib.util
import httpx
import logging
from cachetools import TTLCache, cached
import asyncio # Ensure asyncio is imported for cachetools with async
from prompt_generator import generate_prompt
from scoring import calculate_score, analyze_code_quality
import asyncio
import uuid
import time
import os # Added for os.makedirs
from pydantic import BaseModel
from simulation_manager import SimulationManager, SimulationJob
from results_db import ResultsDB, SimulationResult
from visualization import ResultsVisualizer
from config import default_config, Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(debug=default_config.api.debug)

# Load problems from problems.py
spec = importlib.util.spec_from_file_location("problems", "problems.py")
problems_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(problems_module)
problems: List[Dict] = problems_module.problems

# Initialize the simulation manager
simulation_manager = SimulationManager(num_workers=default_config.simulation.num_workers)
results_db = ResultsDB(db_path=default_config.db_path)

# Dictionary to store status of active simulation runs (orchestrated by _run_genetic_algorithm)
active_simulations: Dict[str, Dict] = {}

# Cache for models endpoint (1 item, 5 minutes TTL)
# models_cache = TTLCache(maxsize=1, ttl=300) # Caching disabled for now

# Lock for async cache updates
# cache_lock = asyncio.Lock() # Removed as it's causing issues with cachetools' @cached decorator

async def _fetch_models_from_proxy_actual():
    """Actual function to fetch models from the proxy."""
    logger.info("Fetching models directly from proxy...")
    async with httpx.AsyncClient() as client:
        # Use the model_proxy_url from the loaded configuration
        # Ensure the URL is correctly formed, e.g., http://localhost:8080/v1/models
        proxy_models_url = f"{default_config.api.model_proxy_url}/models"
        if not default_config.api.model_proxy_url.endswith('/'):
             proxy_models_url = f"{default_config.api.model_proxy_url}/models" # Common case
        else: # if it ends with / e.g. http://host/v1/
             proxy_models_url = f"{default_config.api.model_proxy_url}models"


        logger.info(f"Fetching models from proxy at URL: {proxy_models_url}")
        response = await client.get(proxy_models_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()

# @cached(models_cache) # Caching disabled for now
async def get_models_directly(): # Renamed to reflect it's not cached
    """Function to get models (caching disabled)."""
    logger.info("Fetching models directly (caching disabled).")
    return await _fetch_models_from_proxy_actual()
    
@app.get("/")
async def read_root():
    return {"message": "Benchmark Tool API"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the system status.
    
    Returns:
        dict: A dictionary containing system status information
    """
    try:
        # Check if the database is accessible
        db_status = "ok"
        try:
            results_db.get_problem_summary()
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        # Check if the model proxy is accessible (using the cached models data)
        model_proxy_status = "ok"
        try:
            # Attempt to get models from cache; this will indirectly check proxy if cache is empty/stale
            # Use a timeout for this specific check within health, even if get_cached_models itself might wait longer.
            # However, get_cached_models itself handles the httpx timeout.
            # We just want to ensure the health check doesn't hang indefinitely if the proxy is slow.
            models_data = await asyncio.wait_for(get_models_directly(), timeout=5.0) # 5s timeout for this part of health check
            if not models_data: # Or check for a specific structure if models_data could be empty but valid
                model_proxy_status = "error: no models returned or proxy unreachable"
        except asyncio.TimeoutError:
            model_proxy_status = "error: timeout during models fetch for health check"
            logger.warning("Timeout in health_check waiting for get_models_directly.")
        except httpx.HTTPStatusError as e:
            model_proxy_status = f"error: proxy returned status {e.response.status_code}"
            logger.warning(f"HTTPStatusError in health_check from get_models_directly: {e}")
        except httpx.RequestError as e:
            model_proxy_status = f"error: request error connecting to proxy - {type(e).__name__}"
            logger.warning(f"RequestError in health_check from get_models_directly: {e}")
        except Exception as e:
            model_proxy_status = f"error: {type(e).__name__} - {str(e)}"
            logger.error(f"Unexpected error in health_check during get_models_directly: {e}", exc_info=True)
        
        # Check if the simulation manager is running
        simulation_status = "ok" if simulation_manager.is_running else "not running"
        
        return {
            "status": "ok",
            "timestamp": time.time(),
            "version": "1.0.0",
            "components": {
                "database": db_status,
                "model_proxy": model_proxy_status,
                "simulation_manager": simulation_status
            },
            "config": {
                "workers": default_config.simulation.num_workers,
                "debug": default_config.api.debug
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/problems")
async def list_problems():
    return problems

@app.get("/models")
async def list_models():
    logger.info("Request received for /models endpoint (caching disabled).")
    models_data = await get_models_directly()
    return models_data

@app.get("/generate_prompt/{problem_name}")
async def generate_prompt_endpoint(problem_name: str):
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    return {"prompt": generate_prompt(problem["description"])}

# Initialize the application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Start the simulation manager when the application starts."""
    await simulation_manager.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the simulation manager when the application shuts down."""
    await simulation_manager.stop()
    results_db.close()
    
@app.get("/config")
async def get_configuration():
    """Get the current configuration."""
    return default_config.to_dict()

class UpdateConfigRequest(BaseModel):
    config: Dict[str, Any]

@app.post("/config/update")
async def update_configuration(request: UpdateConfigRequest):
    """Update the configuration."""
    try:
        # Load existing configuration
        new_config = Config.load("config.json")
        # Get the updates dictionary
        config_dict = request.config

        # Update the genetic algorithm configuration
        if "genetic_algorithm" in config_dict:
            for key, value in config_dict["genetic_algorithm"].items():
                if hasattr(new_config.genetic_algorithm, key):
                    setattr(new_config.genetic_algorithm, key, value)

        # Update the simulation configuration
        if "simulation" in config_dict:
            for key, value in config_dict["simulation"].items():
                if hasattr(new_config.simulation, key):
                    setattr(new_config.simulation, key, value)

        # Update the API configuration
        if "api" in config_dict:
            for key, value in config_dict["api"].items():
                if hasattr(new_config.api, key):
                    setattr(new_config.api, key, value)

        # Update the Streamlit configuration
        if "streamlit" in config_dict:
            for key, value in config_dict["streamlit"].items():
                if hasattr(new_config.streamlit, key):
                    setattr(new_config.streamlit, key, value)

        # Update other top-level configuration fields
        for key, value in config_dict.items():
            if key not in ["genetic_algorithm", "simulation", "api", "streamlit"] and hasattr(new_config, key):
                setattr(new_config, key, value)

        # Persist the configuration
        new_config.save("config.json")

        # Replace the in-memory default
        global default_config
        default_config = new_config

        return {"message": "Configuration updated successfully", "config": new_config.to_dict()}
    except Exception as e:
        return {"error": f"Error updating configuration: {e}"}
    
@app.post("/config/reset")
async def reset_configuration():
    """Reset the configuration to default values."""
    try:
        # Create a new default configuration
        new_config = Config()
        
        # Save the configuration
        new_config.save("config.json")
        
        # Update the default configuration
        global default_config
        default_config = new_config
        
        return {"message": "Configuration reset to default values", "config": new_config.to_dict()}
    except Exception as e:
        return {"error": f"Error resetting configuration: {e}"}

@app.get("/streamlit/config")
async def get_streamlit_config():
    """Get the Streamlit configuration."""
    try:
        # Get the Streamlit configuration
        streamlit_config = default_config.streamlit

        # Return the configuration
        return streamlit_config.__dict__
    except Exception as e:
        return {"error": f"Error getting Streamlit configuration: {e}"}

@app.post("/streamlit/config/update")
async def update_streamlit_config(config_data: Dict[str, Any]):
    """Update the Streamlit configuration."""
    try:
        # Update the Streamlit configuration
        for key, value in config_data.items():
            if hasattr(default_config.streamlit, key):
                setattr(default_config.streamlit, key, value)

        # Save the configuration
        default_config.save("config.json")

        # Return the updated configuration
        return {"message": "Streamlit configuration updated", "config": default_config.streamlit.__dict__}
    except Exception as e:
        return {"error": f"Error updating Streamlit configuration: {e}"}

class CodeRequest(BaseModel):
    code: str
    
@app.post("/score_code/{problem_name}")
async def score_code_endpoint(problem_name: str, request: CodeRequest):
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    score = calculate_score(request.code, problem["unit_tests"])
    quality = analyze_code_quality(request.code)
    return {"score": score, "code_quality": quality}

# New endpoints for simulations

class SimulationRequest(BaseModel):
    problem_name: str
    model_id: str
    prompt_template: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    system_prompt: Optional[str] = None
    language: str = "python"
    generation: int = 0

@app.post("/simulations/run")
async def run_simation(request: SimulationRequest):
    """Run a single simulation with the given parameters."""
    # Check if the problem exists
    problem = next((p for p in problems if p["name"] == request.problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    
    # Create a job ID
    job_id = f"job-{uuid.uuid4()}"
    
    # Create a simulation job
    job = SimulationJob(
        id=job_id,
        problem_name=request.problem_name,
        model_id=request.model_id,
        prompt_template=request.prompt_template or "default",
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        system_prompt=request.system_prompt,
        language=request.language,
        status="pending",
        created_at=time.time()
    )
    
    # Submit the job
    await simulation_manager.submit_job(job)
    
    return {
        "job_id": job_id,
        "status": "submitted",
        "message": f"Simulation job {job_id} submitted successfully"
    }

class SimulationStartRequest(BaseModel):
    problem_name: str
    model_id: str
    population_size: int = 20
    num_generations: int = 5
    language: str = "python"

@app.post("/simulations/start")
async def start_simation(request: SimulationStartRequest):
    """Start a simulation with the given parameters."""
    try:
        # Check if the problem exists
        problem = next((p for p in problems if p["name"] == request.problem_name), None)
        if problem is None:
            return {"error": "Problem not found", "status": "error"}
        
        # Check if the model exists
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{default_config.api.model_proxy_url}/models",
                    timeout=2.0
                )
                models = response.json()
                # Handle different model data structures
                # Some APIs return a list of strings, others return objects with an id field
                model_exists = False
                if models and isinstance(models, list):
                    if len(models) > 0:
                        if isinstance(models[0], str):
                            # List of strings
                            model_exists = request.model_id in models
                        elif isinstance(models[0], dict) and "id" in models[0]:
                            # List of objects with id field
                            model_exists = any(m["id"] == request.model_id for m in models)
                        else:
                            # Unknown format, assume model exists
                            logger.warning(f"Unknown model format: {type(models[0])}. Assuming model exists.")
                            model_exists = True
                    else:
                        # Empty list, assume model exists
                        logger.warning("Empty models list. Assuming model exists.")
                        model_exists = True
                else:
                    # Not a list or empty response, assume model exists
                    logger.warning(f"Unexpected models response format: {type(models)}. Assuming model exists.")
                    model_exists = True

                if not model_exists:
                    return {"error": f"Model {request.model_id} not found", "status": "error"}
        except Exception as e:
            # If we can't check models, we'll assume the model exists
            logger.warning(f"Could not verify model existence: {e}")
    
        
        # Create a unique run ID
        run_id = f"sim-{uuid.uuid4()}"

        # Initialize status for this run_id in active_simulations
        active_simulations[run_id] = {
            "job_id": run_id,
            "problem_name": request.problem_name,
            "model_id": request.model_id,
            "status": "starting", # Will be updated to "running" by the task
            "progress": 0.0,
            "current_generation": 0,
            "total_generations": request.num_generations,
            "created_at": time.time(),
            "started_at": 0.0, # Will be set by the task
            "completed_at": 0.0,
            "result": None
        }
        logger.info(f"Initialized active_simulations entry for {run_id}")
        
        # Start the simulation in a background task
        asyncio.create_task(
            _run_genetic_algorithm(
                run_id=run_id, # Pass run_id
                problem_name=request.problem_name,
                pop_size=request.population_size,
                ngen=request.num_generations,
                language=request.language,
                num_workers=default_config.simulation.num_workers,
                model_id=request.model_id
            )
        )
        
        return {
            "run_id": run_id,
            "status": "started",
            "message": f"Simulation started for problem {request.problem_name} with model {request.model_id}",
            "details": {
                "problem_name": request.problem_name,
                "model_id": request.model_id,
                "population_size": request.population_size,
                "num_generations": request.num_generations,
                "language": request.language
            }
        }
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return {
            "error": f"Error starting simulation: {str(e)}",
            "status": "error"
        }

@app.get("/simulations/{job_id}")
async def get_simulation_status(job_id: str):
    """Get the status of a main simulation run."""
    global active_simulations
    simulation_data = active_simulations.get(job_id)
    
    if simulation_data is None:
        return {"error": "Simulation not found or not active"}
    
    return simulation_data

@app.get("/simulations")
async def list_simulations(status: Optional[str] = None):
    """List all simulation jobs, optionally filtered by status."""
    # This endpoint currently lists worker jobs. 
    # For main simulation runs, a new endpoint or modification might be needed,
    # or it could list from active_simulations.
    # For now, keeping original behavior for worker jobs.
    jobs = await simulation_manager.get_all_job_statuses()
    
    if status:
        if status in jobs:
            return {status: [job.to_dict() for job in jobs[status]]}
        return {status: []}
    
    return {
        "pending": [job.to_dict() for job in jobs["pending"]],
        "running": [job.to_dict() for job in jobs["running"]],
        "completed": [job.to_dict() for job in jobs["completed"]],
        "failed": [job.to_dict() for job in jobs["failed"]]
    }

@app.delete("/simulations/{job_id}")
async def cancel_simulation(job_id: str):
    """Cancel a pending simulation job."""
    # This likely refers to worker jobs. Cancelling a main GA run is more complex.
    # For now, keeping original behavior for worker jobs.
    # To cancel a main run, one might remove it from active_simulations
    # and add logic in _run_genetic_algorithm to check for its removal.
    global active_simulations
    if job_id in active_simulations:
        # Rudimentary cancellation: remove from active dict.
        # The _run_genetic_algorithm task should check this.
        del active_simulations[job_id]
        logger.info(f"Main simulation run {job_id} marked for cancellation (removed from active_simulations).")
        return {"message": f"Main simulation run {job_id} cancellation initiated."}

    success = await simulation_manager.cancel_job(job_id) # For worker jobs
    if success:
        return {"message": f"Worker job {job_id} cancelled successfully"}
    return {"error": "Failed to cancel job. Job may not exist or is already running/completed, or not a worker job."}

class GeneticAlgorithmRequest(BaseModel):
    problem_name: str
    population_size: int = 20
    num_generations: int = 5
    language: str = "python"
    num_workers: int = 4

@app.post("/genetic_algorithm/run")
async def run_genetic_algorithm(request: GeneticAlgorithmRequest):
    """Run the genetic algorithm for a problem."""
    # Check if the problem exists
    problem = next((p for p in problems if p["name"] == request.problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    
    # Create a unique ID for this run
    run_id = f"ga-run-{uuid.uuid4()}" # This is a generic GA run, not tied to the main simulation flow with progress
                                     # The main simulation flow uses 'sim-<uuid>'
    
    # Start the genetic algorithm in a background task
    asyncio.create_task(
        _run_genetic_algorithm( # This will use the updated _run_genetic_algorithm
            run_id=run_id,
            problem_name=request.problem_name,
            pop_size=request.population_size,
            ngen=request.num_generations,
            language=request.language,
            num_workers=request.num_workers
            # model_id is not part of GeneticAlgorithmRequest, so it will be None
        )
    )
    
    return {
        "run_id": run_id,
        "status": "started", # This status is for the submission, actual status in active_simulations
        "message": f"Genetic algorithm started for problem {request.problem_name}"
    }

async def _run_genetic_algorithm(run_id: str, problem_name: str, pop_size: int,
                                ngen: int, language: str, num_workers: int, model_id: str = None):
    """Run the genetic algorithm, store results, and update job progress."""
    # Import the run_genetic_algorithm from simulation_manager module
    from simulation_manager import run_genetic_algorithm as run_ga_from_sim_manager
    global active_simulations # Ensure we're using the global dict
    logger.info(f"_run_genetic_algorithm started for run_id: {run_id}")

    # Update status to running and set started_at time
    if run_id in active_simulations:
        active_simulations[run_id]["status"] = "running"
        active_simulations[run_id]["started_at"] = time.time()
        active_simulations[run_id]["total_generations"] = ngen # Ensure total_generations is set
        logger.info(f"Simulation {run_id} marked as running. Initial state: {active_simulations[run_id]}")
    else:
        logger.warning(f"Run ID {run_id} (e.g. from /genetic_algorithm/run) not found in active_simulations at task start. Progress will not be tracked in active_simulations.")
        # If not tracked, we can't update its progress in the shared dict.
        # The GA will run, and results will be saved to file, but UI won't see live progress for this run_id.

    try:
        final_results_stats = None
        logger.info(f"Starting iteration for progress_update from run_ga_from_sim_manager for {run_id}")
        async for progress_update in run_ga_from_sim_manager(
            problem_name=problem_name,
            pop_size=pop_size,
            ngen=ngen,
            language=language,
            num_workers=num_workers,
            model_id=model_id
        ):
            logger.debug(f"Received progress_update for {run_id}: {progress_update}")
            if run_id in active_simulations: 
                if isinstance(progress_update, tuple) and len(progress_update) == 3:
                    current_gen, total_gen_from_yield, _ = progress_update 
                    
                    active_simulations[run_id]["current_generation"] = current_gen
                    # Progress: (number of steps completed) / (total number of steps) * 100
                    # Steps are: initial eval (gen 0) + gen 1 eval + ... + gen ngen eval. Total ngen+1 steps.
                    # current_gen is 0-indexed for these steps. So current_gen+1 steps are done.
                    # total_gen_from_yield is ngen. So total_gen_from_yield+1 total steps.
                    progress_percentage = ((current_gen + 1) / (total_gen_from_yield + 1)) * 100 if (total_gen_from_yield + 1) > 0 else 0
                    active_simulations[run_id]["progress"] = progress_percentage
                    
                    logger.info(f"Updating {run_id}: Gen {current_gen} (step {current_gen+1}/{total_gen_from_yield+1}), Progress: {progress_percentage:.2f}%")
                    logger.debug(f"State of active_simulations[{run_id}] after update: {active_simulations[run_id]}")
                elif isinstance(progress_update, dict): # This should be the final stats dictionary
                    final_results_stats = progress_update
                    logger.info(f"Simulation {run_id} GA loop finished, received final stats dictionary: {final_results_stats}")
                else:
                    logger.warning(f"Unexpected progress_update format for {run_id}: {progress_update}")

            elif isinstance(progress_update, dict): # If not tracked in active_simulations (e.g. ga-run-id), still capture final_results
                 final_results_stats = progress_update
                 logger.info(f"Untracked run {run_id} GA loop finished, received final stats dictionary.")
            else:
                logger.warning(f"Untracked run {run_id} received unexpected progress_update: {progress_update}")
        
        logger.info(f"Finished iterating progress_updates for {run_id}.")


        if run_id in active_simulations: # If it was a tracked simulation
            if final_results_stats:
                active_simulations[run_id]["status"] = "completed"
                active_simulations[run_id]["progress"] = 100.0
                active_simulations[run_id]["result"] = final_results_stats
                active_simulations[run_id]["completed_at"] = time.time()
                logger.info(f"Simulation {run_id} marked as completed with final results.")
            else: # No final stats, but was tracked
                logger.warning(f"No final_results_stats received for tracked simulation {run_id}.")
                active_simulations[run_id]["status"] = "failed"
                active_simulations[run_id]["result"] = {"error": "Genetic algorithm did not produce final results."}
                active_simulations[run_id]["completed_at"] = time.time()
        elif final_results_stats: # Untracked run (e.g. from /genetic_algorithm/run) but got results
            logger.info(f"Untracked run {run_id} completed with final stats.")
        else: # Untracked run and no final stats
            logger.warning(f"Untracked run {run_id} did not produce final results.")


        # Store the final results (full stats) to a file, regardless of tracking type
        if final_results_stats:
            os.makedirs("results", exist_ok=True) 
            with open(f"results/{run_id}.json", "w") as f:
                import json
                json.dump(final_results_stats, f, indent=2)
            logger.info(f"Final results for {run_id} saved to results/{run_id}.json")

            # Populate ResultsDB
            try:
                overall_best_score = final_results_stats.get("best_score")
                overall_best_code = final_results_stats.get("best_code")

                for gen_data in final_results_stats.get("generations", []):
                    gen_num = gen_data["gen"]
                    best_ind_dict = gen_data.get("best_individual", {})
                    best_fitness_this_gen = gen_data.get("best_fitness")

                    if not best_ind_dict or best_fitness_this_gen is None:
                        logger.warning(f"Skipping DB entry for gen {gen_num} of {run_id} due to missing data: {gen_data}")
                        continue

                    # Determine code for this generation's best
                    # This is an approximation; ideally, each evaluated individual's code would be stored.
                    code_for_this_gen_best = None
                    if best_fitness_this_gen == overall_best_score:
                        # If this generation's best is also the overall best, use the overall best code.
                        code_for_this_gen_best = overall_best_code
                    else:
                        # Otherwise, we don't have the specific code for this generation's best in final_results_stats.
                        # Could regenerate it, but that's expensive. Store placeholder or None.
                        code_for_this_gen_best = f"# Code for best of gen {gen_num} not in summary stats, score: {best_fitness_this_gen}"

                    sim_result_id = f"{run_id}-gen-{gen_num}-best"
                    sim_result = SimulationResult(
                        id=sim_result_id,
                        problem_name=problem_name, # from _run_genetic_algorithm args
                        model_id=best_ind_dict.get("model_id"),
                        prompt_template=best_ind_dict.get("prompt_template"),
                        temperature=best_ind_dict.get("temperature"),
                        max_tokens=best_ind_dict.get("max_tokens"),
                        top_p=best_ind_dict.get("top_p"),
                        system_prompt=best_ind_dict.get("system_prompt"),
                        language=language, # from _run_genetic_algorithm args
                        code=code_for_this_gen_best,
                        # test_score and quality_score are not directly available in gen_data for generation's best.
                        # These would come from individual evaluations if those were stored.
                        # For now, setting to None or a sensible default.
                        test_score=None,  # Or best_fitness_this_gen if it represents test pass rate
                        quality_score=None, 
                        final_score=best_fitness_this_gen,
                        generation=gen_num,
                        timestamp=time.time() # Timestamp for this specific result entry
                    )
                    await asyncio.to_thread(results_db.add_result, sim_result)
                    logger.debug(f"Added SimulationResult to DB for {run_id} gen {gen_num}: {sim_result_id}")

                    # The add_generation_stats method in results_db.py takes problem_name, generation, and a stats dict.
                    # The gen_data itself contains avg, min, max, etc.
                    await asyncio.to_thread(results_db.add_generation_stats, problem_name, gen_num, gen_data)
                    logger.debug(f"Added GenerationStats to DB for {run_id} gen {gen_num} using gen_data: {gen_data}")
                logger.info(f"Populated ResultsDB for {run_id} with generation bests and stats.")
            except Exception as db_ex:
                logger.error(f"Error populating ResultsDB for {run_id}: {db_ex}", exc_info=True)
        
        # Generate visualizations (even if it failed, some data might be there for partial viz)
        visualizer = ResultsVisualizer()
        visualizer.plot_generation_progression(problem_name)
        visualizer.plot_model_comparison(problem_name)
        visualizer.plot_score_distribution(problem_name)
        visualizer.generate_html_report(problem_name)
        visualizer.close()
        
    except Exception as e:
        logger.error(f"Error in _run_genetic_algorithm for {run_id}: {e}", exc_info=True)
        if run_id in active_simulations:
            active_simulations[run_id]["status"] = "failed"
            active_simulations[run_id]["result"] = {"error": str(e)}
            active_simulations[run_id]["completed_at"] = time.time()

@app.get("/genetic_algorithm/results/{problem_name}")
async def get_genetic_algorithm_results(problem_name: str):
    """Get the results of genetic algorithm runs for a problem."""
    # Check if the problem exists
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    
    # Get statistics from the database
    stats = results_db.get_problem_statistics(problem_name)
    
    # Get the best result
    best_result = results_db.get_best_result(problem_name)
    
    return {
        "problem_name": problem_name,
        "statistics": stats,
        "best_result": best_result.to_dict() if best_result else None
    }

@app.get("/results/{problem_name}")
async def get_problem_results(problem_name: str):
    """Get all results for a problem."""
    # Check if the problem exists
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    
    # Get results from the database
    results = results_db.get_results_by_problem(problem_name)
    
    return {
        "problem_name": problem_name,
        "results": [result.to_dict() for result in results]
    }

@app.get("/results/{problem_name}/best")
async def get_best_result(problem_name: str):
    """Get the best result for a problem."""
    # Check if the problem exists
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    
    # Get the best result from the database
    best_result = results_db.get_best_result(problem_name)
    if best_result is None:
        return {"error": "No results found for this problem"}
    
    return best_result.to_dict()

@app.get("/results/problem/{problem_name}")
async def get_problem_results_endpoint(problem_name: str):
    """Get all results for a specific problem."""
    # Check if the problem exists
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    
    # Get results from the database
    results = results_db.get_results_by_problem(problem_name)
    
    return {
        "problem_name": problem_name,
        "results": [result.to_dict() for result in results]
    }

@app.get("/results/problem/{problem_name}/generations")
async def get_problem_generations_endpoint(problem_name: str):
    """Get generation statistics for a specific problem."""
    # Check if the problem exists
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}

    # Get generation statistics from the database
    generation_stats = results_db.get_all_generation_stats(problem_name)

    return {
        "problem_name": problem_name,
        "generations": generation_stats
    }
    
@app.get("/results/{problem_name}/best")
async def get_best_result(problem_name: str):
    """Get the best result for a problem."""
    # Check if the problem exists
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    
    # Get generation statistics from the database
    generation_stats = results_db.get_all_generation_stats(problem_name)

    return {
        "problem_name": problem_name,
        "generations": generation_stats
    }

    # Get the best result from the database
    best_result = results_db.get_best_result(problem_name)
    if best_result is None:
        return {"error": "No results found for this problem"}
    
    return best_result.to_dict()

@app.get("/results/{problem_name}/generation/{generation}")
async def get_generation_results(problem_name: str, generation: int):
    """Get results for a specific generation of a problem."""
    # Check if the problem exists
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    
    # Get results from the database
    results = results_db.get_results_by_generation(problem_name, generation)
    
    return {
        "problem_name": problem_name,
        "generation": generation,
        "results": [result.to_dict() for result in results]
    }

@app.get("/visualizations/{problem_name}")
async def get_visualizations(problem_name: str):
    """Get visualization URLs for a problem."""
    # Check if the problem exists
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return {"error": "Problem not found"}
    
    # Generate visualizations
    visualizer = ResultsVisualizer()
    
    # Generate the visualizations
    generation_progression = visualizer.plot_generation_progression(problem_name)
    model_comparison = visualizer.plot_model_comparison(problem_name)
    score_distribution = visualizer.plot_score_distribution(problem_name)
    html_report = visualizer.generate_html_report(problem_name)
    
    visualizer.close()
    
    # Return the URLs
    return {
        "problem_name": problem_name,
        "visualizations": {
            "generation_progression": generation_progression,
            "model_comparison": model_comparison,
            "score_distribution": score_distribution,
            "html_report": html_report
        }
    }
