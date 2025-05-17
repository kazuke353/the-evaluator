
import asyncio
import random
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import genetic_algorithm as ga
from scoring import evaluate_code
from config import default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simulation_manager")

@dataclass
class SimulationJob:
    """Represents a simulation job to be executed."""
    id: str
    problem_name: str
    model_id: str
    prompt_template: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    system_prompt: Optional[str] = None
    language: str = "python"
    status: str = "pending" # e.g., pending, running, completed, failed
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    result: Optional[Dict[str, Any]] = None
    # Add progress tracking
    progress: float = 0.0  # Percentage completion
    current_generation: int = 0
    total_generations: int = 0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self):
        """Convert the job to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        """Create a job from a dictionary."""
        return cls(**data)
    
    @classmethod
    def from_chromosome(cls, job_id: str, problem_name: str, chromosome: ga.Chromosome, language: str = "python"):
        """Create a job from a chromosome."""
        return cls(
            id=job_id,
            problem_name=problem_name,
            model_id=chromosome.model_id,
            prompt_template=chromosome.prompt_template,
            temperature=chromosome.temperature,
            max_tokens=chromosome.max_tokens,
            top_p=chromosome.top_p,
            system_prompt=chromosome.system_prompt,
            language=language
        )


class JobQueue:
    """A queue for managing simulation jobs."""
    
    def __init__(self):
        self.pending_jobs: List[SimulationJob] = []
        self.running_jobs: Dict[str, SimulationJob] = {}
        self.completed_jobs: Dict[str, SimulationJob] = {}
        self.failed_jobs: Dict[str, SimulationJob] = {}
        self._lock = asyncio.Lock()
    
    async def add_job(self, job: SimulationJob) -> str:
        """Add a job to the queue."""
        async with self._lock:
            self.pending_jobs.append(job)
            logger.info(f"Added job {job.id} to the queue")
            return job.id
    
    async def get_next_job(self) -> Optional[SimulationJob]:
        """Get the next job from the queue."""
        async with self._lock:
            if not self.pending_jobs:
                return None
            
            job = self.pending_jobs.pop(0)
            self.running_jobs[job.id] = job
            job.status = "running"
            job.started_at = time.time()
            logger.info(f"Started job {job.id}")
            return job
    
    async def complete_job(self, job_id: str, result: Dict[str, Any]) -> None:
        """Mark a job as completed."""
        async with self._lock:
            if job_id not in self.running_jobs:
                logger.warning(f"Job {job_id} not found in running jobs")
                return
            
            job = self.running_jobs.pop(job_id)
            job.status = "completed"
            job.completed_at = time.time()
            job.result = result
            self.completed_jobs[job_id] = job
            logger.info(f"Completed job {job_id}")
    
    async def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed."""
        async with self._lock:
            if job_id not in self.running_jobs:
                logger.warning(f"Job {job_id} not found in running jobs")
                return
            
            job = self.running_jobs.pop(job_id)
            job.status = "failed"
            job.completed_at = time.time()
            job.result = {"error": error}
            self.failed_jobs[job_id] = job
            logger.warning(f"Failed job {job_id}: {error}")
    
    async def get_job(self, job_id: str) -> Optional[SimulationJob]:
        """Get a job by ID."""
        async with self._lock:
            if job_id in self.pending_jobs:
                return self.pending_jobs[job_id]
            if job_id in self.running_jobs:
                return self.running_jobs[job_id]
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id]
            if job_id in self.failed_jobs:
                return self.failed_jobs[job_id]
            return None
    
    async def get_all_jobs(self) -> Dict[str, List[SimulationJob]]:
        """Get all jobs."""
        async with self._lock:
            return {
                "pending": self.pending_jobs.copy(),
                "running": list(self.running_jobs.values()),
                "completed": list(self.completed_jobs.values()),
                "failed": list(self.failed_jobs.values())
            }
    
    async def clear_completed_jobs(self) -> None:
        """Clear completed jobs."""
        async with self._lock:
            self.completed_jobs.clear()
            self.failed_jobs.clear()
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        async with self._lock:
            for i, job in enumerate(self.pending_jobs):
                if job.id == job_id:
                    self.pending_jobs.pop(i)
                    job.status = "cancelled"
                    logger.info(f"Cancelled job {job_id}")
                    return True
            
            if job_id in self.running_jobs:
                # Can't cancel running jobs
                logger.warning(f"Cannot cancel running job {job_id}")
                return False
            
            logger.warning(f"Job {job_id} not found")
            return False


class SimulationWorker:
    """A worker for executing simulation jobs."""
    
    def __init__(self, worker_id: str, job_queue: JobQueue):
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.running = False
        self._task = None
    
    async def start(self):
        """Start the worker."""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"Started worker {self.worker_id}")
    
    async def stop(self):
        """Stop the worker."""
        if not self.running:
            return
        
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped worker {self.worker_id}")
    
    async def _run(self):
        """Run the worker loop."""
        while self.running:
            job = await self.job_queue.get_next_job()
            if not job:
                # No jobs available, wait a bit
                await asyncio.sleep(0.1)
                continue
            
            try:
                # Execute the job
                result = await self._execute_job(job)
                await self.job_queue.complete_job(job.id, result)
            except Exception as e:
                logger.exception(f"Error executing job {job.id}")
                await self.job_queue.fail_job(job.id, str(e))
    
    async def _execute_job(self, job: SimulationJob) -> Dict[str, Any]:
        """Execute a simulation job."""
        # Get configuration
        config = default_config

        # Generate a prompt using the prompt template
        prompt = await ga.generate_prompt(job.problem_name, job.prompt_template) # await async call

        # Use the model to generate code
        code = await ga.generate_code( # await async call
            job.model_id,
            prompt,
            job.temperature,
            job.max_tokens,
            job.top_p,
            job.system_prompt
        )

        # Get the problem details to extract unit tests
        problems = await ga.fetch_problems() # await async call
        problem = next((p for p in problems if p["name"] == job.problem_name), None)

        if not problem:
            raise ValueError(f"Problem {job.problem_name} not found")

        # Evaluate the code
        unit_tests = problem.get("unit_tests", "")

        # Use weights from configuration
        weights = {
            "test_weight": config.simulation.test_weight,
            "quality_weight": config.simulation.quality_weight
        }

        evaluation = evaluate_code(code, unit_tests, job.language, weights)

        # Return the result
        return {
            "code": code,
            "evaluation": evaluation,
            "score": evaluation.get("final_score", 0)
        }


class SimulationManager:
    """Manages parallel simulation of code generation and evaluation."""
    

    def __init__(self, num_workers: Optional[int] = None):
        # Get configuration
        config = default_config

        # Use provided number of workers or default from configuration
        self.num_workers = num_workers or config.simulation.num_workers

        self.job_queue = JobQueue()
        self.workers: List[SimulationWorker] = []
        self.running = False
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
    

    async def start(self):
        """Start the simulation manager."""
        if self.running:
            return
        
        self.running = True
        
        # Create and start workers
        for i in range(self.num_workers):
            worker = SimulationWorker(f"worker-{i}", self.job_queue)
            self.workers.append(worker)
            await worker.start()
        
        logger.info(f"Started simulation manager with {self.num_workers} workers")
    
    async def stop(self):
        """Stop the simulation manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop all workers
        for worker in self.workers:
            await worker.stop()
        
        self.workers.clear()
        self._executor.shutdown(wait=True)
        
        logger.info("Stopped simulation manager")
    
    async def submit_job(self, job: SimulationJob) -> str:
        """Submit a job to the queue."""
        return await self.job_queue.add_job(job)
    
    async def submit_chromosome(self, problem_name: str, chromosome: ga.Chromosome, language: str = "python") -> str:
        """Submit a chromosome for evaluation."""
        job_id = f"job-{int(time.time())}-{random.randint(1000, 9999)}"
        job = SimulationJob.from_chromosome(job_id, problem_name, chromosome, language)
        return await self.submit_job(job)
    
    async def submit_batch(self, problem_name: str, chromosomes: List[ga.Chromosome], language: str = "python") -> List[str]:
        """Submit a batch of chromosomes for evaluation."""
        job_ids = []
        for chromosome in chromosomes:
            job_id = await self.submit_chromosome(problem_name, chromosome, language)
            job_ids.append(job_id)
        return job_ids
    
    async def wait_for_jobs(self, job_ids: List[str], timeout: float = None) -> Dict[str, Dict[str, Any]]:
        """Wait for jobs to complete and return their results."""
        start_time = time.time()
        pending_jobs = set(job_ids)
        results = {}
        
        while pending_jobs:
            # Check if timeout has been reached
            if timeout and time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for jobs: {pending_jobs}")
                break
            
            # Check for completed jobs
            jobs = await self.job_queue.get_all_jobs()
            completed = {j.id: j.result for j in jobs["completed"] if j.id in pending_jobs}
            failed = {j.id: j.result for j in jobs["failed"] if j.id in pending_jobs}
            
            # Update results and pending jobs
            results.update(completed)
            results.update(failed)
            pending_jobs -= set(completed.keys()) | set(failed.keys())
            
            if not pending_jobs:
                break
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        return results
    
    async def evaluate_population(self, problem_name: str, population: List[ga.Chromosome], language: str = "python") -> List[Tuple[ga.Chromosome, float]]:
        """Evaluate a population of chromosomes."""
        # Submit all chromosomes for evaluation
        job_ids = await self.submit_batch(problem_name, population, language)
        
        # Wait for all jobs to complete
        results = await self.wait_for_jobs(job_ids)
        
        # Combine chromosomes with their scores
        chromosome_scores = []
        for i, chromosome in enumerate(population):
            job_id = job_ids[i]
            if job_id in results:
                score = results[job_id].get("score", 0)
                chromosome_scores.append((chromosome, score))
            else:
                # Job didn't complete for some reason
                chromosome_scores.append((chromosome, 0))
        
        return chromosome_scores
    
    async def get_job_status(self, job_id: str) -> Optional[SimulationJob]:
        """Get the status of a job."""
        return await self.job_queue.get_job(job_id)
    
    async def get_all_job_statuses(self) -> Dict[str, List[SimulationJob]]:
        """Get the status of all jobs."""
        return await self.job_queue.get_all_jobs()
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        return await self.job_queue.cancel_job(job_id)
    
    async def clear_completed_jobs(self) -> None:
        """Clear completed jobs."""
        await self.job_queue.clear_completed_jobs()

async def run_genetic_algorithm(problem_name: str, pop_size: Optional[int] = None,
                               ngen: Optional[int] = None, language: str = "python",
                               num_workers: Optional[int] = None, model_id: Optional[str] = None):
    """
    Run the genetic algorithm using parallel simulation, yielding progress.
    
    Args:
        problem_name: The name of the problem to solve
        pop_size: The size of the population, or None to use the default from configuration
        ngen: The number of generations, or None to use the default from configuration
        language: The programming language to use
        num_workers: The number of workers to use, or None to use the default from configuration
        model_id: The specific model ID to use, or None to use various models
    
    Yields:
        Tuple[int, int, Optional[Dict[str, Any]]]: (current_generation, total_generations, current_best_result_dict)
        The final yield will be the full stats dictionary.
    """
    # Get configuration
    config = default_config
    
    # Use provided parameters or defaults from configuration
    pop_size = pop_size or config.genetic_algorithm.population_size
    ngen = ngen or config.genetic_algorithm.num_generations
    num_workers = num_workers or config.simulation.num_workers
    
    # Initialize the simulation manager
    sim_manager = SimulationManager(num_workers=num_workers)
    await sim_manager.start()
    
    try:
        # Initialize the population
        toolbox = ga.toolbox

        # If a specific model_id is provided, create a population with that model
        if model_id:
            logger.info(f"Creating population with specific model: {model_id}")
            population = []
            for _ in range(pop_size):
                # Create an individual with the specified model_id
                individual = toolbox.individual()
                # Set the model_id using the property we added
                individual.model_id = model_id
                # Keep other attributes as randomly generated
                population.append(individual)
        else:
            # Create a diverse population with different models
            population = toolbox.population(n=pop_size)
    

        # Initialize statistics
        stats = {
            "generations": [],
            "best_fitness": [],
            "avg_fitness": [],
            "best_individual": None,
            "best_code": None,
            "best_score": 0
        }
        
        # Evaluate the initial population
        logger.info(f"Evaluating initial population of {len(population)} individuals")
        chromosome_scores = await sim_manager.evaluate_population(problem_name, population, language)
        
        # Update fitness values
        for i, (chromosome, score) in enumerate(chromosome_scores):
            individual = population[i]
            individual.fitness.values = (score,)
        
        # Record statistics
        fits = [ind.fitness.values[0] for ind in population]
        
        # Use DEAP's tools module to select the best individual
        from deap import tools as deap_tools
        best_ind = deap_tools.selBest(population, 1)[0]
        
        best_chromosome = ga.Chromosome.from_list(best_ind)
        
        current_best_result_for_yield = {
            "model_id": best_chromosome.model_id,
            "final_score": best_ind.fitness.values[0],
            "code": None # Code for best will be generated later if it's overall best
        }

        gen_stats = {
            "gen": 0,
            "avg": sum(fits) / len(fits),
            "min": min(fits),
            "max": max(fits),
            "best_individual": best_chromosome.to_dict(),
            "best_fitness": best_ind.fitness.values[0]
        }
        
        stats["generations"].append(gen_stats)
        stats["best_fitness"].append(best_ind.fitness.values[0])
        stats["avg_fitness"].append(sum(fits) / len(fits))
        
        if best_ind.fitness.values[0] > stats["best_score"]:
            stats["best_individual"] = best_chromosome.to_dict()
            stats["best_score"] = best_ind.fitness.values[0]
            
            # Generate the best code
            prompt = await ga.generate_prompt(problem_name, best_chromosome.prompt_template) # await async call
            code = await ga.generate_code( # await async call
                best_chromosome.model_id,
                prompt,
                best_chromosome.temperature,
                best_chromosome.max_tokens,
                best_chromosome.top_p,
                best_chromosome.system_prompt
            )
            stats["best_code"] = code
            current_best_result_for_yield["code"] = code # Update code for yield
        
        logger.info(f"Initial population evaluated. Best fitness: {best_ind.fitness.values[0]}")
        yield (0, ngen, current_best_result_for_yield)

        # Begin the evolution
        for gen in range(1, ngen + 1):
            logger.info(f"Starting generation {gen}")
            
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for i in range(1, len(offspring), 2):
                if i < len(offspring) - 1:
                    offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            
            for i in range(len(offspring)):
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
            
            # Convert offspring to chromosomes for evaluation
            offspring_chromosomes = [ga.Chromosome.from_list(ind) for ind in offspring]
            
            # Evaluate the offspring
            logger.info(f"Evaluating {len(offspring)} offspring")
            chromosome_scores = await sim_manager.evaluate_population(problem_name, offspring_chromosomes, language)
            
            # Update fitness values
            for i, (chromosome, score) in enumerate(chromosome_scores):
                offspring[i].fitness.values = (score,)
            
            # Replace the population with the offspring
            population[:] = offspring
            
            # Record statistics
            fits = [ind.fitness.values[0] for ind in population]
            
            # Use DEAP's tools module to select the best individual
            from deap import tools as deap_tools
            best_ind = deap_tools.selBest(population, 1)[0]
            best_chromosome = ga.Chromosome.from_list(best_ind)

            current_best_result_for_yield = {
                "model_id": best_chromosome.model_id,
                "final_score": best_ind.fitness.values[0],
                "code": None 
            }
            
            gen_stats = {
                "gen": gen,
                "avg": sum(fits) / len(fits),
                "min": min(fits),
                "max": max(fits),
                "best_individual": best_chromosome.to_dict(),
                "best_fitness": best_ind.fitness.values[0]
            }
            
            stats["generations"].append(gen_stats)
            stats["best_fitness"].append(best_ind.fitness.values[0])
            stats["avg_fitness"].append(sum(fits) / len(fits))
            
            if best_ind.fitness.values[0] > stats["best_score"]:
                stats["best_individual"] = best_chromosome.to_dict()
                stats["best_score"] = best_ind.fitness.values[0]
                
                # Generate the best code
                prompt = await ga.generate_prompt(problem_name, best_chromosome.prompt_template) # await async call
                code = await ga.generate_code( # await async call
                    best_chromosome.model_id,
                    prompt,
                    best_chromosome.temperature,
                    best_chromosome.max_tokens,
                    best_chromosome.top_p,
                    best_chromosome.system_prompt
                )
                stats["best_code"] = code
                current_best_result_for_yield["code"] = code
            
            logger.info(f"Generation {gen} complete. Best fitness: {best_ind.fitness.values[0]}")
            yield (gen, ngen, current_best_result_for_yield)
        
        yield stats # Final yield is the complete stats dictionary
    finally:
        # Stop the simulation manager
        await sim_manager.stop()

if __name__ == "__main__":
    async def main():
        # Example usage
        problem_name = "fibonacci"
        
        # Run the genetic algorithm
        results = await run_genetic_algorithm(
            problem_name=problem_name,
            pop_size=10,
            ngen=3,
            language="python",
            num_workers=4
        )
        
        # Print the results
        print(f"Best fitness: {results['best_score']}")
        print(f"Best individual: {results['best_individual']}")
        print(f"Best code:\n{results['best_code']}")
        
        # Plot the fitness over generations
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(results["best_fitness"])), results["best_fitness"], label="Best Fitness")
        plt.plot(range(len(results["avg_fitness"])), results["avg_fitness"], label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(f"Fitness Evolution for {problem_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{problem_name}_fitness.png")
        plt.close()
    
    # Run the example
    asyncio.run(main())
