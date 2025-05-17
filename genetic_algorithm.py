
import random
import numpy as np
from deap import base, creator, tools, algorithms
import httpx # Changed from requests
import json
import copy
import asyncio # Added for async operations
from config import default_config

# API base URL from config
def get_api_base_url():
    # Ensure we are using the potentially updated default_config
    # This helps if default_config was updated after initial module load
    # by the main application context.
    from config import default_config as current_default_config
    return f"http://{current_default_config.api.host}:{current_default_config.api.port}"

# Define the chromosome representation
class Chromosome:
    """
    Represents a chromosome in the genetic algorithm.
    
    Attributes:
        model_id (str): The ID of the model to use
        prompt_template (str): The prompt template to use
        temperature (float): The temperature parameter for the model
        max_tokens (int): The maximum number of tokens to generate
        top_p (float): The top_p parameter for the model
        system_prompt (str): The system prompt to use
    """
    def __init__(self, model_id=None, prompt_template=None, temperature=0.7, 
                 max_tokens=1000, top_p=0.9, system_prompt=None):
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.system_prompt = system_prompt
    
    def to_list(self):
        """Convert the chromosome to a list representation."""
        return [
            self.model_id,
            self.prompt_template,
            self.temperature,
            self.max_tokens,
            self.top_p,
            self.system_prompt
        ]
    
    @classmethod
    def from_list(cls, chromosome_list):
        """Create a chromosome from a list representation."""
        return cls(
            model_id=chromosome_list[0],
            prompt_template=chromosome_list[1],
            temperature=chromosome_list[2],
            max_tokens=chromosome_list[3],
            top_p=chromosome_list[4],
            system_prompt=chromosome_list[5]
        )
    
    def __str__(self):
        return (f"Chromosome(model_id={self.model_id}, "
                f"prompt_template={self.prompt_template}, "
                f"temperature={self.temperature}, "
                f"max_tokens={self.max_tokens}, "
                f"top_p={self.top_p})")
    
    def to_dict(self):
        """Convert the chromosome to a dictionary representation."""
        return {
            "model_id": self.model_id,
            "prompt_template": self.prompt_template,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "system_prompt": self.system_prompt
        }

# Define the async fitness function
async def fitness(individual, problem_name): # Made async
    """
    Calculate the fitness of an individual. (async)
    
    Args:
        individual (list): The individual to evaluate
        problem_name (str): The name of the problem to solve
        
    Returns:
        tuple: A tuple containing the fitness score
    """
    # Convert the individual to a Chromosome
    chromosome = Chromosome.from_list(individual)
    
    # Ensure all attributes are properly set from the individual
    # This is a safeguard in case the property access isn't used directly
    if hasattr(individual, 'model_id'):
        chromosome.model_id = individual.model_id
    if hasattr(individual, 'prompt_template'):
        chromosome.prompt_template = individual.prompt_template
    if hasattr(individual, 'temperature'):
        chromosome.temperature = individual.temperature
    if hasattr(individual, 'max_tokens'):
        chromosome.max_tokens = individual.max_tokens
    if hasattr(individual, 'top_p'):
        chromosome.top_p = individual.top_p
    if hasattr(individual, 'system_prompt'):
        chromosome.system_prompt = individual.system_prompt
    
    # Generate a prompt using the prompt template
    prompt = await generate_prompt(problem_name, chromosome.prompt_template) # await async call
    
    # Use the model to generate code
    code = await generate_code( # await async call
        chromosome.model_id, 
        prompt, 
        chromosome.temperature, 
        int(chromosome.max_tokens),  # Ensure max_tokens is an integer
        chromosome.top_p,
        chromosome.system_prompt
    )
    
    # Score the generated code
    score = await score_code(problem_name, code) # await async call
    
    return score,

# Define the async function to generate a prompt
async def generate_prompt(problem_name, prompt_template): # Changed to async
    """
    Generate a prompt for a problem. (async)
    
    Args:
        problem_name (str): The name of the problem
        prompt_template (str): The prompt template to use
        
    Returns:
        str: The generated prompt
    """
    try:
        async with httpx.AsyncClient() as client: # Changed to httpx.AsyncClient
            response = await client.get(f"{get_api_base_url()}/generate_prompt/{problem_name}", timeout=15) # Increased timeout
            if response.status_code == 200:
                return response.json()["prompt"]
    except (httpx.RequestError, ValueError, json.JSONDecodeError) as e: # Changed to httpx.RequestError, added json.JSONDecodeError
        print(f"Warning: Could not generate prompt from API: {e}")
        print("Falling back to direct problem description")
    
    # Fallback to using the problem description directly
    problems = await fetch_problems(use_api=False) # Changed to await
    problem = next((p for p in problems if p["name"] == problem_name), None)
    if problem is None:
        return f"Write code to solve the following problem: {problem_name}"
    
    return f"Write code to solve the following problem: {problem['description']}"

# Define the async function to generate code using a model
async def generate_code(model_id, prompt, temperature=0.7, max_tokens=1000, top_p=0.9, system_prompt=None): # Changed to async
    """
    Generate code using a model. (async)
    
    Args:
        model_id (str): The ID of the model to use
        prompt (str): The prompt to use
        temperature (float): The temperature parameter for the model
        max_tokens (int): The maximum number of tokens to generate
        top_p (float): The top_p parameter for the model
        system_prompt (str): The system prompt to use
        
    Returns:
        str: The generated code
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful coding assistant. Write clean, efficient code that solves the given problem."
        )
    
    # Ensure max_tokens is an integer
    max_tokens = int(max_tokens)
    
    try:
        async with httpx.AsyncClient() as client: # Changed to httpx.AsyncClient
            response = await client.post( # Changed to await client.post
                "http://localhost:8080/v1/chat/completions",
                json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": True  # Changed to True to correctly handle SSE
            },
            timeout=30  # Increased timeout for potentially longer streams
        )

        # Check if the response is valid
        if response.status_code != 200:
            error_text = await response.aread()
            print(f"Error: Model API returned status code {response.status_code}")
            print(f"Response: {error_text.decode(errors='ignore')[:500]}")
            return _generate_fallback_code(prompt)

        # Process the SSE stream
        accumulated_content = ""
        try:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data:"):
                    data_content = line[len("data:"):].strip()
                    if data_content == "[DONE]":
                        break  # Stream finished
                    try:
                        chunk = json.loads(data_content)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"] is not None:
                                accumulated_content += delta["content"]
                            # Handle other parts of the chunk if necessary, e.g., finish_reason
                            if chunk["choices"][0].get("finish_reason") is not None:
                                # Potentially log finish_reason or handle specific cases
                                pass 
                        elif "error" in chunk: # Handle error messages within the stream
                            error_message = chunk["error"].get("message", "Unknown error in stream")
                            print(f"Error in model stream: {error_message}")
                            # Depending on the error, might want to return fallback or raise an exception
                            # For now, we'll break and return what we have, or fallback if nothing.
                            if not accumulated_content:
                                return _generate_fallback_code(prompt, f"Stream error: {error_message}")
                            break # Stop processing on error
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from stream line: {data_content}")
                        # Decide how to handle non-JSON data lines if they are expected or problematic
                else:
                    print(f"Warning: Received non-SSE line: {line}")


            if not accumulated_content:
                print("Warning: Stream completed but no content was accumulated.")
                return _generate_fallback_code(prompt, "Empty stream response")
            return accumulated_content

        except httpx.ReadTimeout:
            print("Error: Read timeout while streaming model response.")
            if accumulated_content:
                print("Returning partially accumulated content due to read timeout.")
                return accumulated_content
            return _generate_fallback_code(prompt, "Read timeout during stream")
        except Exception as e_stream:
            print(f"Error processing model stream: {e_stream}")
            if accumulated_content:
                print(f"Returning partially accumulated content due to stream error: {e_stream}")
                return accumulated_content
            return _generate_fallback_code(prompt, f"Stream processing error: {e_stream}")

    except httpx.RequestError as e: # Changed to httpx.RequestError
        print(f"Error connecting to model API: {e}")
        return _generate_fallback_code(prompt, f"API connection error: {e}")


def _generate_fallback_code(prompt, reason="Model API failure"):
    """
    Generate fallback code when the model API fails.
    
    Args:
        prompt (str): The original prompt
        
    Returns:
        str: Simple fallback code
    """
    # Fallback code generated due to model API failure
    problem_description = prompt.split("Write code to solve the following problem:")[-1].strip()

    return f"""
# Fallback code generated: {reason}
# Problem: {problem_description}

def solve_problem():
    # This is a fallback implementation
    # The model API was unavailable or returned an error
    print("Model API unavailable. This is fallback code.")
    return None

# Call the function
if __name__ == "__main__":
    solve_problem()
"""

# Define the async function to score the generated code
async def score_code(problem_name, code): # Changed to async
    """
    Score the generated code. (async)
    
    Args:
        problem_name (str): The name of the problem
        code (str): The generated code
        
    Returns:
        float: The score of the code
    """
    try:
        async with httpx.AsyncClient() as client: # Changed to httpx.AsyncClient
            response = await client.post( # Changed to await client.post
                f"{get_api_base_url()}/score_code/{problem_name}",
                json={"code": code},
                timeout=15 # Increased timeout
            )
            if response.status_code == 200:
                return response.json()["score"]
    except (httpx.RequestError, ValueError, json.JSONDecodeError) as e: # Changed to httpx.RequestError, added json.JSONDecodeError
        print(f"Warning: Could not score code using API: {e}")
        print("Falling back to basic scoring")
    
    # Fallback to a basic scoring mechanism
    # This is a very simple scoring function that just checks if the code compiles
    try:
        # Try to compile the code to check for syntax errors
        compile(code, "<string>", "exec")
        # If it compiles, give it a basic score
        return 0.5  # Medium score for compilable code
    except Exception:
        # If it doesn't compile, give it a low score
        return 0.1  # Low score for code with syntax errors
    

# Custom crossover operator
def custom_crossover(ind1, ind2):
    """
    Custom crossover operator for the genetic algorithm.
    
    This operator performs a two-point crossover on the continuous parameters
    (temperature, max_tokens, top_p) and a uniform crossover on the discrete
    parameters (model_id, prompt_template, system_prompt).
    
    Args:
        ind1 (list): The first individual
        ind2 (list): The second individual
        
    Returns:
        tuple: A tuple containing the two offspring
    """
    # Create copies of the individuals
    child1 = copy.deepcopy(ind1)
    child2 = copy.deepcopy(ind2)
    
    # Perform uniform crossover on discrete parameters (model_id, prompt_template, system_prompt)
    if random.random() < 0.5:
        child1[0], child2[0] = child2[0], child1[0]  # model_id
    if random.random() < 0.5:
        child1[1], child2[1] = child2[1], child1[1]  # prompt_template
    if random.random() < 0.5:
        child1[5], child2[5] = child2[5], child1[5]  # system_prompt
    
    # Perform two-point crossover on continuous parameters (temperature, max_tokens, top_p)
    # Convert to numpy arrays for easier manipulation
    continuous_params1 = np.array([child1[2], child1[3], child1[4]])
    continuous_params2 = np.array([child2[2], child2[3], child2[4]])
    
    # Perform crossover
    alpha = random.random()
    continuous_params1_new = alpha * continuous_params1 + (1 - alpha) * continuous_params2
    continuous_params2_new = (1 - alpha) * continuous_params1 + alpha * continuous_params2
    
    # Update the individuals, ensuring max_tokens is integer
    child1[2] = float(continuous_params1_new[0])
    child1[3] = int(round(continuous_params1_new[1]))
    child1[4] = float(continuous_params1_new[2])
    child2[2] = float(continuous_params2_new[0])
    child2[3] = int(round(continuous_params2_new[1]))
    child2[4] = float(continuous_params2_new[2])
    
    return child1, child2

# Custom mutation operator
def custom_mutation(individual, indpb=0.2):
    """
    Custom mutation operator for the genetic algorithm.
    
    This operator mutates each parameter with a probability of indpb.
    For continuous parameters, it adds a random value from a normal distribution.
    For discrete parameters, it randomly selects a new value from the available options.
    
    Args:
        individual (list): The individual to mutate
        indpb (float): The probability of each parameter being mutated
        
    Returns:
        list: The mutated individual
    """

    # Get configuration
    config_ga_mutation = default_config.genetic_algorithm

    # Use available models from configuration for mutation
    model_ids_for_mutation = config_ga_mutation.available_models
    if not model_ids_for_mutation: # If empty, use a hardcoded default. Relies on config being populated.
        print("Warning: custom_mutation - available_models in config is empty. Using fallback 'default-model'.")
        model_ids_for_mutation = ["default-model"]

    prompt_templates = config_ga_mutation.prompt_templates
    system_prompts = config_ga_mutation.system_prompts # Corrected: Use config_ga_mutation

    # Get parameter ranges from configuration
    temp_min, temp_max = config_ga_mutation.temperature_range # Corrected: Use config_ga_mutation
    tokens_min, tokens_max = config_ga_mutation.max_tokens_range # Corrected: Use config_ga_mutation
    top_p_min, top_p_max = config_ga_mutation.top_p_range # Corrected: Use config_ga_mutation

    # Mutate model_id
    if random.random() < indpb:
        individual[0] = random.choice(model_ids_for_mutation)

    # Mutate prompt_template
    if random.random() < indpb:
        individual[1] = random.choice(prompt_templates)

    # Mutate temperature (between temp_min and temp_max)
    if random.random() < indpb:
        individual[2] = max(temp_min, min(temp_max, individual[2] + random.gauss(0, 0.2)))

    # Mutate max_tokens (between tokens_min and tokens_max)
    if random.random() < indpb:
        new_val = individual[3] + random.gauss(0, 200)
        individual[3] = int(max(tokens_min, min(tokens_max, new_val)))

    # Mutate top_p (between top_p_min and top_p_max)
    if random.random() < indpb:
        individual[4] = max(top_p_min, min(top_p_max, individual[4] + random.gauss(0, 0.2)))

    # Mutate system_prompt
    if random.random() < indpb:
        individual[5] = random.choice(system_prompts)

    return individual,

# Create the fitness class
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize the toolbox
toolbox = base.Toolbox()

# Define the attributes for the individual

# Get configuration for genetic algorithm
config_ga = default_config.genetic_algorithm

# Initialize model_ids from configuration, with a fallback
# This list is used by attr_model_id and potentially custom_mutation
model_ids_global_init = config_ga.available_models
if not model_ids_global_init:
    print("Warning: genetic_algorithm.py - No available_models in config at module load. Using fallback 'default-model'.")
    model_ids_global_init = ["default-model"]

# Define the attribute generators
def attr_model_id():
    # Use the globally initialized model_ids_global_init
    # Re-access config in case it was updated by main app context, though module-level init might be fixed.
    # For safety, could re-fetch from default_config here, but let's rely on initial load for now.
    if not model_ids_global_init: 
        return "default-model" # Should not be reached if fallback above works
    return random.choice(model_ids_global_init)

# fetch_models_dynamic function is removed as custom_mutation now relies on pre-populated config.

def attr_prompt_template():
    return random.choice(config_ga.prompt_templates) # Use config_ga defined earlier

def attr_temperature():
    temp_min, temp_max = config_ga.temperature_range # Use config_ga
    return random.uniform(temp_min, temp_max)

def attr_max_tokens():
    tokens_min, tokens_max = config_ga.max_tokens_range # Use config_ga
    return random.randint(tokens_min, tokens_max)
    
def attr_top_p():
    top_p_min, top_p_max = config_ga.top_p_range # Use config_ga
    return random.uniform(top_p_min, top_p_max)

def attr_system_prompt():
    return random.choice(config_ga.system_prompts) # Use config_ga
    
# Register the attribute generators
toolbox.register("attr_model_id", attr_model_id)
toolbox.register("attr_prompt_template", attr_prompt_template)
toolbox.register("attr_temperature", attr_temperature)
toolbox.register("attr_max_tokens", attr_max_tokens)
toolbox.register("attr_top_p", attr_top_p)
toolbox.register("attr_system_prompt", attr_system_prompt)

# Define the individual and population
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_model_id, toolbox.attr_prompt_template, 
                  toolbox.attr_temperature, toolbox.attr_max_tokens, 
                  toolbox.attr_top_p, toolbox.attr_system_prompt), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Add property access for all attributes to Individual class
# Add model_id property to Individual class for direct access
def get_model_id(self):
    return self[0]

def set_model_id(self, value):
    self[0] = value

def get_prompt_template(self):
    return self[1]

def set_prompt_template(self, value):
    self[1] = value

def get_temperature(self):
    return self[2]

def set_temperature(self, value):
    self[2] = value

def get_max_tokens(self):
    return self[3]

def set_max_tokens(self, value):
    self[3] = value

def get_top_p(self):
    return self[4]

def set_top_p(self, value):
    self[4] = value

def get_system_prompt(self):
    return self[5]

def set_system_prompt(self, value):
    self[5] = value

# Add the properties to the Individual class
# Add the property to the Individual class
creator.Individual.model_id = property(get_model_id, set_model_id)
creator.Individual.prompt_template = property(get_prompt_template, set_prompt_template)
creator.Individual.temperature = property(get_temperature, set_temperature)
creator.Individual.max_tokens = property(get_max_tokens, set_max_tokens)
creator.Individual.top_p = property(get_top_p, set_top_p)
creator.Individual.system_prompt = property(get_system_prompt, set_system_prompt)

# Define the genetic operators

# Define the async function to fetch problems with lazy loading and fallback
async def fetch_problems(use_api=True): # Changed to async
    """
    Fetch problems with lazy loading and fallback to local file. (async)
    
    Args:
        use_api (bool): Whether to try fetching from API first
        
    Returns:
        list: A list of problem dictionaries
    """
    if not hasattr(fetch_problems, "_problems_cache"):
        fetch_problems._problems_cache = None
    
    # Return cached problems if available
    if fetch_problems._problems_cache is not None:
        return fetch_problems._problems_cache
    
    # Try to fetch from API if requested
    if use_api:
        try:
            async with httpx.AsyncClient() as client: # Changed to httpx.AsyncClient
                response = await client.get(f"{get_api_base_url()}/problems", timeout=10) # Increased timeout
                if response.status_code == 200:
                    fetch_problems._problems_cache = response.json()
                    return fetch_problems._problems_cache
        except (httpx.RequestError, ValueError, json.JSONDecodeError) as e: # Changed to httpx.RequestError, added json.JSONDecodeError
            print(f"Warning: Could not fetch problems from API: {e}")
            print("Falling back to local problems.py file")
    
    # Fallback to loading from problems.py
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("problems", "problems.py")
        problems_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(problems_module)
        fetch_problems._problems_cache = problems_module.problems
        return fetch_problems._problems_cache
    except Exception as e:
        print(f"Error loading problems from problems.py: {e}")
        # Return a minimal default problem if all else fails
        return [{"name": "default_problem", "description": "Write a simple function."}]

# Defer problem loading until needed
async def get_problem_names(): # Made async
    """Get problem names with lazy loading."""
    problems = await fetch_problems(use_api=False)  # await async call, Use local file during initialization
    return [problem["name"] for problem in problems]

async def fitness_wrapper(individual): # Made async
    """Wrapper for the fitness function that selects a random problem."""
    problem_names = await get_problem_names() # await async call
    return await fitness(individual, random.choice(problem_names)) # await async call

# Custom map function for asyncio
async def async_map(func, iterable):
    coroutines = [func(item) for item in iterable]
    results = await asyncio.gather(*coroutines)
    return results

# Register the genetic operators
toolbox.register("map", async_map) # Register custom async map
toolbox.register("evaluate", fitness_wrapper) # fitness_wrapper is now async
toolbox.register("mate", custom_crossover)
toolbox.register("mutate", custom_mutation, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Population management
class PopulationManager:
    """
    Manages the population for the genetic algorithm.
    
    This class handles the evolution of the population, including selection,
    crossover, mutation, and elitism.
    
    Attributes:
        toolbox (base.Toolbox): The DEAP toolbox
        pop_size (int): The size of the population
        elite_size (int): The number of elite individuals to preserve
        cxpb (float): The probability of crossover
        mutpb (float): The probability of mutation
    """
    def __init__(self, toolbox, pop_size=50, elite_size=5, cxpb=0.5, mutpb=0.2):
        self.toolbox = toolbox
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.population = None
        self.hall_of_fame = tools.HallOfFame(elite_size)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.stats.register("std", np.std)
        self.logbook = tools.Logbook()
    
    def initialize(self):
        """Initialize the population."""
        self.population = self.toolbox.population(n=self.pop_size)
        return self.population
    
    async def evaluate_population(self): # Made async
        """Evaluate the fitness of the population."""
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        # Use await with toolbox.map since it's now our async_map
        fitnesses = await self.toolbox.map(self.toolbox.evaluate, invalid_ind) # await async map
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the best individuals
        self.hall_of_fame.update(self.population)
        
        # Record the statistics
        record = self.stats.compile(self.population)
        self.logbook.record(gen=len(self.logbook), **record)
        
        return self.population
    
    async def evolve_population(self): # Made async
        """Evolve the population to the next generation."""
        # Select the next generation individuals
        offspring = self.toolbox.select(self.population, len(self.population) - self.elite_size)
        
        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if i < len(offspring) - 1 and random.random() < self.cxpb:
                offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values
        
        for i in range(len(offspring)):
            if random.random() < self.mutpb:
                offspring[i], = self.toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
        
        # Add the elite individuals to the offspring
        elite = list(map(self.toolbox.clone, self.hall_of_fame))
        
        # Replace the population with the offspring and elite individuals
        self.population = offspring + elite
        
        return self.population
    
    def get_best_individual(self):
        """Get the best individual from the hall of fame."""
        if len(self.hall_of_fame) > 0:
            return self.hall_of_fame[0]
        return None
    
    def get_statistics(self):
        """Get the statistics of the evolution."""
        return self.logbook

# Initialize the population and evolve (now async)

async def evolve(pop_size=None, ngen=None, elite_size=None, cxpb=None, mutpb=None): # Made async
    """
    Evolve the population for a given number of generations. (async)
    
    Args:
        pop_size (int): The size of the population
        ngen (int): The number of generations
        elite_size (int): The number of elite individuals to preserve
        cxpb (float): The probability of crossover
        mutpb (float): The probability of mutation
        
    Returns:
        tuple: A tuple containing the final population, the logbook, and the best individual
    """
    # Get configuration
    config = default_config.genetic_algorithm

    # Use provided parameters or defaults from configuration
    pop_size = pop_size or config.population_size
    ngen = ngen or config.num_generations
    elite_size = elite_size or config.elite_size
    cxpb = cxpb or config.crossover_probability
    mutpb = mutpb or config.mutation_probability

    # Initialize the population manager
    pop_manager = PopulationManager(
        toolbox=toolbox,
        pop_size=pop_size,
        elite_size=elite_size,
        cxpb=cxpb,
        mutpb=mutpb
    )
    

    # Initialize the population
    pop_manager.initialize()
    
    # Evaluate the initial population
    await pop_manager.evaluate_population() # await async call
    
    # Print the initial statistics
    print(f"Initial population statistics: {pop_manager.logbook[-1]}")
    
    # Evolve the population for ngen generations
    for gen in range(ngen):
        # Evolve the population
        await pop_manager.evolve_population() # await async call (evolve_population itself calls evaluate_population)
        
        # Evaluate the new population (already done by evolve_population if it calls evaluate_population)
        # If evolve_population doesn't call evaluate_population internally after mutation/crossover, then:
        await pop_manager.evaluate_population() # await async call
        
        # Print the statistics
        print(f"Generation {gen+1} statistics: {pop_manager.logbook[-1]}")
    
    # Get the best individual
    best_ind = pop_manager.get_best_individual()
    
    # Print the best individual
    if best_ind is not None:
        chromosome = Chromosome.from_list(best_ind)
        print(f"Best individual: {chromosome}")
        print(f"Fitness: {best_ind.fitness.values[0]}")
    
    return pop_manager.population, pop_manager.logbook, best_ind

if __name__ == "__main__":
    # Set the random seed for reproducibility
    random.seed(42)

    async def main_async(): # Create an async main for asyncio.run
        # Evolve the population
        pop, log, best = await evolve(pop_size=50, ngen=10) # await async call
        
        # Print the best individual
        if best is not None:
            chromosome = Chromosome.from_list(best)
            print(f"Best individual: {chromosome}")
            print(f"Fitness: {best.fitness.values[0]}")
            
            # Get problem names with lazy loading
            problem_names = await get_problem_names() # await async call
            
            # Generate code using the best individual
            problem_name = random.choice(problem_names)
            prompt = await generate_prompt(problem_name, chromosome.prompt_template) # await async call
            code = await generate_code( # await async call
                chromosome.model_id, 
                prompt, 
                chromosome.temperature, 
                chromosome.max_tokens, 
                chromosome.top_p,
                chromosome.system_prompt
            )
            
            # Score the generated code
            score = await score_code(problem_name, code) # await async call
            
            print(f"Problem: {problem_name}")
            print(f"Score: {score}")
            print(f"Code:\n{code}")

    asyncio.run(main_async()) # Run the async main
