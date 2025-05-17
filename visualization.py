import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
from results_db import ResultsDB, SimulationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("visualization")

class ResultsVisualizer:
    """A class for visualizing simulation results."""
    
    def __init__(self, db_path="results.db", output_dir="visualizations"):
        """Initialize the visualizer."""
        self.db = ResultsDB(db_path)
        self.output_dir = output_dir
        
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def close(self):
        """Close the database connection."""
        self.db.close()
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
    
    def _save_figure(self, fig, filename):
        """Save a figure to the output directory."""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved figure to {filepath}")
        return filepath
    
    def plot_generation_progression(self, problem_name: str) -> Optional[str]:
        """
        Plot the progression of scores across generations for a problem.
        
        Args:
            problem_name: The name of the problem
            
        Returns:
            The path to the saved figure, or None if an error occurred
        """
        try:
            # Get generation stats
            stats = self.db.get_all_generation_stats(problem_name)
            if not stats:
                logger.warning(f"No generation stats found for problem {problem_name}")
                return None
            
            # Extract data
            generations = [stat["generation"] for stat in stats]
            avg_scores = [stat["avg_score"] for stat in stats]
            max_scores = [stat["max_score"] for stat in stats]
            min_scores = [stat["min_score"] for stat in stats]
            
            # Create the figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the data
            ax.plot(generations, avg_scores, 'o-', label='Average Score', color='blue')
            ax.plot(generations, max_scores, 'o-', label='Max Score', color='green')
            ax.plot(generations, min_scores, 'o-', label='Min Score', color='red')
            
            # Fill the area between min and max
            ax.fill_between(generations, min_scores, max_scores, alpha=0.2, color='gray')
            
            # Set labels and title
            ax.set_xlabel('Generation')
            ax.set_ylabel('Score')
            ax.set_title(f'Score Progression Across Generations for {problem_name}')
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Save the figure
            filename = f"{problem_name}_generation_progression.png"
            return self._save_figure(fig, filename)
        
        except Exception as e:
            logger.error(f"Error plotting generation progression: {e}")
            return None
    
    def plot_model_comparison(self, problem_name: str) -> Optional[str]:
        """
        Plot a comparison of model performance for a problem.
        
        Args:
            problem_name: The name of the problem
            
        Returns:
            The path to the saved figure, or None if an error occurred
        """
        try:
            # Get results for the problem
            results = self.db.get_results_by_problem(problem_name)
            if not results:
                logger.warning(f"No results found for problem {problem_name}")
                return None
            
            # Group results by model
            model_scores = {}
            for result in results:
                if result.model_id not in model_scores:
                    model_scores[result.model_id] = []
                model_scores[result.model_id].append(result.final_score)
            
            # Calculate statistics for each model
            models = []
            avg_scores = []
            std_scores = []
            
            for model_id, scores in model_scores.items():
                if len(scores) < 2:
                    continue  # Skip models with too few data points
                
                models.append(model_id)
                avg_scores.append(np.mean(scores))
                std_scores.append(np.std(scores))
            
            if not models:
                logger.warning(f"Not enough data for model comparison for problem {problem_name}")
                return None
            
            # Sort by average score
            sorted_indices = np.argsort(avg_scores)[::-1]  # Descending order
            models = [models[i] for i in sorted_indices]
            avg_scores = [avg_scores[i] for i in sorted_indices]
            std_scores = [std_scores[i] for i in sorted_indices]
            
            # Create the figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot the data
            x = np.arange(len(models))
            width = 0.6
            
            bars = ax.bar(x, avg_scores, width, yerr=std_scores, align='center',
                         alpha=0.7, ecolor='black', capsize=10)
            
            # Set labels and title
            ax.set_xlabel('Model')
            ax.set_ylabel('Average Score')
            ax.set_title(f'Model Performance Comparison for {problem_name}')
            
            # Set x-axis ticks
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Add grid and value labels
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add value labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{avg_scores[i]:.1f}',
                        ha='center', va='bottom', rotation=0)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            filename = f"{problem_name}_model_comparison.png"
            return self._save_figure(fig, filename)
        
        except Exception as e:
            logger.error(f"Error plotting model comparison: {e}")
            return None
    
    def plot_parameter_influence(self, problem_name: str, parameter: str) -> Optional[str]:
        """
        Plot the influence of a parameter on the score.
        
        Args:
            problem_name: The name of the problem
            parameter: The parameter to analyze (temperature, max_tokens, top_p)
            
        Returns:
            The path to the saved figure, or None if an error occurred
        """
        try:
            # Get results for the problem
            results = self.db.get_results_by_problem(problem_name)
            if not results:
                logger.warning(f"No results found for problem {problem_name}")
                return None
            
            # Extract parameter values and scores
            param_values = []
            scores = []
            
            for result in results:
                if parameter == "temperature":
                    param_values.append(result.temperature)
                elif parameter == "max_tokens":
                    param_values.append(result.max_tokens)
                elif parameter == "top_p":
                    param_values.append(result.top_p)
                else:
                    logger.warning(f"Unknown parameter: {parameter}")
                    return None
                
                scores.append(result.final_score)
            
            # Create the figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the data as a scatter plot
            scatter = ax.scatter(param_values, scores, alpha=0.7, c=scores, cmap='viridis')
            
            # Add a trend line
            if len(param_values) > 1:
                z = np.polyfit(param_values, scores, 1)
                p = np.poly1d(z)
                ax.plot(sorted(param_values), p(sorted(param_values)), "r--", alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel(parameter.capitalize())
            ax.set_ylabel('Score')
            ax.set_title(f'Influence of {parameter.capitalize()} on Score for {problem_name}')
            
            # Add grid and colorbar
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Score')
            
            # Save the figure
            filename = f"{problem_name}_{parameter}_influence.png"
            return self._save_figure(fig, filename)
        
        except Exception as e:
            logger.error(f"Error plotting parameter influence: {e}")
            return None
    
    def plot_score_distribution(self, problem_name: str, generation: Optional[int] = None) -> Optional[str]:
        """
        Plot the distribution of scores for a problem.
        
        Args:
            problem_name: The name of the problem
            generation: The generation to plot, or None for all generations
            
        Returns:
            The path to the saved figure, or None if an error occurred
        """
        try:
            # Get results for the problem
            if generation is not None:
                results = self.db.get_results_by_generation(problem_name, generation)
                title_suffix = f" (Generation {generation})"
                filename_prefix = f"{problem_name}_gen{generation}"
            else:
                results = self.db.get_results_by_problem(problem_name)
                title_suffix = ""
                filename_prefix = problem_name
            
            if not results:
                logger.warning(f"No results found for problem {problem_name}{title_suffix}")
                return None
            
            # Extract scores
            test_scores = [result.test_score for result in results]
            quality_scores = [result.quality_score for result in results]
            final_scores = [result.final_score for result in results]
            
            # Create the figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the data as histograms
            bins = np.linspace(0, 100, 21)  # 0 to 100 in steps of 5
            
            ax.hist(test_scores, bins=bins, alpha=0.5, label='Test Score', color='blue')
            ax.hist(quality_scores, bins=bins, alpha=0.5, label='Quality Score', color='green')
            ax.hist(final_scores, bins=bins, alpha=0.5, label='Final Score', color='red')
            
            # Set labels and title
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Score Distribution for {problem_name}{title_suffix}')
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Save the figure
            filename = f"{filename_prefix}_score_distribution.png"
            return self._save_figure(fig, filename)
        
        except Exception as e:
            logger.error(f"Error plotting score distribution: {e}")
            return None
    
    def plot_best_solutions_over_time(self, problem_name: str) -> Optional[str]:
        """
        Plot the best solution score over time.
        
        Args:
            problem_name: The name of the problem
            
        Returns:
            The path to the saved figure, or None if an error occurred
        """
        try:
            # Get results for the problem
            results = self.db.get_results_by_problem(problem_name)
            if not results:
                logger.warning(f"No results found for problem {problem_name}")
                return None
            
            # Sort results by timestamp
            results.sort(key=lambda r: r.timestamp)
            
            # Track the best score over time
            timestamps = []
            best_scores = []
            best_score = 0
            
            for result in results:
                if result.final_score > best_score:
                    best_score = result.final_score
                    timestamps.append(result.timestamp)
                    best_scores.append(best_score)
            
            if not timestamps:
                logger.warning(f"No improvement in scores for problem {problem_name}")
                return None
            
            # Convert timestamps to relative time in hours
            start_time = timestamps[0]
            rel_times = [(t - start_time) / 3600 for t in timestamps]  # Convert to hours
            
            # Create the figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the data
            ax.plot(rel_times, best_scores, 'o-', color='blue')
            
            # Set labels and title
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Best Score')
            ax.set_title(f'Best Solution Score Over Time for {problem_name}')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save the figure
            filename = f"{problem_name}_best_over_time.png"
            return self._save_figure(fig, filename)
        
        except Exception as e:
            logger.error(f"Error plotting best solutions over time: {e}")
            return None
    
    def generate_report(self, problem_name: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report for a problem.
        
        Args:
            problem_name: The name of the problem
            
        Returns:
            A dictionary containing the report data and paths to generated figures
        """
        report = {
            "problem_name": problem_name,
            "timestamp": time.time(),
            "figures": {},
            "statistics": None,
            "best_solution": None
        }
        
        # Generate figures
        report["figures"]["generation_progression"] = self.plot_generation_progression(problem_name)
        report["figures"]["model_comparison"] = self.plot_model_comparison(problem_name)
        report["figures"]["temperature_influence"] = self.plot_parameter_influence(problem_name, "temperature")
        report["figures"]["top_p_influence"] = self.plot_parameter_influence(problem_name, "top_p")
        report["figures"]["score_distribution"] = self.plot_score_distribution(problem_name)
        report["figures"]["best_over_time"] = self.plot_best_solutions_over_time(problem_name)
        
        # Get problem statistics
        report["statistics"] = self.db.get_problem_statistics(problem_name)
        
        # Get the best solution
        best_result = self.db.get_best_result(problem_name)
        if best_result:
            report["best_solution"] = best_result.to_dict()
        
        # Save the report as JSON
        report_path = os.path.join(self.output_dir, f"{problem_name}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated report for problem {problem_name} at {report_path}")
        
        return report
    
    def generate_html_report(self, problem_name: str) -> Optional[str]:
        """
        Generate an HTML report for a problem.
        
        Args:
            problem_name: The name of the problem
            
        Returns:
            The path to the HTML report, or None if an error occurred
        """
        try:
            # Generate the report data
            report = self.generate_report(problem_name)
            
            # Create the HTML content
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Benchmark Report: {problem_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .section {{ margin-bottom: 30px; }}
                    .figure {{ margin-bottom: 20px; text-align: center; }}
                    .figure img {{ max-width: 100%; border: 1px solid #ddd; }}
                    .stats {{ margin-bottom: 20px; }}
                    .stats table {{ width: 100%; border-collapse: collapse; }}
                    .stats th, .stats td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .stats th {{ background-color: #f2f2f2; }}
                    .code {{ font-family: monospace; white-space: pre-wrap; background-color: #f5f5f5; 
                             padding: 15px; border: 1px solid #ddd; overflow-x: auto; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Benchmark Report: {problem_name}</h1>
                    <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report["timestamp"]))}</p>
                    
                    <div class="section">
                        <h2>Performance Overview</h2>
            """
            
            # Add statistics if available
            if report["statistics"]:
                stats = report["statistics"]
                html += f"""
                        <div class="stats">
                            <h3>Overall Statistics</h3>
                            <table>
                                <tr>
                                    <th>Total Evaluations</th>
                                    <th>Average Score</th>
                                    <th>Max Score</th>
                                    <th>Min Score</th>
                                </tr>
                                <tr>
                                    <td>{stats["overall"]["count"]}</td>
                                    <td>{f'{stats["overall"]["avg_score"]:.2f}' if stats["overall"]["avg_score"] is not None else 'N/A'}</td>
                                    <td>{f'{stats["overall"]["max_score"]:.2f}' if stats["overall"]["max_score"] is not None else 'N/A'}</td>
                                    <td>{f'{stats["overall"]["min_score"]:.2f}' if stats["overall"]["min_score"] is not None else 'N/A'}</td>
                                </tr>
                            </table>
                        </div>
                """
                
                if stats["best_model"]:
                    html += f"""
                        <div class="stats">
                            <h3>Best Performing Model</h3>
                            <table>
                                <tr>
                                    <th>Model ID</th>
                                    <th>Average Score</th>
                                    <th>Number of Evaluations</th>
                                </tr>
                                <tr>
                                    <td>{stats["best_model"]["model_id"]}</td>
                                    <td>{f'{stats["best_model"]["avg_score"]:.2f}' if stats["best_model"]["avg_score"] is not None else 'N/A'}</td>
                                    <td>{stats["best_model"]["count"]}</td>
                                </tr>
                            </table>
                        </div>
                    """
            
            # Add figures
            html += """
                    </div>
                    
                    <div class="section">
                        <h2>Performance Visualizations</h2>
            """
            
            for name, path in report["figures"].items():
                if path:
                    # Convert the path to a relative path for the HTML
                    rel_path = os.path.basename(path)
                    html += f"""
                        <div class="figure">
                            <h3>{name.replace('_', ' ').title()}</h3>
                            <img src="{rel_path}" alt="{name}">
                        </div>
                    """
            
            # Add best solution if available
            if report["best_solution"]:
                best = report["best_solution"]
                html += f"""
                    <div class="section">
                        <h2>Best Solution</h2>
                        <div class="stats">
                            <table>
                                <tr>
                                    <th>Model ID</th>
                                    <th>Final Score</th>
                                    <th>Test Score</th>
                                    <th>Quality Score</th>
                                    <th>Generation</th>
                                </tr>
                                <tr>
                                    <td>{best["model_id"]}</td>
                                    <td>{f'{best["final_score"]:.2f}' if best["final_score"] is not None else 'N/A'}</td>
                                    <td>{f'{best["test_score"]:.2f}' if best["test_score"] is not None else 'N/A'}</td>
                                    <td>{f'{best["quality_score"]:.2f}' if best["quality_score"] is not None else 'N/A'}</td>
                                    <td>{best["generation"]}</td>
                                </tr>
                            </table>
                        </div>
                        
                        <h3>Parameters</h3>
                        <div class="stats">
                            <table>
                                <tr>
                                    <th>Temperature</th>
                                    <th>Max Tokens</th>
                                    <th>Top P</th>
                                </tr>
                                <tr>
                                    <td>{f'{best["temperature"]:.2f}' if best["temperature"] is not None else 'N/A'}</td>
                                    <td>{best["max_tokens"]}</td>
                                    <td>{f'{best["top_p"]:.2f}' if best["top_p"] is not None else 'N/A'}</td>
                                </tr>
                            </table>
                        </div>
                        
                        <h3>Generated Code</h3>
                        <div class="code">{best["code"]}</div>
                    </div>
                """
            
            # Close the HTML
            html += """
                </div>
            </body>
            </html>
            """
            
            # Save the HTML report
            html_path = os.path.join(self.output_dir, f"{problem_name}_report.html")
            with open(html_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Generated HTML report for problem {problem_name} at {html_path}")
            
            return html_path
        
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return None


def track_evolution_progress(db: ResultsDB, problem_name: str, generation: int, 
                            population: List[Any], fitness_values: List[float]) -> None:
    """
    Track the progress of the genetic algorithm evolution.
    
    Args:
        db: The results database
        problem_name: The name of the problem
        generation: The current generation
        population: The population of chromosomes
        fitness_values: The fitness values of the population
    """
    # Calculate statistics
    stats = {
        "avg": float(np.mean(fitness_values)),
        "max": float(np.max(fitness_values)),
        "min": float(np.min(fitness_values)),
        "std": float(np.std(fitness_values)),
        "median": float(np.median(fitness_values)),
        "population_size": len(population)
    }
    
    # Add generation statistics to the database
    db.add_generation_stats(problem_name, generation, stats)
    
    # Log the progress
    logger.info(f"Generation {generation} stats: avg={stats['avg']:.2f}, max={stats['max']:.2f}, min={stats['min']:.2f}")


def save_simulation_result(db: ResultsDB, job_id: str, problem_name: str, model_id: str,
                          prompt_template: str, temperature: float, max_tokens: int,
                          top_p: float, system_prompt: str, language: str, code: str,
                          evaluation: Dict[str, Any], generation: int) -> None:
    """
    Save a simulation result to the database.
    
    Args:
        db: The results database
        job_id: The ID of the simulation job
        problem_name: The name of the problem
        model_id: The ID of the model
        prompt_template: The prompt template used
        temperature: The temperature parameter
        max_tokens: The max tokens parameter
        top_p: The top_p parameter
        system_prompt: The system prompt used
        language: The programming language
        code: The generated code
        evaluation: The evaluation results
        generation: The generation number
    """
    # Create a simulation result
    result = SimulationResult(
        id=job_id,
        problem_name=problem_name,
        model_id=model_id,
        prompt_template=prompt_template,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        system_prompt=system_prompt,
        language=language,
        code=code,
        test_score=evaluation.get("test_score", 0),
        quality_score=evaluation.get("quality_score", 0),
        final_score=evaluation.get("final_score", 0),
        generation=generation,
        timestamp=time.time()
    )
    
    # Add the result to the database
    db.add_result(result)
    
    logger.info(f"Saved simulation result {job_id} to database")



def generate_streamlit_figures(problem_name: str) -> Dict[str, Any]:
    """
    Generate figures for Streamlit display.
    
    Args:
        problem_name: The name of the problem
        
    Returns:
        A dictionary containing the generated figures
    """
    figures = {}
    
    try:
        # Create a visualizer
        with ResultsVisualizer() as visualizer:
            # Generate figures
            figures["generation_progression"] = visualizer.plot_generation_progression(problem_name)
            figures["model_comparison"] = visualizer.plot_model_comparison(problem_name)
            figures["temperature_influence"] = visualizer.plot_parameter_influence(problem_name, "temperature")
            figures["top_p_influence"] = visualizer.plot_parameter_influence(problem_name, "top_p")
            figures["score_distribution"] = visualizer.plot_score_distribution(problem_name)
            figures["best_over_time"] = visualizer.plot_best_solutions_over_time(problem_name)
            
            # Get problem statistics
            figures["statistics"] = visualizer.db.get_problem_statistics(problem_name)
            
            # Get the best solution
            best_result = visualizer.db.get_best_result(problem_name)
            if best_result:
                figures["best_solution"] = best_result.to_dict()
    
    except Exception as e:
        logger.error(f"Error generating Streamlit figures: {e}")
    
    return figures

def get_streamlit_generation_data(problem_name: str) -> List[Dict[str, Any]]:
    """
    Get generation data for Streamlit display.
    
    Args:
        problem_name: The name of the problem
        
    Returns:
        A list of dictionaries containing generation data
    """
    try:
        with ResultsDB() as db:
            return db.get_all_generation_stats(problem_name)
    except Exception as e:
        logger.error(f"Error getting generation data for Streamlit: {e}")
        return []

def get_streamlit_results_data(problem_name: str) -> List[Dict[str, Any]]:
    """
    Get results data for Streamlit display.
    
    Args:
        problem_name: The name of the problem
        
    Returns:
        A list of dictionaries containing results data
    """
    try:
        with ResultsDB() as db:
            results = db.get_results_by_problem(problem_name)
            return [result.to_dict() for result in results]
    except Exception as e:
        logger.error(f"Error getting results data for Streamlit: {e}")
        return []

def get_streamlit_best_result(problem_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the best result for Streamlit display.
    
    Args:
        problem_name: The name of the problem
        
    Returns:
        A dictionary containing the best result, or None if no results are available
    """
    try:
        with ResultsDB() as db:
            best_result = db.get_best_result(problem_name)
            if best_result:
                return best_result.to_dict()
            return None
    except Exception as e:
        logger.error(f"Error getting best result for Streamlit: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    with ResultsVisualizer() as visualizer:
        # Generate a report for a problem
        html_path = visualizer.generate_html_report("fibonacci")
        if html_path:
            print(f"Generated HTML report at {html_path}")
