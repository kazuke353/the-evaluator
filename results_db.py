
import sqlite3
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("results_db")

@dataclass
class SimulationResult:
    """Represents a simulation result."""
    id: str
    problem_name: str
    model_id: str
    prompt_template: str
    temperature: float
    max_tokens: int
    top_p: float
    system_prompt: str
    language: str
    code: str
    test_score: float
    quality_score: float
    final_score: float
    generation: int
    timestamp: float
    
    @classmethod
    def from_dict(cls, data):
        """Create a result from a dictionary."""
        return cls(**data)
    
    def to_dict(self):
        """Convert the result to a dictionary."""
        return {
            "id": self.id,
            "problem_name": self.problem_name,
            "model_id": self.model_id,
            "prompt_template": self.prompt_template,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "system_prompt": self.system_prompt,
            "language": self.language,
            "code": self.code,
            "test_score": self.test_score,
            "quality_score": self.quality_score,
            "final_score": self.final_score,
            "generation": self.generation,
            "timestamp": self.timestamp
        }

class ResultsDB:
    """A database for storing simulation results."""
    
    def __init__(self, db_path="results.db"):
        """Initialize the database."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database connection and tables."""
        try:
            # Create the database directory if it doesn't exist
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Connect to the database
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            
            # Create the results table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    problem_name TEXT,
                    model_id TEXT,
                    prompt_template TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    top_p REAL,
                    system_prompt TEXT,
                    language TEXT,
                    code TEXT,
                    test_score REAL,
                    quality_score REAL,
                    final_score REAL,
                    generation INTEGER,
                    timestamp REAL
                )
            ''')
            
            # Create the generations table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem_name TEXT,
                    generation INTEGER,
                    avg_score REAL,
                    max_score REAL,
                    min_score REAL,
                    std_score REAL,
                    timestamp REAL,
                    stats TEXT
                )
            ''')
            
            # Create indices for faster queries
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_problem ON results (problem_name)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_generation ON results (generation)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_model ON results (model_id)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_generations_problem ON generations (problem_name)')
            
            self.conn.commit()
            logger.info(f"Initialized database at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
    
    def add_result(self, result: SimulationResult) -> bool:
        """Add a result to the database."""
        try:
            self.cursor.execute('''
                INSERT INTO results (
                    id, problem_name, model_id, prompt_template, temperature,
                    max_tokens, top_p, system_prompt, language, code,
                    test_score, quality_score, final_score, generation, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.id, result.problem_name, result.model_id, result.prompt_template, result.temperature,
                result.max_tokens, result.top_p, result.system_prompt, result.language, result.code,
                result.test_score, result.quality_score, result.final_score, result.generation, result.timestamp
            ))
            self.conn.commit()
            logger.info(f"Added result {result.id} to database")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding result to database: {e}")
            return False
    
    def add_generation_stats(self, problem_name: str, generation: int, stats: Dict[str, Any]) -> bool:
        """Add generation statistics to the database."""
        try:
            self.cursor.execute('''
                INSERT INTO generations (
                    problem_name, generation, avg_score, max_score, min_score, std_score, timestamp, stats
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                problem_name, generation, stats.get("avg", 0), stats.get("max", 0),
                stats.get("min", 0), stats.get("std", 0), time.time(), json.dumps(stats)
            ))
            self.conn.commit()
            logger.info(f"Added generation {generation} stats for problem {problem_name} to database")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding generation stats to database: {e}")
            return False
    
    def get_result(self, result_id: str) -> Optional[SimulationResult]:
        """Get a result by ID."""
        try:
            self.cursor.execute('SELECT * FROM results WHERE id = ?', (result_id,))
            row = self.cursor.fetchone()
            if row:
                return SimulationResult.from_dict(dict(row))
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting result from database: {e}")
            return None
    
    def get_results_by_problem(self, problem_name: str) -> List[SimulationResult]:
        """Get all results for a problem."""
        try:
            self.cursor.execute('SELECT * FROM results WHERE problem_name = ? ORDER BY generation, final_score DESC', (problem_name,))
            rows = self.cursor.fetchall()
            return [SimulationResult.from_dict(dict(row)) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error getting results from database: {e}")
            return []
    
    def get_results_by_generation(self, problem_name: str, generation: int) -> List[SimulationResult]:
        """Get all results for a generation of a problem."""
        try:
            self.cursor.execute(
                'SELECT * FROM results WHERE problem_name = ? AND generation = ? ORDER BY final_score DESC',
                (problem_name, generation)
            )
            rows = self.cursor.fetchall()
            return [SimulationResult.from_dict(dict(row)) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error getting results from database: {e}")
            return []
    
    def get_best_result(self, problem_name: str) -> Optional[SimulationResult]:
        """Get the best result for a problem."""
        try:
            self.cursor.execute(
                'SELECT * FROM results WHERE problem_name = ? ORDER BY final_score DESC LIMIT 1',
                (problem_name,)
            )
            row = self.cursor.fetchone()
            if row:
                return SimulationResult.from_dict(dict(row))
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting best result from database: {e}")
            return None
    
    def get_best_result_by_generation(self, problem_name: str, generation: int) -> Optional[SimulationResult]:
        """Get the best result for a generation of a problem."""
        try:
            self.cursor.execute(
                'SELECT * FROM results WHERE problem_name = ? AND generation = ? ORDER BY final_score DESC LIMIT 1',
                (problem_name, generation)
            )
            row = self.cursor.fetchone()
            if row:
                return SimulationResult.from_dict(dict(row))
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting best result from database: {e}")
            return None
    
    def get_generation_stats(self, problem_name: str, generation: int) -> Optional[Dict[str, Any]]:
        """Get statistics for a generation of a problem."""
        try:
            self.cursor.execute(
                'SELECT * FROM generations WHERE problem_name = ? AND generation = ?',
                (problem_name, generation)
            )
            row = self.cursor.fetchone()
            if row:
                result = dict(row)
                result["stats"] = json.loads(result["stats"])
                return result
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting generation stats from database: {e}")
            return None
    
    def get_all_generation_stats(self, problem_name: str) -> List[Dict[str, Any]]:
        """Get statistics for all generations of a problem."""
        try:
            self.cursor.execute(
                'SELECT * FROM generations WHERE problem_name = ? ORDER BY generation',
                (problem_name,)
            )
            rows = self.cursor.fetchall()
            results = []
            for row in rows:
                result = dict(row)
                result["stats"] = json.loads(result["stats"])
                results.append(result)
            return results
        except sqlite3.Error as e:
            logger.error(f"Error getting generation stats from database: {e}")
            return []
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get performance statistics for a model across all problems."""
        try:
            self.cursor.execute(
                '''
                SELECT 
                    problem_name,
                    COUNT(*) as count,
                    AVG(final_score) as avg_score,
                    MAX(final_score) as max_score,
                    MIN(final_score) as min_score
                FROM results 
                WHERE model_id = ? 
                GROUP BY problem_name
                ''',
                (model_id,)
            )
            rows = self.cursor.fetchall()
            return {row["problem_name"]: {
                "count": row["count"],
                "avg_score": row["avg_score"],
                "max_score": row["max_score"],
                "min_score": row["min_score"]
            } for row in rows}
        except sqlite3.Error as e:
            logger.error(f"Error getting model performance from database: {e}")
            return {}
    
    def get_problem_statistics(self, problem_name: str) -> Dict[str, Any]:
        """Get statistics for a problem across all generations."""
        try:
            # Get overall statistics
            self.cursor.execute(
                '''
                SELECT 
                    COUNT(*) as count,
                    AVG(final_score) as avg_score,
                    MAX(final_score) as max_score,
                    MIN(final_score) as min_score
                FROM results 
                WHERE problem_name = ?
                ''',
                (problem_name,)
            )
            overall = dict(self.cursor.fetchone())
            
            # Get best model
            self.cursor.execute(
                '''
                SELECT 
                    model_id,
                    AVG(final_score) as avg_score,
                    COUNT(*) as count
                FROM results 
                WHERE problem_name = ? 
                GROUP BY model_id
                ORDER BY avg_score DESC
                LIMIT 1
                ''',
                (problem_name,)
            )
            best_model_row = self.cursor.fetchone()
            best_model = dict(best_model_row) if best_model_row else None
            
            # Get generation progression
            self.cursor.execute(
                '''
                SELECT 
                    generation,
                    AVG(final_score) as avg_score,
                    MAX(final_score) as max_score,
                    MIN(final_score) as min_score
                FROM results 
                WHERE problem_name = ? 
                GROUP BY generation
                ORDER BY generation
                ''',
                (problem_name,)
            )
            generations = [dict(row) for row in self.cursor.fetchall()]
            
            return {
                "overall": overall,
                "best_model": best_model,
                "generations": generations
            }
        except sqlite3.Error as e:
            logger.error(f"Error getting problem statistics from database: {e}")
            return {}

def get_all_problems() -> List[str]:
    """
    Get a list of all problems in the database.
    
    Returns:
        A list of problem names
    """
    try:
        with ResultsDB() as db:
            db.cursor.execute('SELECT DISTINCT problem_name FROM results')
            rows = db.cursor.fetchall()
            return [row["problem_name"] for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error getting all problems from database: {e}")
        return []

def get_problem_summary() -> List[Dict[str, Any]]:
    """
    Get a summary of all problems in the database.
    
    Returns:
        A list of dictionaries containing problem summaries
    """
    try:
        with ResultsDB() as db:
            db.cursor.execute('''
                SELECT 
                    problem_name,
                    COUNT(*) as count,
                    AVG(final_score) as avg_score,
                    MAX(final_score) as max_score,
                    MIN(final_score) as min_score,
                    MAX(generation) as max_generation
                FROM results 
                GROUP BY problem_name
            ''')
            rows = db.cursor.fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error getting problem summary from database: {e}")
        return []

def get_model_summary() -> List[Dict[str, Any]]:
    """
    Get a summary of all models in the database.
    
    Returns:
        A list of dictionaries containing model summaries
    """
    try:
        with ResultsDB() as db:
            db.cursor.execute('''
                SELECT 
                    model_id,
                    COUNT(*) as count,
                    AVG(final_score) as avg_score,
                    MAX(final_score) as max_score,
                    MIN(final_score) as min_score,
                    COUNT(DISTINCT problem_name) as num_problems
                FROM results 
                GROUP BY model_id
            ''')
            rows = db.cursor.fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error getting model summary from database: {e}")
        return []

def get_recent_simulations(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get a list of recent simulations.
    
    Args:
        limit: The maximum number of simulations to return
        
    Returns:
        A list of dictionaries containing simulation data
    """
    try:
        with ResultsDB() as db:
            db.cursor.execute('''
                SELECT 
                    problem_name,
                    model_id,
                    MAX(generation) as max_generation,
                    COUNT(*) as count,
                    AVG(final_score) as avg_score,
                    MAX(final_score) as max_score,
                    MAX(timestamp) as last_updated
                FROM results 
                GROUP BY problem_name, model_id
                ORDER BY last_updated DESC
                LIMIT ?
            ''', (limit,))
            rows = db.cursor.fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error getting recent simulations from database: {e}")
        return []

def get_problem_summary() -> List[Dict[str, Any]]:
    """
    Get a summary of all problems in the database.
    
    Returns:
        A list of dictionaries containing problem summaries
    """
    try:
        with ResultsDB() as db:
            db.cursor.execute('''
                SELECT 
                    problem_name,
                    COUNT(*) as count,
                    AVG(final_score) as avg_score,
                    MAX(final_score) as max_score,
                    MIN(final_score) as min_score,
                    MAX(generation) as max_generation
                FROM results 
                GROUP BY problem_name
            ''')
            rows = db.cursor.fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error getting problem summary from database: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    with ResultsDB() as db:
        # Add a sample result
        result = SimulationResult(
            id="test-result-1",
            problem_name="fibonacci",
            model_id="model1",
            prompt_template="template1",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            system_prompt="You are a helpful coding assistant.",
            language="python",
            code="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            test_score=80.0,
            quality_score=70.0,
            final_score=77.0,
            generation=1,
            timestamp=time.time()
        )
        db.add_result(result)
        
        # Add generation stats
        db.add_generation_stats("fibonacci", 1, {
            "avg": 75.0,
            "max": 90.0,
            "min": 60.0,
            "std": 10.0,
            "population_size": 10
        })
        
        # Retrieve the result
        retrieved = db.get_result("test-result-1")
        if retrieved:
            print(f"Retrieved result: {retrieved.to_dict()}")
        
        # Get generation stats
        stats = db.get_generation_stats("fibonacci", 1)
        if stats:
            print(f"Generation stats: {stats}")
