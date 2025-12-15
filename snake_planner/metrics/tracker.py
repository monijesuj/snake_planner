"""
Metrics tracking and comparison for planning algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
import time

from ..planners.base_planner import PlanningResult


@dataclass
class GameMetrics:
    """Metrics for a single game run."""
    algorithm: str
    score: int
    snake_length: int
    total_plans: int
    successful_plans: int
    failed_plans: int
    total_path_length: int
    total_nodes_expanded: int
    total_computation_time_ms: float
    game_duration_s: float
    
    @property
    def success_rate(self) -> float:
        """Calculate planning success rate."""
        if self.total_plans == 0:
            return 0.0
        return self.successful_plans / self.total_plans * 100
    
    @property
    def avg_path_length(self) -> float:
        """Average path length per successful plan."""
        if self.successful_plans == 0:
            return 0.0
        return self.total_path_length / self.successful_plans
    
    @property
    def avg_nodes_expanded(self) -> float:
        """Average nodes expanded per plan."""
        if self.total_plans == 0:
            return 0.0
        return self.total_nodes_expanded / self.total_plans
    
    @property
    def avg_computation_time(self) -> float:
        """Average computation time per plan (ms)."""
        if self.total_plans == 0:
            return 0.0
        return self.total_computation_time_ms / self.total_plans
    
    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            'algorithm': self.algorithm,
            'score': self.score,
            'snake_length': self.snake_length,
            'total_plans': self.total_plans,
            'successful_plans': self.successful_plans,
            'failed_plans': self.failed_plans,
            'success_rate': self.success_rate,
            'total_path_length': self.total_path_length,
            'avg_path_length': self.avg_path_length,
            'total_nodes_expanded': self.total_nodes_expanded,
            'avg_nodes_expanded': self.avg_nodes_expanded,
            'total_computation_time_ms': self.total_computation_time_ms,
            'avg_computation_time_ms': self.avg_computation_time,
            'game_duration_s': self.game_duration_s
        }


class MetricsTracker:
    """
    Tracks and aggregates metrics during gameplay.
    """
    
    def __init__(self, algorithm_name: str):
        """
        Initialize metrics tracker.
        
        Args:
            algorithm_name: Name of the algorithm being used
        """
        self.algorithm_name = algorithm_name
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.planning_results: List[PlanningResult] = []
        self.current_result: Optional[PlanningResult] = None
        self.game_start_time = time.time()
        
        # Aggregates
        self.total_plans = 0
        self.successful_plans = 0
        self.failed_plans = 0
        self.total_path_length = 0
        self.total_nodes_expanded = 0
        self.total_computation_time_ms = 0.0
    
    def record_plan(self, result: PlanningResult):
        """
        Record a planning result.
        
        Args:
            result: Planning result to record
        """
        self.planning_results.append(result)
        self.current_result = result
        
        self.total_plans += 1
        self.total_nodes_expanded += result.nodes_expanded
        self.total_computation_time_ms += result.computation_time_ms
        
        if result.success:
            self.successful_plans += 1
            self.total_path_length += result.path_length
        else:
            self.failed_plans += 1
    
    def get_summary(self) -> dict:
        """Get current metrics summary."""
        return {
            'algorithm': self.algorithm_name,
            'total_plans': self.total_plans,
            'successful_plans': self.successful_plans,
            'failed_plans': self.failed_plans,
            'success_rate': (self.successful_plans / self.total_plans * 100) if self.total_plans > 0 else 0,
            'total_path_length': self.total_path_length,
            'avg_path_length': self.total_path_length / self.successful_plans if self.successful_plans > 0 else 0,
            'total_nodes_expanded': self.total_nodes_expanded,
            'avg_nodes_expanded': self.total_nodes_expanded / self.total_plans if self.total_plans > 0 else 0,
            'total_computation_time_ms': self.total_computation_time_ms,
            'avg_computation_time': self.total_computation_time_ms / self.total_plans if self.total_plans > 0 else 0,
        }
    
    def finalize(self, score: int, snake_length: int) -> GameMetrics:
        """
        Finalize metrics at end of game.
        
        Args:
            score: Final game score
            snake_length: Final snake length
            
        Returns:
            Complete game metrics
        """
        game_duration = time.time() - self.game_start_time
        
        return GameMetrics(
            algorithm=self.algorithm_name,
            score=score,
            snake_length=snake_length,
            total_plans=self.total_plans,
            successful_plans=self.successful_plans,
            failed_plans=self.failed_plans,
            total_path_length=self.total_path_length,
            total_nodes_expanded=self.total_nodes_expanded,
            total_computation_time_ms=self.total_computation_time_ms,
            game_duration_s=game_duration
        )
    
    def __repr__(self) -> str:
        return f"MetricsTracker({self.algorithm_name}, plans={self.total_plans})"


class ComparisonTracker:
    """
    Tracks metrics across multiple algorithm runs for comparison.
    """
    
    def __init__(self):
        """Initialize comparison tracker."""
        self.game_results: List[GameMetrics] = []
    
    def add_result(self, metrics: GameMetrics):
        """Add a game result."""
        self.game_results.append(metrics)
    
    def get_comparison(self) -> Dict[str, Dict]:
        """
        Get comparison of all algorithms.
        
        Returns:
            Dictionary mapping algorithm names to their metrics
        """
        comparison = {}
        for result in self.game_results:
            if result.algorithm not in comparison:
                comparison[result.algorithm] = []
            comparison[result.algorithm].append(result.to_dict())
        
        # Calculate averages for each algorithm
        summary = {}
        for algo, results in comparison.items():
            n = len(results)
            summary[algo] = {
                'runs': n,
                'avg_score': sum(r['score'] for r in results) / n,
                'avg_path_length': sum(r['avg_path_length'] for r in results) / n,
                'avg_nodes_expanded': sum(r['avg_nodes_expanded'] for r in results) / n,
                'avg_computation_time_ms': sum(r['avg_computation_time_ms'] for r in results) / n,
                'avg_success_rate': sum(r['success_rate'] for r in results) / n,
            }
        
        return summary
    
    def export_json(self, filepath: str):
        """Export results to JSON file."""
        data = {
            'games': [r.to_dict() for r in self.game_results],
            'comparison': self.get_comparison()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_comparison(self):
        """Print comparison table to console."""
        comparison = self.get_comparison()
        
        if not comparison:
            print("No results to compare.")
            return
        
        print("\n" + "=" * 80)
        print("ALGORITHM COMPARISON")
        print("=" * 80)
        
        headers = ["Algorithm", "Runs", "Avg Score", "Avg Path", "Avg Nodes", "Avg Time(ms)", "Success%"]
        print(f"{headers[0]:<12} {headers[1]:<6} {headers[2]:<10} {headers[3]:<10} {headers[4]:<12} {headers[5]:<12} {headers[6]:<10}")
        print("-" * 80)
        
        for algo, stats in comparison.items():
            print(
                f"{algo:<12} "
                f"{stats['runs']:<6} "
                f"{stats['avg_score']:<10.1f} "
                f"{stats['avg_path_length']:<10.1f} "
                f"{stats['avg_nodes_expanded']:<12.1f} "
                f"{stats['avg_computation_time_ms']:<12.3f} "
                f"{stats['avg_success_rate']:<10.1f}"
            )
        
        print("=" * 80 + "\n")
