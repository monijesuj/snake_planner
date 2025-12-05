"""
Base class for all planning algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional
import time


Position = Tuple[int, int]


@dataclass
class PlanningResult:
    """Result of a planning operation."""
    success: bool
    path: List[Position] = field(default_factory=list)
    explored_nodes: Set[Position] = field(default_factory=set)
    nodes_expanded: int = 0
    computation_time_ms: float = 0.0
    path_length: int = 0
    
    def __post_init__(self):
        if self.path:
            self.path_length = len(self.path) - 1  # Number of moves, not positions


class BasePlanner(ABC):
    """
    Abstract base class for path planning algorithms.
    
    All planners must implement the plan() method.
    """
    
    def __init__(self, grid_size: int):
        """
        Initialize planner.
        
        Args:
            grid_size: Size of the NxN grid
        """
        self.grid_size = grid_size
        self.name = self.__class__.__name__
    
    @abstractmethod
    def plan(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position]
    ) -> PlanningResult:
        """
        Plan a path from start to goal avoiding obstacles.
        
        Args:
            start: Starting position
            goal: Goal position
            obstacles: Set of obstacle positions to avoid
            
        Returns:
            PlanningResult with path and metrics
        """
        pass
    
    def is_valid(self, pos: Position, obstacles: Set[Position]) -> bool:
        """
        Check if position is valid (in bounds and not an obstacle).
        
        Args:
            pos: Position to check
            obstacles: Set of obstacle positions
            
        Returns:
            True if valid
        """
        x, y = pos
        return (
            0 <= x < self.grid_size and
            0 <= y < self.grid_size and
            pos not in obstacles
        )
    
    def get_neighbors(self, pos: Position, obstacles: Set[Position]) -> List[Position]:
        """
        Get valid 4-directional neighbors.
        
        Args:
            pos: Current position
            obstacles: Set of obstacle positions
            
        Returns:
            List of valid neighbor positions
        """
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if self.is_valid(new_pos, obstacles):
                neighbors.append(new_pos)
        return neighbors
    
    def reconstruct_path(
        self,
        came_from: dict,
        current: Position
    ) -> List[Position]:
        """
        Reconstruct path from came_from dictionary.
        
        Args:
            came_from: Dictionary mapping positions to their predecessors
            current: Goal position
            
        Returns:
            Path from start to goal
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _timed_plan(
        self,
        plan_func,
        start: Position,
        goal: Position,
        obstacles: Set[Position]
    ) -> PlanningResult:
        """
        Wrapper to time the planning function.
        
        Args:
            plan_func: Planning function to call
            start: Start position
            goal: Goal position
            obstacles: Obstacles set
            
        Returns:
            PlanningResult with timing information
        """
        start_time = time.perf_counter()
        result = plan_func(start, goal, obstacles)
        end_time = time.perf_counter()
        result.computation_time_ms = (end_time - start_time) * 1000
        return result
    
    def __repr__(self) -> str:
        return f"{self.name}(grid_size={self.grid_size})"
