from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Set
from snake_planner.common.geometry import Position, get_neighbors

@dataclass
class PlanningResult:
    success: bool
    path: List[Position] = field(default_factory=list)
    explored_nodes: Set[Position] = field(default_factory=set)
    nodes_expanded: int = 0
    computation_time_ms: float = 0.0
    path_length: int = 0
    
    def __post_init__(self):
        if self.path:
            self.path_length = len(self.path) - 1

class BasePlanner(ABC):
    """
    Abstract base class for path planning algorithms.
    """
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.name = self.__class__.__name__

    @abstractmethod
    def plan(self, start: Position, goal: Position, obstacles: Set[Position]) -> PlanningResult:
        pass

    def get_valid_neighbors(self, pos: Position, obstacles: Set[Position]) -> List[Position]:
        """Shared logic for finding walkable neighbors."""
        candidates = get_neighbors(pos, self.grid_size)
        return [p for p in candidates if p not in obstacles]

    def reconstruct_path(self, came_from: dict, current: Position) -> List[Position]:
        """
        Reconstruct path from came_from dictionary.
        Handles both implicit roots (key missing) and explicit roots (value is None).
        """
        path = [current]
        while current in came_from:
            parent = came_from[current]
            
            # Fix: RRT uses {start: None}, so we must stop if parent is None
            if parent is None:
                break
                
            current = parent
            path.append(current)
            
        path.reverse()
        return path