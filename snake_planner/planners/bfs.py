"""
BFS (Breadth-First Search) path planning algorithm.
"""

from collections import deque
from typing import Set, Tuple, Dict
import time

from .base_planner import BasePlanner, PlanningResult


Position = Tuple[int, int]


class BFSPlanner(BasePlanner):
    """
    BFS (Breadth-First Search) path planning algorithm.
    
    Classic baseline algorithm. Guarantees shortest path on unweighted
    grid. Explores in layers/wavefront from start.
    """
    
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.name = "BFS"
    
    def plan(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position]
    ) -> PlanningResult:
        """
        Plan path using BFS algorithm.
        
        Args:
            start: Starting position
            goal: Goal position
            obstacles: Set of obstacle positions
            
        Returns:
            PlanningResult with path and metrics
        """
        start_time = time.perf_counter()
        
        # Check trivial cases
        if start == goal:
            return PlanningResult(
                success=True,
                path=[start],
                explored_nodes={start},
                nodes_expanded=1,
                computation_time_ms=0,
                path_length=0
            )
        
        if goal in obstacles:
            return PlanningResult(
                success=False,
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # BFS queue
        queue = deque([start])
        
        # Tracking
        visited: Set[Position] = {start}
        came_from: Dict[Position, Position] = {}
        
        # Explored nodes for visualization
        explored = {start}
        nodes_expanded = 0
        
        while queue:
            current = queue.popleft()
            nodes_expanded += 1
            
            # Check if we reached the goal
            if current == goal:
                path = self.reconstruct_path(came_from, current)
                end_time = time.perf_counter()
                return PlanningResult(
                    success=True,
                    path=path,
                    explored_nodes=explored,
                    nodes_expanded=nodes_expanded,
                    computation_time_ms=(end_time - start_time) * 1000,
                    path_length=len(path) - 1
                )
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current, obstacles):
                if neighbor not in visited:
                    visited.add(neighbor)
                    explored.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
        
        # No path found
        end_time = time.perf_counter()
        return PlanningResult(
            success=False,
            explored_nodes=explored,
            nodes_expanded=nodes_expanded,
            computation_time_ms=(end_time - start_time) * 1000
        )
