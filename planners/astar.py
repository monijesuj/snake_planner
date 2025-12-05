"""
A* (A-star) path planning algorithm.
"""

import heapq
from typing import Set, Tuple, Dict
import time

from .base_planner import BasePlanner, PlanningResult


Position = Tuple[int, int]


class AStarPlanner(BasePlanner):
    """
    A* path planning algorithm.
    
    Uses Manhattan distance as heuristic. Guarantees optimal path
    on a grid with uniform edge costs.
    """
    
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.name = "A*"
    
    def plan(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position]
    ) -> PlanningResult:
        """
        Plan path using A* algorithm.
        
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
        
        # Priority queue: (f_score, counter, position)
        # Counter is used for tie-breaking to ensure consistent ordering
        counter = 0
        open_set = [(self.manhattan_distance(start, goal), counter, start)]
        
        # Tracking dictionaries
        came_from: Dict[Position, Position] = {}
        g_score: Dict[Position, float] = {start: 0}
        f_score: Dict[Position, float] = {start: self.manhattan_distance(start, goal)}
        
        # Set for O(1) lookup of open set membership
        open_set_hash = {start}
        
        # Explored nodes for visualization
        explored = set()
        nodes_expanded = 0
        
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            
            nodes_expanded += 1
            explored.add(current)
            
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
                tentative_g = g_score[current] + 1  # Uniform cost of 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Found better path to neighbor
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.manhattan_distance(neighbor, goal)
                    f_score[neighbor] = f
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        end_time = time.perf_counter()
        return PlanningResult(
            success=False,
            explored_nodes=explored,
            nodes_expanded=nodes_expanded,
            computation_time_ms=(end_time - start_time) * 1000
        )
