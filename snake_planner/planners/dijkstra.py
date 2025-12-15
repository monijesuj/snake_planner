"""
Dijkstra's path planning algorithm.
"""

import heapq
from typing import Set, Tuple, Dict
import time

from .base_planner import BasePlanner, PlanningResult


Position = Tuple[int, int]


class DijkstraPlanner(BasePlanner):
    """
    Dijkstra's shortest path algorithm.
    
    Guarantees optimal path. Explores more nodes than A* since
    it doesn't use a heuristic.
    """
    
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.name = "Dijkstra"
    
    def plan(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position]
    ) -> PlanningResult:
        """
        Plan path using Dijkstra's algorithm.
        
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
        
        # Priority queue: (distance, counter, position)
        counter = 0
        open_set = [(0, counter, start)]
        
        # Tracking dictionaries
        came_from: Dict[Position, Position] = {}
        distance: Dict[Position, float] = {start: 0}
        
        # Visited set
        visited: Set[Position] = set()
        
        # Explored nodes for visualization
        explored = set()
        nodes_expanded = 0
        
        while open_set:
            # Get node with lowest distance
            dist, _, current = heapq.heappop(open_set)
            
            # Skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
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
                if neighbor in visited:
                    continue
                
                new_dist = distance[current] + 1  # Uniform cost of 1
                
                if neighbor not in distance or new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_set, (new_dist, counter, neighbor))
        
        # No path found
        end_time = time.perf_counter()
        return PlanningResult(
            success=False,
            explored_nodes=explored,
            nodes_expanded=nodes_expanded,
            computation_time_ms=(end_time - start_time) * 1000
        )
