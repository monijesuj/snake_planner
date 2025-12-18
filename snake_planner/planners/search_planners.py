import heapq
import time
from collections import deque
from typing import Set, Dict, List, Tuple
from abc import abstractmethod

from .base_planner import BasePlanner, PlanningResult
from snake_planner.common.geometry import Position, manhattan_distance

class GraphSearchPlanner(BasePlanner):
    """
    Abstract base for graph search algorithms (BFS, A*, Dijkstra).
    Implements the core search loop pattern.
    """
    
    def plan(self, start: Position, goal: Position, obstacles: Set[Position]) -> PlanningResult:
        start_time = time.perf_counter()
        
        if start == goal:
            return PlanningResult(True, [start], {start}, 1, 0, 0)
        if goal in obstacles:
            return PlanningResult(False, computation_time_ms=(time.perf_counter() - start_time) * 1000)

        # Initialize core structures
        container = self._init_container(start)
        came_from: Dict[Position, Position] = {}
        cost_so_far: Dict[Position, float] = {start: 0}
        
        explored = set()
        nodes_expanded = 0

        while not self._is_container_empty(container):
            current = self._pop_container(container)
            
            # Optimization: Skip if we found a shorter path to this node already (only for PriorityQueue)
            # For BFS/Unweighted, this check is implicit in 'if neighbor not in cost_so_far'
            
            nodes_expanded += 1
            explored.add(current)

            if current == goal:
                path = self.reconstruct_path(came_from, current)
                return PlanningResult(
                    True, path, explored, nodes_expanded,
                    (time.perf_counter() - start_time) * 1000
                )

            for neighbor in self.get_valid_neighbors(current, obstacles):
                new_cost = cost_so_far[current] + 1
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = self._calculate_priority(new_cost, neighbor, goal)
                    came_from[neighbor] = current
                    self._push_container(container, neighbor, priority)
        
        return PlanningResult(False, explored_nodes=explored, nodes_expanded=nodes_expanded, 
                            computation_time_ms=(time.perf_counter() - start_time) * 1000)

    @abstractmethod
    def _init_container(self, start: Position): pass
    
    @abstractmethod
    def _is_container_empty(self, container) -> bool: pass
    
    @abstractmethod
    def _pop_container(self, container) -> Position: pass
    
    @abstractmethod
    def _push_container(self, container, item: Position, priority: float): pass
    
    def _calculate_priority(self, g_cost: float, current: Position, goal: Position) -> float:
        return g_cost # Default for unweighted

# --- Implementations ---

class WeightedGraphPlanner(GraphSearchPlanner):
    """Base for A* and Dijkstra (using Heap)."""
    
    def _init_container(self, start: Position):
        # Heap stores: (priority, tie_breaker, item)
        self.tie_breaker = 0
        return [(0, 0, start)]

    def _is_container_empty(self, container) -> bool:
        return len(container) == 0

    def _pop_container(self, container) -> Position:
        return heapq.heappop(container)[2]

    def _push_container(self, container, item: Position, priority: float):
        self.tie_breaker += 1
        heapq.heappush(container, (priority, self.tie_breaker, item))

class AStarPlanner(WeightedGraphPlanner):
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.name = "A*"

    def _calculate_priority(self, g_cost: float, current: Position, goal: Position) -> float:
        # f = g + h
        return g_cost + manhattan_distance(current, goal)

class DijkstraPlanner(WeightedGraphPlanner):
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.name = "Dijkstra"
    
    def _calculate_priority(self, g_cost: float, current: Position, goal: Position) -> float:
        return g_cost

class BFSPlanner(GraphSearchPlanner):
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.name = "BFS"

    def _init_container(self, start: Position):
        return deque([start])

    def _is_container_empty(self, container) -> bool:
        return len(container) == 0

    def _pop_container(self, container) -> Position:
        return container.popleft()

    def _push_container(self, container, item: Position, priority: float):
        container.append(item)