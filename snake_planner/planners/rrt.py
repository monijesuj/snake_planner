import random
import time
from typing import Set, List, Optional
from .base_planner import BasePlanner, PlanningResult
from snake_planner.common.geometry import Position, manhattan_distance

class RRTPlanner(BasePlanner):
    """
    RRT (Rapidly-exploring Random Tree) adapted for discrete grid.
    """
    
    def __init__(self, grid_size: int, max_iterations: int = 5000, goal_bias: float = 0.1):
        super().__init__(grid_size)
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias

    def plan(self, start: Position, goal: Position, obstacles: Set[Position]) -> PlanningResult:
        start_time = time.perf_counter()
        
        # Trivial cases
        if start == goal:
            return PlanningResult(True, [start], {start}, 1, 0, 0)
        
        if goal in obstacles:
            return PlanningResult(
                False, 
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )

        # Initialize tree: Maps child -> parent
        tree = {start: None}
        tree_nodes = [start]  # List for efficient random sampling
        explored = {start}
        
        for _ in range(self.max_iterations):
            # 1. Sample
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._random_sample()
            
            # 2. Nearest Neighbor
            # FIX: Use imported function, NOT self.manhattan_distance
            nearest = min(tree_nodes, key=lambda n: manhattan_distance(n, sample))
            
            # 3. Extend
            new_node = self._extend(nearest, sample, obstacles)
            
            if new_node and new_node not in tree:
                tree[new_node] = nearest
                tree_nodes.append(new_node)
                explored.add(new_node)
                
                # 4. Check Goal
                # Either we reached it, or we are adjacent and can connect
                if new_node == goal or self._can_connect(new_node, goal, obstacles):
                    if new_node != goal:
                        tree[goal] = new_node
                    
                    path = self.reconstruct_path(tree, goal)
                    return PlanningResult(
                        True, 
                        path, 
                        explored, 
                        len(tree), 
                        (time.perf_counter() - start_time) * 1000
                    )
        
        # Failed
        return PlanningResult(
            False, 
            explored_nodes=explored, 
            nodes_expanded=len(tree), 
            computation_time_ms=(time.perf_counter() - start_time) * 1000
        )

    def _random_sample(self) -> Position:
        return (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1)
        )

    def _extend(self, from_node: Position, to_node: Position, obstacles: Set[Position]) -> Optional[Position]:
        if from_node == to_node:
            return None
            
        dx = to_node[0] - from_node[0]
        dy = to_node[1] - from_node[1]
        
        # Determine step direction (Manhattan-ish steering)
        # Try to move in the axis with the largest difference first
        moves = []
        if abs(dx) >= abs(dy):
            moves = [(1 if dx > 0 else -1, 0), (0, 1 if dy > 0 else -1)]
        else:
            moves = [(0, 1 if dy > 0 else -1), (1 if dx > 0 else -1, 0)]

        for move_x, move_y in moves:
            if move_x == 0 and move_y == 0:
                continue
                
            candidate = (from_node[0] + move_x, from_node[1] + move_y)
            
            # Check bounds and obstacles
            if (0 <= candidate[0] < self.grid_size and 
                0 <= candidate[1] < self.grid_size and 
                candidate not in obstacles):
                return candidate
                
        return None

    def _can_connect(self, u: Position, v: Position, obstacles: Set[Position]) -> bool:
        """Check if two nodes are adjacent and the target is free."""
        dist = manhattan_distance(u, v)
        return dist == 1 and v not in obstacles