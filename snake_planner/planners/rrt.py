"""
RRT (Rapidly-exploring Random Tree) path planning algorithm adapted for grid.
"""

import random
from typing import Set, Tuple, Dict, List, Optional
import time

from .base_planner import BasePlanner, PlanningResult


Position = Tuple[int, int]


class RRTPlanner(BasePlanner):
    """
    RRT (Rapidly-exploring Random Tree) algorithm adapted for discrete grid.
    
    This is a sampling-based algorithm that builds a tree by randomly
    sampling positions and connecting them. Adapted for grid by using
    Manhattan distance and grid-aligned extensions.
    
    Note: RRT is not guaranteed to find the optimal path, making it
    interesting for comparison with optimal algorithms like A*.
    """
    
    def __init__(
        self,
        grid_size: int,
        max_iterations: int = 5000,
        goal_bias: float = 0.1,
        step_size: int = 1
    ):
        """
        Initialize RRT planner.
        
        Args:
            grid_size: Size of the NxN grid
            max_iterations: Maximum iterations before giving up
            goal_bias: Probability of sampling goal instead of random point
            step_size: Maximum step size for extension (1 for grid)
        """
        super().__init__(grid_size)
        self.name = "RRT"
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
        self.step_size = step_size
    
    def plan(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position]
    ) -> PlanningResult:
        """
        Plan path using RRT algorithm.
        
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
        
        # Initialize tree with start node
        # tree[node] = parent
        tree: Dict[Position, Optional[Position]] = {start: None}
        tree_nodes: List[Position] = [start]  # List for random selection
        
        explored = {start}
        nodes_expanded = 0
        
        for iteration in range(self.max_iterations):
            nodes_expanded += 1
            
            # Sample random point (with goal bias)
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._random_sample()
            
            # Find nearest node in tree
            nearest = self._find_nearest(tree_nodes, sample)
            
            # Extend towards sample
            new_node = self._extend(nearest, sample, obstacles)
            
            if new_node is not None and new_node not in tree:
                tree[new_node] = nearest
                tree_nodes.append(new_node)
                explored.add(new_node)
                
                # Check if we reached the goal
                if new_node == goal:
                    path = self._reconstruct_path(tree, goal)
                    end_time = time.perf_counter()
                    return PlanningResult(
                        success=True,
                        path=path,
                        explored_nodes=explored,
                        nodes_expanded=nodes_expanded,
                        computation_time_ms=(end_time - start_time) * 1000,
                        path_length=len(path) - 1
                    )
                
                # Check if we can connect to goal
                if self._can_connect(new_node, goal, obstacles):
                    tree[goal] = new_node
                    explored.add(goal)
                    path = self._reconstruct_path(tree, goal)
                    end_time = time.perf_counter()
                    return PlanningResult(
                        success=True,
                        path=path,
                        explored_nodes=explored,
                        nodes_expanded=nodes_expanded,
                        computation_time_ms=(end_time - start_time) * 1000,
                        path_length=len(path) - 1
                    )
        
        # Failed to find path within max iterations
        end_time = time.perf_counter()
        return PlanningResult(
            success=False,
            explored_nodes=explored,
            nodes_expanded=nodes_expanded,
            computation_time_ms=(end_time - start_time) * 1000
        )
    
    def _random_sample(self) -> Position:
        """Generate random sample position."""
        return (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1)
        )
    
    def _find_nearest(self, nodes: List[Position], target: Position) -> Position:
        """Find nearest node in tree to target."""
        return min(nodes, key=lambda n: self.manhattan_distance(n, target))
    
    def _extend(
        self,
        from_node: Position,
        to_node: Position,
        obstacles: Set[Position]
    ) -> Optional[Position]:
        """
        Extend from from_node towards to_node by one step.
        
        For grid-based RRT, we move one cell in the direction of to_node.
        
        Args:
            from_node: Starting node
            to_node: Target node
            obstacles: Set of obstacles
            
        Returns:
            New node position, or None if blocked
        """
        if from_node == to_node:
            return None
        
        # Calculate direction to target
        dx = to_node[0] - from_node[0]
        dy = to_node[1] - from_node[1]
        
        # Prioritize the larger displacement
        # Move one step in the dominant direction
        if abs(dx) >= abs(dy):
            # Move horizontally
            step_x = 1 if dx > 0 else -1
            new_node = (from_node[0] + step_x, from_node[1])
        else:
            # Move vertically
            step_y = 1 if dy > 0 else -1
            new_node = (from_node[0], from_node[1] + step_y)
        
        # Check if valid
        if self.is_valid(new_node, obstacles):
            return new_node
        
        # Try the other direction
        if abs(dx) >= abs(dy):
            if dy != 0:
                step_y = 1 if dy > 0 else -1
                new_node = (from_node[0], from_node[1] + step_y)
                if self.is_valid(new_node, obstacles):
                    return new_node
        else:
            if dx != 0:
                step_x = 1 if dx > 0 else -1
                new_node = (from_node[0] + step_x, from_node[1])
                if self.is_valid(new_node, obstacles):
                    return new_node
        
        return None
    
    def _can_connect(
        self,
        from_node: Position,
        to_node: Position,
        obstacles: Set[Position]
    ) -> bool:
        """
        Check if we can directly connect from_node to to_node.
        
        For grid, we check if to_node is adjacent and not blocked.
        """
        dist = self.manhattan_distance(from_node, to_node)
        if dist != 1:
            return False
        return self.is_valid(to_node, obstacles)
    
    def _reconstruct_path(
        self,
        tree: Dict[Position, Optional[Position]],
        goal: Position
    ) -> List[Position]:
        """Reconstruct path from tree."""
        path = []
        current: Optional[Position] = goal
        while current is not None:
            path.append(current)
            current = tree.get(current)
        path.reverse()
        return path
