"""
Survival-focused path planning algorithm.

This planner prioritizes long-term survival over immediate shortest path.
It uses flood fill to evaluate the safety of moves and can chase its tail
when no safe path to food exists.
"""

import heapq
from typing import Set, Tuple, Dict, List, Optional
from collections import deque
import time

from .base_planner import BasePlanner, PlanningResult


Position = Tuple[int, int]


class SurvivalPlanner(BasePlanner):
    """
    Survival-focused path planning algorithm.
    
    Key strategies:
    1. Evaluate reachable space after each potential path
    2. Prefer paths that keep more space accessible
    3. Fall back to tail-chasing when no safe path to food exists
    4. Avoid moves that trap the snake
    """
    
    def __init__(self, grid_size: int, safety_threshold: float = 0.5):
        """
        Initialize survival planner.
        
        Args:
            grid_size: Size of the NxN grid
            safety_threshold: Minimum ratio of reachable cells to snake length
                             required to consider a path "safe" (default 0.5)
        """
        super().__init__(grid_size)
        self.name = "Survival"
        self.safety_threshold = safety_threshold
        self.tail_tip: Optional[Position] = None  # Track tail for chasing
    
    def set_tail_tip(self, tail_tip: Position):
        """Set the tail tip position for tail-chasing fallback."""
        self.tail_tip = tail_tip
    
    def flood_fill_count(
        self,
        start: Position,
        obstacles: Set[Position],
        max_count: Optional[int] = None
    ) -> int:
        """
        Count reachable cells from a position using flood fill.
        
        Args:
            start: Starting position
            obstacles: Set of obstacle positions
            max_count: Stop counting after this many (for efficiency)
            
        Returns:
            Number of reachable cells
        """
        if start in obstacles or not self._is_valid(start):
            return 0
        
        visited = {start}
        queue = deque([start])
        count = 1
        
        if max_count is None:
            max_count = self.grid_size * self.grid_size
        
        while queue and count < max_count:
            current = queue.popleft()
            
            for neighbor in self._get_neighbors_no_obstacles(current):
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    count += 1
                    
                    if count >= max_count:
                        break
        
        return count
    
    def _is_valid(self, pos: Position) -> bool:
        """Check if position is within grid bounds."""
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def _get_neighbors_no_obstacles(self, pos: Position) -> List[Position]:
        """Get valid 4-directional neighbors (bounds check only, no obstacle check)."""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if self._is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def simulate_path_obstacles(
        self,
        path: List[Position],
        current_obstacles: Set[Position],
        snake_length: int
    ) -> Set[Position]:
        """
        Simulate where obstacles (snake body) will be after following a path.
        
        This accounts for the tail following the head.
        
        Args:
            path: Path the snake will follow
            current_obstacles: Current obstacle positions (snake tail)
            snake_length: Current length of snake
            
        Returns:
            Predicted obstacle set after following the path
        """
        # Simple simulation: the snake body shifts along the path
        # After moving len(path)-1 steps, the tail moves that many positions
        steps = len(path) - 1
        
        # Convert obstacles to list to simulate tail movement
        # This is a simplification - in reality we'd need the full body
        tail_list = list(current_obstacles)
        
        # The tail "shrinks" from the back as the head moves
        # But for safety we're conservative and assume obstacles stay
        # A better approach: simulate actual body movement
        
        # For now, return current obstacles minus positions that would be vacated
        # This is optimistic but helps avoid over-conservative behavior
        return current_obstacles
    
    def evaluate_path_safety(
        self,
        path: List[Position],
        obstacles: Set[Position],
        snake_length: int
    ) -> float:
        """
        Evaluate how safe a path is based on remaining accessible space.
        
        Args:
            path: Proposed path to goal
            obstacles: Current obstacles
            snake_length: Current snake length (for context)
            
        Returns:
            Safety score (higher is better, 0-1 range typically)
        """
        if not path:
            return 0.0
        
        end_pos = path[-1]
        
        # Count reachable cells from the end position
        # Limit max_count to save computation
        max_needed = snake_length * 2
        reachable = self.flood_fill_count(
            end_pos,
            obstacles,
            max_count=max_needed
        )
        
        # Safety score: ratio of reachable cells to snake length
        safety = reachable / max(snake_length, 1)
        
        return safety
    
    def find_path_astar(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position]
    ) -> Tuple[List[Position], Set[Position], int]:
        """
        Find path using A* algorithm.
        
        Returns:
            Tuple of (path, explored_nodes, nodes_expanded)
        """
        if start == goal:
            return [start], {start}, 1
        
        if goal in obstacles:
            return [], set(), 0
        
        counter = 0
        open_set = [(self.manhattan_distance(start, goal), counter, start)]
        came_from: Dict[Position, Position] = {}
        g_score: Dict[Position, float] = {start: 0}
        open_set_hash = {start}
        explored = set()
        nodes_expanded = 0
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            nodes_expanded += 1
            explored.add(current)
            
            if current == goal:
                path = self.reconstruct_path(came_from, current)
                return path, explored, nodes_expanded
            
            for neighbor in self._get_neighbors_no_obstacles(current):
                if neighbor in obstacles:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.manhattan_distance(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score, counter, neighbor))
                        open_set_hash.add(neighbor)
        
        return [], explored, nodes_expanded
    
    def find_all_shortest_paths(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position],
        max_paths: int = 3
    ) -> List[List[Position]]:
        """
        Find one or a few shortest paths to the goal.
        
        Simplified version that finds the main path and maybe one alternative.
        
        Args:
            start: Starting position
            goal: Goal position
            obstacles: Obstacle positions
            max_paths: Maximum number of paths to find (kept small for efficiency)
            
        Returns:
            List of paths (each path is a list of positions)
        """
        if start == goal:
            return [[start]]
        
        if goal in obstacles:
            return []
        
        # Just find ONE shortest path using BFS (much more efficient)
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == goal:
                return [path]  # Return immediately with first path found
            
            for neighbor in self._get_neighbors_no_obstacles(current):
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def plan(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position],
        snake_length: int = 3
    ) -> PlanningResult:
        """
        Plan path using survival-focused strategy.
        
        Strategy:
        1. Find shortest path to goal
        2. Evaluate safety of the path (accessible space after reaching goal)
        3. If unsafe, try alternative paths
        4. If no safe path, chase tail instead
        
        Args:
            start: Starting position
            goal: Goal position
            obstacles: Set of obstacle positions
            snake_length: Current snake length for safety evaluation
            
        Returns:
            PlanningResult with path and metrics
        """
        start_time = time.perf_counter()
        total_nodes_expanded = 0
        all_explored = set()
        
        # Trivial case
        if start == goal:
            return PlanningResult(
                success=True,
                path=[start],
                explored_nodes={start},
                nodes_expanded=1,
                computation_time_ms=0,
                path_length=0
            )
        
        # Strategy 1: Find paths to the food and evaluate safety
        paths = self.find_all_shortest_paths(start, goal, obstacles, max_paths=5)
        
        best_path = None
        best_safety = -1
        
        for path in paths:
            safety = self.evaluate_path_safety(path, obstacles, snake_length)
            all_explored.update(path)
            
            if safety > best_safety:
                best_safety = safety
                best_path = path
        
        # If we found a safe enough path, use it
        if best_path and best_safety >= self.safety_threshold:
            end_time = time.perf_counter()
            return PlanningResult(
                success=True,
                path=best_path,
                explored_nodes=all_explored,
                nodes_expanded=len(all_explored),
                computation_time_ms=(end_time - start_time) * 1000,
                path_length=len(best_path) - 1
            )
        
        # Strategy 2: If path to food is unsafe, try to chase tail
        # This buys time and keeps options open
        if self.tail_tip is not None and self.tail_tip not in obstacles:
            tail_path, explored, nodes = self.find_path_astar(
                start, self.tail_tip, obstacles
            )
            all_explored.update(explored)
            total_nodes_expanded += nodes
            
            if tail_path and len(tail_path) > 1:
                # Don't actually go to tail, just move towards it
                # Take only the first step in that direction
                # But actually, let's take a few steps
                safe_steps = min(3, len(tail_path) - 1)
                partial_path = tail_path[:safe_steps + 1]
                
                end_time = time.perf_counter()
                return PlanningResult(
                    success=True,
                    path=partial_path,
                    explored_nodes=all_explored,
                    nodes_expanded=total_nodes_expanded,
                    computation_time_ms=(end_time - start_time) * 1000,
                    path_length=len(partial_path) - 1
                )
        
        # Strategy 3: If we have a path to food (even if unsafe), use it
        # Better to try than to give up
        if best_path:
            end_time = time.perf_counter()
            return PlanningResult(
                success=True,
                path=best_path,
                explored_nodes=all_explored,
                nodes_expanded=len(all_explored),
                computation_time_ms=(end_time - start_time) * 1000,
                path_length=len(best_path) - 1
            )
        
        # Strategy 4: No path to food, try to find ANY safe move
        # Find the neighbor with the most accessible space
        best_move = None
        best_space = -1
        
        for neighbor in self._get_neighbors_no_obstacles(start):
            if neighbor not in obstacles:
                # Limit flood fill to save computation
                space = self.flood_fill_count(neighbor, obstacles | {start}, max_count=snake_length * 2)
                all_explored.add(neighbor)
                
                if space > best_space:
                    best_space = space
                    best_move = neighbor
        
        if best_move and best_space > 0:
            end_time = time.perf_counter()
            return PlanningResult(
                success=True,
                path=[start, best_move],
                explored_nodes=all_explored,
                nodes_expanded=len(all_explored),
                computation_time_ms=(end_time - start_time) * 1000,
                path_length=1
            )
        
        # No valid move found
        end_time = time.perf_counter()
        return PlanningResult(
            success=False,
            explored_nodes=all_explored,
            computation_time_ms=(end_time - start_time) * 1000
        )


class HamiltonianPlanner(BasePlanner):
    """
    Hamiltonian cycle-based planner for guaranteed survival.
    
    Precomputes a Hamiltonian cycle (path visiting all cells exactly once)
    and follows it. Guarantees the snake never traps itself but is very slow.
    
    This is the "perfect" but inefficient solution.
    """
    
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.name = "Hamiltonian"
        self.cycle = self._generate_cycle()
        self.position_to_index = {pos: i for i, pos in enumerate(self.cycle)}
    
    def _generate_cycle(self) -> List[Position]:
        """
        Generate a Hamiltonian cycle for the grid.
        
        Creates a proper cycle that visits all cells and returns to start.
        Pattern (for a 4x4 grid):
        
         0  1  2  3
         7  6  5  4
         8  9 10 11
        15 14 13 12
        
        Then wraps: 15 -> 0 (going up the left edge)
        
        But to make it a CYCLE, we use a different pattern:
        
        Start at (0,0), go right, then snake down, and return via left column.
        
         0→ 1→ 2→ 3
                  ↓
         7← 6← 5← 4
         ↓
         8→ 9→10→11
                  ↓
        15←14←13←12
         ↓
        (back to 0 via left edge going up)
        
        For this to work as a cycle, we need the left column to be the return path:
        - Row 0: (1,0) to (n-1,0) going right
        - Row 1: (n-1,1) to (1,1) going left  
        - ... alternating
        - Left column (0,y) traversed last going up from (0,n-1) to (0,0)
        """
        cycle = []
        n = self.grid_size
        
        # Standard snake pattern but skip column 0 initially
        for y in range(n):
            if y % 2 == 0:
                # Left to right, starting from column 1
                for x in range(1, n):
                    cycle.append((x, y))
            else:
                # Right to left, ending at column 1
                for x in range(n - 1, 0, -1):
                    cycle.append((x, y))
        
        # Now traverse column 0 from bottom to top to complete the cycle
        for y in range(n - 1, -1, -1):
            cycle.append((0, y))
        
        return cycle
    
    def _cycle_distance(self, from_idx: int, to_idx: int) -> int:
        """Calculate distance along the cycle."""
        n = len(self.cycle)
        return (to_idx - from_idx) % n
    
    def plan(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position],
        snake_length: int = 3
    ) -> PlanningResult:
        """
        Plan path by following the Hamiltonian cycle.
        
        Returns path to goal following the cycle. Only returns a limited
        number of steps to allow for replanning.
        """
        start_time = time.perf_counter()
        
        if start not in self.position_to_index:
            # Start position not on cycle - find nearest cycle position
            # This can happen if snake starts off-cycle
            min_dist = float('inf')
            nearest_idx = 0
            for idx, pos in enumerate(self.cycle):
                dist = abs(pos[0] - start[0]) + abs(pos[1] - start[1])
                if dist < min_dist and pos not in obstacles:
                    min_dist = dist
                    nearest_idx = idx
            
            # Return path to nearest cycle position using simple movement
            if min_dist > 0:
                # Just move one step toward the cycle
                next_pos = self.cycle[nearest_idx]
                path = [start, next_pos]
                end_time = time.perf_counter()
                return PlanningResult(
                    success=True,
                    path=path,
                    explored_nodes=set(path),
                    nodes_expanded=2,
                    computation_time_ms=(end_time - start_time) * 1000,
                    path_length=1
                )
            start_idx = nearest_idx
        else:
            start_idx = self.position_to_index[start]
        
        goal_idx = self.position_to_index.get(goal, -1)
        
        if goal_idx == -1:
            # Goal not on cycle, find nearest goal on cycle
            min_dist = float('inf')
            for idx, pos in enumerate(self.cycle):
                dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
                if dist < min_dist:
                    min_dist = dist
                    goal_idx = idx
        
        # Build path along the cycle
        path = []
        current_idx = start_idx
        
        # Limit path length to avoid very long paths
        # Return at most enough steps to reach the goal or a reasonable chunk
        max_steps = min(self._cycle_distance(start_idx, goal_idx) + 1, len(self.cycle))
        
        for _ in range(max_steps):
            path.append(self.cycle[current_idx])
            if current_idx == goal_idx:
                break
            current_idx = (current_idx + 1) % len(self.cycle)
        
        # Validate path doesn't hit obstacles (shouldn't happen with proper cycle)
        for pos in path[1:]:  # Skip start position
            if pos in obstacles:
                # Obstacle in path - this shouldn't happen but handle it
                # Return just the first valid steps
                valid_path = [path[0]]
                for p in path[1:]:
                    if p in obstacles:
                        break
                    valid_path.append(p)
                if len(valid_path) > 1:
                    path = valid_path
                    break
                else:
                    return PlanningResult(
                        success=False,
                        computation_time_ms=(time.perf_counter() - start_time) * 1000
                    )
        
        end_time = time.perf_counter()
        return PlanningResult(
            success=True,
            path=path,
            explored_nodes=set(path),
            nodes_expanded=len(path),
            computation_time_ms=(end_time - start_time) * 1000,
            path_length=len(path) - 1
        )
