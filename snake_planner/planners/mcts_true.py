"""
True Monte Carlo Tree Search (MCTS) path planning algorithm.

This implements the classic MCTS algorithm with all 4 phases:
1. Selection - UCB1-guided tree traversal
2. Expansion - Add new child nodes
3. Simulation - Random playouts
4. Backpropagation - Update statistics up the tree
"""

import math
import random
import time
from typing import Set, Tuple, List, Optional, Dict
from dataclasses import dataclass

from .base_planner import BasePlanner, PlanningResult


Position = Tuple[int, int]

# Directions: up, down, left, right
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


@dataclass
class SimState:
    """Lightweight simulation state for MCTS."""
    head: Position
    body: List[Position]
    food: Position
    grid_size: int
    
    def copy(self) -> 'SimState':
        return SimState(
            head=self.head,
            body=self.body.copy(),
            food=self.food,
            grid_size=self.grid_size
        )
    
    def get_body_set(self) -> Set[Position]:
        return set(self.body)
    
    def is_valid_pos(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def get_valid_moves(self) -> List[Position]:
        """Get list of valid next positions."""
        valid = []
        body_set = self.get_body_set()
        
        for dx, dy in DIRECTIONS:
            new_pos = (self.head[0] + dx, self.head[1] + dy)
            if not self.is_valid_pos(new_pos):
                continue
            # Can move to tail tip (it will move away)
            future_body = set(self.body[:-1]) if len(self.body) > 1 else set()
            if new_pos in future_body:
                continue
            valid.append(new_pos)
        
        return valid
    
    def move(self, new_head: Position) -> bool:
        """Move snake to new position. Returns True if food was eaten."""
        ate_food = (new_head == self.food)
        self.body.insert(0, new_head)
        self.head = new_head
        if not ate_food:
            self.body.pop()
        return ate_food


class TreeNode:
    """
    A node in the MCTS search tree.
    
    Each node represents a game state and tracks:
    - Visit count (N)
    - Total reward (Q)
    - Children for each action
    """
    
    def __init__(self, state: SimState, parent: 'TreeNode' = None, 
                 action: Position = None):
        self.state = state
        self.parent = parent
        self.action = action  # The move that led here from parent
        
        self.children: Dict[Position, 'TreeNode'] = {}
        self.visits: int = 0
        self.total_reward: float = 0.0
        
        # Untried actions for expansion
        self.untried_actions: List[Position] = state.get_valid_moves()
    
    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    @property
    def is_terminal(self) -> bool:
        return len(self.state.get_valid_moves()) == 0
    
    def ucb1_score(self, exploration_weight: float) -> float:
        """
        Upper Confidence Bound for Trees (UCB1).
        
        UCB1 = Q/N + c * sqrt(ln(parent_N) / N)
        
        - Q/N: exploitation (average reward)
        - sqrt term: exploration bonus
        - c: exploration weight (typically sqrt(2) ≈ 1.41)
        """
        if self.visits == 0:
            return float('inf')  # Always try unvisited nodes
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def select_child(self, exploration_weight: float) -> 'TreeNode':
        """Select child with highest UCB1 score."""
        return max(
            self.children.values(),
            key=lambda child: child.ucb1_score(exploration_weight)
        )
    
    def add_child(self, action: Position, child_state: SimState) -> 'TreeNode':
        """Add a child node for the given action."""
        child = TreeNode(child_state, parent=self, action=action)
        self.children[action] = child
        self.untried_actions.remove(action)
        return child
    
    def update(self, reward: float):
        """Update node statistics with simulation result."""
        self.visits += 1
        self.total_reward += reward
    
    def best_action(self) -> Position:
        """
        Return the best action based on visit count.
        
        Using visit count (not average reward) is more robust
        because it represents how much we've explored that branch.
        """
        if not self.children:
            return None
        return max(
            self.children.items(),
            key=lambda item: item[1].visits
        )[0]


class TrueMCTSPlanner(BasePlanner):
    """
    True Monte Carlo Tree Search planner.
    
    Implements the classic MCTS algorithm:
    
    For num_simulations iterations:
        1. SELECTION: Start at root, use UCB1 to traverse tree
           until reaching a node that isn't fully expanded
        2. EXPANSION: Add one new child node for an untried action
        3. SIMULATION: Random playout from new node until terminal
        4. BACKPROPAGATION: Update visit counts and rewards up the tree
    
    Finally, return the most-visited child of root.
    """
    
    def __init__(self, grid_size: int,
                 num_simulations: int = 500,
                 max_simulation_depth: int = 50,
                 exploration_weight: float = 1.41):
        """
        Initialize True MCTS planner.
        
        Args:
            grid_size: Size of the NxN grid
            num_simulations: Number of MCTS iterations
            max_simulation_depth: Max depth of random playouts
            exploration_weight: UCB1 exploration constant (sqrt(2) is theoretically optimal)
        """
        super().__init__(grid_size)
        self.name = "TrueMCTS"
        self.num_simulations = num_simulations
        self.max_simulation_depth = max_simulation_depth
        self.exploration_weight = exploration_weight
        self.snake_body: List[Position] = []
    
    def set_snake_body(self, body: List[Position]):
        """Set the current snake body for simulation."""
        self.snake_body = list(body)
    
    def _create_state(self, start: Position, goal: Position,
                      obstacles: Set[Position]) -> SimState:
        """Create simulation state from game state."""
        body = self.snake_body.copy() if self.snake_body else [start] + list(obstacles)
        return SimState(head=start, body=body, food=goal, grid_size=self.grid_size)
    
    # ========== PHASE 1: SELECTION ==========
    def _select(self, node: TreeNode) -> TreeNode:
        """
        Selection phase: traverse tree using UCB1.
        
        Keep going down until we find a node that:
        - Has untried actions (not fully expanded), OR
        - Is terminal (game over)
        """
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node  # Found a node to expand
            # All children tried, select best by UCB1
            node = node.select_child(self.exploration_weight)
        return node
    
    # ========== PHASE 2: EXPANSION ==========
    def _expand(self, node: TreeNode) -> TreeNode:
        """
        Expansion phase: add a new child node.
        
        Pick a random untried action and create a child node
        representing the state after taking that action.
        """
        if node.is_terminal or not node.untried_actions:
            return node  # Can't expand terminal or fully expanded
        
        # Pick random untried action
        action = random.choice(node.untried_actions)
        
        # Create child state
        child_state = node.state.copy()
        ate_food = child_state.move(action)
        
        # If food eaten, spawn new food
        if ate_food:
            child_state.food = self._spawn_food(child_state)
        
        return node.add_child(action, child_state)
    
    # ========== PHASE 3: SIMULATION ==========
    def _simulate(self, state: SimState) -> float:
        """
        Simulation phase: random playout from given state.
        
        Play random moves until:
        - Snake dies (no valid moves)
        - Max depth reached
        - Food is eaten (early termination bonus)
        
        Returns a reward in [0, 1] range.
        """
        sim = state.copy()
        initial_food = sim.food
        food_eaten = 0
        steps = 0
        
        # Initial distance to food
        initial_dist = self._manhattan_distance(sim.head, sim.food)
        
        for step in range(self.max_simulation_depth):
            valid_moves = sim.get_valid_moves()
            
            if not valid_moves:
                # Snake died - return low reward based on survival
                return 0.1 * (steps / self.max_simulation_depth)
            
            # Semi-random move selection (biased toward food)
            action = self._simulation_policy(sim, valid_moves)
            ate_food = sim.move(action)
            steps += 1
            
            if ate_food:
                food_eaten += 1
                # Early termination - eating food is the goal!
                # Return high reward scaled by how quickly we got it
                speed_bonus = 1.0 - (steps / self.max_simulation_depth)
                return 0.7 + 0.3 * speed_bonus
        
        # Reached max depth without dying
        # Reward based on: survival (good) + distance improvement
        final_dist = self._manhattan_distance(sim.head, sim.food)
        dist_improvement = (initial_dist - final_dist) / (initial_dist + 1)
        
        # Survival is worth something, getting closer is bonus
        survival_reward = 0.3
        distance_reward = 0.3 * max(0, dist_improvement)
        
        return survival_reward + distance_reward
    
    def _simulation_policy(self, state: SimState, 
                           valid_moves: List[Position]) -> Position:
        """
        Policy for selecting moves during simulation.
        
        Epsilon-greedy: mostly pick best move, sometimes random.
        """
        epsilon = 0.2  # 20% random
        
        if random.random() < epsilon:
            return random.choice(valid_moves)
        
        # Greedy: pick move closest to food
        return min(
            valid_moves,
            key=lambda m: self._manhattan_distance(m, state.food)
        )
    
    # ========== PHASE 4: BACKPROPAGATION ==========
    def _backpropagate(self, node: TreeNode, reward: float):
        """
        Backpropagation phase: update statistics up the tree.
        
        Walk from the given node back to root, updating
        visit count and total reward at each node.
        """
        while node is not None:
            node.update(reward)
            node = node.parent
    
    # ========== MAIN PLANNING LOOP ==========
    def plan(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position],
        snake_length: int = 3
    ) -> PlanningResult:
        """
        Plan using Monte Carlo Tree Search.
        
        The MCTS loop:
        1. Selection - UCB1 tree traversal
        2. Expansion - Add new child
        3. Simulation - Random playout
        4. Backpropagation - Update tree statistics
        """
        start_time = time.perf_counter()
        
        # Create root node
        initial_state = self._create_state(start, goal, obstacles)
        root = TreeNode(initial_state)
        
        # Check for valid moves
        if not initial_state.get_valid_moves():
            return PlanningResult(
                success=False,
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # Check if food is one step away
        for move in initial_state.get_valid_moves():
            if move == goal:
                return PlanningResult(
                    success=True,
                    path=[start, goal],
                    explored_nodes={start, goal},
                    nodes_expanded=1,
                    computation_time_ms=(time.perf_counter() - start_time) * 1000,
                    path_length=1
                )
        
        # ===== MCTS MAIN LOOP =====
        for iteration in range(self.num_simulations):
            # 1. SELECTION: Traverse tree to find expandable node
            node = self._select(root)
            
            # 2. EXPANSION: Add a new child (if not terminal)
            if not node.is_terminal:
                node = self._expand(node)
            
            # 3. SIMULATION: Random playout from this node
            reward = self._simulate(node.state)
            
            # 4. BACKPROPAGATION: Update tree with result
            self._backpropagate(node, reward)
        
        # Get best action from root (most visited)
        best_action = root.best_action()
        
        if best_action is None:
            return PlanningResult(
                success=False,
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        end_time = time.perf_counter()
        
        # Collect explored nodes for visualization
        explored = {start}
        self._collect_explored(root, explored)
        
        return PlanningResult(
            success=True,
            path=[start, best_action],
            explored_nodes=explored,
            nodes_expanded=root.visits,
            computation_time_ms=(end_time - start_time) * 1000,
            path_length=1
        )
    
    # ========== HELPER METHODS ==========
    def _manhattan_distance(self, a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _spawn_food(self, state: SimState) -> Position:
        """Spawn food in random empty position."""
        body_set = state.get_body_set()
        for _ in range(100):
            pos = (random.randint(0, state.grid_size - 1),
                   random.randint(0, state.grid_size - 1))
            if pos not in body_set:
                return pos
        return state.food
    
    def _collect_explored(self, node: TreeNode, explored: Set[Position]):
        """Recursively collect all explored positions."""
        explored.add(node.state.head)
        for child in node.children.values():
            self._collect_explored(child, explored)
    
    def get_tree_stats(self, root: TreeNode) -> Dict:
        """Get statistics about the search tree (for debugging)."""
        stats = {
            'root_visits': root.visits,
            'children': {}
        }
        for action, child in root.children.items():
            stats['children'][action] = {
                'visits': child.visits,
                'avg_reward': child.total_reward / child.visits if child.visits > 0 else 0,
                'ucb1': child.ucb1_score(self.exploration_weight) if child.visits > 0 else float('inf')
            }
        return stats
