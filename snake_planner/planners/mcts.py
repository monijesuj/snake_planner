"""
Monte Carlo Tree Search (MCTS) path planning algorithm.

MCTS simulates many possible futures to find the move that maximizes
long-term survival and food collection.
"""

import math
import random
import time
from typing import Set, Tuple, List, Optional, Dict
from collections import deque
from dataclasses import dataclass

from .base_planner import BasePlanner, PlanningResult


Position = Tuple[int, int]

# Directions: up, down, left, right
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


@dataclass
class SimulationState:
    """Represents a simulated game state."""
    head: Position
    body: List[Position]  # List of body positions (head is body[0])
    food: Position
    grid_size: int
    
    def copy(self) -> 'SimulationState':
        return SimulationState(
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
        """Get list of valid next positions. Disallow moving directly into the neck (backwards)."""
        valid = []
        # neck is the second segment (body[1]) when snake length > 1
        neck = self.body[1] if len(self.body) > 1 else None

        for dx, dy in DIRECTIONS:
            new_pos = (self.head[0] + dx, self.head[1] + dy)
            # Disallow moving into the neck (backwards)
            if neck is not None and new_pos == neck:
                continue
            # Check bounds
            if not self.is_valid_pos(new_pos):
                continue
            # Check self-collision (excluding tail tip which will move)
            future_body = set(self.body[:-1]) if len(self.body) > 1 else set()
            if new_pos in future_body:
                continue
            valid.append(new_pos)

        return valid
    
    def move(self, new_head: Position) -> bool:
        """
        Move snake to new position.
        Returns True if food was eaten.
        """
        ate_food = (new_head == self.food)
        
        # Add new head
        self.body.insert(0, new_head)
        self.head = new_head
        
        # Remove tail unless we ate food
        if not ate_food:
            self.body.pop()
        
        return ate_food


class MCTSNode:
    """A node in the MCTS tree."""
    
    def __init__(self, state: SimulationState, parent: 'MCTSNode' = None, 
                 move: Position = None):
        self.state = state
        self.parent = parent
        self.move = move  # The move that led to this state
        self.children: Dict[Position, 'MCTSNode'] = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves: List[Position] = state.get_valid_moves()
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        return len(self.state.get_valid_moves()) == 0
    
    def ucb1(self, exploration_weight: float = 1.41) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def best_child(self, exploration_weight: float = 1.41) -> 'MCTSNode':
        """Select child with highest UCB1 score."""
        return max(
            self.children.values(),
            key=lambda c: c.ucb1(exploration_weight)
        )
    
    def best_move(self) -> Position:
        """Return the move with the highest visit count (most robust)."""
        if not self.children:
            return None
        return max(
            self.children.items(),
            key=lambda item: item[1].visits
        )[0]


class MCTSPlanner(BasePlanner):
    """
    Monte Carlo Tree Search planner.
    
    Uses MCTS to simulate many possible futures and select the move
    that leads to the best long-term outcomes.
    """
    
    def __init__(self, grid_size: int, 
                 num_simulations: int = 200,
                 max_simulation_depth: int = 30,
                 exploration_weight: float = 1.0):
        """
        Initialize MCTS planner.
        
        Args:
            grid_size: Size of the NxN grid
            num_simulations: Number of MCTS simulations per planning call
            max_simulation_depth: Maximum depth of each simulation
            exploration_weight: UCB1 exploration parameter (higher = more exploration)
        """
        super().__init__(grid_size)
        self.name = "MCTS"
        self.num_simulations = num_simulations
        self.max_simulation_depth = max_simulation_depth
        self.exploration_weight = exploration_weight
        
        # Store snake body for simulation
        self.snake_body: List[Position] = []
    
    def set_snake_body(self, body: List[Position]):
        """Set the current snake body for simulation."""
        self.snake_body = list(body)
    
    def _create_initial_state(self, start: Position, goal: Position, 
                               obstacles: Set[Position]) -> SimulationState:
        """Create initial simulation state from current game state."""
        # Reconstruct body from obstacles + head
        if self.snake_body:
            body = self.snake_body.copy()
        else:
            # Fallback: head + obstacles as body
            body = [start] + list(obstacles)
        
        return SimulationState(
            head=start,
            body=body,
            food=goal,
            grid_size=self.grid_size
        )
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCB1."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child(self.exploration_weight)
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase: add a new child node."""
        if not node.untried_moves:
            return node
        
        # Pick a random untried move
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        
        # Create new state
        new_state = node.state.copy()
        new_state.move(move)
        
        # If food was eaten, spawn new food
        if new_state.head == new_state.food:
            new_state.food = self._spawn_food(new_state)
        
        # Create child node
        child = MCTSNode(new_state, parent=node, move=move)
        node.children[move] = child
        
        return child
    
    def _simulate(self, state: SimulationState) -> float:
        """
        Simulation phase: play out moves and return reward.
        
        Heavily rewards:
        1. Eating food (huge bonus)
        2. Getting closer to food with FIRST move
        3. Survival
        """
        sim_state = state.copy()
        food_collected = 0
        moves_survived = 0
        
        # Track distance to food at start
        initial_dist = abs(sim_state.head[0] - sim_state.food[0]) + abs(sim_state.head[1] - sim_state.food[1])
        first_move_dist = initial_dist  # Will be updated after first move
        
        for step in range(self.max_simulation_depth):
            valid_moves = sim_state.get_valid_moves()
            
            if not valid_moves:
                # Dead
                break
            
            # Smart random: strongly prefer moves toward food
            move = self._smart_random_move(sim_state, valid_moves)
            
            ate_food = sim_state.move(move)
            moves_survived += 1
            
            # Track distance after first move (most important!)
            if step == 0:
                first_move_dist = abs(sim_state.head[0] - sim_state.food[0]) + abs(sim_state.head[1] - sim_state.food[1])
            
            if ate_food:
                food_collected += 1
                # Spawn new food
                sim_state.food = self._spawn_food(sim_state)
                # Early exit bonus - we got food!
                break
        
        # REWARD CALCULATION
        reward = 0.0
        
        # 1. HUGE reward for eating food
        if food_collected > 0:
            reward += 100.0
        
        # 2. BIG reward/penalty for first move direction
        # This is the KEY - the first move determines if we're going toward food
        first_move_improvement = initial_dist - first_move_dist
        reward += first_move_improvement * 20.0  # +20 for getting closer, -20 for farther
        
        # 3. Smaller reward for final distance (if no food collected)
        if food_collected == 0:
            final_dist = abs(sim_state.head[0] - sim_state.food[0]) + abs(sim_state.head[1] - sim_state.food[1])
            total_improvement = initial_dist - final_dist
            reward += total_improvement * 2.0
        
        # 4. Small survival bonus
        reward += moves_survived * 0.1
        
        # Normalize to [0, 1]
        # Max possible: 100 + 20 + 40 + 3 = ~163
        max_reward = 163.0
        normalized = (reward + 40.0) / max_reward  # Shift to handle negative
        return max(0.0, min(1.0, normalized))
    
    def _smart_random_move(self, state: SimulationState, 
                           valid_moves: List[Position]) -> Position:
        """
        Choose a move with some intelligence.
        Strongly biased toward food.
        """
        if not valid_moves:
            return None
        
        # Only 10% chance of pure random (was 30%)
        if random.random() < 0.1:
            return random.choice(valid_moves)
        
        # Score each move
        best_move = valid_moves[0]
        best_score = -1000
        
        for move in valid_moves:
            score = 0
            
            # STRONGLY prefer moves closer to food
            food_dist = abs(move[0] - state.food[0]) + abs(move[1] - state.food[1])
            current_dist = abs(state.head[0] - state.food[0]) + abs(state.head[1] - state.food[1])
            
            # Big bonus for getting closer
            if food_dist < current_dist:
                score += 10
            elif food_dist > current_dist:
                score -= 5
            
            # Huge bonus if this move gets the food!
            if move == state.food:
                score += 100
            
            # Penalize moves with few escape routes (avoid traps)
            temp_state = state.copy()
            temp_state.move(move)
            future_moves = len(temp_state.get_valid_moves())
            score += future_moves * 2
            
            # Light penalty for walls
            x, y = move
            if x == 0 or x == state.grid_size - 1:
                score -= 1
            if y == 0 or y == state.grid_size - 1:
                score -= 1
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _spawn_food(self, state: SimulationState) -> Position:
        """Spawn food in a random free position."""
        body_set = state.get_body_set()
        attempts = 0
        while attempts < 100:
            x = random.randint(0, state.grid_size - 1)
            y = random.randint(0, state.grid_size - 1)
            if (x, y) not in body_set:
                return (x, y)
            attempts += 1
        # Fallback: return current food position
        return state.food
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagation phase: update statistics up the tree."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def plan(
        self,
        start: Position,
        goal: Position,
        obstacles: Set[Position],
        snake_length: int = 3
    ) -> PlanningResult:
        """
        Plan the next move using MCTS.
        
        Simplified approach: directly evaluate each possible move
        by running simulations from each move's resulting state.
        """
        start_time = time.perf_counter()
        
        # Create initial state
        initial_state = self._create_initial_state(start, goal, obstacles)
        
        # Check if any moves are possible
        valid_moves = initial_state.get_valid_moves()
        if not valid_moves:
            return PlanningResult(
                success=False,
                computation_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # If only one move, just take it
        if len(valid_moves) == 1:
            return PlanningResult(
                success=True,
                path=[start, valid_moves[0]],
                explored_nodes={start, valid_moves[0]},
                nodes_expanded=1,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
                path_length=1
            )
        
        # Check if food is directly reachable in one move
        for move in valid_moves:
            if move == goal:
                return PlanningResult(
                    success=True,
                    path=[start, move],
                    explored_nodes={start, move},
                    nodes_expanded=1,
                    computation_time_ms=(time.perf_counter() - start_time) * 1000,
                    path_length=1
                )
        
        # SIMPLIFIED MCTS: Run simulations directly for each possible move
        # and pick the move with the best average outcome
        move_scores = {}
        sims_per_move = self.num_simulations // len(valid_moves)
        
        for move in valid_moves:
            # Create state after making this move
            state_after_move = initial_state.copy()
            ate_food = state_after_move.move(move)
            
            # If this move eats food, huge bonus
            if ate_food:
                move_scores[move] = 1000.0
                continue
            
            # Run simulations from this state
            total_reward = 0.0
            for _ in range(sims_per_move):
                reward = self._simulate_from_state(state_after_move, goal)
                total_reward += reward
            
            # Also add immediate heuristic: distance to food
            dist_to_food = abs(move[0] - goal[0]) + abs(move[1] - goal[1])
            current_dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            
            # Bonus for getting closer to food
            direction_bonus = (current_dist - dist_to_food) * 10.0
            
            # Check if move leads to trap (few future moves)
            future_moves = len(state_after_move.get_valid_moves())
            safety_bonus = future_moves * 5.0
            
            avg_reward = total_reward / sims_per_move if sims_per_move > 0 else 0
            move_scores[move] = avg_reward + direction_bonus + safety_bonus
        
        # Pick the best move
        best_move = max(move_scores.keys(), key=lambda m: move_scores[m])
        
        end_time = time.perf_counter()
        
        return PlanningResult(
            success=True,
            path=[start, best_move],
            explored_nodes={start} | set(valid_moves),
            nodes_expanded=self.num_simulations,
            computation_time_ms=(end_time - start_time) * 1000,
            path_length=1
        )
    
    def _simulate_from_state(self, state: SimulationState, original_goal: Position) -> float:
        """
        Run a simulation from the given state and return reward.
        """
        sim_state = state.copy()
        food_collected = 0
        moves_survived = 0
        
        initial_dist = abs(sim_state.head[0] - sim_state.food[0]) + abs(sim_state.head[1] - sim_state.food[1])
        
        for _ in range(self.max_simulation_depth):
            valid_moves = sim_state.get_valid_moves()
            
            if not valid_moves:
                # Dead - penalize
                return -10.0 + moves_survived * 0.1
            
            # Pick move toward food (greedy with some randomness)
            move = self._smart_random_move(sim_state, valid_moves)
            
            ate_food = sim_state.move(move)
            moves_survived += 1
            
            if ate_food:
                food_collected += 1
                sim_state.food = self._spawn_food(sim_state)
        
        # Reward calculation
        reward = food_collected * 50.0
        
        # Distance improvement
        final_dist = abs(sim_state.head[0] - sim_state.food[0]) + abs(sim_state.head[1] - sim_state.food[1])
        reward += (initial_dist - final_dist) * 1.0
        
        # Survival bonus
        reward += moves_survived * 0.5
        
        return reward
    
    def get_move_statistics(self, root: MCTSNode) -> Dict[Position, Dict]:
        """Get statistics for each possible move (for debugging)."""
        stats = {}
        for move, child in root.children.items():
            stats[move] = {
                'visits': child.visits,
                'avg_reward': child.total_reward / child.visits if child.visits > 0 else 0,
                'ucb1': child.ucb1() if child.visits > 0 else float('inf')
            }
        return stats
