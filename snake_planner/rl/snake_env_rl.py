import numpy as np
from typing import Tuple, Optional, Set
from snake_planner.game import GameState
from snake_planner.config import GameConfig, Direction
from snake_planner.common.geometry import Position, manhattan_distance

class SnakeRLEnv:
    def __init__(self, config: GameConfig):
        self.config = config
        self.game = GameState(config)

    def reset(self) -> np.ndarray:
        self.game.reset()
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        old_dist = manhattan_distance(self.game.snake.head, self.game.destination)
        # Mapping 0: Straight, 1: Right, 2: Left
        clock_dirs = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        curr_idx = clock_dirs.index(self.game.snake.direction)
        
        if action == 1: # Right
            new_dir = clock_dirs[(curr_idx + 1) % 4]
        elif action == 2: # Left
            new_dir = clock_dirs[(curr_idx - 1) % 4]
        else:
            new_dir = self.game.snake.direction

        dx, dy = new_dir.value
        head = self.game.snake.head
        next_pos = (head[0] + dx, head[1] + dy)

        # Physics/Rule Check delegated to Game/Environment
        if not self.game.environment.is_valid_position(next_pos):
            return self._get_state(), -10.0, True # Wall
            
        if next_pos in self.game.snake.get_future_obstacles():
            return self._get_state(), -10.0, True # Self

        # 3. Execute Move
        self.game.snake.move_to(next_pos)
        
        # 4. Calculate New Distance
        new_dist = manhattan_distance(next_pos, self.game.destination)
        
        # 5. SHAPED REWARD
        # Base penalty for living
        reward = -0.1 
        
        # Bonus for getting closer to food, penalty for moving away
        if new_dist < old_dist:
            reward += 0.2  # Encouragement
        else:
            reward -= 0.2  # Discouragement
            
        if next_pos == self.game.destination:
            reward = 20.0  # Increased from 10.0
            self.game.score += 1
            self.game.snake.grow(self.config.growth_per_food)
            try:
                self.game.environment.spawn_destination(self.game.snake.get_body_set())
            except RuntimeError:
                return self._get_state(), 50.0, True # Big win bonus

        return self._get_state(), reward, False

    def _get_state(self, head=None, food=None, direction=None, obstacles=None) -> np.ndarray:
        """
        Generate state vector. 
        Arguments allow overriding current game state for planning/simulation.
        """
        # Resolve defaults if not provided
        if head is None: head = self.game.snake.head
        if food is None: food = self.game.destination
        if direction is None: direction = self.game.snake.direction
        if obstacles is None: obstacles = self.game.snake.get_body_set()

        # Helper to check safety of a generic point
        def is_unsafe(pos):
            return (not self.game.environment.is_valid_position(pos) or 
                    pos in obstacles)

        # Determine absolute positions of neighbors relative to head
        dx, dy = direction.value
        
        # Standard relative checks
        straight = (head[0] + dx, head[1] + dy)
        right = (head[0] - dy, head[1] + dx) # 90 deg rotation vector math: (x,y) -> (-y, x)
        left = (head[0] + dy, head[1] - dx)  # 90 deg rotation vector math: (x,y) -> (y, -x)

        state = [
            is_unsafe(straight), is_unsafe(right), is_unsafe(left),
            direction == Direction.LEFT,
            direction == Direction.RIGHT,
            direction == Direction.UP,
            direction == Direction.DOWN,
            food[0] < head[0], food[0] > head[0],
            food[1] < head[1], food[1] > head[1]
        ]
        return np.array(state, dtype=np.float32)

    def build_state_from_game(self, head, goal, obstacles, current_dir):
        """
        Public helper for DQNPlanner to get a state vector for a hypothetical scenario
        without mutating the actual game state.
        """
        return self._get_state(head=head, food=goal, direction=current_dir, obstacles=obstacles)