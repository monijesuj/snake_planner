
import numpy as np
from typing import Tuple

from ..game import GameState
from ..config import GameConfig, Direction


class SnakeRLEnv:
    """
    RL wrapper around the existing Snake game logic
    using an 11-dimensional state representation and
    relative action space (straight, right, left).
    """

    # Relative actions
    ACTION_STRAIGHT = 0
    ACTION_RIGHT = 1
    ACTION_LEFT = 2

    def __init__(self, config: GameConfig):
        self.config = config
        self.game = GameState(config)

        # Initial direction (arbitrary, will be updated)
        self.current_direction = Direction.RIGHT

    # -----------------------
    # Core RL API
    # -----------------------

    def reset(self) -> np.ndarray:
        self.game.reset()
        self.current_direction = Direction.RIGHT
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one relative action.
        Returns: next_state, reward, done
        """

        # Convert relative action:  absolute direction
        new_direction = self._relative_to_absolute(action)
        self.current_direction = new_direction

        dx, dy = new_direction.value
        hx, hy = self.game.snake.head
        next_pos = (hx + dx, hy + dy)

        # Default step penalty
        reward = -0.1
        done = False

        # Wall collision
        if not self.game.environment.is_valid_position(next_pos):
            return self._get_state(), -10.0, True

        # Self collision
        if next_pos in self.game.snake.get_body_set():
            return self._get_state(), -10.0, True

        # Move snake
        self.game.snake.move_to(next_pos)

        # Food
        if next_pos == self.game.destination:
            self.game.score += 1
            self.game.snake.grow(self.config.growth_per_food)
            self.game.environment.spawn_destination(
                self.game.snake.get_body_set()
            )
            reward = 10.0

        return self._get_state(), reward, done

    # -----------------------
    # State Representation
    # -----------------------

    def _get_state(self) -> np.ndarray:
        """
        11D state vector:
        [
            danger_straight, danger_right, danger_left,
            moving_left, moving_right, moving_up, moving_down,
            food_left, food_right, food_up, food_down
        ]
        """

        head_x, head_y = self.game.snake.head
        food_x, food_y = self.game.destination

        # Direction flags
        dir_left = self.current_direction == Direction.LEFT
        dir_right = self.current_direction == Direction.RIGHT
        dir_up = self.current_direction == Direction.UP
        dir_down = self.current_direction == Direction.DOWN

        # Danger detection
        danger_straight = self._is_danger(self.current_direction)
        danger_right = self._is_danger(self._turn_right(self.current_direction))
        danger_left = self._is_danger(self._turn_left(self.current_direction))

        state = np.array([
            danger_straight,
            danger_right,
            danger_left,

            dir_left,
            dir_right,
            dir_up,
            dir_down,

            food_x < head_x,   # food left
            food_x > head_x,   # food right
            food_y < head_y,   # food up
            food_y > head_y    # food down
        ], dtype=np.float32)

        return state

    # -----------------------
    # Helpers
    # -----------------------

    def _is_danger(self, direction: Direction) -> float:
        dx, dy = direction.value
        hx, hy = self.game.snake.head
        next_pos = (hx + dx, hy + dy)

        if not self.game.environment.is_valid_position(next_pos):
            return 1.0
        if next_pos in self.game.snake.get_body_set():
            return 1.0
        return 0.0

    def _relative_to_absolute(self, action: int) -> Direction:
        if action == self.ACTION_STRAIGHT:
            return self.current_direction
        elif action == self.ACTION_RIGHT:
            return self._turn_right(self.current_direction)
        elif action == self.ACTION_LEFT:
            return self._turn_left(self.current_direction)
        else:
            raise ValueError("Invalid action")

    @staticmethod
    def _turn_right(direction: Direction) -> Direction:
        mapping = {
            Direction.UP: Direction.RIGHT,
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP
        }
        return mapping[direction]

    @staticmethod
    def _turn_left(direction: Direction) -> Direction:
        mapping = {
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
            Direction.RIGHT: Direction.UP
        }
        return mapping[direction]

    # -----------------------
    # Compatibility with planner
    # -----------------------
    def build_state_from_game(self, head, food, obstacles, current_direction):
        hx, hy = head
        fx, fy = food

        # Direction flags
        dir_left = current_direction == Direction.LEFT
        dir_right = current_direction == Direction.RIGHT
        dir_up = current_direction == Direction.UP
        dir_down = current_direction == Direction.DOWN

        danger_straight = self._is_danger(current_direction)
        danger_right = self._is_danger(self._turn_right(current_direction))
        danger_left = self._is_danger(self._turn_left(current_direction))

        state = np.array([
            danger_straight,
            danger_right,
            danger_left,
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            fx < hx,
            fx > hx,
            fy < hy,
            fy > hy
        ], dtype=np.float32)

        return state

    # def sync_from_game(self, head, goal, obstacles):
    #     """
    #     Sync environment with external game state (used by planner).
    #     """
    #     self.game.snake.head = head
    #     self.game.destination = goal
