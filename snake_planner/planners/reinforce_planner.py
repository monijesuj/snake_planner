import torch
from .base_planner import PlanningResult
from ..config import Direction
from ..rl.reinforce_agent import ReinforceAgent
from ..rl.snake_env_rl import SnakeRLEnv

# Relative to Absolute mapping
TURN_RIGHT = {
    Direction.UP: Direction.RIGHT,
    Direction.RIGHT: Direction.DOWN,
    Direction.DOWN: Direction.LEFT,
    Direction.LEFT: Direction.UP,
}

TURN_LEFT = {
    Direction.UP: Direction.LEFT,
    Direction.LEFT: Direction.DOWN,
    Direction.DOWN: Direction.RIGHT,
    Direction.RIGHT: Direction.UP,
}


class ReinforcePlanner:
    name = "REINFORCE (Policy Gradient)"

    def __init__(self, game):
        self.game = game
        self.env = SnakeRLEnv(game.config)
        self.env.game = game  # Share the same game instance

        # State: 11, Action: 3
        self.agent = ReinforceAgent(state_dim=11, action_dim=3)

        # Look for the model in the root directory (where train.py saves it)
        try:
            self.agent.load("reinforce_snake.pt")
        except FileNotFoundError:
            print(
                "Warning: reinforce_snake.pt not found. Ensure you have run training."
            )

    def plan(self, start, goal, obstacles):
        """
        Policy Gradient returns a single-step action.
        """
        # Build state from the current perspective
        state = self.env.build_state_from_game(
            start, goal, obstacles, self.game.snake.direction
        )

        # FIXED: Unpack 3 values (action, log_prob, entropy)
        # During planning/inference, we only need the action.
        action, _, _ = self.agent.act(state)

        # Current heading from the REAL game
        current_dir = self.game.snake.direction

        # Convert relative action to absolute grid direction
        if action == 0:  # straight
            new_dir = current_dir
        elif action == 1:  # right
            new_dir = TURN_RIGHT[current_dir]
        elif action == 2:  # left
            new_dir = TURN_LEFT[current_dir]
        else:
            new_dir = current_dir

        dx, dy = new_dir.value
        next_pos = (start[0] + dx, start[1] + dy)

        # Return as a path containing current head and next step
        return PlanningResult(
            success=True, path=[start, next_pos], explored_nodes=set()
        )
