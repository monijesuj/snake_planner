
import torch

from .base_planner import PlanningResult
from ..config import Direction
from ..rl.dqn_agent import DQNAgent
from ..rl.snake_env_rl import SnakeRLEnv


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


class DQNPlanner:
    name = "DQN (Learned Policy)"

    def __init__(self, game):
        self.game = game
        self.env = SnakeRLEnv(game.config)
        self.env.game = game  # share the same game instance

        # input and output dimensions
        state_dim = 11
        action_dim = 3  # straight, right, left

        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim
        )

        checkpoint = torch.load("./models/dqn_snake.pt", map_location="cpu")
        self.agent.q_net.load_state_dict(checkpoint)
        self.agent.q_net.eval()

    def plan(self, start, goal, obstacles):
        """
        DQN returns a single-step policy action.
        """

        # Build 11D state from current game
        state = self.env.build_state_from_game(start, goal, obstacles, self.game.snake.direction)


        action = self.agent.act(state, epsilon=0)

        # Current heading from REAL game
        current_dir = self.game.snake.direction

        # Convert relative action to absolute direction
        if action == 0:          # straight
            new_dir = current_dir
        elif action == 1:        # right
            new_dir = TURN_RIGHT[current_dir]
        elif action == 2:        # left
            new_dir = TURN_LEFT[current_dir]
        else:
            raise ValueError(f"Invalid DQN action: {action}")

        dx, dy = new_dir.value
        next_pos = (start[0] + dx, start[1] + dy)
        print("DQN action:", action, "Direction:", new_dir)


        return PlanningResult(
            success=True,
            path=[start, next_pos],
            explored_nodes=set()
        )
