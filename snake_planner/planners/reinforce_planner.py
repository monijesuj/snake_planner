import torch
from .base_planner import PlanningResult
from ..config import Direction
from ..rl.reinforce_agent import ReinforceAgent
from ..rl.snake_env_rl import SnakeRLEnv

# Relative to Absolute mapping (Mirroring DQN logic)
TURN_RIGHT = {Direction.UP: Direction.RIGHT, Direction.RIGHT: Direction.DOWN, Direction.DOWN: Direction.LEFT, Direction.LEFT: Direction.UP}
TURN_LEFT = {Direction.UP: Direction.LEFT, Direction.LEFT: Direction.DOWN, Direction.DOWN: Direction.RIGHT, Direction.RIGHT: Direction.UP}

class ReinforcePlanner:
    name = "REINFORCE (Policy Gradient)"

    def __init__(self, game):
        self.game = game
        self.env = SnakeRLEnv(game.config)
        self.agent = ReinforceAgent(state_dim=11, action_dim=3)
        
        try:
            self.agent.load("./models/reinforce_snake.pt")
            print("Loaded REINFORCE model.")
        except FileNotFoundError:
            print("Warning: REINFORCE model not found. Using untrained policy.")

    def plan(self, start, goal, obstacles):
        state = self.env.build_state_from_game(start, goal, obstacles, self.game.snake.direction)
        
        # act returns (action, log_prob); we only need the action for inference
        action, _ = self.agent.act(state)
        current_dir = self.game.snake.direction

        if action == 0:    new_dir = current_dir
        elif action == 1:  new_dir = TURN_RIGHT[current_dir]
        elif action == 2:  new_dir = TURN_LEFT[current_dir]
        
        dx, dy = new_dir.value
        next_pos = (start[0] + dx, start[1] + dy)
        
        return PlanningResult(success=True, path=[start, next_pos], explored_nodes=set())