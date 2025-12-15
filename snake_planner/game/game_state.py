"""
Game state management for snake planning.
"""

from typing import Optional, List, Tuple
from enum import Enum

from .snake import Snake
from .environment import Environment
from snake_planner.config import GameConfig, DEFAULT_CONFIG


Position = Tuple[int, int]


class GameStatus(Enum):
    """Current game status."""
    RUNNING = "running"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    PLANNING = "planning"
    NO_PATH = "no_path"


class GameState:
    """
    Manages the complete game state.
    
    Coordinates snake, environment, and game logic.
    """
    
    def __init__(self, config: GameConfig = DEFAULT_CONFIG):
        """
        Initialize game state.
        
        Args:
            config: Game configuration
        """
        self.config = config
        self.environment = Environment(config.grid_size)
        
        # Start snake in the middle-left area
        start_x = config.initial_length + 2
        start_y = config.grid_size // 2
        self.snake = Snake((start_x, start_y), config.initial_length)
        
        # Game state
        self.status = GameStatus.RUNNING
        self.score = 0
        
        # Current planned path
        self.current_path: List[Position] = []
        self.path_index = 0
        
        # Explored nodes (for visualization)
        self.explored_nodes: set = set()
        
        # Spawn first destination
        self.environment.spawn_destination(self.snake.get_body_set())
    
    def reset(self):
        """Reset game to initial state."""
        self.snake.reset()
        self.status = GameStatus.RUNNING
        self.score = 0
        self.current_path = []
        self.path_index = 0
        self.explored_nodes = set()
        self.environment.spawn_destination(self.snake.get_body_set())
    
    @property
    def destination(self) -> Optional[Position]:
        """Get current destination."""
        return self.environment.destination
    
    @property
    def is_running(self) -> bool:
        """Check if game is currently running."""
        return self.status == GameStatus.RUNNING
    
    @property
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.status == GameStatus.GAME_OVER
    
    def set_path(self, path: List[Position], explored: set = None):
        """
        Set the planned path for the snake to follow.
        
        Args:
            path: List of positions from current head to destination
            explored: Set of explored nodes during planning (for visualization)
        """
        self.current_path = path
        self.path_index = 0
        if explored:
            self.explored_nodes = explored
        else:
            self.explored_nodes = set()
    
    def get_next_move(self) -> Optional[Position]:
        """
        Get the next position to move to from the planned path.
        
        Returns:
            Next position, or None if no path or path completed
        """
        if not self.current_path:
            return None
        
        # Path includes current position, so we skip index 0
        if self.path_index + 1 < len(self.current_path):
            self.path_index += 1
            return self.current_path[self.path_index]
        
        return None
    
    def step(self) -> bool:
        """
        Execute one game step.
        
        Returns:
            True if step was successful, False if game over or no move
        """
        if self.status != GameStatus.RUNNING:
            return False
        
        # Get next position from path
        next_pos = self.get_next_move()
        
        if next_pos is None:
            # No path or path completed - need to replan
            return False
        
        # Move snake
        if not self.snake.move_to(next_pos):
            # Collision with self - game over
            self.status = GameStatus.GAME_OVER
            return False
        
        # Check bounds
        if not self.environment.is_valid_position(self.snake.head):
            self.status = GameStatus.GAME_OVER
            return False
        
        # Check if reached destination
        if self.snake.head == self.destination:
            self.score += 1
            self.snake.grow(self.config.growth_per_food)
            
            # Spawn new destination
            try:
                self.environment.spawn_destination(self.snake.get_body_set())
            except RuntimeError:
                # No space left - player wins!
                self.status = GameStatus.GAME_OVER
                return False
            
            # Clear path - will trigger replanning
            self.current_path = []
            self.path_index = 0
            self.explored_nodes = set()
        
        return True
    
    def needs_replanning(self) -> bool:
        """Check if path needs to be replanned."""
        return (
            self.status == GameStatus.RUNNING and
            (not self.current_path or self.path_index >= len(self.current_path) - 1)
        )
    
    def get_obstacles(self) -> set:
        """
        Get current obstacles (snake tail) for planning.
        
        Note: We use tail only (not head) since head will move away.
        """
        return self.snake.get_tail_set()
    
    def toggle_pause(self):
        """Toggle pause state."""
        if self.status == GameStatus.RUNNING:
            self.status = GameStatus.PAUSED
        elif self.status == GameStatus.PAUSED:
            self.status = GameStatus.RUNNING
    
    def __repr__(self) -> str:
        return (
            f"GameState(status={self.status.value}, score={self.score}, "
            f"snake_length={self.snake.length}, path_len={len(self.current_path)})"
        )
