"""
Configuration parameters for the Snake Planning project.
"""

from dataclasses import dataclass
from enum import Enum


class Algorithm(Enum):
    """Available planning algorithms."""
    ASTAR = "astar"
    DIJKSTRA = "dijkstra"
    RRT = "rrt"
    BFS = "bfs"


class Direction(Enum):
    """Movement directions (4-directional)."""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    
    @property
    def opposite(self):
        """Return the opposite direction."""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites[self]


@dataclass
class GameConfig:
    """Game configuration parameters."""
    # Grid settings
    grid_size: int = 20  # NxN grid
    cell_size: int = 30  # Pixels per cell
    
    # Snake settings
    initial_length: int = 3  # Starting snake length
    growth_per_food: int = 1  # Tail growth when reaching destination
    
    # Game settings
    speed: int = 150  # Milliseconds per move
    
    # Display settings
    window_padding: int = 50  # Extra space for metrics display
    fps: int = 60
    
    @property
    def window_width(self) -> int:
        return self.grid_size * self.cell_size + self.window_padding * 2
    
    @property
    def window_height(self) -> int:
        return self.grid_size * self.cell_size + self.window_padding + 150  # Extra space for metrics


@dataclass
class Colors:
    """Color definitions for rendering."""
    BACKGROUND = (40, 44, 52)  # Dark gray
    GRID_LINE = (60, 64, 72)  # Slightly lighter
    
    SNAKE_HEAD = (97, 175, 239)  # Blue
    SNAKE_BODY = (152, 195, 121)  # Green
    SNAKE_TAIL = (229, 192, 123)  # Yellow/gold
    
    DESTINATION = (224, 108, 117)  # Red/coral
    
    PATH_PLANNED = (198, 120, 221)  # Purple (planned path)
    PATH_EXPLORED = (86, 182, 194)  # Cyan (explored nodes)
    
    TEXT = (255, 255, 255)  # White
    TEXT_DIM = (150, 150, 150)  # Gray
    
    GAME_OVER = (224, 108, 117)  # Red


# Default configuration instance
DEFAULT_CONFIG = GameConfig()
