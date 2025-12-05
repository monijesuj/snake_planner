"""
Grid environment for the snake planning problem.
"""

import random
from typing import Set, Tuple, Optional, List
from config import Direction


Position = Tuple[int, int]


class Environment:
    """
    Discrete grid environment.
    
    Manages the grid, obstacles (snake tail), and destination placement.
    """
    
    def __init__(self, grid_size: int = 20):
        """
        Initialize environment.
        
        Args:
            grid_size: Size of the NxN grid
        """
        self.grid_size = grid_size
        self.destination: Optional[Position] = None
    
    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
    
    def is_free(self, pos: Position, obstacles: Set[Position]) -> bool:
        """
        Check if position is valid and not occupied.
        
        Args:
            pos: Position to check
            obstacles: Set of obstacle positions (e.g., snake body)
            
        Returns:
            True if position is free
        """
        return self.is_valid_position(pos) and pos not in obstacles
    
    def get_neighbors(self, pos: Position, obstacles: Set[Position]) -> List[Position]:
        """
        Get valid neighboring positions (4-directional).
        
        Args:
            pos: Current position
            obstacles: Set of obstacle positions
            
        Returns:
            List of valid neighbor positions
        """
        neighbors = []
        for direction in Direction:
            dx, dy = direction.value
            neighbor = (pos[0] + dx, pos[1] + dy)
            if self.is_free(neighbor, obstacles):
                neighbors.append(neighbor)
        return neighbors
    
    def get_all_neighbors(self, pos: Position) -> List[Position]:
        """
        Get all valid neighboring positions (ignoring obstacles).
        
        Args:
            pos: Current position
            
        Returns:
            List of valid neighbor positions within bounds
        """
        neighbors = []
        for direction in Direction:
            dx, dy = direction.value
            neighbor = (pos[0] + dx, pos[1] + dy)
            if self.is_valid_position(neighbor):
                neighbors.append(neighbor)
        return neighbors
    
    def spawn_destination(self, occupied: Set[Position]) -> Position:
        """
        Spawn a new destination at random free position.
        
        Args:
            occupied: Set of positions that cannot be used (snake body)
            
        Returns:
            New destination position
        """
        free_positions = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        
        if not free_positions:
            raise RuntimeError("No free positions available for destination!")
        
        self.destination = random.choice(free_positions)
        return self.destination
    
    def manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate Euclidean distance between two positions."""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5
    
    def get_random_position(self) -> Position:
        """Get a random position within the grid."""
        return (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1)
        )
    
    def get_random_free_position(self, obstacles: Set[Position]) -> Optional[Position]:
        """
        Get a random free position.
        
        Args:
            obstacles: Set of obstacle positions
            
        Returns:
            Random free position, or None if grid is full
        """
        for _ in range(1000):  # Avoid infinite loop
            pos = self.get_random_position()
            if pos not in obstacles:
                return pos
        return None
    
    def __repr__(self) -> str:
        return f"Environment(grid_size={self.grid_size}, destination={self.destination})"
