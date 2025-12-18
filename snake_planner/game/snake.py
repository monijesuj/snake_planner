from collections import deque
from enum import Enum
from typing import List, Tuple, Optional
from snake_planner.config import Direction
from snake_planner.common.geometry import Position

class Snake:
    def __init__(self, start_pos: Position, initial_length: int = 3):
        self.initial_length = initial_length
        self.start_pos = start_pos
        self.reset()

    def reset(self):
        self.body: deque = deque()
        x, y = self.start_pos
        # Snake starts horizontally, extending to the left
        for i in range(self.initial_length):
            self.body.append((x - i, y))
        self.grow_pending = 0
        self.direction: Direction = Direction.RIGHT

    @property
    def length(self) -> int:
        """Get current snake length."""
        return len(self.body)

    @property
    def head(self) -> Position:
        """Get head position."""
        return self.body[0]

    def get_body_set(self) -> set:
        """Get set of all body positions."""
        return set(self.body)
    
    def get_tail_set(self) -> set:
        """Returns body parts that count as obstacles (head isn't an obstacle to itself)."""
        return set(list(self.body)[1:])

    def get_future_obstacles(self) -> set:
        """Returns where the body will be after a move (accounts for tail moving)."""
        if self.grow_pending > 0:
            # If growing, the tail tip stays where it is
            return set(self.body)
        # If not growing, the tail tip will move, so it's not an obstacle for the *next* step
        return set(list(self.body)[:-1])

    def move_to(self, position: Position) -> bool:
        """Move snake head to adjacent position."""
        dx, dy = position[0] - self.head[0], position[1] - self.head[1]
        
        # Update direction for visualization
        for d in Direction:
            if d.value == (dx, dy):
                self.direction = d
                break
        
        # Check collision against future state
        if position in self.get_future_obstacles():
            return False

        self.body.appendleft(position)
        
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            self.body.pop() # Remove tail
            
        return True

    def grow(self, amount: int = 1):
        """Schedule growth."""
        self.grow_pending += amount