"""
Snake class representing the robot with head and tail.
"""

from typing import List, Tuple, Optional
from collections import deque
from config import Direction


Position = Tuple[int, int]


class Snake:
    """
    Snake robot with head and tail.
    
    The head moves according to planned path, and the tail follows
    the exact path the head took.
    """
    
    def __init__(self, start_pos: Position, initial_length: int = 3):
        """
        Initialize snake at starting position.
        
        Args:
            start_pos: (x, y) starting position for the head
            initial_length: Initial length of the snake
        """
        self.initial_length = initial_length
        self.start_pos = start_pos
        self.reset()
    
    def reset(self):
        """Reset snake to initial state."""
        # Body is a deque where index 0 is the head
        # Snake starts horizontally, extending to the left
        self.body: deque = deque()
        x, y = self.start_pos
        for i in range(self.initial_length):
            self.body.append((x - i, y))
        
        self.grow_pending = 0  # Segments to grow
        self.direction: Optional[Direction] = Direction.RIGHT
    
    @property
    def head(self) -> Position:
        """Get head position."""
        return self.body[0]
    
    @property
    def tail(self) -> List[Position]:
        """Get tail positions (everything except head)."""
        return list(self.body)[1:]
    
    @property
    def length(self) -> int:
        """Get current snake length."""
        return len(self.body)
    
    def get_body_set(self) -> set:
        """Get set of all body positions for fast collision checking."""
        return set(self.body)
    
    def get_tail_set(self) -> set:
        """Get set of tail positions (excluding head) for collision checking."""
        return set(list(self.body)[1:])
    
    def move(self, direction: Direction) -> bool:
        """
        Move snake one step in the given direction.
        
        Args:
            direction: Direction to move
            
        Returns:
            True if move was valid (no self-collision), False otherwise
        """
        # Calculate new head position
        dx, dy = direction.value
        old_head = self.head
        new_head = (old_head[0] + dx, old_head[1] + dy)
        
        # Check for self-collision BEFORE moving
        # We need to check against where the tail WILL be after the move
        # If we're not growing, the last segment will be removed
        future_tail = set(list(self.body)[:-1]) if self.grow_pending == 0 else set(self.body)
        
        if new_head in future_tail:
            return False  # Collision with own body
        
        # Add new head
        self.body.appendleft(new_head)
        
        # Handle tail: grow or maintain length
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            self.body.pop()  # Remove last segment
        
        self.direction = direction
        return True
    
    def move_to(self, position: Position) -> bool:
        """
        Move snake head to adjacent position.
        
        Args:
            position: Target position (must be adjacent to head)
            
        Returns:
            True if move was valid, False otherwise
        """
        # Determine direction from current head to target
        dx = position[0] - self.head[0]
        dy = position[1] - self.head[1]
        
        # Find matching direction
        for direction in Direction:
            if direction.value == (dx, dy):
                return self.move(direction)
        
        return False  # Invalid move (not adjacent)
    
    def grow(self, amount: int = 1):
        """
        Schedule snake to grow by given amount.
        
        Args:
            amount: Number of segments to add
        """
        self.grow_pending += amount
    
    def will_collide(self, position: Position) -> bool:
        """
        Check if moving to position would cause self-collision.
        
        Args:
            position: Position to check
            
        Returns:
            True if collision would occur
        """
        # Future tail after one move (last segment removed if not growing)
        if self.grow_pending > 0:
            future_tail = set(self.body)
        else:
            future_tail = set(list(self.body)[:-1])
        
        return position in future_tail
    
    def get_valid_moves(self, grid_size: int) -> List[Position]:
        """
        Get list of valid adjacent positions the head can move to.
        
        Args:
            grid_size: Size of the grid (for boundary checking)
            
        Returns:
            List of valid positions
        """
        valid = []
        for direction in Direction:
            dx, dy = direction.value
            new_pos = (self.head[0] + dx, self.head[1] + dy)
            
            # Check bounds
            if 0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size:
                # Check self-collision
                if not self.will_collide(new_pos):
                    valid.append(new_pos)
        
        return valid
    
    def __repr__(self) -> str:
        return f"Snake(head={self.head}, length={self.length}, grow_pending={self.grow_pending})"
