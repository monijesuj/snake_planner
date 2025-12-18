import random
from typing import Set, Optional
from snake_planner.common.geometry import Position, get_neighbors, manhattan_distance


class Environment:
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.destination: Optional[Position] = None

    def is_valid_position(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def is_free(self, pos: Position, obstacles: Set[Position]) -> bool:
        return self.is_valid_position(pos) and pos not in obstacles

    def spawn_destination(self, occupied: Set[Position]) -> Position:
        # Optimized selection for sparsely populated grids
        for _ in range(100):
            pos = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if pos not in occupied:
                self.destination = pos
                return pos

        # Fallback for dense grids
        free = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        if not free:
            raise RuntimeError("No free positions!")
        self.destination = random.choice(free)
        return self.destination
