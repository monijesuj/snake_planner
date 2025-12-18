import math
from typing import Tuple

Position = Tuple[int, int]

def manhattan_distance(pos1: Position, pos2: Position) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean_distance(pos1: Position, pos2: Position) -> float:
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def get_neighbors(pos: Position, grid_size: int = None) -> list[Position]:
    """Get 4-directional neighbors. Checks bounds if grid_size provided."""
    x, y = pos
    neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    if grid_size is not None:
        return [p for p in neighbors if 0 <= p[0] < grid_size and 0 <= p[1] < grid_size]
    return neighbors