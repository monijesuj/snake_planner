# Snake Planning Algorithm Comparison

A graduate-level planning algorithms project that implements a Snake game where the robot (snake head) must navigate to destinations while avoiding its own tail (dynamic obstacle).

## Project Overview

This project compares different path planning algorithms in a discrete grid environment inspired by the classic Nokia Snake game. The snake's tail follows the exact path of the head and grows by 1 segment each time a destination is reached.

## Features

- **Discrete Grid Environment**: Configurable grid size (default 20x20)
- **4-Directional Movement**: Up, Down, Left, Right
- **Dynamic Obstacles**: Snake tail acts as obstacle
- **Multiple Planning Algorithms**: 
  - A* (A-star)
  - Dijkstra
  - RRT (Rapidly-exploring Random Tree)
  - BFS (Breadth-First Search)
- **Real-time Visualization**: Pygame-based animation
- **Metrics Comparison**: Path length, nodes expanded, computation time, score

## Installation

```bash
cd snake_planner
pip install -r requirements.txt
```

## Usage

```bash
# Run with A* algorithm (default)
python main.py

# Run with specific algorithm
python main.py --algorithm astar
python main.py --algorithm dijkstra
python main.py --algorithm rrt
python main.py --algorithm bfs

# Custom grid size
python main.py --grid-size 30

# Custom speed (milliseconds per move)
python main.py --speed 100

# Run comparison of all algorithms
python main.py --compare
```

## Project Structure

```
snake_planner/
├── main.py                 # Entry point with CLI
├── config.py               # Configuration parameters
├── game/
│   ├── __init__.py
│   ├── snake.py            # Snake class (head + tail management)
│   ├── environment.py      # Grid environment
│   └── game_state.py       # Game state management
├── planners/
│   ├── __init__.py
│   ├── base_planner.py     # Abstract base class for planners
│   ├── astar.py            # A* algorithm
│   ├── dijkstra.py         # Dijkstra's algorithm
│   ├── rrt.py              # RRT algorithm (grid-adapted)
│   └── bfs.py              # Breadth-First Search
├── visualization/
│   ├── __init__.py
│   └── renderer.py         # Pygame visualization
├── metrics/
│   ├── __init__.py
│   └── tracker.py          # Metrics collection and comparison
└── requirements.txt
```

## Algorithms

### A* (A-star)
Heuristic-based search using Manhattan distance. Guarantees optimal path.

### Dijkstra
Classic shortest path algorithm. Optimal but explores more nodes than A*.

### RRT (Rapidly-exploring Random Tree)
Sampling-based algorithm adapted for grid. Not guaranteed optimal but interesting for comparison.

### BFS (Breadth-First Search)
Baseline algorithm. Guarantees shortest path on unweighted grid.

## Metrics Tracked

- **Path Length**: Number of steps in planned path
- **Nodes Expanded**: Cells visited during planning
- **Computation Time**: Time to compute path (ms)
- **Score**: Destinations reached before game over
- **Success Rate**: Percentage of successful path plans

## Controls

- **ESC**: Quit game
- **SPACE**: Pause/Resume
- **R**: Restart game

## Author

Graduate Planning Algorithms Course Project
