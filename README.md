# Snake Planning Algorithm Comparison

A graduate-level planning algorithms project that implements a Snake game where the robot (snake head) must navigate to destinations while avoiding its own tail (dynamic obstacle).

## Demos

Below are short recordings of the planners and gameplay. Large media are tracked with Git LFS.

<table>
  <tr>
    <td><img src="media/snake_astar_20251212_191217.gif" alt="A* Demo" style="max-width:100%; height:auto;"></td>
    <td><img src="media/snake_survival_20251212_193258.gif" alt="Survival Demo" style="max-width:100%; height:auto;"></td>
  </tr>
  <tr>
    <td><img src="media/snake_dqn_20251218_175533.gif" alt="DQN Demo" style="max-width:100%; height:auto;"></td>
    <td><img src="media/snake_mcts_20251218_175753.gif" alt="Monte Carlo Demo" style="max-width:100%; height:auto;"></td>
  </tr>
</table>

<details>
<summary>Optional video (MP4)</summary>

[Survival run (MP4)](media/snake_survival_20251212_155349.mp4)

</details>

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
# Option 1: requirements.txt (simple)
pip install -r requirements.txt

# Option 2: editable install (project has a pyproject.toml)
pip install -e .
```

Note about media: this repo uses Git LFS for large files under `media/`.
If you don’t see GIF/MP4 assets after cloning, run:

```bash
git lfs install
git lfs pull
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
.
├── main.py                        # Entry point with CLI
├── requirements.txt               # Simple install option
├── pyproject.toml                 # Project metadata / packaging
├── media/                         # Demo GIFs and videos (Git LFS)
├── docs/                          # Additional docs
├── test_planners.py               # Basic tests
├── snake_planner/                 # Python package
│   ├── __init__.py
│   ├── config.py                  # Configuration parameters
│   ├── game/
│   │   ├── __init__.py
│   │   ├── snake.py               # Snake class (head + tail management)
│   │   ├── environment.py         # Grid environment
│   │   └── game_state.py          # Game state management
│   ├── planners/
│   │   ├── __init__.py
│   │   ├── base_planner.py        # Abstract base class for planners
│   │   ├── astar.py               # A* algorithm
│   │   ├── dijkstra.py            # Dijkstra's algorithm
│   │   ├── rrt.py                 # RRT algorithm (grid-adapted)
│   │   ├── bfs.py                 # Breadth-First Search
│   │   └── dqn_planner.py         # DQN-based planner (experimental)
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── renderer.py            # Pygame visualization
│   └── metrics/
│       ├── __init__.py
│       └── tracker.py             # Metrics collection and comparison
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
