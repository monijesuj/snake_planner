#!/usr/bin/env python3
"""
Snake Planning Algorithm Comparison

A graduate-level planning algorithms project demonstrating path planning
in a dynamic environment (snake's tail as obstacle).

Usage:
    python main.py                      # Run with A* (default)
    python main.py --algorithm dijkstra # Run with Dijkstra
    python main.py --algorithm rrt      # Run with RRT
    python main.py --algorithm bfs      # Run with BFS
    python main.py --compare            # Run comparison of all algorithms
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from config import GameConfig, Algorithm
from game import GameState
from game.game_state import GameStatus
from planners import AStarPlanner, DijkstraPlanner, RRTPlanner, BFSPlanner
from visualization import Renderer
from metrics import MetricsTracker
from metrics.tracker import ComparisonTracker


def get_planner(algorithm: Algorithm, grid_size: int):
    """
    Factory function to create planner based on algorithm choice.
    
    Args:
        algorithm: Algorithm enum value
        grid_size: Size of the grid
        
    Returns:
        Planner instance
    """
    planners = {
        Algorithm.ASTAR: lambda: AStarPlanner(grid_size),
        Algorithm.DIJKSTRA: lambda: DijkstraPlanner(grid_size),
        Algorithm.RRT: lambda: RRTPlanner(grid_size, max_iterations=5000, goal_bias=0.15),
        Algorithm.BFS: lambda: BFSPlanner(grid_size),
    }
    return planners[algorithm]()


def run_game(config: GameConfig, algorithm: Algorithm, headless: bool = False) -> dict:
    """
    Run a single game with the specified algorithm.
    
    Args:
        config: Game configuration
        algorithm: Planning algorithm to use
        headless: If True, run without visualization (for batch comparison)
        
    Returns:
        Final game metrics dictionary
    """
    # Initialize components
    game_state = GameState(config)
    planner = get_planner(algorithm, config.grid_size)
    metrics = MetricsTracker(planner.name)
    
    if not headless:
        renderer = Renderer(config)
    
    # Game loop timing
    last_move_time = pygame.time.get_ticks()
    running = True
    
    print(f"\nStarting game with {planner.name} algorithm...")
    print(f"Grid size: {config.grid_size}x{config.grid_size}")
    print("Controls: SPACE=pause, R=restart, ESC=quit\n")
    
    while running:
        current_time = pygame.time.get_ticks()
        
        # Handle events
        if not headless:
            events = renderer.handle_events()
            
            if events['quit']:
                running = False
                continue
            
            if events['restart']:
                game_state.reset()
                metrics.reset()
                last_move_time = current_time
                continue
            
            if events['pause']:
                game_state.toggle_pause()
        
        # Game logic (only if running)
        if game_state.status == GameStatus.RUNNING:
            # Check if we need to replan
            if game_state.needs_replanning():
                # Plan path to destination
                result = planner.plan(
                    start=game_state.snake.head,
                    goal=game_state.destination,
                    obstacles=game_state.get_obstacles()
                )
                
                metrics.record_plan(result)
                
                if result.success:
                    game_state.set_path(result.path, result.explored_nodes)
                else:
                    # No path found - game over (trapped)
                    print(f"No path found! Snake is trapped.")
                    game_state.status = GameStatus.GAME_OVER
            
            # Move snake at configured speed
            if current_time - last_move_time >= config.speed:
                game_state.step()
                last_move_time = current_time
        
        # Render
        if not headless:
            renderer.render(game_state, metrics, planner.name)
            renderer.tick()
        
        # Check for game over in headless mode
        if headless and game_state.is_game_over:
            running = False
    
    # Finalize metrics
    final_metrics = metrics.finalize(game_state.score, game_state.snake.length)
    
    if not headless:
        # Keep window open on game over until user quits
        waiting = game_state.is_game_over
        while waiting:
            events = renderer.handle_events()
            if events['quit']:
                waiting = False
            elif events['restart']:
                # Restart the game
                return run_game(config, algorithm, headless)
            renderer.render(game_state, metrics, planner.name)
            renderer.tick()
        
        renderer.cleanup()
    
    return final_metrics


def run_comparison(config: GameConfig, num_runs: int = 1):
    """
    Run comparison of all algorithms.
    
    Args:
        config: Game configuration
        num_runs: Number of runs per algorithm
    """
    comparison = ComparisonTracker()
    algorithms = [Algorithm.ASTAR, Algorithm.DIJKSTRA, Algorithm.BFS, Algorithm.RRT]
    
    print("\n" + "=" * 60)
    print("RUNNING ALGORITHM COMPARISON")
    print(f"Grid Size: {config.grid_size}x{config.grid_size}")
    print(f"Runs per algorithm: {num_runs}")
    print("=" * 60 + "\n")
    
    for algo in algorithms:
        print(f"\n--- Testing {algo.value.upper()} ---")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)
            metrics = run_game(config, algo, headless=False)
            comparison.add_result(metrics)
            print(f"Score: {metrics.score}")
    
    # Print comparison
    comparison.print_comparison()
    
    # Export results
    comparison.export_json("comparison_results.json")
    print("Results exported to comparison_results.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Snake Planning Algorithm Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Run with A* (default)
  python main.py --algorithm dijkstra   # Run with Dijkstra
  python main.py --algorithm rrt        # Run with RRT
  python main.py --algorithm bfs        # Run with BFS
  python main.py --grid-size 30         # Use 30x30 grid
  python main.py --speed 100            # Faster snake (100ms per move)
  python main.py --compare              # Compare all algorithms
        """
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=['astar', 'dijkstra', 'rrt', 'bfs'],
        default='astar',
        help='Planning algorithm to use (default: astar)'
    )
    
    parser.add_argument(
        '--grid-size', '-g',
        type=int,
        default=20,
        help='Grid size NxN (default: 20)'
    )
    
    parser.add_argument(
        '--speed', '-s',
        type=int,
        default=150,
        help='Snake speed in ms per move (default: 150)'
    )
    
    parser.add_argument(
        '--initial-length', '-l',
        type=int,
        default=3,
        help='Initial snake length (default: 3)'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Run comparison of all algorithms'
    )
    
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=1,
        help='Number of runs per algorithm in comparison mode (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = GameConfig(
        grid_size=args.grid_size,
        speed=args.speed,
        initial_length=args.initial_length
    )
    
    # Initialize pygame
    pygame.init()
    
    try:
        if args.compare:
            run_comparison(config, args.runs)
        else:
            algorithm = Algorithm(args.algorithm)
            metrics = run_game(config, algorithm)
            
            print("\n" + "=" * 40)
            print("FINAL RESULTS")
            print("=" * 40)
            for key, value in metrics.to_dict().items():
                print(f"  {key}: {value}")
            print("=" * 40 + "\n")
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
