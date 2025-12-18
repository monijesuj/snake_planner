#!/usr/bin/env python3
"""
Record gameplay videos of different planning algorithms.

Records each algorithm playing until game over and saves as GIF.
Requires: pip install pillow numpy
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: pillow not installed. Install with: pip install pillow")

from snake_planner.config import GameConfig, Algorithm
from snake_planner.game import GameState
from snake_planner.game.game_state import GameStatus
from snake_planner.planners import (
    AStarPlanner, DijkstraPlanner, RRTPlanner, BFSPlanner,
    SurvivalPlanner, HamiltonianPlanner, MCTSPlanner, DQNPlanner
)
from snake_planner.visualization import Renderer
from snake_planner.metrics import MetricsTracker


def get_planner(algorithm: Algorithm, grid_size: int, game_state=None):
    """Factory function to create planner based on algorithm choice.

    `game_state` is required for planners that need access to the live game (e.g. DQN).
    """
    planners = {
        Algorithm.ASTAR: lambda gs: AStarPlanner(grid_size),
        Algorithm.DIJKSTRA: lambda gs: DijkstraPlanner(grid_size),
        Algorithm.RRT: lambda gs: RRTPlanner(grid_size, max_iterations=5000, goal_bias=0.15),
        Algorithm.BFS: lambda gs: BFSPlanner(grid_size),
        Algorithm.SURVIVAL: lambda gs: SurvivalPlanner(grid_size),
        Algorithm.HAMILTONIAN: lambda gs: HamiltonianPlanner(grid_size),
        Algorithm.MCTS: lambda gs: MCTSPlanner(grid_size, num_simulations=200),
        Algorithm.DQN: lambda gs: DQNPlanner(gs),
    }
    factory = planners.get(algorithm)
    if factory is None:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return factory(game_state)


def surface_to_pil(surface):
    """Convert pygame surface to PIL Image."""
    # Get the raw pixel data
    data = pygame.image.tostring(surface, 'RGB')
    size = surface.get_size()
    return Image.frombytes('RGB', size, data)


def record_game(config: GameConfig, algorithm: Algorithm, output_path: str, 
                max_frames: int = None, target_fps: int = 15) -> dict:
    """
    Record a game with the specified algorithm.
    
    Args:
        config: Game configuration
        algorithm: Planning algorithm to use
        output_path: Path to save the GIF file
        max_frames: Maximum frames to record (None for unlimited)
        target_fps: Target FPS for the GIF
        
    Returns:
        Final game metrics dictionary
    """
    if not HAS_PIL:
        print("Error: pillow required for recording. Install with: pip install pillow")
        return None
    
    # Initialize components
    game_state = GameState(config)
    planner = get_planner(algorithm, config.grid_size, game_state)
    metrics = MetricsTracker(planner.name)
    renderer = Renderer(config)
    
    # Collect frames
    frames = []
    
    # Game loop timing
    last_move_time = pygame.time.get_ticks()
    frame_count = 0
    running = True
    frame_skip = 2  # Capture every Nth frame to reduce GIF size
    
    print(f"\nRecording {planner.name} algorithm to {output_path}...")
    print(f"Grid size: {config.grid_size}x{config.grid_size}")
    print("Press ESC to stop early, or wait for game over.\n")
    
    # Use faster speed for recording
    record_speed = max(30, config.speed // 3)
    
    while running:
        current_time = pygame.time.get_ticks()
        
        # Handle events (allow ESC to quit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Game logic
        if game_state.status == GameStatus.RUNNING:
            # Check if we need to replan
            if game_state.needs_replanning():
                # Set tail tip for survival planner
                if hasattr(planner, 'set_tail_tip') and game_state.snake.tail:
                    planner.set_tail_tip(game_state.snake.tail[-1])
                
                # Set snake body for MCTS planner
                if hasattr(planner, 'set_snake_body'):
                    planner.set_snake_body(list(game_state.snake.body))
                
                # Plan path to destination
                try:
                    result = planner.plan(
                        start=game_state.snake.head,
                        goal=game_state.destination,
                        obstacles=game_state.get_obstacles(),
                        snake_length=game_state.snake.length
                    )
                except TypeError:
                    result = planner.plan(
                        start=game_state.snake.head,
                        goal=game_state.destination,
                        obstacles=game_state.get_obstacles()
                    )
                
                metrics.record_plan(result)
                
                if result.success:
                    game_state.set_path(result.path, result.explored_nodes)
                else:
                    print(f"No path found! Snake is trapped.")
                    game_state.status = GameStatus.GAME_OVER
            
            # Move snake
            if current_time - last_move_time >= record_speed:
                game_state.step()
                last_move_time = current_time
        
        # Render
        renderer.render(game_state, metrics, planner.name)
        
        # Capture frame (with frame skipping for smaller GIF)
        frame_count += 1
        if frame_count % frame_skip == 0:
            surface = pygame.display.get_surface()
            pil_image = surface_to_pil(surface)
            # Resize to reduce file size
            pil_image = pil_image.resize(
                (pil_image.width // 2, pil_image.height // 2), 
                Image.Resampling.LANCZOS
            )
            frames.append(pil_image)
        
        # Print progress
        if frame_count % 100 == 0:
            print(f"  Frames: {len(frames)}, Score: {game_state.score}, Length: {game_state.snake.length}")
        
        # Check limits
        if max_frames and len(frames) >= max_frames:
            print("Max frames reached.")
            running = False
        
        if game_state.is_game_over:
            # Capture a few more frames showing game over
            for _ in range(target_fps):  # ~1 second of game over
                renderer.render(game_state, metrics, planner.name)
                surface = pygame.display.get_surface()
                pil_image = surface_to_pil(surface)
                pil_image = pil_image.resize(
                    (pil_image.width // 2, pil_image.height // 2),
                    Image.Resampling.LANCZOS
                )
                frames.append(pil_image)
            running = False
        
        renderer.tick()
    
    # Save GIF
    if frames:
        print(f"\nSaving GIF with {len(frames)} frames...")
        # Calculate duration in milliseconds
        duration = int(1000 / target_fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,  # Loop forever
            optimize=True
        )
        print(f"GIF saved: {output_path}")
    
    # Cleanup
    renderer.cleanup()
    
    # Finalize metrics
    final_metrics = metrics.finalize(game_state.score, game_state.snake.length)
    
    print(f"\nRecording complete: {output_path}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Final score: {game_state.score}")
    print(f"  Snake length: {game_state.snake.length}")
    
    return final_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Record Snake Planning Algorithm gameplay to GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python record_comparison.py                    # Record survival and hamiltonian
  python record_comparison.py --algorithms astar survival
  python record_comparison.py --grid-size 15 --speed 100
  python record_comparison.py --output-dir ./videos
        """
    )
    
    parser.add_argument(
        '--algorithms', '-a',
        nargs='+',
        type=str,
        choices=['astar', 'dijkstra', 'rrt', 'bfs', 'survival', 'hamiltonian', 'mcts', 'dqn'],
        default=['survival', 'hamiltonian'],
        help='Algorithms to record (default: survival hamiltonian)'
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
        default=100,
        help='Snake speed in ms per move for recording (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='.',
        help='Output directory for GIFs (default: current directory)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=15,
        help='GIF FPS (default: 15)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to record per GIF (default: unlimited)'
    )
    
    args = parser.parse_args()
    
    # Check for pillow
    if not HAS_PIL:
        print("\nError: pillow is required for recording.")
        print("Install with: pip install pillow")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration
    config = GameConfig(
        grid_size=args.grid_size,
        speed=args.speed,
        initial_length=3
    )
    
    # Initialize pygame
    pygame.init()
    
    results = {}
    
    try:
        for algo_name in args.algorithms:
            algorithm = Algorithm(algo_name)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                args.output_dir, 
                f"snake_{algo_name}_{timestamp}.gif"
            )
            
            print(f"\n{'='*60}")
            print(f"Recording: {algo_name.upper()}")
            print(f"{'='*60}")
            
            # Re-initialize pygame for each recording
            pygame.quit()
            pygame.init()
            
            metrics = record_game(
                config, 
                algorithm, 
                output_path,
                max_frames=args.max_frames,
                target_fps=args.fps
            )
            
            if metrics:
                results[algo_name] = {
                    'video': output_path,
                    'metrics': metrics
                }
        
        # Print summary
        print(f"\n{'='*60}")
        print("RECORDING SUMMARY")
        print(f"{'='*60}")
        for algo_name, data in results.items():
            print(f"\n{algo_name.upper()}:")
            print(f"  GIF: {data['video']}")
            if hasattr(data['metrics'], 'to_dict'):
                for key, value in data['metrics'].to_dict().items():
                    print(f"  {key}: {value}")
            else:
                print(f"  Score: {data['metrics'].get('score', 'N/A')}")
        print(f"\n{'='*60}")
        
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
