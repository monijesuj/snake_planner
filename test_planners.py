#!/usr/bin/env python3
"""
Quick test script to verify all algorithms can find paths.
Tests basic functionality without running full game.
"""

import sys
from config import GameConfig
from game import Snake, Environment
from planners import AStarPlanner, DijkstraPlanner, RRTPlanner, BFSPlanner


def test_planner(planner, name):
    """Test a single planner."""
    print(f"\nTesting {name}...")
    
    config = GameConfig(grid_size=10)
    env = Environment(config.grid_size)
    snake = Snake((3, 5), initial_length=3)
    
    # Simple test: plan from snake head to a nearby goal
    start = snake.head
    goal = (7, 5)
    obstacles = snake.get_tail_set()
    
    print(f"  Start: {start}, Goal: {goal}")
    print(f"  Obstacles: {len(obstacles)} cells")
    
    result = planner.plan(start, goal, obstacles)
    
    if result.success:
        print(f"  ✓ Path found: {result.path_length} steps")
        print(f"  ✓ Nodes expanded: {result.nodes_expanded}")
        print(f"  ✓ Time: {result.computation_time_ms:.2f}ms")
        return True
    else:
        print(f"  ✗ Failed to find path!")
        return False


def main():
    """Run tests."""
    print("=" * 60)
    print("TESTING PLANNING ALGORITHMS")
    print("=" * 60)
    
    grid_size = 10
    planners = [
        (AStarPlanner(grid_size), "A*"),
        (DijkstraPlanner(grid_size), "Dijkstra"),
        (BFSPlanner(grid_size), "BFS"),
        (RRTPlanner(grid_size, max_iterations=1000), "RRT"),
    ]
    
    results = []
    for planner, name in planners:
        success = test_planner(planner, name)
        results.append((name, success))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name:12} {status}")
    
    all_passed = all(success for _, success in results)
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run the game.")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
