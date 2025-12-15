"""
Pygame-based visualization for snake planning.
"""

import pygame
from typing import List, Set, Tuple, Optional

from snake_planner.game.game_state import GameState, GameStatus
from snake_planner.config import GameConfig, Colors, DEFAULT_CONFIG
from snake_planner.metrics.tracker import MetricsTracker


Position = Tuple[int, int]


class Renderer:
    """
    Pygame renderer for the snake planning visualization.
    
    Handles all drawing: grid, snake, destination, planned path,
    explored nodes, and metrics overlay.
    """
    
    def __init__(self, config: GameConfig = DEFAULT_CONFIG):
        """
        Initialize renderer.
        
        Args:
            config: Game configuration
        """
        self.config = config
        
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Snake Planning Algorithm Comparison")
        
        self.screen = pygame.display.set_mode(
            (config.window_width, config.window_height)
        )
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 24)
        
        # Grid offset (padding)
        self.grid_offset_x = config.window_padding
        self.grid_offset_y = config.window_padding
    
    def grid_to_screen(self, pos: Position) -> Tuple[int, int]:
        """Convert grid position to screen coordinates."""
        return (
            self.grid_offset_x + pos[0] * self.config.cell_size,
            self.grid_offset_y + pos[1] * self.config.cell_size
        )
    
    def draw_grid(self):
        """Draw the grid lines."""
        for i in range(self.config.grid_size + 1):
            # Vertical lines
            x = self.grid_offset_x + i * self.config.cell_size
            pygame.draw.line(
                self.screen,
                Colors.GRID_LINE,
                (x, self.grid_offset_y),
                (x, self.grid_offset_y + self.config.grid_size * self.config.cell_size)
            )
            # Horizontal lines
            y = self.grid_offset_y + i * self.config.cell_size
            pygame.draw.line(
                self.screen,
                Colors.GRID_LINE,
                (self.grid_offset_x, y),
                (self.grid_offset_x + self.config.grid_size * self.config.cell_size, y)
            )
    
    def draw_cell(self, pos: Position, color: Tuple[int, int, int], shrink: int = 1):
        """
        Draw a filled cell at grid position.
        
        Args:
            pos: Grid position
            color: RGB color tuple
            shrink: Pixels to shrink from each side
        """
        screen_x, screen_y = self.grid_to_screen(pos)
        rect = pygame.Rect(
            screen_x + shrink,
            screen_y + shrink,
            self.config.cell_size - 2 * shrink,
            self.config.cell_size - 2 * shrink
        )
        pygame.draw.rect(self.screen, color, rect)
    
    def draw_explored_nodes(self, explored: Set[Position]):
        """Draw explored nodes (for visualization of algorithm)."""
        for pos in explored:
            self.draw_cell(pos, Colors.PATH_EXPLORED, shrink=8)
    
    def draw_planned_path(self, path: List[Position]):
        """Draw the planned path."""
        if len(path) < 2:
            return
        
        # Draw path as connected line through cell centers
        points = []
        for pos in path:
            screen_x, screen_y = self.grid_to_screen(pos)
            center_x = screen_x + self.config.cell_size // 2
            center_y = screen_y + self.config.cell_size // 2
            points.append((center_x, center_y))
        
        pygame.draw.lines(self.screen, Colors.PATH_PLANNED, False, points, 3)
        
        # Draw small dots at each path point
        for point in points[1:-1]:  # Skip start and end
            pygame.draw.circle(self.screen, Colors.PATH_PLANNED, point, 4)
    
    def draw_snake(self, game_state: GameState):
        """Draw the snake (head and body)."""
        snake = game_state.snake
        
        # Draw body segments (from tail to head, so head is on top)
        body_list = list(snake.body)
        
        for i, pos in enumerate(reversed(body_list)):
            if i == len(body_list) - 1:
                # Head
                self.draw_cell(pos, Colors.SNAKE_HEAD, shrink=2)
                # Draw direction indicator
                self._draw_head_direction(pos, snake.direction)
            elif i == 0:
                # Tail tip
                self.draw_cell(pos, Colors.SNAKE_TAIL, shrink=3)
            else:
                # Body
                self.draw_cell(pos, Colors.SNAKE_BODY, shrink=2)
    
    def _draw_head_direction(self, pos: Position, direction):
        """Draw an indicator showing snake's direction."""
        if direction is None:
            return
        
        screen_x, screen_y = self.grid_to_screen(pos)
        center_x = screen_x + self.config.cell_size // 2
        center_y = screen_y + self.config.cell_size // 2
        
        # Draw eyes based on direction
        eye_offset = 6
        eye_radius = 3
        
        dx, dy = direction.value
        
        if dx == 0:  # Moving vertically
            # Eyes on left and right
            eye1 = (center_x - eye_offset, center_y + dy * 4)
            eye2 = (center_x + eye_offset, center_y + dy * 4)
        else:  # Moving horizontally
            # Eyes on top and bottom
            eye1 = (center_x + dx * 4, center_y - eye_offset)
            eye2 = (center_x + dx * 4, center_y + eye_offset)
        
        pygame.draw.circle(self.screen, Colors.BACKGROUND, eye1, eye_radius)
        pygame.draw.circle(self.screen, Colors.BACKGROUND, eye2, eye_radius)
    
    def draw_destination(self, destination: Optional[Position]):
        """Draw the destination/food."""
        if destination is None:
            return
        
        screen_x, screen_y = self.grid_to_screen(destination)
        center_x = screen_x + self.config.cell_size // 2
        center_y = screen_y + self.config.cell_size // 2
        
        # Draw as a star/diamond shape
        radius = self.config.cell_size // 2 - 4
        pygame.draw.circle(self.screen, Colors.DESTINATION, (center_x, center_y), radius)
        
        # Inner highlight
        pygame.draw.circle(
            self.screen,
            (255, 150, 150),
            (center_x - 3, center_y - 3),
            radius // 3
        )
    
    def draw_metrics(self, metrics: MetricsTracker, game_state: GameState, algorithm_name: str):
        """Draw metrics overlay at bottom of screen."""
        y_start = self.grid_offset_y + self.config.grid_size * self.config.cell_size + 20
        x = self.grid_offset_x
        
        # Algorithm name and score
        title = f"Algorithm: {algorithm_name}  |  Score: {game_state.score}  |  Length: {game_state.snake.length}"
        title_surface = self.font_large.render(title, True, Colors.TEXT)
        self.screen.blit(title_surface, (x, y_start))
        
        y_start += 35
        
        # Current planning metrics
        if metrics.current_result:
            result = metrics.current_result
            metrics_text = (
                f"Path: {result.path_length} steps  |  "
                f"Nodes: {result.nodes_expanded}  |  "
                f"Time: {result.computation_time_ms:.2f}ms"
            )
            metrics_surface = self.font_medium.render(metrics_text, True, Colors.TEXT_DIM)
            self.screen.blit(metrics_surface, (x, y_start))
        
        y_start += 30
        
        # Aggregate stats
        stats = metrics.get_summary()
        if stats['total_plans'] > 0:
            summary = (
                f"Plans: {stats['total_plans']}  |  "
                f"Avg Path: {stats['avg_path_length']:.1f}  |  "
                f"Avg Nodes: {stats['avg_nodes_expanded']:.1f}  |  "
                f"Avg Time: {stats['avg_computation_time']:.2f}ms"
            )
            summary_surface = self.font_small.render(summary, True, Colors.TEXT_DIM)
            self.screen.blit(summary_surface, (x, y_start))
    
    def draw_game_over(self, score: int):
        """Draw game over screen."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.config.window_width, self.config.window_height))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(180)
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        game_over_text = self.font_large.render("GAME OVER", True, Colors.GAME_OVER)
        text_rect = game_over_text.get_rect(
            center=(self.config.window_width // 2, self.config.window_height // 2 - 30)
        )
        self.screen.blit(game_over_text, text_rect)
        
        # Score
        score_text = self.font_medium.render(f"Final Score: {score}", True, Colors.TEXT)
        score_rect = score_text.get_rect(
            center=(self.config.window_width // 2, self.config.window_height // 2 + 10)
        )
        self.screen.blit(score_text, score_rect)
        
        # Instructions
        restart_text = self.font_small.render("Press R to restart, ESC to quit", True, Colors.TEXT_DIM)
        restart_rect = restart_text.get_rect(
            center=(self.config.window_width // 2, self.config.window_height // 2 + 50)
        )
        self.screen.blit(restart_text, restart_rect)
    
    def draw_paused(self):
        """Draw pause overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.config.window_width, self.config.window_height))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(150)
        self.screen.blit(overlay, (0, 0))
        
        pause_text = self.font_large.render("PAUSED", True, Colors.TEXT)
        text_rect = pause_text.get_rect(
            center=(self.config.window_width // 2, self.config.window_height // 2)
        )
        self.screen.blit(pause_text, text_rect)
        
        hint_text = self.font_small.render("Press SPACE to resume", True, Colors.TEXT_DIM)
        hint_rect = hint_text.get_rect(
            center=(self.config.window_width // 2, self.config.window_height // 2 + 40)
        )
        self.screen.blit(hint_text, hint_rect)
    
    def draw_no_path(self):
        """Draw no path found message."""
        text = self.font_medium.render("No path found!", True, Colors.GAME_OVER)
        text_rect = text.get_rect(
            center=(self.config.window_width // 2, self.grid_offset_y - 20)
        )
        self.screen.blit(text, text_rect)
    
    def render(
        self,
        game_state: GameState,
        metrics: MetricsTracker,
        algorithm_name: str,
        show_explored: bool = True,
        show_path: bool = True
    ):
        """
        Render complete frame.
        
        Args:
            game_state: Current game state
            metrics: Metrics tracker
            algorithm_name: Name of current algorithm
            show_explored: Whether to show explored nodes
            show_path: Whether to show planned path
        """
        # Clear screen
        self.screen.fill(Colors.BACKGROUND)
        
        # Draw grid
        self.draw_grid()
        
        # Draw explored nodes (behind everything)
        if show_explored and game_state.explored_nodes:
            self.draw_explored_nodes(game_state.explored_nodes)
        
        # Draw planned path
        if show_path and game_state.current_path:
            self.draw_planned_path(game_state.current_path)
        
        # Draw destination
        self.draw_destination(game_state.destination)
        
        # Draw snake
        self.draw_snake(game_state)
        
        # Draw metrics
        self.draw_metrics(metrics, game_state, algorithm_name)
        
        # Draw overlays based on status
        if game_state.status == GameStatus.GAME_OVER:
            self.draw_game_over(game_state.score)
        elif game_state.status == GameStatus.PAUSED:
            self.draw_paused()
        elif game_state.status == GameStatus.NO_PATH:
            self.draw_no_path()
        
        # Update display
        pygame.display.flip()
    
    def handle_events(self) -> dict:
        """
        Handle pygame events.
        
        Returns:
            Dict with event flags: quit, restart, pause
        """
        events = {
            'quit': False,
            'restart': False,
            'pause': False
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events['quit'] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    events['quit'] = True
                elif event.key == pygame.K_r:
                    events['restart'] = True
                elif event.key == pygame.K_SPACE:
                    events['pause'] = True
        
        return events
    
    def tick(self):
        """Limit frame rate."""
        self.clock.tick(self.config.fps)
    
    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()
