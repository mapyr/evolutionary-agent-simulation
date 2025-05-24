import pygame

from src.config import GRID_SIZE


class AgentRenderer:
    """
    Handles rendering of an agent on a Pygame surface.
    Responsible for visualizing agent's current position and optional highlight.
    """

    HIGHLIGHT_COLOR = (255, 255, 255)
    HIGHLIGHT_WIDTH = 2

    def __init__(self, agent):
        """
        Initializes the renderer with the given agent instance.
        Args:
            agent: The agent object to render. Expected to have attributes x, y, color.
        """
        self.agent = agent

    def draw(self, surf, highlight=False):
        """
        Draws the agent on the given surface.
        Args:
            surf: Pygame surface to draw on.
            highlight (bool): If True, draws a highlight border around the agent.
        """
        gx = self.agent.x * GRID_SIZE
        gy = self.agent.y * GRID_SIZE
        rect = (gx, gy, GRID_SIZE, GRID_SIZE)

        pygame.draw.rect(surf, self.agent.color, rect)
        if highlight:
            pygame.draw.rect(surf, self.HIGHLIGHT_COLOR, rect, self.HIGHLIGHT_WIDTH)
