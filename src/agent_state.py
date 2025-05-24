import numpy as np
from collections import deque
import random

from src.config import ENERGY_START, NN_LAYERS, NN_HIDDEN
from src.utils import random_personality

class AgentState:
    """
    Container for all mutable agent data.
    Holds agent's core attributes, learning state, and lineage information.
    Intended for read/write by agent logic and components only.
    """

    def __init__(
        self, sim, x, y, color, energy=ENERGY_START,
        food_radius=3, agent_radius=3, personality=None
    ):
        """
        Initializes all agent state fields.
        Args:
            sim: Simulation instance reference.
            x (int): Initial x-coordinate.
            y (int): Initial y-coordinate.
            color (tuple): Agent color.
            energy (float): Initial energy.
            food_radius (int): Perception radius for food.
            agent_radius (int): Perception radius for agents.
            personality: Agent personality string or object.
        """
        self.sim = sim
        self.x = x
        self.y = y
        self.color = color
        self.energy = float(energy)
        self.personality = personality if personality else random_personality()
        self.age = 0
        self.offspring_count = 0
        self.id = random.randint(0, 1_000_000)
        self.parent_id = None
        self.death_reason = None
        self.visited_last_10 = deque(maxlen=10)
        self.last_move = (0, 0)
        self.lstm_hidden = np.zeros((NN_LAYERS, NN_HIDDEN), dtype=np.float32)
        self.lstm_cell = np.zeros((NN_LAYERS, NN_HIDDEN), dtype=np.float32)
        self.food_radius = food_radius
        self.agent_radius = agent_radius
        # Add additional fields required by components as needed.
