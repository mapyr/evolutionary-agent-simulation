import random

from src.config import (
    ENERGY_PER_FOOD,
    MAX_AGENT_AGE,
    ENERGY_TO_REPRODUCE,
    SENSOR_RADIUS_RANGE,
    MUTATION_RANGE,
    ENERGY_AFTER_REPRO
)
from src.utils import clamp, mutate_personality


class AgentBody:
    """
    Represents the physical body and environment interaction logic for an agent.
    Handles movement, eating, reproduction, and aging.
    """

    def __init__(self, state, senses, agent_class):
        """
        Initializes the AgentBody with a given state, senses, and agent class constructor.
        Args:
            state: Mutable agent state object (position, energy, etc.).
            senses: Senses object holding sensory data.
            agent_class: Callable to create a new agent (typically the owning agent class).
        """
        self.state = state
        self.senses = senses
        self.agent_class = agent_class

    def apply_move(self, action, hidden, cell, taken, trace_map, tick):
        """
        Applies an action (movement) to the agent, updating its position and energy accordingly.
        Args:
            action: Integer index representing the move direction.
            hidden: Updated LSTM hidden state.
            cell: Updated LSTM cell state.
            taken: Set of positions already occupied by other agents.
            trace_map: Map for tracing movement history.
            tick: Current simulation tick.
        """
        dx, dy = self._decode_action(action)
        sim = self.state.sim
        nx = clamp(self.state.x + dx, 0, sim.world_w - 1)
        ny = clamp(self.state.y + dy, 0, sim.world_h - 1)
        self.state.lstm_hidden = hidden
        self.state.lstm_cell = cell

        if (nx, ny) == (self.state.x, self.state.y) or (nx, ny) in taken:
            # Agent cannot move; apply idle cost
            self.state.energy -= sim.IDLE_COST
        else:
            # Update position and movement history
            trace_map[(self.state.x, self.state.y)] = tick
            self.state.visited_last_10.append((self.state.x, self.state.y))
            self.state.last_move = (dx, dy)
            self.state.x, self.state.y = nx, ny
            self.state.energy -= sim.MOVE_COST

    @staticmethod
    def _decode_action(action):
        """
        Decodes an action index into a (dx, dy) tuple representing movement direction.
        Args:
            action: Integer in [0, 3] representing direction (up, down, left, right).
        Returns:
            Tuple[int, int]: Change in x and y coordinates.
        """
        # Up, Down, Left, Right
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        return directions[action]

    def eat(self, food):
        """
        Consumes food at the agent's current position if available, increasing energy.
        Args:
            food: Set of food positions (x, y tuples).
        """
        pos = (self.state.x, self.state.y)
        if pos in food:
            food.remove(pos)
            self.state.energy += ENERGY_PER_FOOD

    def step(self):
        """
        Advances the agent's age by one tick and checks for death conditions.
        """
        self.state.age += 1
        sim = self.state.sim

        if sim.MAX_NEIGHBORS > 0 and self.senses.sense_agents > sim.MAX_NEIGHBORS:
            self.state.death_reason = "crowd"
            self.state.energy = -1
        elif self.state.age >= MAX_AGENT_AGE:
            self.state.death_reason = "old_age"
            self.state.energy = -1
        elif self.state.energy <= 0:
            self.state.death_reason = "energy"

    def can_reproduce(self):
        """
        Checks if the agent has sufficient energy to reproduce.
        Returns:
            bool: True if reproduction is possible.
        """
        return self.state.energy >= ENERGY_TO_REPRODUCE

    def reproduce(self):
        """
        Spawns a new agent with mutated sensory radii and color.
        Returns:
            New agent instance.
        """
        def mutate_radius(value):
            return clamp(value + random.randint(-1, 1), *SENSOR_RADIUS_RANGE)

        new_color = tuple(
            clamp(c + random.randint(-MUTATION_RANGE, MUTATION_RANGE), 0, 255)
            for c in self.state.color
        )

        child = self.agent_class(
            self.state.sim,
            self.state.x,
            self.state.y,
            new_color,
            ENERGY_AFTER_REPRO,
            food_radius=mutate_radius(self.state.food_radius),
            agent_radius=mutate_radius(self.state.agent_radius),
            personality=mutate_personality(self.state.personality),
        )
        child.parent_id = self.state.id
        self.state.energy = ENERGY_AFTER_REPRO
        self.state.offspring_count += 1
        return child
