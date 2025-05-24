from src.agent_components.agent_body import AgentBody
from src.agent_components.agent_brain import AgentBrain
from src.agent_components.agent_renderer import AgentRenderer
from src.agent_components.agent_senses import AgentSenses
from src.agent_state import AgentState
from src.config import ENERGY_START

class Agent:
    """
    Main agent entity. Aggregates state, senses, brain, body, and rendering.
    Provides property-based interface to state and senses for external usage.
    """

    def __init__(
        self, sim, x, y, color, energy=ENERGY_START,
        food_radius=3, agent_radius=3, personality=None
    ):
        """
        Constructs a new agent with all core components (state, senses, brain, body, renderer).
        Args:
            sim: Simulation instance reference.
            x (int): Initial x-coordinate.
            y (int): Initial y-coordinate.
            color (tuple): Agent color.
            energy (float): Starting energy.
            food_radius (int): Sensing radius for food.
            agent_radius (int): Sensing radius for other agents.
            personality: Agent behavioral type.
        """
        self.state = AgentState(sim, x, y, color, energy, food_radius, agent_radius, personality)
        self.senses = AgentSenses(self.state)
        self.brain = AgentBrain(self.state, self.senses)
        self.body = AgentBody(self.state, self.senses, agent_class=type(self))
        self.renderer = AgentRenderer(self.state)

    # ---- Core State Properties (proxy to AgentState) ----

    @property
    def sim(self):
        return self.state.sim

    @property
    def x(self):
        return self.state.x
    @x.setter
    def x(self, value):
        self.state.x = value

    @property
    def y(self):
        return self.state.y
    @y.setter
    def y(self, value):
        self.state.y = value

    @property
    def color(self):
        return self.state.color

    @property
    def energy(self):
        return self.state.energy
    @energy.setter
    def energy(self, value):
        self.state.energy = value

    @property
    def personality(self):
        return self.state.personality

    @property
    def age(self):
        return self.state.age

    @property
    def offspring_count(self):
        return self.state.offspring_count

    @property
    def id(self):
        return self.state.id

    @property
    def parent_id(self):
        return self.state.parent_id
    @parent_id.setter
    def parent_id(self, value):
        self.state.parent_id = value

    @property
    def death_reason(self):
        return self.state.death_reason
    @death_reason.setter
    def death_reason(self, value):
        self.state.death_reason = value

    @property
    def visited_last_10(self):
        return self.state.visited_last_10

    @property
    def last_move(self):
        return self.state.last_move

    @property
    def lstm_hidden(self):
        return self.state.lstm_hidden
    @lstm_hidden.setter
    def lstm_hidden(self, value):
        self.state.lstm_hidden = value

    @property
    def lstm_cell(self):
        return self.state.lstm_cell
    @lstm_cell.setter
    def lstm_cell(self, value):
        self.state.lstm_cell = value

    @property
    def food_radius(self):
        return self.state.food_radius

    @property
    def agent_radius(self):
        return self.state.agent_radius

    # ---- Senses Properties (proxy to AgentSenses) ----

    @property
    def sense_food(self):
        return self.senses.sense_food

    @property
    def sense_agents(self):
        return self.senses.sense_agents

    @property
    def food_up(self):
        return self.senses.food_up

    @property
    def food_down(self):
        return self.senses.food_down

    @property
    def food_left(self):
        return self.senses.food_left

    @property
    def food_right(self):
        return self.senses.food_right

    @property
    def food_up_dist(self):
        return self.senses.food_up_dist

    @property
    def food_down_dist(self):
        return self.senses.food_down_dist

    @property
    def food_left_dist(self):
        return self.senses.food_left_dist

    @property
    def food_right_dist(self):
        return self.senses.food_right_dist

    @property
    def sense_friends(self):
        return self.senses.sense_friends

    @property
    def sense_others(self):
        return self.senses.sense_others

    @property
    def avg_energy(self):
        return self.senses.avg_energy

    @property
    def edge_distance_x(self):
        return self.senses.edge_distance_x

    @property
    def edge_distance_y(self):
        return self.senses.edge_distance_y

    # ---- Agent Operations (delegation to brain/body/renderer) ----

    def sense(self, food_grid, agent_grid):
        """
        Updates sensory inputs from the environment.
        Args:
            food_grid: Set of (x, y) food positions.
            agent_grid: Dict[(x, y)] -> List[Agent].
        """
        self.senses.update(food_grid, agent_grid)

    def get_inputs(self):
        """
        Returns the input vector for the agent's neural network (from AgentBrain).
        """
        return self.brain.get_inputs()

    def apply_move(self, action, hidden, cell, taken, trace_map, tick):
        """
        Moves the agent according to a chosen action.
        Args:
            action: Integer movement action.
            hidden, cell: RNN state.
            taken: Set of occupied positions.
            trace_map: Dict[(x, y)] -> tick.
            tick: Simulation step.
        """
        self.body.apply_move(action, hidden, cell, taken, trace_map, tick)

    def eat(self, food):
        """
        Consumes food at the agent's location if present.
        Args:
            food: Set of (x, y) food positions.
        """
        self.body.eat(food)

    def step(self):
        """
        Advances agent's age and checks death conditions.
        """
        self.body.step()

    def can_reproduce(self):
        """
        Checks if the agent can reproduce (enough energy).
        Returns: bool
        """
        return self.body.can_reproduce()

    def reproduce(self):
        """
        Produces a mutated offspring agent.
        Returns: New agent instance.
        """
        return self.body.reproduce()

    def draw(self, surf, highlight=False):
        """
        Renders the agent to a surface.
        Args:
            surf: Pygame Surface.
            highlight (bool): If True, draw with highlight.
        """
        self.renderer.draw(surf, highlight)
