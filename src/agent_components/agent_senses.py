from src.utils import clamp

class AgentSenses:
    """
    Aggregates and processes environmental sensory data for an agent.
    Computes local food, agents, friend/other counts, directional food, chemical signal, and normalized distances to map edge.
    """

    def __init__(self, state):
        """
        Initializes the AgentSenses with the given agent state.
        Args:
            state: Agent's state object.
        """
        self.state = state
        self.reset()

    def reset(self):
        """
        Resets all sensory data to default (empty) values.
        """
        self.sense_food = 0
        self.sense_agents = 0
        self.sense_friends = 0
        self.sense_others = 0
        self.food_up = self.food_down = self.food_left = self.food_right = 0
        self.food_up_dist = self.food_down_dist = self.food_left_dist = self.food_right_dist = None
        self.avg_energy = 0.0
        self.max_energy = 0.0
        self.chemo_signal = 0.0
        self.edge_distance_x = 0.0
        self.edge_distance_y = 0.0

    def update(self, food_grid, agent_grid, chemo_grid=None):
        """
        Updates all sensory data from the current environment grids.
        Args:
            food_grid: Set of (x, y) positions containing food.
            agent_grid: Dict mapping (x, y) to list of agent objects at those positions.
            chemo_grid: Optional 2D array of chemical concentrations.
        """
        self.reset()
        fr = self.state.food_radius
        energy_sum = 0
        energy_max = 0
        energy_count = 0
        friends = 0
        others = 0

        for dx in range(-fr, fr + 1):
            for dy in range(-fr, fr + 1):
                tx = clamp(self.state.x + dx, 0, self.state.sim.world_w - 1)
                ty = clamp(self.state.y + dy, 0, self.state.sim.world_h - 1)
                agents_here = agent_grid.get((tx, ty), [])
                self.sense_agents += len(agents_here)
                for ag in agents_here:
                    # "Group" defined by color by default; replace as needed.
                    if ag.color == self.state.color:
                        friends += 1
                    else:
                        others += 1
                    energy_sum += ag.energy
                    if ag.energy > energy_max:
                        energy_max = ag.energy
                    energy_count += 1
                if (tx, ty) in food_grid:
                    self.sense_food += 1
                    if dx == 0 and dy == 0:
                        continue
                    self._update_direction(dx, dy)
                    self._update_distance(dx, dy)
                if chemo_grid is not None:
                    self.chemo_signal += chemo_grid[tx][ty]

        self.sense_friends = friends
        self.sense_others = others
        self.avg_energy = (energy_sum / energy_count) if energy_count else 0.0
        self.max_energy = energy_max
        self._normalize_distances()

        # Normalized edge distance [0, 1]: distance to nearest map edge along each axis
        self.edge_distance_x = min(
            self.state.x, self.state.sim.world_w - 1 - self.state.x
        ) / (self.state.sim.world_w - 1)
        self.edge_distance_y = min(
            self.state.y, self.state.sim.world_h - 1 - self.state.y
        ) / (self.state.sim.world_h - 1)

    def _update_direction(self, dx, dy):
        """
        Increments food direction counters based on dx, dy.
        """
        if abs(dx) >= abs(dy):
            if dx > 0:
                self.food_right += 1
            elif dx < 0:
                self.food_left += 1
        if abs(dy) >= abs(dx):
            if dy > 0:
                self.food_down += 1
            elif dy < 0:
                self.food_up += 1

    def _update_distance(self, dx, dy):
        """
        Updates the minimum normalized distance to food in each cardinal direction.
        """
        if dx == 0 and dy < 0:
            self.food_up_dist = self._min_dist(self.food_up_dist, abs(dy))
        if dx == 0 and dy > 0:
            self.food_down_dist = self._min_dist(self.food_down_dist, abs(dy))
        if dy == 0 and dx < 0:
            self.food_left_dist = self._min_dist(self.food_left_dist, abs(dx))
        if dy == 0 and dx > 0:
            self.food_right_dist = self._min_dist(self.food_right_dist, abs(dx))

    @staticmethod
    def _min_dist(old, new):
        """
        Returns the smaller of the two distances, treating None as infinity.
        """
        return new if old is None or new < old else old

    def _normalize_distances(self):
        """
        Normalizes food direction distances to [0, 1] scale.
        """
        r = self.state.food_radius + 1
        self.food_up_dist = (self.food_up_dist or r) / r
        self.food_down_dist = (self.food_down_dist or r) / r
        self.food_left_dist = (self.food_left_dist or r) / r
        self.food_right_dist = (self.food_right_dist or r) / r

    # --- Property access for compatibility with agent code ---

    @property
    def friends(self):
        return self.sense_friends

    @property
    def others(self):
        return self.sense_others

    @property
    def avg_energy_level(self):
        return self.avg_energy

    @property
    def max_energy_level(self):
        return self.max_energy

    @property
    def chemo(self):
        return self.chemo_signal

    @property
    def edge_x(self):
        return self.edge_distance_x

    @property
    def edge_y(self):
        return self.edge_distance_y
