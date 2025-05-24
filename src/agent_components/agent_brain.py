import numpy as np

from src.config import N_HISTORY


class AgentBrain:
    """
    Encapsulates sensory processing and input vector construction for an agent.
    Maintains input history and applies personality bias to the agent's perception.
    """

    def __init__(self, state, senses):
        """
        Initializes the AgentBrain with agent state and sensory interface.
        Args:
            state: Mutable state object for the agent.
            senses: Senses object holding the latest sensory data.
        """
        self.state = state
        self.senses = senses
        self.last_inputs = []
        self._init_input_shape()

    def _init_input_shape(self):
        """
        Defines the input feature keys and total input vector size,
        including history window.
        """
        self.input_keys = [
            "sense_food",
            "sense_agents",
            "sense_friends",
            "sense_others",
            "avg_energy",
            "max_energy",
            "chemo_signal",
            "edge_distance_x",
            "edge_distance_y",
            "food_up",
            "food_down",
            "food_left",
            "food_right",
            "food_up_dist",
            "food_down_dist",
            "food_left_dist",
            "food_right_dist",
            "visited_here",
            "last_move_x",
            "last_move_y",
            "crowd_global",
            "energy_global",
            "old_age_global",
        ]
        self.n_base_inputs = len(self.input_keys)
        self.n_history = N_HISTORY
        self.n_inputs = self.n_base_inputs * (1 + self.n_history)

    def get_inputs(self):
        """
        Returns the input vector for the agent's decision process.
        Applies normalization and appends history.
        Adds bias based on agent personality.
        Returns:
            np.ndarray: Normalized input vector of fixed length.
        """
        s = self.senses
        st = self.state

        base_inputs = [
            s.sense_food,
            s.sense_agents,
            s.sense_friends,
            s.sense_others,
            s.avg_energy,
            s.max_energy,
            s.chemo_signal,
            s.edge_distance_x,
            s.edge_distance_y,
            s.food_up,
            s.food_down,
            s.food_left,
            s.food_right,
            s.food_up_dist,
            s.food_down_dist,
            s.food_left_dist,
            s.food_right_dist,
            1.0 if (st.x, st.y) in st.visited_last_10 else 0.0,
            st.last_move[0],
            st.last_move[1],
            st.sim.ema_crowd,
            st.sim.ema_energy,
            st.sim.ema_old_age,
        ]

        # Normalize inputs; ranges should be matched to simulation conventions
        base_inputs[0] /= 10.0    # sense_food
        base_inputs[1] /= 10.0    # sense_agents
        base_inputs[2] /= 10.0    # sense_friends
        base_inputs[3] /= 10.0    # sense_others
        base_inputs[4] /= 200.0   # avg_energy
        base_inputs[5] /= 200.0   # max_energy
        base_inputs[6] /= 5.0     # chemo_signal
        # edge_distance_x, edge_distance_y: assumed in [0, 1]
        base_inputs[9] /= 5.0     # food_up
        base_inputs[10] /= 5.0    # food_down
        base_inputs[11] /= 5.0    # food_left
        base_inputs[12] /= 5.0    # food_right
        # food_*_dist: assumed in [0, 1]

        assert len(base_inputs) == self.n_base_inputs

        # Concatenate historical inputs
        flat_hist = []
        for old in self.last_inputs:
            flat_hist.extend(old)
        while len(flat_hist) < self.n_base_inputs * self.n_history:
            flat_hist.append(0.0)

        inputs = np.array(base_inputs + flat_hist, dtype=np.float32)
        assert len(inputs) == self.n_inputs

        # Update input history
        self.last_inputs.append(base_inputs)
        if len(self.last_inputs) > self.n_history:
            self.last_inputs = self.last_inputs[-self.n_history:]

        # Apply personality-based bias to specific features
        boost = 1.2
        if st.personality == "explorer":
            # Prioritize forward exploration
            inputs[13] *= boost  # food_up_dist
        elif st.personality == "survivor":
            # Focus on average energy
            inputs[4] *= boost
        elif st.personality == "feeder":
            # Prioritize immediate food detection
            inputs[0] *= boost
        elif st.personality == "loner":
            # Heightened awareness of other agents
            inputs[1] *= boost
        elif st.personality == "social":
            # Aversion to crowds
            inputs[1] *= -boost

        return inputs
