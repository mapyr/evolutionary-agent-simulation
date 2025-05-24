"""
Simulation configuration constants.

Defines environment, agent, neural network, and evolutionary hyperparameters.
Modify this file to adjust simulation mechanics, resource values, agent neural architecture, and population dynamics.
"""

# ---- Environment ----

GRID_SIZE = 16  # Size of each grid cell (pixels)
food_zones = [
    (0.0, 0.5, 0.0, 1.0),    # Left half
    (0.0, 1.0, 0.0, 1.0),    # Full map
    (0.5, 1.0, 0.0, 1.0),    # Right half
    (0.25, 0.75, 0.25, 0.75) # Center
]

# ---- Agent Energy & Life ----

ENERGY_START = 45               # Initial energy per agent
ENERGY_PER_FOOD = 60            # Energy gain per food
ENERGY_TO_REPRODUCE = 120       # Minimum energy to reproduce
ENERGY_AFTER_REPRO = 45         # Energy after reproduction
MAX_AGENT_AGE = 1500            # Maximum agent lifespan (ticks)

# ---- Mutation & Evolution ----

MUTATION_RANGE = 16             # RGB color mutation range
SENSOR_RADIUS_RANGE = (0, 20)   # Allowed sensory radius mutation bounds
NN_MUTATION = 0.10              # Neural network weight mutation rate
PERSONALITY_TYPES = [
    'explorer', 'survivor', 'feeder', 'loner', 'social'
]
PERSONALITY_MUTATION_RATE = 0.05

# ---- Neural Network Architecture ----

N_BASE_INPUTS = 23              # Base input features (see AgentBrain)
N_HISTORY = 20                  # Steps of input history included
NN_INPUTS = N_BASE_INPUTS * (1 + N_HISTORY)
NN_LAYERS = 10                  # LSTM layers (deep, for testing; can set lower)
NN_HIDDEN = 64                  # Hidden units per LSTM layer
NN_OUTPUTS = 4                  # Action output dimension

# ---- Agent Trace / World Trace ----

TRACE_LENGTH = 8                # Length of agent's visual trace (optional)

# ---- Population Dynamics ----

TARGET_CROWD_RATIO = 0.25       # Desired ratio of "crowded" deaths
TARGET_ENERGY_RATIO = 0.25      # Desired ratio of "energy" deaths
TARGET_OLD_AGE_RATIO = 0.15     # Desired ratio of "old age" deaths
MIN_POP = 40                    # Minimum allowed population
MAX_POP = 800                   # Maximum allowed population
