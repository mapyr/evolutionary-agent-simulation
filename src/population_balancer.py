import random

from src.config import TARGET_CROWD_RATIO, TARGET_ENERGY_RATIO, MIN_POP
from src.utils import clamp

class ResourceLimits:
    """
    Resource limits container for simulation parameterization.
    Used to transfer batch resource constraints to simulation instance.
    """
    def __init__(self, max_neighbors, food_count, move_cost, idle_cost, max_pop):
        self.max_neighbors = max_neighbors
        self.food_count = food_count
        self.move_cost = move_cost
        self.idle_cost = idle_cost
        self.max_pop = max_pop

class PopulationBalancer:
    """
    Dynamically tunes simulation resource parameters (food, move cost, crowding, pop cap)
    to maintain target population and death ratios.
    Implements deadlock-breaking and adaptive resource injection/withdrawal.
    """

    MAXN_MIN, MAXN_MAX = 8, 30
    FOOD_MIN, FOOD_MAX = 300, 1200
    MOVE_MIN, MOVE_MAX = 0.8, 2.5
    POP_MIN, POP_MAX = 200, 2000
    DEADLOCK_LIMIT = 80

    def __init__(self, simulation):
        self.sim = simulation
        self._deadlock_ticks = 0

    def balance(self):
        """
        Balances the simulation resource limits and agent parameters
        according to recent death statistics and population size.
        Implements deadlock prevention and resource feedback control.
        """
        sim = self.sim

        # Skip balancing if insufficient data
        if len(sim.agents) == 0 or len(sim.recent_deaths) < 50:
            return

        # --- Gather recent death statistics ---
        crowd = sum(1 for d in sim.recent_deaths if d == ("crowd", "cull"))
        energy = sum(1 for d in sim.recent_deaths if d == "energy")
        old_age = sum(1 for d in sim.recent_deaths if d == "old_age")
        total = crowd + energy + old_age + 1

        current_crowd_ratio = crowd / total
        current_energy_ratio = energy / total
        current_old_age_ratio = old_age / total

        # --- Update EMA (Exponential Moving Average) for each death type ---
        dynamic_alpha = 0.03
        sim.ema_crowd += dynamic_alpha * (current_crowd_ratio - sim.ema_crowd)
        sim.ema_energy += dynamic_alpha * (current_energy_ratio - sim.ema_energy)
        sim.ema_old_age += dynamic_alpha * (current_old_age_ratio - sim.ema_old_age)

        # --- Apply proportional feedback to resource parameters ---
        neighbor_change = 1.0 + (sim.ema_crowd - TARGET_CROWD_RATIO) * 0.1
        neighbor_change = clamp(neighbor_change, 0.98, 1.02)
        food_change = 1.0 + (sim.ema_energy - TARGET_ENERGY_RATIO) * 0.1
        food_change = clamp(food_change, 0.98, 1.02)
        move_cost_change = 1.0 + (sim.ema_crowd - TARGET_CROWD_RATIO) * 0.05
        move_cost_change = clamp(move_cost_change, 0.98, 1.02)
        pop_limit_change = 1.0 - (sim.ema_crowd - TARGET_CROWD_RATIO) * 0.05
        pop_limit_change = clamp(pop_limit_change, 0.95, 1.05)

        max_neighbors = int(clamp(sim.MAX_NEIGHBORS * neighbor_change, self.MAXN_MIN, self.MAXN_MAX))
        food_count = int(clamp(sim.FOOD_COUNT * food_change, self.FOOD_MIN, self.FOOD_MAX))
        move_cost = clamp(sim.MOVE_COST * move_cost_change, self.MOVE_MIN, self.MOVE_MAX)
        idle_cost = move_cost * 0.6
        max_pop = int(clamp(sim.MAX_POP * pop_limit_change, self.POP_MIN, self.POP_MAX))

        # --- Deadlock detection & breaker ---
        is_deadlocked = (
            abs(max_neighbors - self.MAXN_MIN) < 1.1 and
            abs(food_count - self.FOOD_MIN) < 5.1 and
            abs(move_cost - self.MOVE_MAX) < 0.09 and
            abs(max_pop - self.POP_MIN) < 15 and
            sim.ema_crowd > 0.96
        )
        if is_deadlocked:
            self._deadlock_ticks += 1
        else:
            self._deadlock_ticks = 0

        if self._deadlock_ticks >= self.DEADLOCK_LIMIT:
            # Inject resources and reset limits to break the deadlock.
            print(f"[DEADLOCK BREAKER] t={sim.tick} – resetting limits and injecting resources.")
            max_neighbors = random.randint(20, self.MAXN_MAX)
            food_count = random.randint(self.FOOD_MAX // 2, self.FOOD_MAX)
            move_cost = self.MOVE_MIN
            idle_cost = move_cost * 0.6
            max_pop = self.POP_MAX
            sim.ema_crowd = 0.3
            self._deadlock_ticks = 0

        # --- Apply new limits via API ---
        limits = ResourceLimits(
            max_neighbors=max_neighbors,
            food_count=food_count,
            move_cost=move_cost,
            idle_cost=idle_cost,
            max_pop=max_pop
        )
        sim.apply_resource_limits(limits)

        # --- Hard cull if population vastly exceeds limit ---
        if len(sim.agents) > int(sim.MAX_POP * 1.3):
            overpop = len(sim.agents) - sim.MAX_POP
            sim.cull_agents(overpop)

        # --- Soft resource control when population above limit ---
        if len(sim.agents) > sim.MAX_POP:
            food_count = max(sim.FOOD_COUNT - 50, self.FOOD_MIN)
            move_cost = min(sim.MOVE_COST + 0.05, self.MOVE_MAX)
            limits = ResourceLimits(
                max_neighbors=sim.MAX_NEIGHBORS,
                food_count=food_count,
                move_cost=move_cost,
                idle_cost=move_cost * 0.6,
                max_pop=sim.MAX_POP
            )
            sim.apply_resource_limits(limits)
            print(f"[POP CONTROL] t={sim.tick} POP={len(sim.agents)} – moderate resource reduction.")

        # --- Gentle resource increase when population is low ---
        if len(sim.agents) < MIN_POP:
            food_count = min(sim.FOOD_COUNT + 50, self.FOOD_MAX)
            move_cost = max(sim.MOVE_COST - 0.05, self.MOVE_MIN)
            limits = ResourceLimits(
                max_neighbors=sim.MAX_NEIGHBORS,
                food_count=food_count,
                move_cost=move_cost,
                idle_cost=move_cost * 0.6,
                max_pop=sim.MAX_POP
            )
            sim.apply_resource_limits(limits)
            print(f"[POP RECOVERY] t={sim.tick} POP={len(sim.agents)} – gentle resource support.")

        # --- Debug: current simulation resource statistics ---
        print(
            f"[AUTO-EMA] t={sim.tick} pop={len(sim.agents)} "
            f"EMA[crowd]={sim.ema_crowd:.2f} EMA[energy]={sim.ema_energy:.2f} EMA[old]={sim.ema_old_age:.2f} | "
            f"FOOD={sim.FOOD_COUNT} MOVE={sim.MOVE_COST:.2f} IDLE={sim.IDLE_COST:.2f} "
            f"MAXN={sim.MAX_NEIGHBORS} MAX_POP={sim.MAX_POP} DEADLOCK={self._deadlock_ticks}"
        )
