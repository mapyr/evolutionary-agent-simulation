import collections
import random
import sys
import time

import numpy as np
import pygame
import torch

from src.agent import Agent
from src.batched_lstm import BatchedLSTM
from src.config import (
    GRID_SIZE, SENSOR_RADIUS_RANGE, NN_INPUTS, NN_LAYERS, MAX_AGENT_AGE,
    TRACE_LENGTH, food_zones, NN_OUTPUTS, NN_HIDDEN, MAX_POP
)
from src.population_balancer import PopulationBalancer, ResourceLimits
from src.utils import clamp, random_color, random_pos_in_zone, random_personality, spawn_food, print_population_stats


def safe_fmt(val, fmt=".2f", fallback="–"):
    """
    Safely format possibly-None or malformed values for display.
    """
    if val is None:
        return fallback
    try:
        return format(val, fmt)
    except Exception:
        return fallback


class Simulation:
    """
    Main simulation engine.
    Orchestrates agents, neural decision-making, resource balancing,
    and user interface (Pygame visualization & interaction).
    """

    def __init__(self):
        self.WIDTH, self.HEIGHT = 1720, 1320
        self.AGENT_COUNT = 160
        self.food_zone_idx = 0
        self.food_zone_duration = 1000
        self.EMA_ALPHA = 0.05

        # Resource-limited fields (modulated by PopulationBalancer)
        self._MAX_NEIGHBORS = 15
        self._FOOD_COUNT = 600
        self._MOVE_COST = 1.0
        self._IDLE_COST = 0.6
        self._MAX_POP = MAX_POP

        self.ema_crowd = 0.0
        self.ema_energy = 0.0
        self.ema_old_age = 0.0

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.brain = BatchedLSTM(NN_INPUTS, NN_HIDDEN, NN_OUTPUTS, num_layers=NN_LAYERS, device=self.device)

        self.world_w, self.world_h = self.WIDTH // GRID_SIZE, self.HEIGHT // GRID_SIZE

        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 14)
        self.font_big = pygame.font.SysFont('monospace', 22, bold=True)
        self.pinned_agent = None  # Agent currently highlighted (tooltip); None if none

        self.tick = 0
        self.paused = False
        self.trace_map = {}
        self.recent_deaths = collections.deque(maxlen=2000)
        self.genome_stats = collections.defaultdict(list)
        self.deaths = collections.defaultdict(list)

        self.agents = [
            Agent(
                self, *random_pos_in_zone(*self._current_food_zone()),
                random_color(),
                food_radius=random.randint(*SENSOR_RADIUS_RANGE),
                agent_radius=random.randint(*SENSOR_RADIUS_RANGE),
                personality=random_personality()
            )
            for _ in range(self.AGENT_COUNT)
        ]
        self.food_set = set()

        self.balancer = PopulationBalancer(self)

        # UI stats
        self._last_tick = time.time()
        self._last_draw = time.time()
        self._fps = 0.0
        self._tps = 0.0

        print_population_stats(0, self.agents, self.genome_stats, self.deaths)
        sys.stdout.flush()

    # --- API: Resource control properties ---

    @property
    def MAX_NEIGHBORS(self):
        return self._MAX_NEIGHBORS

    @property
    def FOOD_COUNT(self):
        return self._FOOD_COUNT

    @property
    def MOVE_COST(self):
        return self._MOVE_COST

    @property
    def IDLE_COST(self):
        return self._IDLE_COST

    @property
    def MAX_POP(self):
        return self._MAX_POP

    def apply_resource_limits(self, limits: ResourceLimits):
        """
        Sets resource control fields (called by PopulationBalancer).
        """
        self._MAX_NEIGHBORS = limits.max_neighbors
        self._FOOD_COUNT = limits.food_count
        self._MOVE_COST = limits.move_cost
        self._IDLE_COST = limits.idle_cost
        self._MAX_POP = limits.max_pop

    def cull_agents(self, count: int):
        """
        Hard-culls 'count' oldest agents (sets their energy to -1 and death_reason='cull').
        """
        if count <= 0:
            return
        sorted_agents = sorted(self.agents, key=lambda a: -a.state.age)
        kill = sorted_agents[:count]
        for a in kill:
            a.state.energy = -1
            a.state.death_reason = "cull"
        print(f"[HARD CULL] t={self.tick} – removed {count} agents (max pop {self._MAX_POP})")

    def _current_food_zone(self):
        """
        Returns (x0, x1, y0, y1) tuple for current food zone bounds.
        """
        x0f, x1f, y0f, y1f = food_zones[self.food_zone_idx]
        return (
            int(x0f * self.world_w), int(x1f * self.world_w),
            int(y0f * self.world_h), int(y1f * self.world_h)
        )

    def _spawn_food(self):
        """
        Replenishes food within the current food zone.
        """
        zone = self._current_food_zone()
        taken = {(a.x, a.y) for a in self.agents}
        spawn_food(self.food_set, self._FOOD_COUNT, zone, taken)

    def _handle_events(self):
        """
        Handles user and system events (quit, pause, mouse selection).
        Returns False if simulation should terminate, True otherwise.
        """
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                self.paused = not self.paused
                print("[PAUSE]" if self.paused else "[RUN]")
                sys.stdout.flush()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = pygame.mouse.get_pos()
                ax, ay = mx // GRID_SIZE, my // GRID_SIZE
                agent = next((a for a in self.agents if a.x == ax and a.y == ay), None)
                if agent:
                    self.pinned_agent = None if self.pinned_agent is agent else agent
                else:
                    self.pinned_agent = None
        return True

    def _draw_overlay(self, highlight):
        """
        Draws simulation statistics and agent tooltip overlay.
        """
        surf = self.screen
        y = 4
        pad = 8
        lineh = 20

        def draw_line(txt, bold=False, color=(200, 255, 255)):
            nonlocal y
            font = self.font_big if bold else self.font
            surf.blit(font.render(txt, True, color), (pad, y))
            y += lineh

        # Population/stats
        draw_line(f"Population: {len(self.agents)}   (MAX: {self._MAX_POP})", bold=True)
        ages = [a.age for a in self.agents]
        draw_line(f"Mean age: {np.mean(ages):.1f}, max: {max(ages) if ages else 0}")
        personalities = collections.Counter(a.personality for a in self.agents)
        draw_line(f"Personalities: {dict(personalities)}")

        # Deaths
        deaths_total = sum(len(self.deaths[k]) for k in ('energy', 'old_age', 'crowd'))
        deaths_energy = len(self.deaths.get('energy', []))
        deaths_old = len(self.deaths.get('old_age', []))
        deaths_crowd = len(self.deaths.get('crowd', []))
        draw_line(f"Deaths: {deaths_total}   (E:{deaths_energy} O:{deaths_old} C:{deaths_crowd})")

        # Top genome
        if self.genome_stats:
            g, lst = max(self.genome_stats.items(), key=lambda kv: len(kv[1]))
            draw_line(f"Top genome: {g}, count {len(lst)}")

        # Tick/FPS
        draw_line(
            f"Tick: {self.tick}   FPS: {safe_fmt(self._fps)}   TPS: {safe_fmt(self._tps)}",
            bold=True, color=(255, 255, 0)
        )

        # Pause
        if self.paused:
            draw_line("== PAUSED ==", bold=True, color=(255, 80, 80))

        # Agent tooltip
        if highlight:
            fx, fy = highlight.x * GRID_SIZE, highlight.y * GRID_SIZE
            lines = [
                f'ID={highlight.id}  P={highlight.parent_id}  Pers={highlight.personality}',
                f'Pos=({highlight.x},{highlight.y})   Age={highlight.age}/{MAX_AGENT_AGE}  Kids={highlight.offspring_count}',
                f'Energy={safe_fmt(highlight.energy)}  Mem={1 if (highlight.x, highlight.y) in highlight.visited_last_10 else 0}  LastMove={highlight.last_move}',
                f'FoodRadius={highlight.food_radius}  AgentRadius={highlight.agent_radius}',
                f'FoodCount={highlight.sense_food}  Friends={highlight.sense_friends}  Others={highlight.sense_others}  Agents={highlight.sense_agents}',
                f'F[U/D/L/R]={highlight.food_up}/{highlight.food_down}/{highlight.food_left}/{highlight.food_right}',
                f'FoodDist[U/D/L/R]={safe_fmt(highlight.food_up_dist)}/'
                f'{safe_fmt(highlight.food_down_dist)}/'
                f'{safe_fmt(highlight.food_left_dist)}/'
                f'{safe_fmt(highlight.food_right_dist)}',
                f'AvgEnergy={safe_fmt(highlight.senses.avg_energy)}  MaxEnergy={safe_fmt(highlight.senses.max_energy)}',
                f'ChemoSignal={safe_fmt(highlight.senses.chemo_signal)}',
                f'EdgeDistX={safe_fmt(highlight.senses.edge_distance_x)}  EdgeDistY={safe_fmt(highlight.senses.edge_distance_y)}'
            ]
            for i, line in enumerate(lines):
                self.screen.blit(self.font.render(line, True, (255, 255, 255)), (fx, fy - 18 - i * 18))

            pygame.draw.circle(self.screen, (100, 255, 100),
                               (fx + GRID_SIZE // 2, fy + GRID_SIZE // 2),
                               highlight.food_radius * GRID_SIZE, 1)
            pygame.draw.circle(self.screen, (100, 100, 255),
                               (fx + GRID_SIZE // 2, fy + GRID_SIZE // 2),
                               highlight.agent_radius * GRID_SIZE, 1)

    def _draw(self, highlight):
        """
        Renders world (agents, traces, food, overlays).
        """
        self.screen.fill((0, 0, 0))
        for (tx, ty), tval in self.trace_map.items():
            age = self.tick - tval
            if age < TRACE_LENGTH:
                alpha = int(48 * (1 - age / TRACE_LENGTH)) + 24
                s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                s.fill((120, 120, 120, alpha))
                self.screen.blit(s, (tx * GRID_SIZE, ty * GRID_SIZE))

        # Agent highlight
        if self.pinned_agent in self.agents:
            hl = self.pinned_agent
        else:
            mx, my = pygame.mouse.get_pos()
            hl = next((a for a in self.agents if a.x == mx // GRID_SIZE and a.y == my // GRID_SIZE), None)

        if highlight:
            for (vx, vy) in highlight.visited_last_10:
                s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                s.fill((255, 255, 128, 90))
                self.screen.blit(s, (vx * GRID_SIZE, vy * GRID_SIZE))
        for fx, fy in self.food_set:
            pygame.draw.rect(
                self.screen, (0, 200, 0),
                (fx * GRID_SIZE + 4, fy * GRID_SIZE + 4, GRID_SIZE - 8, GRID_SIZE - 8)
            )
        for a in self.agents:
            a.draw(self.screen, highlight is a)
        self._draw_overlay(hl)
        pygame.display.flip()

    def run(self):
        """
        Main simulation loop. Handles ticks, neural inference, world updates, agent logic, UI and drawing.
        """
        running = True
        prev_tick_time = time.time()
        prev_draw_time = time.time()
        tick_counter = 0
        draw_counter = 0

        while running:
            running = self._handle_events()
            mx, my = pygame.mouse.get_pos()
            highlight = next((a for a in self.agents if a.x == mx // GRID_SIZE and a.y == my // GRID_SIZE), None)

            now = time.time()
            if not self.paused:
                # --- Simulation tick ---
                if self.tick % self.food_zone_duration == 0:
                    self.food_zone_idx = (self.food_zone_idx + 1) % len(food_zones)
                    print(f"[ZONE] now {self.food_zone_idx} -> {food_zones[self.food_zone_idx]}")
                    self.food_set.clear()
                self._spawn_food()
                food_grid = set(self.food_set)
                agent_grid = collections.defaultdict(list)
                for a in self.agents:
                    agent_grid[(a.x, a.y)].append(a)
                for a in self.agents:
                    a.sense(food_grid, agent_grid)
                inputs, hiddens, cells, active_idx = [], [], [], []
                for idx, a in enumerate(self.agents):
                    if a.energy > 0:
                        inputs.append(a.get_inputs())
                        hiddens.append(a.lstm_hidden)
                        cells.append(a.lstm_cell)
                        active_idx.append(idx)
                if inputs:
                    inputs_np = np.stack(inputs, axis=0)
                    hiddens_np = np.stack(hiddens, axis=0)
                    cells_np = np.stack(cells, axis=0)
                    inputs_t = torch.from_numpy(inputs_np).to(dtype=torch.float32, device=self.device).unsqueeze(1)
                    hiddens_t = torch.from_numpy(hiddens_np).to(dtype=torch.float32, device=self.device).transpose(0, 1)
                    cells_t = torch.from_numpy(cells_np).to(dtype=torch.float32, device=self.device).transpose(0, 1)
                    with torch.no_grad():
                        probs, h_new, c_new = self.brain(inputs_t, hiddens_t, cells_t)
                    probs = probs.cpu().numpy()
                    h_new = h_new.cpu().numpy().transpose(1, 0, 2)
                    c_new = c_new.cpu().numpy().transpose(1, 0, 2)
                    taken = {(a.x, a.y) for a in self.agents}
                    self.trace_map = {}
                    for i, agent_idx in enumerate(active_idx):
                        a = self.agents[agent_idx]
                        action = np.random.choice(4, p=probs[i])
                        a.apply_move(action, h_new[i], c_new[i], taken, self.trace_map, self.tick)
                        taken.add((a.x, a.y))
                for a in self.agents:
                    if a.energy > 0:
                        a.eat(self.food_set)
                        a.step()
                dead = [a for a in self.agents if a.energy <= 0]
                for a in dead:
                    self.genome_stats[(a.color, a.food_radius, a.agent_radius, a.personality)].append(
                        (a.age, a.offspring_count))
                    self.deaths[a.death_reason].append(a.age)
                    self.deaths['all'].append(a.age)
                    self.recent_deaths.append(a.death_reason)
                self.agents = [a for a in self.agents if a.energy > 0]
                # Reproduction
                new_agents = []
                taken = {(a.x, a.y) for a in self.agents}
                for a in self.agents:
                    if not a.can_reproduce():
                        continue
                    empty = []
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = clamp(a.x + dx, 0, self.world_w - 1), clamp(a.y + dy, 0, self.world_h - 1)
                        if ((nx, ny) not in taken) and ((nx, ny) not in self.food_set):
                            empty.append((nx, ny))
                    if empty:
                        nx, ny = random.choice(empty)
                        child = a.reproduce()
                        child.state.x, child.state.y = nx, ny
                        taken.add((nx, ny))
                        new_agents.append(child)
                self.agents.extend(new_agents)
                if self.tick % 10 == 0:
                    self.balancer.balance()
                if self.tick % 500 == 0:
                    print_population_stats(self.tick, self.agents, self.genome_stats, self.deaths)
                    sys.stdout.flush()
                self.tick += 1
                tick_counter += 1
            # --- Always draw board & stats (even when paused) ---
            self._draw(highlight)
            draw_counter += 1

            now2 = time.time()
            if now2 - prev_draw_time > 1.0:
                self._fps = draw_counter / (now2 - prev_draw_time)
                self._tps = tick_counter / (now2 - prev_tick_time) if not self.paused else 0.0
                prev_draw_time = now2
                prev_tick_time = now2
                tick_counter = 0
                draw_counter = 0
            self.clock.tick(30)
        pygame.quit()
        print_population_stats(self.tick, self.agents, self.genome_stats, self.deaths)
        sys.stdout.flush()
