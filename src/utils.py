import random
from collections import Counter

import numpy as np

from src.config import PERSONALITY_TYPES, PERSONALITY_MUTATION_RATE


def clamp(val, low, high):
    """
    Clamp a value to the given interval [low, high].
    """
    return max(low, min(high, val))


def random_color():
    """
    Returns a random RGB color as a tuple of three ints [0, 255).
    """
    return tuple(np.random.randint(0, 255, 3))


def random_pos_in_zone(x0, x1, y0, y1):
    """
    Returns a random (x, y) position within the rectangular zone [x0, x1), [y0, y1).
    """
    return random.randint(x0, x1 - 1), random.randint(y0, y1 - 1)


def random_personality():
    """
    Returns a random agent personality string.
    """
    return random.choice(PERSONALITY_TYPES)


def mutate_personality(parent):
    """
    With probability PERSONALITY_MUTATION_RATE, mutate personality; otherwise, return parent.
    """
    return random_personality() if random.random() < PERSONALITY_MUTATION_RATE else parent


def spawn_food(food_set, desired_count, zone, taken):
    """
    Spawns food at random positions within a zone, avoiding conflicts with taken positions.
    Modifies food_set in-place.
    Args:
        food_set: set to add food positions to (set of (x, y))
        desired_count: number of food items to spawn
        zone: (x0, x1, y0, y1) rectangular bounds
        taken: set of forbidden positions (occupied by agents or other food)
    """
    x0, x1, y0, y1 = zone
    tries = 0
    while len(food_set) < desired_count and tries < desired_count * 10:
        pos = random_pos_in_zone(x0, x1, y0, y1)
        if pos in food_set or pos in taken:
            tries += 1
            continue
        food_set.add(pos)
        tries += 1


def print_population_stats(tick, agents, genome_stats, deaths):
    """
    Prints summary statistics for the current population and deaths.
    Intended for debugging and monitoring in console.
    """
    print(f"\n--- Tick {tick} ---")
    print(f"Population: {len(agents)}")
    if agents:
        ages = [a.age for a in agents]
        print(f"Mean age: {np.mean(ages):.1f}, max: {max(ages)}")
        personalities = Counter(a.personality for a in agents)
        print(f"Personalities: {dict(personalities)}")
        if genome_stats:
            g, lst = max(genome_stats.items(), key=lambda kv: len(kv[1]))
            print(f"Top genome {g}, count {len(lst)}")
    if deaths:
        for k in ('energy', 'old_age', 'crowd'):
            print(f"Deaths {k}: {len(deaths.get(k, []))}")
        print(f"All deaths: {len(deaths.get('all', []))}")
