import numpy as np

class Config:
    """Configuration class for simulation parameters."""
    
    def __init__(self, 
                 seed: int = 42,
                 width: int = 100,
                 height: int = 100,
                 max_steps: int = 1000,
                 init_herbivores: int = 20,
                 init_carnivores: int = 10,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.1,
                 plant_growth_rate: float = 0.05,
                 plant_cap: float = 10.0,
                 max_energy: float = 200.0,
                 min_energy: float = 0.0,
                 herbivore_bite_cap: float = 3.0,
                 carnivore_gain_eff: float = 0.8,
                 reproduce_threshold: float = 120.0,
                 reproduce_cost_frac: float = 0.5,
                 move_cost_base: float = 0.1,
                 grid_cell: int = 8,
                 snapshot_every: int = 100,
                 # Trait bounds
                 speed_min: float = 0.1,
                 speed_max: float = 3.0,
                 size_min: float = 0.5,
                 size_max: float = 2.0,
                 aggression_min: float = 0.0,
                 aggression_max: float = 1.0,
                 perception_min: float = 0.5,
                 perception_max: float = 5.0,
                 energy_efficiency_min: float = 0.5,
                 energy_efficiency_max: float = 2.0,
                 reproduction_threshold_min: float = 60.0,
                 reproduction_threshold_max: float = 150.0,
                 lifespan_min: int = 500,
                 lifespan_max: int = 2000):
        """Initialize configuration with default values."""
        self.seed = seed
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.init_herbivores = init_herbivores
        self.init_carnivores = init_carnivores
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.plant_growth_rate = plant_growth_rate
        self.plant_cap = plant_cap
        self.max_energy = max_energy
        self.min_energy = min_energy
        self.herbivore_bite_cap = herbivore_bite_cap
        self.carnivore_gain_eff = carnivore_gain_eff
        self.reproduce_threshold = reproduce_threshold
        self.reproduce_cost_frac = reproduce_cost_frac
        self.move_cost_base = move_cost_base
        self.grid_cell = grid_cell
        self.snapshot_every = snapshot_every
        
        # Trait bounds
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.size_min = size_min
        self.size_max = size_max
        self.aggression_min = aggression_min
        self.aggression_max = aggression_max
        self.perception_min = perception_min
        self.perception_max = perception_max
        self.energy_efficiency_min = energy_efficiency_min
        self.energy_efficiency_max = energy_efficiency_max
        self.reproduction_threshold_min = reproduction_threshold_min
        self.reproduction_threshold_max = reproduction_threshold_max
        self.lifespan_min = lifespan_min
        self.lifespan_max = lifespan_max


class RNG:
    """Central deterministic RNG wrapper."""
    def __init__(self, seed: int):
        self.seed = seed
        self.rs = np.random.RandomState(seed)

    # Basic distributions
    def normal(self, loc=0.0, scale=1.0, size=None):
        return self.rs.normal(loc, scale, size)

    def rand(self, *shape):
        return self.rs.rand(*shape)

    def randint(self, low, high=None, size=None):
        return self.rs.randint(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        return self.rs.choice(a, size, replace, p)

    def random_bool(self, p=0.5, size=None):
        return self.rs.rand(*(size if isinstance(size, tuple) else (() if size is None else (size,)))) < p
