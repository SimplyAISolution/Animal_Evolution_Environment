import numpy as np

class Config:
    """Configuration class for the AI Animal Evolution Environment."""
    
    def __init__(self, **kwargs):
        # World parameters
        self.width = kwargs.get('width', 100)
        self.height = kwargs.get('height', 100)
        self.seed = kwargs.get('seed', 42)
        
        # Simulation parameters
        self.max_steps = kwargs.get('max_steps', 1000)
        self.snapshot_every = kwargs.get('snapshot_every', 50)
        
        # Population parameters
        self.init_herbivores = kwargs.get('init_herbivores', 50)
        self.init_carnivores = kwargs.get('init_carnivores', 25)
        
        # Energy parameters
        self.min_energy = kwargs.get('min_energy', 0.0)
        self.max_energy = kwargs.get('max_energy', 150.0)
        
        # Environment parameters
        self.plant_growth_rate = kwargs.get('plant_growth_rate', 0.1)
        self.plant_cap = kwargs.get('plant_cap', 10.0)
        
        # Feeding parameters
        self.herbivore_bite_cap = kwargs.get('herbivore_bite_cap', 3.0)
        self.carnivore_gain_eff = kwargs.get('carnivore_gain_eff', 0.8)
        
        # Movement parameters
        self.move_cost_base = kwargs.get('move_cost_base', 0.1)
        
        # Reproduction parameters
        self.reproduce_cost_frac = kwargs.get('reproduce_cost_frac', 0.5)
        
        # Evolution parameters
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.mutation_strength = kwargs.get('mutation_strength', 0.1)
        self.perception_max = kwargs.get('perception_max', 5.0)
        
        # Spatial hash parameters
        self.grid_cell = kwargs.get('grid_cell', 10.0)

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
