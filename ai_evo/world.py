import numpy as np
from .config import Config
from .rng import RNG

class Environment:
    """2D toroidal grid with plant resource & temperature placeholder."""
    def __init__(self, cfg: Config, rng: RNG):
        self.cfg, self.rng = cfg, rng
        self.width, self.height = cfg.width, cfg.height
        self.food = np.zeros((self.height, self.width), dtype=np.float32)
        self.temperature = 20.0
        self.t = 0

    def grow_food(self):
        self.food += self.cfg.plant_growth_rate
        np.minimum(self.food, self.cfg.plant_cap, out=self.food)

    def wrap(self, x, y):
        return x % self.width, y % self.height
        
    def step(self):
        """Advance environment by one time step."""
        self.t += 1
        self.grow_food()
        
    def consume_food_at(self, x: int, y: int, amount: float) -> float:
        """Consume food at given position, return amount actually consumed."""
        x, y = int(x), int(y)  # Convert to integers
        x, y = self.wrap(x, y)
        available = self.food[y, x]
        consumed = min(available, amount)
        self.food[y, x] -= consumed
        return consumed
