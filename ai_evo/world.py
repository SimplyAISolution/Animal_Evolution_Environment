"""Environment simulation with food growth and resource management."""

import numpy as np
from .config import Config
from .rng import RNG


class Environment:
    """2D toroidal grid with plant resource & temperature management."""
    
    def __init__(self, cfg: Config, rng: RNG):
        """Initialize environment.
        
        Args:
            cfg: Configuration object
            rng: Random number generator
        """
        self.cfg = cfg
        self.rng = rng
        self.width = cfg.width
        self.height = cfg.height
        
        # Food grid - plants that regrow over time
        self.food = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Environmental parameters
        self.temperature = cfg.temperature
        self.step_count = 0
        
        # Initialize with some random food distribution
        self._initialize_food()
    
    def _initialize_food(self):
        """Initialize food distribution across the world."""
        # Start with random food distribution
        self.food = self.rng.rand(self.height, self.width) * self.cfg.plant_max_density * 0.5
    
    def step(self):
        """Update environment for one time step."""
        self.step_count += 1
        self._grow_food()
        self._update_temperature()
    
    def _grow_food(self):
        """Grow plants based on growth rate."""
        # Add growth, but cap at maximum density
        growth = self.cfg.plant_growth_rate * (1.0 + 0.1 * self.rng.normal(size=self.food.shape))
        self.food = np.minimum(self.food + growth, self.cfg.plant_max_density)
        
        # Ensure non-negative values
        self.food = np.maximum(self.food, 0.0)
    
    def _update_temperature(self):
        """Update environmental temperature with seasonal variation."""
        # Simple seasonal temperature variation
        seasonal_cycle = np.sin(self.step_count * 0.01) * 5.0
        self.temperature = self.cfg.temperature + seasonal_cycle
    
    def get_food_at(self, x: int, y: int) -> float:
        """Get food amount at specific coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Food amount at location
        """
        x, y = self._wrap_coordinates(x, y)
        return self.food[y, x]
    
    def consume_food(self, x: int, y: int, amount: float) -> float:
        """Consume food at location and return amount actually consumed.
        
        Args:
            x: X coordinate
            y: Y coordinate  
            amount: Amount of food to consume
            
        Returns:
            Amount of food actually consumed
        """
        x, y = self._wrap_coordinates(x, y)
        available = self.food[y, x]
        consumed = min(available, amount)
        self.food[y, x] -= consumed
        return consumed
    
    def _wrap_coordinates(self, x: int, y: int):
        """Handle toroidal wrapping of coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Wrapped (x, y) coordinates
        """
        return x % self.width, y % self.height
    
    def get_statistics(self) -> dict:
        """Get environment statistics.
        
        Returns:
            Dictionary with environment metrics
        """
        return {
            'total_food': float(np.sum(self.food)),
            'avg_food': float(np.mean(self.food)),
            'max_food': float(np.max(self.food)),
            'min_food': float(np.min(self.food)),
            'temperature': self.temperature,
            'step_count': self.step_count,
            'width': self.width,
            'height': self.height
        }
    
    def get_food_in_radius(self, center_x: float, center_y: float, radius: float) -> float:
        """Get total food within radius of a point.
        
        Args:
            center_x: Center X coordinate
            center_y: Center Y coordinate
            radius: Search radius
            
        Returns:
            Total food in radius
        """
        total_food = 0.0
        radius_sq = radius * radius
        
        for dx in range(-int(radius)-1, int(radius)+2):
            for dy in range(-int(radius)-1, int(radius)+2):
                if dx*dx + dy*dy <= radius_sq:
                    x = int(center_x + dx)
                    y = int(center_y + dy)
                    x, y = self._wrap_coordinates(x, y)
                    total_food += self.food[y, x]
        
        return total_food
