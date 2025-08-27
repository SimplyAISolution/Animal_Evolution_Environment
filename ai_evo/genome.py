"""Genome class representing creature genetics."""

import numpy as np
from typing import Optional


class Genome:
    """Represents the genetic information of a creature."""
    
    def __init__(self, 
                 speed: float = 1.0,
                 size: float = 1.0, 
                 aggression: float = 0.5,
                 perception: float = 2.0,
                 energy_efficiency: float = 1.0,
                 reproduction_threshold: float = 90.0,
                 lifespan: int = 1000,
                 brain_weights: Optional[np.ndarray] = None):
        """Initialize genome with trait values."""
        self.speed = speed
        self.size = size
        self.aggression = aggression
        self.perception = perception
        self.energy_efficiency = energy_efficiency
        self.reproduction_threshold = reproduction_threshold
        self.lifespan = lifespan
        self.brain_weights = brain_weights if brain_weights is not None else np.random.randn(10)
        
    def copy(self) -> 'Genome':
        """Create a copy of this genome."""
        return Genome(
            speed=self.speed,
            size=self.size,
            aggression=self.aggression,
            perception=self.perception,
            energy_efficiency=self.energy_efficiency,
            reproduction_threshold=self.reproduction_threshold,
            lifespan=self.lifespan,
            brain_weights=self.brain_weights.copy()
        )
