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
        
        # Initialize brain weights as tuple (W1, b1, W2, b2) or simple array
        if brain_weights is None:
            # Create default neural network weights
            input_size = 8
            hidden_size = 6
            output_size = 4
            self.brain_weights = (
                np.random.randn(input_size, hidden_size) * 0.5,  # W1
                np.random.randn(hidden_size) * 0.5,              # b1
                np.random.randn(hidden_size, output_size) * 0.5, # W2
                np.random.randn(output_size) * 0.5               # b2
            )
        else:
            self.brain_weights = brain_weights
        
    def copy(self) -> 'Genome':
        """Create a copy of this genome."""
        if isinstance(self.brain_weights, tuple):
            copied_weights = tuple(w.copy() for w in self.brain_weights)
        else:
            copied_weights = self.brain_weights.copy()
            
        return Genome(
            speed=self.speed,
            size=self.size,
            aggression=self.aggression,
            perception=self.perception,
            energy_efficiency=self.energy_efficiency,
            reproduction_threshold=self.reproduction_threshold,
            lifespan=self.lifespan,
            brain_weights=copied_weights
        )
