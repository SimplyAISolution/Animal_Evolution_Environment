"""Creature class for individual animals in the simulation."""

import numpy as np
from typing import Optional, Tuple
from .genome import Genome


class Creature:
    """Represents an individual creature with genetics, behavior, and state."""
    
    def __init__(self, id: str, genome: Genome, species: str, position: np.ndarray, energy: float):
        """Initialize a creature with basic properties."""
        self.id = id
        self.genome = genome
        self.species = species
        self.position = position.copy()
        self.energy = energy
        self.age = 0
        self.generation = 0
        self.alive = True
        
    def can_reproduce(self) -> bool:
        """Check if creature has enough energy to reproduce."""
        return self.energy >= self.genome.reproduction_threshold
        
    def is_alive(self) -> bool:
        """Check if creature is still alive."""
        return self.alive and self.energy > 0
        
    def consume_energy(self, amount: float, min_energy: float, max_energy: float):
        """Consume energy, ensuring it stays within bounds."""
        self.energy = max(min_energy, self.energy - amount)
        if self.energy <= min_energy:
            self.alive = False
            
    def gain_energy(self, amount: float, max_energy: float):
        """Gain energy, ensuring it doesn't exceed maximum."""
        self.energy = min(max_energy, self.energy + amount)
        
    def update_position(self, dx: float, dy: float, world_width: int, world_height: int):
        """Update position with world boundary wrapping."""
        self.position[0] = (self.position[0] + dx) % world_width
        self.position[1] = (self.position[1] + dy) % world_height
