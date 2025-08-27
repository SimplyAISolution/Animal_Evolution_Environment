"""Creature entities with genomes and behavior."""

import numpy as np
from typing import List, Optional
from .genome import Genome


class Creature:
    """Individual creature with genome, position, and energy."""
    
    def __init__(self, id: str, genome: Genome, species: str, 
                 position: np.ndarray, energy: float, generation: int = 0,
                 parent_ids: Optional[List[str]] = None):
        """Initialize creature.
        
        Args:
            id: Unique identifier
            genome: Creature's genetic traits
            species: "herbivore" or "carnivore"
            position: [x, y] position in world
            energy: Current energy level
            generation: Generation number
            parent_ids: List of parent IDs (for sexual reproduction)
        """
        self.id = id
        self.genome = genome
        self.species = species
        self.position = position.astype(np.float32)
        self.energy = energy
        self.generation = generation
        self.parent_ids = parent_ids or []
        
        # State tracking
        self.age = 0
        self.births = 0
        self.kills = 0
        
    def is_alive(self) -> bool:
        """Check if creature is still alive."""
        return (self.energy > 0 and 
                self.age < self.genome.lifespan)
    
    def consume_energy(self, amount: float, min_energy: float, max_energy: float) -> None:
        """Consume energy with bounds checking.
        
        Args:
            amount: Energy to consume
            min_energy: Minimum energy level
            max_energy: Maximum energy level
        """
        self.energy = max(min_energy, min(max_energy, self.energy - amount))
    
    def gain_energy(self, amount: float, min_energy: float, max_energy: float) -> None:
        """Gain energy with bounds checking.
        
        Args:
            amount: Energy to gain
            min_energy: Minimum energy level  
            max_energy: Maximum energy level
        """
        self.energy = max(min_energy, min(max_energy, self.energy + amount))
    
    def can_reproduce(self) -> bool:
        """Check if creature has enough energy to reproduce."""
        return self.energy >= self.genome.reproduction_threshold
    
    def move(self, dx: float, dy: float, world_width: float, world_height: float) -> None:
        """Move creature with toroidal wrapping.
        
        Args:
            dx: Change in x position
            dy: Change in y position
            world_width: Width of the world
            world_height: Height of the world
        """
        # Apply movement with speed scaling
        movement_scale = self.genome.speed * 0.5  # Scale movement by speed trait
        self.position[0] = (self.position[0] + dx * movement_scale) % world_width
        self.position[1] = (self.position[1] + dy * movement_scale) % world_height
    
    def get_state_dict(self) -> dict:
        """Get creature state as dictionary for serialization."""
        return {
            'id': self.id,
            'species': self.species,
            'position': self.position.tolist(),
            'energy': self.energy,
            'age': self.age,
            'generation': self.generation,
            'genome': self.genome.__dict__,
            'parent_ids': self.parent_ids,
            'births': self.births,
            'kills': self.kills
        }
