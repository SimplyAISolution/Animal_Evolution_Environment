"""Genome representation and mutation mechanics."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Genome:
    """Genetic traits that define creature capabilities."""
    
    # Physical traits
    speed: float = 1.0              # Movement speed multiplier
    size: float = 1.0               # Size affects energy costs and combat
    aggression: float = 0.5         # Tendency to attack (carnivores)
    perception: float = 2.0         # Sensory range
    
    # Efficiency traits
    energy_efficiency: float = 1.0  # Energy cost reduction
    
    # Life cycle
    reproduction_threshold: float = 80.0  # Energy needed to reproduce
    lifespan: int = 1000            # Maximum age
    
    def __post_init__(self):
        """Ensure trait values are within valid ranges."""
        self.speed = max(0.1, min(3.0, self.speed))
        self.size = max(0.5, min(2.0, self.size)) 
        self.aggression = max(0.0, min(1.0, self.aggression))
        self.perception = max(1.0, min(10.0, self.perception))
        self.energy_efficiency = max(0.5, min(2.0, self.energy_efficiency))
        self.reproduction_threshold = max(50.0, min(150.0, self.reproduction_threshold))
        self.lifespan = max(500, min(2000, self.lifespan))
    
    def mutate(self, rng, mutation_rate: float, mutation_strength: float) -> 'Genome':
        """Create a mutated copy of this genome.
        
        Args:
            rng: Random number generator
            mutation_rate: Probability of each trait mutating
            mutation_strength: Standard deviation of mutations
            
        Returns:
            New mutated genome
        """
        new_genome = Genome(
            speed=self.speed,
            size=self.size,
            aggression=self.aggression,
            perception=self.perception,
            energy_efficiency=self.energy_efficiency,
            reproduction_threshold=self.reproduction_threshold,
            lifespan=self.lifespan
        )
        
        # Mutate each trait independently
        if rng.random_bool(mutation_rate):
            new_genome.speed += rng.normal(0, mutation_strength)
            
        if rng.random_bool(mutation_rate):
            new_genome.size += rng.normal(0, mutation_strength)
            
        if rng.random_bool(mutation_rate):
            new_genome.aggression += rng.normal(0, mutation_strength)
            
        if rng.random_bool(mutation_rate):
            new_genome.perception += rng.normal(0, mutation_strength)
            
        if rng.random_bool(mutation_rate):
            new_genome.energy_efficiency += rng.normal(0, mutation_strength)
            
        if rng.random_bool(mutation_rate):
            new_genome.reproduction_threshold += rng.normal(0, mutation_strength * 10)
            
        if rng.random_bool(mutation_rate):
            new_genome.lifespan += int(rng.normal(0, mutation_strength * 100))
        
        # Re-apply constraints
        new_genome.__post_init__()
        return new_genome
    
    def crossover(self, other: 'Genome', rng) -> 'Genome':
        """Create offspring genome through crossover with another genome.
        
        Args:
            other: Other parent genome
            rng: Random number generator
            
        Returns:
            New offspring genome
        """
        # Simple uniform crossover - randomly choose each trait from either parent
        return Genome(
            speed=self.speed if rng.random_bool() else other.speed,
            size=self.size if rng.random_bool() else other.size,
            aggression=self.aggression if rng.random_bool() else other.aggression,
            perception=self.perception if rng.random_bool() else other.perception,
            energy_efficiency=self.energy_efficiency if rng.random_bool() else other.energy_efficiency,
            reproduction_threshold=self.reproduction_threshold if rng.random_bool() else other.reproduction_threshold,
            lifespan=self.lifespan if rng.random_bool() else other.lifespan
        )
