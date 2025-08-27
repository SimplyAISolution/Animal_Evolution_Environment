"""Evolution engine for genome creation and mutation."""

import numpy as np
from .genome import Genome
from .rng import RNG
from .config import Config


class EvolutionEngine:
    """Handles genome creation, mutation and evolution mechanics."""
    
    def __init__(self, cfg: Config, rng: RNG):
        """Initialize evolution engine.
        
        Args:
            cfg: Configuration object
            rng: Random number generator
        """
        self.cfg = cfg
        self.rng = rng

    def create_random_genome(self, species: str) -> Genome:
        """Create a random genome for a given species.
        
        Args:
            species: "herbivore" or "carnivore"
            
        Returns:
            New random genome
        """
        # Base traits with some species-specific tendencies
        if species == "herbivore":
            base_aggression = 0.2
            base_speed = 1.2
        else:  # carnivore
            base_aggression = 0.7
            base_speed = 1.0
        
        return Genome(
            speed=max(0.1, min(3.0, base_speed + self.rng.normal(0, 0.3))),
            size=max(0.5, min(2.0, 1.0 + self.rng.normal(0, 0.2))),
            aggression=max(0.0, min(1.0, base_aggression + self.rng.normal(0, 0.2))),
            perception=max(1.0, min(10.0, 2.0 + self.rng.normal(0, 0.5))),
            energy_efficiency=max(0.5, min(2.0, 1.0 + self.rng.normal(0, 0.2))),
            reproduction_threshold=max(50.0, min(150.0, 80.0 + self.rng.normal(0, 15.0))),
            lifespan=max(500, min(2000, 1000 + int(self.rng.normal(0, 200))))
        )

    def mutate(self, genome: Genome) -> Genome:
        """Create a mutated copy of a genome.
        
        Args:
            genome: Parent genome to mutate
            
        Returns:
            Mutated genome
        """
        return genome.mutate(self.rng, self.cfg.mutation_rate, self.cfg.mutation_strength)

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Create offspring genome from two parents.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Offspring genome
        """
        # Create base offspring through crossover
        offspring = parent1.crossover(parent2, self.rng)
        
        # Apply mutation to offspring
        return self.mutate(offspring)

    def get_fitness_score(self, creature) -> float:
        """Calculate fitness score for a creature.
        
        Args:
            creature: Creature to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        # Fitness based on survival, energy, reproduction success
        age_factor = min(creature.age / 500.0, 1.0)  # Reward longevity
        energy_factor = creature.energy / 100.0      # Reward high energy
        reproduction_factor = creature.births * 2.0  # Reward reproductive success
        
        # For carnivores, also reward hunting success
        if creature.species == "carnivore":
            hunting_factor = creature.kills * 1.5
        else:
            hunting_factor = 0
        
        return age_factor + energy_factor + reproduction_factor + hunting_factor
