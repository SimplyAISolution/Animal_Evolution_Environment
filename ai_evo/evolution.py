import numpy as np
from .genome import Genome
from .rng import RNG
from .config import Config

class EvolutionEngine:
    """Handles mutation (crossover hook reserved)."""
    def __init__(self, cfg: Config, rng: RNG):
        self.cfg, self.rng = cfg, rng

    def create_random_genome(self, species: str) -> Genome:
        """Create a random genome for a given species."""
        return Genome(
            speed=self.rng.rand() * (self.cfg.speed_max - self.cfg.speed_min) + self.cfg.speed_min,
            size=self.rng.rand() * (self.cfg.size_max - self.cfg.size_min) + self.cfg.size_min,
            aggression=self.rng.rand() * (self.cfg.aggression_max - self.cfg.aggression_min) + self.cfg.aggression_min,
            perception=self.rng.rand() * (self.cfg.perception_max - self.cfg.perception_min) + self.cfg.perception_min,
            energy_efficiency=self.rng.rand() * (self.cfg.energy_efficiency_max - self.cfg.energy_efficiency_min) + self.cfg.energy_efficiency_min,
            reproduction_threshold=self.rng.rand() * (self.cfg.reproduction_threshold_max - self.cfg.reproduction_threshold_min) + self.cfg.reproduction_threshold_min,
            lifespan=int(self.rng.rand() * (self.cfg.lifespan_max - self.cfg.lifespan_min) + self.cfg.lifespan_min)
        )

    def mutate(self, g: Genome) -> Genome:
        mr = self.cfg.mutation_rate
        ms = self.cfg.mutation_strength

        def mt(val, lo, hi):
            if self.rng.rand() < mr:
                val += self.rng.normal(0, ms)
            return float(min(hi, max(lo, val)))

        new_weights = self._mutate_weights(g.brain_weights, mr, ms)
        return Genome(
            speed=mt(g.speed, self.cfg.speed_min, self.cfg.speed_max),
            size=mt(g.size, self.cfg.size_min, self.cfg.size_max),
            aggression=mt(g.aggression, self.cfg.aggression_min, self.cfg.aggression_max),
            perception=mt(g.perception, self.cfg.perception_min, self.cfg.perception_max),
            energy_efficiency=mt(g.energy_efficiency, self.cfg.energy_efficiency_min, self.cfg.energy_efficiency_max),
            reproduction_threshold=mt(g.reproduction_threshold, self.cfg.reproduction_threshold_min, self.cfg.reproduction_threshold_max),
            lifespan=int(mt(g.lifespan, self.cfg.lifespan_min, self.cfg.lifespan_max)),
            brain_weights=new_weights
        )

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Create offspring through crossover of two parents."""
        # Simple average crossover for most traits
        offspring = Genome(
            speed=(parent1.speed + parent2.speed) / 2,
            size=(parent1.size + parent2.size) / 2,
            aggression=(parent1.aggression + parent2.aggression) / 2,
            perception=(parent1.perception + parent2.perception) / 2,
            energy_efficiency=(parent1.energy_efficiency + parent2.energy_efficiency) / 2,
            reproduction_threshold=(parent1.reproduction_threshold + parent2.reproduction_threshold) / 2,
            lifespan=int((parent1.lifespan + parent2.lifespan) / 2),
            brain_weights=self._crossover_weights(parent1.brain_weights, parent2.brain_weights)
        )
        return offspring
        
    def _crossover_weights(self, weights1, weights2):
        """Crossover brain weights between two parents."""
        if weights1 is None or weights2 is None:
            return weights1 if weights1 is not None else weights2
            
        if isinstance(weights1, tuple) and isinstance(weights2, tuple):
            # Handle tuple format (W1, b1, W2, b2)
            W1_1, b1_1, W2_1, b2_1 = weights1
            W1_2, b1_2, W2_2, b2_2 = weights2
            
            def crossover_array(arr1, arr2):
                mask = self.rng.random_bool(0.5, arr1.shape)
                result = arr1.copy()
                result[mask] = arr2[mask]
                return result
                
            return (
                crossover_array(W1_1, W1_2),
                crossover_array(b1_1, b1_2),
                crossover_array(W2_1, W2_2),
                crossover_array(b2_1, b2_2)
            )
        else:
            # Handle simple array format
            mask = self.rng.random_bool(0.5, weights1.shape)
            result = weights1.copy()
            result[mask] = weights2[mask]
            return result

    def _mutate_weights(self, weights, mr, ms):
        if weights is None:
            return None
            
        if isinstance(weights, tuple) and len(weights) == 4:
            # Handle tuple format (W1, b1, W2, b2)
            W1, b1, W2, b2 = weights
            def mutate_array(arr):
                mask = self.rng.random_bool(mr, arr.shape)
                arr2 = arr.copy()
                arr2[mask] += self.rng.normal(0, ms, mask.sum())
                return np.clip(arr2, -2.0, 2.0)
            return (
                mutate_array(W1),
                mutate_array(b1),
                mutate_array(W2),
                mutate_array(b2)
            )
        else:
            # Handle simple array format
            mask = self.rng.random_bool(mr, weights.shape)
            arr2 = weights.copy()
            arr2[mask] += self.rng.normal(0, ms, mask.sum())
            return np.clip(arr2, -2.0, 2.0)
