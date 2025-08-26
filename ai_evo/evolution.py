import numpy as np
from .genome import Genome
from .rng import RNG
from .config import Config

class EvolutionEngine:
    """Handles mutation (crossover hook reserved)."""
    def __init__(self, cfg: Config, rng: RNG):
        self.cfg, self.rng = cfg, rng

    def mutate(self, g: Genome) -> Genome:
        mr = self.cfg.mutation_rate
        ms = self.cfg.mutation_strength

        def mt(val, lo, hi):
            if self.rng.rand() < mr:
                val += self.rng.normal(0, ms)
            return float(min(hi, max(lo, val)))

        new_weights = self._mutate_weights(g.brain_weights, mr, ms)
        return Genome(
            speed=mt(g.speed, 0.1, 3.0),
            size=mt(g.size, 0.4, 2.5),
            aggression=mt(g.aggression, 0.0, 1.0),
            perception=mt(g.perception, 0.5, self.cfg.perception_max),
            energy_efficiency=mt(g.energy_efficiency, 0.4, 2.5),
            reproduction_threshold=mt(g.reproduction_threshold, 50.0, 160.0),
            lifespan=int(mt(g.lifespan, 300, 2400)),
            brain_weights=new_weights
        )

    def _mutate_weights(self, weights, mr, ms):
        if weights is None:
            return None
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
