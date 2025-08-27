"""Test deterministic behavior across simulation runs."""

import pytest
import numpy as np
from ai_evo.simulation import Simulation
from ai_evo.config import Config

class TestDeterminism:
    """Test suite for deterministic simulation behavior."""
    
    def test_identical_seeds_produce_identical_results(self):
        """Test that same seed produces identical simulation outcomes."""
        config = Config(seed=12345, max_steps=100, width=50, height=50)
        
        # Run first simulation
        sim1 = Simulation(config)
        results1 = []
        
        for _ in range(100):
            sim1.step()
            results1.append({
                "step": sim1.step_count,
                "population": len(sim1.creatures),
                "herbivores": sum(1 for c in sim1.creatures if c.species == "herbivore"),
                "carnivores": sum(1 for c in sim1.creatures if c.species == "carnivore"),
                "total_food": float(np.sum(sim1.environment.food))
            })
        
        # Run second simulation with same seed
        sim2 = Simulation(config)
        results2 = []
        
        for _ in range(100):
            sim2.step()
            results2.append({
                "step": sim2.step_count,
                "population": len(sim2.creatures),
                "herbivores": sum(1 for c in sim2.creatures if c.species == "herbivore"),
                "carnivores": sum(1 for c in sim2.creatures if c.species == "carnivore"),
                "total_food": float(np.sum(sim2.environment.food))
            })
        
        # Compare results
        assert len(results1) == len(results2)
        
        for r1, r2 in zip(results1, results2):
            assert r1["step"] == r2["step"]
            assert r1["population"] == r2["population"]
            assert r1["herbivores"] == r2["herbivores"]
            assert r1["carnivores"] == r2["carnivores"]
            assert abs(r1["total_food"] - r2["total_food"]) < 1e-6
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different outcomes."""
        config1 = Config(seed=12345, max_steps=50)
        config2 = Config(seed=54321, max_steps=50)
        
        sim1 = Simulation(config1)
        sim2 = Simulation(config2)
        
        # Run both simulations
        for _ in range(50):
            sim1.step()
            sim2.step()
        
        # Results should be different
        pop1 = len(sim1.creatures)
        pop2 = len(sim2.creatures)
        
        food1 = float(np.sum(sim1.environment.food))
        food2 = float(np.sum(sim2.environment.food))
        
        # At least one metric should be different
        assert pop1 != pop2 or abs(food1 - food2) > 1e-6
    
    def test_creature_reproduction_determinism(self):
        """Test that creature reproduction is deterministic."""
        config = Config(seed=42, max_steps=200, init_herbivores=10, init_carnivores=5)
        
        sim1 = Simulation(config)
        sim2 = Simulation(config)
        
        birth_sequence1 = []
        birth_sequence2 = []
        
        for _ in range(200):
            births1 = sim1.total_births
            births2 = sim2.total_births
            
            sim1.step()
            sim2.step()
            
            new_births1 = sim1.total_births - births1
            new_births2 = sim2.total_births - births2
            
            birth_sequence1.append(new_births1)
            birth_sequence2.append(new_births2)
        
        assert birth_sequence1 == birth_sequence2
    
    def test_mutation_determinism(self):
        """Test that genetic mutations are deterministic."""
        from ai_evo.evolution import EvolutionEngine
        from ai_evo.genome import Genome
        from ai_evo.rng import RNG
        
        config = Config(seed=999, mutation_rate=0.5, mutation_strength=0.1)
        
        # Create identical starting genomes
        genome1 = Genome(
            speed=1.0, size=1.0, aggression=0.5, perception=2.0,
            energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
        )
        genome2 = Genome(
            speed=1.0, size=1.0, aggression=0.5, perception=2.0,
            energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
        )
        
        # Create identical evolution engines
        rng1 = RNG(config.seed)
        rng2 = RNG(config.seed)
        
        evo1 = EvolutionEngine(config, rng1)
        evo2 = EvolutionEngine(config, rng2)
        
        # Apply mutations
        mutated1 = evo1.mutate(genome1)
        mutated2 = evo2.mutate(genome2)
        
        # Results should be identical
        assert abs(mutated1.speed - mutated2.speed) < 1e-10
        assert abs(mutated1.size - mutated2.size) < 1e-10
        assert abs(mutated1.aggression - mutated2.aggression) < 1e-10
        assert abs(mutated1.perception - mutated2.perception) < 1e-10
        assert abs(mutated1.energy_efficiency - mutated2.energy_efficiency) < 1e-10
        assert abs(mutated1.reproduction_threshold - mutated2.reproduction_threshold) < 1e-10
        assert mutated1.lifespan == mutated2.lifespan
    
    def test_environment_determinism(self):
        """Test that environment updates are deterministic."""
        from ai_evo.world import Environment
        from ai_evo.rng import RNG
        
        config = Config(seed=777, width=20, height=20)
        rng1 = RNG(config.seed)
        rng2 = RNG(config.seed)
        
        env1 = Environment(config, rng1)
        env2 = Environment(config, rng2)
        
        # Initial food should be identical
        assert np.allclose(env1.food, env2.food)
        
        # Step both environments
        for _ in range(50):
            env1.step()
            env2.step()
            
            # Food grids should remain identical
            assert np.allclose(env1.food, env2.food)
            assert abs(env1.temperature - env2.temperature) < 1e-10
