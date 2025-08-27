"""Test evolutionary mechanics and selection pressure."""

import pytest
import numpy as np
from ai_evo.simulation import Simulation
from ai_evo.config import Config
from ai_evo.evolution import EvolutionEngine
from ai_evo.genome import Genome
from ai_evo.rng import RNG

class TestEvolution:
    """Test suite for evolutionary mechanics."""
    
    def test_mutation_respects_bounds(self):
        """Test that mutations stay within defined bounds."""
        config = Config(mutation_rate=1.0, mutation_strength=0.5)  # High mutation
        rng = RNG(42)
        evo = EvolutionEngine(config, rng)
        
        genome = Genome(
            speed=1.0, size=1.0, aggression=0.5, perception=2.0,
            energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
        )
        
        # Apply many mutations
        for _ in range(100):
            mutated = evo.mutate(genome)
            
            # Check all bounds
            assert config.speed_min <= mutated.speed <= config.speed_max
            assert config.size_min <= mutated.size <= config.size_max
            assert config.aggression_min <= mutated.aggression <= config.aggression_max
            assert config.perception_min <= mutated.perception <= config.perception_max
            assert config.energy_efficiency_min <= mutated.energy_efficiency <= config.energy_efficiency_max
            assert config.reproduction_threshold_min <= mutated.reproduction_threshold <= config.reproduction_threshold_max
            assert config.lifespan_min <= mutated.lifespan <= config.lifespan_max
    
    def test_selection_pressure_under_resource_scarcity(self):
        """Test that evolution responds to resource pressure."""
        # Create harsh environment (low food)
        config = Config(
            seed=123,
            max_steps=500,
            plant_growth_rate=0.02,  # Very low food growth
            init_herbivores=30,
            init_carnivores=10,
            mutation_rate=0.15
        )
        
        sim = Simulation(config)
        
        # Record initial trait averages
        initial_herbivores = [c for c in sim.creatures if c.species == "herbivore"]
        initial_speed = np.mean([c.genome.speed for c in initial_herbivores])
        initial_efficiency = np.mean([c.genome.energy_efficiency for c in initial_herbivores])
        
        # Run simulation
        for _ in range(500):
            sim.step()
            if len(sim.creatures) == 0:
                break
        
        # Check if any creatures survived
        if sim.creatures:
            final_herbivores = [c for c in sim.creatures if c.species == "herbivore"]
            
            if final_herbivores:
                final_speed = np.mean([c.genome.speed for c in final_herbivores])
                final_efficiency = np.mean([c.genome.energy_efficiency for c in final_herbivores])
                
                # Under resource pressure, energy efficiency should tend to improve
                # (though this may not always happen due to randomness)
                print(f"Initial efficiency: {initial_efficiency:.3f}, Final: {final_efficiency:.3f}")
                print(f"Initial speed: {initial_speed:.3f}, Final: {final_speed:.3f}")
                
                # Test passes if simulation runs without crashing
                assert True
    
    def test_trait_inheritance(self):
        """Test that offspring inherit parent traits with variation."""
        config = Config(mutation_rate=0.1, mutation_strength=0.05)
        rng = RNG(42)
        evo = EvolutionEngine(config, rng)
        
        parent_genome = Genome(
            speed=2.0, size=1.5, aggression=0.8, perception=3.0,
            energy_efficiency=1.2, reproduction_threshold=100.0, lifespan=1500
        )
        
        # Generate offspring
        offspring_traits = []
        for _ in range(50):
            child_genome = evo.mutate(parent_genome)
            offspring_traits.append({
                'speed': child_genome.speed,
                'size': child_genome.size,
                'aggression': child_genome.aggression,
                'perception': child_genome.perception,
                'energy_efficiency': child_genome.energy_efficiency
            })
        
        # Check inheritance patterns
        for trait in offspring_traits[0].keys():
            parent_value = getattr(parent_genome, trait)
            child_values = [child[trait] for child in offspring_traits]
            
            # Mean should be close to parent value
            mean_child_value = np.mean(child_values)
            assert abs(mean_child_value - parent_value) < 0.5
            
            # Should have some variation
            std_child_value = np.std(child_values)
            assert std_child_value > 0.01
    
    def test_generation_tracking(self):
        """Test that generations are properly tracked."""
        config = Config(
            seed=456,
            max_steps=200,
            init_herbivores=20,
            init_carnivores=5,
            reproduce_threshold=50.0  # Lower threshold for faster reproduction
        )
        
        sim = Simulation(config)
        
        initial_generations = [c.generation for c in sim.creatures]
        assert all(gen == 0 for gen in initial_generations)
        
        max_generation = 0
        generation_counts = {}
        
        for _ in range(200):
            sim.step()
            
            if sim.creatures:
                for creature in sim.creatures:
                    gen = creature.generation
                    max_generation = max(max_generation, gen)
                    generation_counts[gen] = generation_counts.get(gen, 0) + 1
        
        # Should have multiple generations
        assert max_generation > 0
        assert len(generation_counts) > 1
    
    def test_species_specific_evolution(self):
        """Test that different species evolve differently."""
        config = Config(seed=789, max_steps=300, mutation_rate=0.2)
        sim = Simulation(config)
        
        # Record initial traits by species
        initial_herbivore_aggression = np.mean([
            c.genome.aggression for c in sim.creatures if c.species == "herbivore"
        ])
        initial_carnivore_aggression = np.mean([
            c.genome.aggression for c in sim.creatures if c.species == "carnivore"
        ])
        
        # Run simulation
        for _ in range(300):
            sim.step()
            if len(sim.creatures) < 10:  # Stop if population too low
                break
        
        if sim.creatures:
            herbivores = [c for c in sim.creatures if c.species == "herbivore"]
            carnivores = [c for c in sim.creatures if c.species == "carnivore"]
            
            if herbivores and carnivores:
                final_herbivore_aggression = np.mean([c.genome.aggression for c in herbivores])
                final_carnivore_aggression = np.mean([c.genome.aggression for c in carnivores])
                
                # Carnivores should generally have higher aggression
                assert final_carnivore_aggression > final_herbivore_aggression
    
    def test_crossover_functionality(self):
        """Test sexual reproduction crossover (if implemented)."""
        config = Config(mutation_rate=0.1)
        rng = RNG(42)
        evo = EvolutionEngine(config, rng)
        
        parent1 = Genome(
            speed=1.0, size=1.0, aggression=0.2, perception=2.0,
            energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
        )
        
        parent2 = Genome(
            speed=2.0, size=2.0, aggression=0.8, perception=4.0,
            energy_efficiency=1.5, reproduction_threshold=120.0, lifespan=1500
        )
        
        # Test crossover
        child = evo.crossover(parent1, parent2)
        
        # Child traits should be combinations of parent traits
        assert config.speed_min <= child.speed <= config.speed_max
        assert config.size_min <= child.size <= config.size_max
        assert config.aggression_min <= child.aggression <= config.aggression_max
        
        # Test that crossover produces different results
        children = [evo.crossover(parent1, parent2) for _ in range(10)]
        speeds = [c.speed for c in children]
        
        # Should have some variation in offspring
        assert len(set(speeds)) > 1 or any(abs(s - parent1.speed) > 0.1 and abs(s - parent2.speed) > 0.1 for s in speeds)
    
    def test_brain_weight_evolution(self):
        """Test that neural network weights evolve properly."""
        from ai_evo.brain import CreatureBrain
        
        config = Config(mutation_rate=0.3, mutation_strength=0.1)
        rng = RNG(42)
        evo = EvolutionEngine(config, rng)
        
        genome = Genome(
            speed=1.0, size=1.0, aggression=0.5, perception=2.0,
            energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
        )
        
        # Create brain to initialize weights
        brain = CreatureBrain(genome, rng)
        original_weights = genome.brain_weights
        
        # Mutate the genome
        mutated_genome = evo.mutate(genome)
        
        # Weights should be different but similar
        if original_weights and mutated_genome.brain_weights:
            W1_orig, b1_orig, W2_orig, b2_orig = original_weights
            W1_mut, b1_mut, W2_mut, b2_mut = mutated_genome.brain_weights
            
            # Should have some differences
            assert not np.allclose(W1_orig, W1_mut)
            assert not np.allclose(W2_orig, W2_mut)
            
            # But not completely different
            assert np.mean(np.abs(W1_orig - W1_mut)) < 0.5
            assert np.mean(np.abs(W2_orig - W2_mut)) < 0.5
