"""Integration tests for end-to-end simulation functionality."""

import pytest
import numpy as np
from ai_evo.simulation import Simulation
from ai_evo.config import Config

class TestIntegration:
    """Integration test suite for complete simulation workflows."""
    
    def test_complete_simulation_run(self):
        """Test that a complete simulation runs without errors."""
        config = Config(
            seed=42,
            max_steps=500,
            init_herbivores=40,
            init_carnivores=20,
            width=80,
            height=80,
            plant_growth_rate=0.1,
            mutation_rate=0.1
        )
        
        sim = Simulation(config)
        
        # Track simulation progress
        step_data = []
        exception_count = 0
        
        for step in range(500):
            try:
                should_continue = sim.step()
                
                # Record step data
                step_info = {
                    'step': sim.step_count,
                    'population': len(sim.creatures),
                    'herbivores': sum(1 for c in sim.creatures if c.species == "herbivore"),
                    'carnivores': sum(1 for c in sim.creatures if c.species == "carnivore"),
                    'avg_generation': np.mean([c.generation for c in sim.creatures]) if sim.creatures else 0,
                    'total_food': float(np.sum(sim.environment.food)),
                    'births': sim.total_births,
                    'deaths': sim.total_deaths
                }
                step_data.append(step_info)
                
                # Validate step consistency
                assert sim.step_count == step + 1
                assert step_info['population'] >= 0
                assert step_info['herbivores'] >= 0
                assert step_info['carnivores'] >= 0
                assert step_info['births'] >= 0
                assert step_info['deaths'] >= 0
                
                if not should_continue:
                    break
                    
            except Exception as e:
                exception_count += 1
                print(f"Exception at step {step}: {e}")
                if exception_count > 5:  # Too many exceptions
                    raise
        
        # Simulation should run for reasonable duration
        assert len(step_data) >= 50
        assert exception_count == 0
        
        # Should have evolutionary activity
        final_data = step_data[-1]
        assert final_data['births'] > 0
        
        # Should have multiple generations if run long enough
        if len(step_data) > 200:
            assert final_data['avg_generation'] > 0
    
    def test_ecosystem_dynamics(self):
        """Test that ecosystem shows realistic population dynamics."""
        config = Config(
            seed=123,
            max_steps=400,
            init_herbivores=60,
            init_carnivores=30,
            plant_growth_rate=0.12,
            width=100,
            height=100
        )
        
        sim = Simulation(config)
        
        population_history = {
            'herbivores': [],
            'carnivores': [],
            'total_food': []
        }
        
        for _ in range(400):
            # Record populations
            herbivores = sum(1 for c in sim.creatures if c.species == "herbivore")
            carnivores = sum(1 for c in sim.creatures if c.species == "carnivore")
            total_food = float(np.sum(sim.environment.food))
            
            population_history['herbivores'].append(herbivores)
            population_history['carnivores'].append(carnivores)
            population_history['total_food'].append(total_food)
            
            should_continue = sim.step()
            if not should_continue:
                break
        
        # Analyze ecosystem dynamics
        h_pop = population_history['herbivores']
        c_pop = population_history['carnivores']
        food = population_history['total_food']
        
        # Should have population fluctuations (not static)
        if len(h_pop) > 100:
            h_variance = np.var(h_pop[50:])  # Exclude initial settling period
            c_variance = np.var(c_pop[50:])
            
            assert h_variance > 0  # Herbivore population should vary
            assert c_variance > 0  # Carnivore population should vary
        
        # Food levels should be dynamic
        food_variance = np.var(food)
        assert food_variance > 0
        
        # Should maintain some population for reasonable time
        non_zero_steps = sum(1 for h, c in zip(h_pop, c_pop) if h > 0 or c > 0)
        assert non_zero_steps > len(h_pop) * 0.5  # At least 50% of steps with creatures
    
    def test_evolutionary_pressure_response(self):
        """Test that populations evolve in response to environmental pressure."""
        # High mutation rate for faster evolution
        config = Config(
            seed=456,
            max_steps=600,
            init_herbivores=50,
            init_carnivores=25,
            mutation_rate=0.2,
            mutation_strength=0.1,
            plant_growth_rate=0.08,  # Lower food = selection pressure
            width=80,
            height=80
        )
        
        sim = Simulation(config)
        
        # Track trait evolution
        trait_history = {
            'herbivore_speed': [],
            'herbivore_efficiency': [],
            'carnivore_aggression': [],
            'carnivore_speed': []
        }
        
        generation_milestones = []
        
        for step in range(600):
            sim.step()
            
            # Record traits every 20 steps
            if step % 20 == 0:
                herbivores = [c for c in sim.creatures if c.species == "herbivore"]
                carnivores = [c for c in sim.creatures if c.species == "carnivore"]
                
                if herbivores:
                    trait_history['herbivore_speed'].append(
                        np.mean([c.genome.speed for c in herbivores])
                    )
                    trait_history['herbivore_efficiency'].append(
                        np.mean([c.genome.energy_efficiency for c in herbivores])
                    )
                
                if carnivores:
                    trait_history['carnivore_aggression'].append(
                        np.mean([c.genome.aggression for c in carnivores])
                    )
                    trait_history['carnivore_speed'].append(
                        np.mean([c.genome.speed for c in carnivores])
                    )
                
                # Track generation progress
                if sim.creatures:
                    avg_generation = np.mean([c.generation for c in sim.creatures])
                    generation_milestones.append(avg_generation)
            
            if len(sim.creatures) == 0:
                break
        
        # Should achieve multiple generations
        assert max(generation_milestones) > 1.0
        
        # Should show trait variation over time (evolution happening)
        for trait_name, trait_values in trait_history.items():
            if len(trait_values) > 10:
                trait_variance = np.var(trait_values)
                assert trait_variance > 0, f"{trait_name} should show variation over time"
    
    def test_simulation_reset_functionality(self):
        """Test that simulation reset works correctly."""
        config = Config(seed=789, max_steps=100)
        sim = Simulation(config)
        
        # Run simulation for some steps
        initial_population = len(sim.creatures)
        
        for _ in range(50):
            sim.step()
        
        mid_population = len(sim.creatures)
        mid_step_count = sim.step_count
        mid_births = sim.total_births
        
        # Reset simulation
        sim.reset()
        
        # Check reset state
        assert sim.step_count == 0
        assert sim.total_births == 0
        assert sim.total_deaths == 0
        assert len(sim.creatures) == initial_population
        
        # All creatures should be generation 0 again
        assert all(c.generation == 0 for c in sim.creatures)
        
        # Environment should be reset
        assert sim.environment.time_step == 0
        
        # Statistics should be reset
        assert len(sim.statistics.step_data) == 1  # Initial recording
    
    def test_data_export_and_statistics(self):
        """Test data export and statistics generation."""
        config = Config(
            seed=999,
            max_steps=200,
            snapshot_every=10  # Frequent snapshots
        )
        
        sim = Simulation(config)
        
        # Run simulation
        for _ in range(200):
            sim.step()
            if len(sim.creatures) == 0:
                break
        
        # Test statistics generation
        stats = sim.statistics
        
        # Should have recorded multiple snapshots
        assert len(stats.step_data) > 5
        
        # Test population history
        pop_history = stats.get_population_history()
        assert len(pop_history['steps']) > 0
        assert len(pop_history['total']) == len(pop_history['steps'])
        assert len(pop_history['herbivores']) == len(pop_history['steps'])
        assert len(pop_history['carnivores']) == len(pop_history['steps'])
        
        # Test environment history
        env_history = stats.get_environment_history()
        assert len(env_history['steps']) > 0
        assert len(env_history['total_food']) == len(env_history['steps'])
        
        # Test trait evolution (if any species survived)
        for species in ['herbivore', 'carnivore']:
            trait_data = stats.get_trait_evolution(species, 'speed')
            # Should either have data or be empty (if species extinct)
            assert isinstance(trait_data['steps'], list)
            assert isinstance(trait_data['mean'], list)
            assert isinstance(trait_data['std'], list)
        
        # Test report generation
        report = stats.generate_report()
        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial report
        assert "Evolution Simulation Report" in report
    
    def test_edge_case_handling(self):
        """Test handling of edge cases and boundary conditions."""
        # Test with minimal population
        config = Config(
            seed=111,
            init_herbivores=2,
            init_carnivores=1,
            max_steps=100,
            width=20,
            height=20
        )
        
        sim = Simulation(config)
        
        # Should handle small populations without crashing
        for _ in range(100):
            try:
                should_continue = sim.step()
                if not should_continue:
                    break
            except Exception as e:
                pytest.fail(f"Small population simulation failed: {e}")
        
        # Test with extreme configuration
        extreme_config = Config(
            seed=222,
            init_herbivores=1,
            init_carnivores=0,  # No carnivores
            max_steps=50,
            plant_growth_rate=0.0,  # No plant growth
            width=10,
            height=10
        )
        
        extreme_sim = Simulation(extreme_config)
        
        # Should handle extreme conditions
        for _ in range(50):
            try:
                should_continue = extreme_sim.step()
                if not should_continue:
                    break
            except Exception as e:
                pytest.fail(f"Extreme configuration simulation failed: {e}")
        
        # Test boundary wrapping
        config.width = 5
        config.height = 5
        boundary_sim = Simulation(config)
        
        # Move creatures to test boundary wrapping
        for creature in boundary_sim.creatures:
            creature.update_position(-1, -1, config.width, config.height)
            assert 0 <= creature.position[0] < config.width
            assert 0 <= creature.position[1] < config.height
            
            creature.update_position(config.width + 1, config.height + 1, 
                                   config.width, config.height)
            assert 0 <= creature.position[0] < config.width
            assert 0 <= creature.position[1] < config.height
