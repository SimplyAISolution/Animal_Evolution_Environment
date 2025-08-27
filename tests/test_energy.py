"""Test energy conservation and flow mechanics."""

import pytest
import numpy as np
from ai_evo.simulation import Simulation
from ai_evo.config import Config
from ai_evo.creature import Creature
from ai_evo.genome import Genome

class TestEnergy:
    """Test suite for energy conservation and flow mechanics."""
    
    def test_energy_conservation(self):
        """Test that total system energy is conserved (within expected bounds)."""
        config = Config(
            seed=42,
            max_steps=100,
            init_herbivores=30,
            init_carnivores=15,
            plant_growth_rate=0.1,
            width=60,
            height=60
        )
        
        sim = Simulation(config)
        
        energy_history = []
        
        for step in range(100):
            # Calculate total system energy
            creature_energy = sum(c.energy for c in sim.creatures)
            food_energy = float(np.sum(sim.environment.food)) * 8.0  # Food->energy conversion
            total_energy = creature_energy + food_energy
            
            energy_history.append({
                'step': step,
                'creature_energy': creature_energy,
                'food_energy': food_energy,
                'total_energy': total_energy,
                'population': len(sim.creatures)
            })
            
            sim.step()
            
            if len(sim.creatures) == 0:
                break
        
        # Analyze energy flow
        if len(energy_history) > 10:
            initial_energy = energy_history[0]['total_energy']
            final_energy = energy_history[-1]['total_energy']
            
            # Energy should increase due to plant growth but not explode
            energy_change = final_energy - initial_energy
            print(f"Initial energy: {initial_energy:.1f}")
            print(f"Final energy: {final_energy:.1f}")
            print(f"Energy change: {energy_change:.1f}")
            
            # Should not have extreme energy changes (indicating bugs)
            assert abs(energy_change) < initial_energy * 2.0
    
    def test_creature_energy_bounds(self):
        """Test that creature energy stays within configured bounds."""
        config = Config(
            seed=123,
            max_steps=200,
            min_energy=0.0,
            max_energy=200.0
        )
        
        sim = Simulation(config)
        
        for step in range(200):
            # Check all creature energies
            for creature in sim.creatures:
                assert creature.energy >= config.min_energy
                assert creature.energy <= config.max_energy
            
            sim.step()
            
            if len(sim.creatures) == 0:
                break
    
    def test_herbivore_feeding_mechanics(self):
        """Test herbivore food consumption and energy gain."""
        from ai_evo.world import Environment
        from ai_evo.rng import RNG
        
        config = Config(herbivore_bite_cap=2.0, max_energy=200.0)
        rng = RNG(42)
        env = Environment(config, rng)
        
        # Create test herbivore
        genome = Genome(
            speed=1.0, size=1.0, aggression=0.5, perception=2.0,
            energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
        )
        
        herbivore = Creature(
            id="test_herbivore",
            genome=genome,
            species="herbivore",
            position=np.array([10.0, 10.0]),
            energy=50.0
        )
        
        # Set food at herbivore location
        env.food[10, 10] = 5.0
        initial_food = env.food[10, 10]
        initial_energy = herbivore.energy
        
        # Simulate feeding
        x, y = herbivore.position[0], herbivore.position[1]
        food_consumed = env.consume_food_at(x, y, config.herbivore_bite_cap)
        
        if food_consumed > 0:
            energy_gained = food_consumed * 8.0  # Energy conversion
            herbivore.gain_energy(energy_gained, config.max_energy)
        
        # Verify energy gain and food consumption
        assert food_consumed > 0
        assert food_consumed <= config.herbivore_bite_cap
        assert food_consumed <= initial_food
        assert herbivore.energy > initial_energy
        assert env.food[10, 10] < initial_food
    
    def test_carnivore_hunting_mechanics(self):
        """Test carnivore hunting and energy transfer."""
        config = Config(carnivore_gain_eff=0.75, max_energy=200.0)
        
        # Create predator
        predator_genome = Genome(
            speed=2.0, size=1.5, aggression=0.9, perception=3.0,
            energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
        )
        
        predator = Creature(
            id="predator",
            genome=predator_genome,
            species="carnivore",
            position=np.array([10.0, 10.0]),
            energy=60.0
        )
        
        # Create prey
        prey_genome = Genome(
            speed=1.0, size=0.8, aggression=0.2, perception=2.0,
            energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
        )
        
        prey = Creature(
            id="prey",
            genome=prey_genome,
            species="herbivore",
            position=np.array([10.5, 10.5]),  # Close to predator
            energy=80.0
        )
        
        initial_predator_energy = predator.energy
        prey_energy = prey.energy
        
        # Simulate hunting (simplified)
        distance = predator.distance_to(prey)
        assert distance < 2.0  # Within attack range
        
        # Combat resolution
        attack_power = predator.genome.aggression * predator.genome.size
        defense_power = prey.genome.size * 0.5
        
        if attack_power > defense_power:
            # Successful hunt
            energy_gained = prey_energy * config.carnivore_gain_eff
            predator.gain_energy(energy_gained, config.max_energy)
            
            # Verify energy transfer
            assert predator.energy > initial_predator_energy
            expected_energy = min(
                config.max_energy,
                initial_predator_energy + energy_gained
            )
            assert abs(predator.energy - expected_energy) < 0.001
    
    def test_reproduction_energy_cost(self):
        """Test that reproduction properly costs energy."""
        config = Config(
            reproduce_threshold=100.0,
            reproduce_cost_frac=0.6,
            max_energy=200.0
        )
        
        # Create creature with enough energy to reproduce
        genome = Genome(
            speed=1.0, size=1.0, aggression=0.5, perception=2.0,
            energy_efficiency=1.0, reproduction_threshold=100.0, lifespan=1000
        )
        
        parent = Creature(
            id="parent",
            genome=genome,
            species="herbivore",
            position=np.array([20.0, 20.0]),
            energy=120.0
        )
        
        initial_energy = parent.energy
        
        # Check reproduction eligibility
        assert parent.can_reproduce()
        
        # Simulate reproduction cost
        reproduction_cost = parent.genome.reproduction_threshold * config.reproduce_cost_frac
        parent.consume_energy(reproduction_cost, 0.0, config.max_energy)
        
        # Verify energy was consumed
        expected_final_energy = initial_energy - reproduction_cost
        assert abs(parent.energy - expected_final_energy) < 0.001
        assert parent.energy < initial_energy
        
        # Should still be alive
        assert parent.energy > 0
    
    def test_movement_energy_cost(self):
        """Test that movement consumes energy based on traits."""
        config = Config(move_cost_base=0.2, max_energy=200.0, min_energy=0.0)
        
        # Create creatures with different traits
        fast_genome = Genome(
            speed=2.5, size=1.0, aggression=0.5, perception=2.0,
            energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
        )
        
        slow_genome = Genome(
            speed=0.5, size=1.0, aggression=0.5, perception=2.0,
            energy_efficiency=2.0, reproduction_threshold=90.0, lifespan=1000  # More efficient
        )
        
        fast_creature = Creature(
            id="fast", genome=fast_genome, species="herbivore",
            position=np.array([10.0, 10.0]), energy=100.0
        )
        
        slow_creature = Creature(
            id="slow", genome=slow_genome, species="herbivore",
            position=np.array([10.0, 10.0]), energy=100.0
        )
        
        # Simulate movement actions
        movement_intensity = 1.0  # Full movement
        
        # Calculate movement costs
        fast_move_cost = (config.move_cost_base * fast_creature.genome.size * 
                         (0.5 + 0.5 * movement_intensity))
        fast_metabolic_cost = fast_creature.genome.size * 0.1
        fast_total_cost = (fast_move_cost + fast_metabolic_cost) / fast_creature.genome.energy_efficiency
        
        slow_move_cost = (config.move_cost_base * slow_creature.genome.size * 
                         (0.5 + 0.5 * movement_intensity))
        slow_metabolic_cost = slow_creature.genome.size * 0.1
        slow_total_cost = (slow_move_cost + slow_metabolic_cost) / slow_creature.genome.energy_efficiency
        
        # Apply costs
        fast_creature.consume_energy(fast_total_cost, config.min_energy, config.max_energy)
        slow_creature.consume_energy(slow_total_cost, config.min_energy, config.max_energy)
        
        # Efficient creature should consume less energy
        fast_energy_loss = 100.0 - fast_creature.energy
        slow_energy_loss = 100.0 - slow_creature.energy
        
        assert slow_energy_loss < fast_energy_loss
        # Both creatures should have lost some energy
        assert fast_energy_loss > 0
        assert slow_energy_loss > 0
    
    def test_starvation_mechanics(self):
        """Test that creatures die when energy reaches zero."""
        config = Config(min_energy=0.0, move_cost_base=1.0)  # High movement cost
        
        genome = Genome(
            speed=3.0, size=2.0, aggression=0.5, perception=2.0,
            energy_efficiency=0.5, reproduction_threshold=90.0, lifespan=1000  # Inefficient
        )
        
        creature = Creature(
            id="test", genome=genome, species="herbivore",
            position=np.array([10.0, 10.0]), energy=5.0  # Low initial energy
        )
        
        # Creature should be alive initially
        assert creature.is_alive()
        
        # Apply high energy cost to simulate starvation
        high_cost = 10.0
        creature.consume_energy(high_cost, config.min_energy, 200.0)
        
        # Creature should now have zero energy and be dead
        assert creature.energy == config.min_energy
        assert not creature.is_alive()
    
    def test_energy_efficiency_trait_impact(self):
        """Test that energy efficiency trait affects actual energy consumption."""
        config = Config(move_cost_base=0.5, min_energy=0.0, max_energy=200.0)
        
        # Create creatures with different energy efficiencies
        efficient_genome = Genome(
            speed=1.0, size=1.0, aggression=0.5, perception=2.0,
            energy_efficiency=2.0, reproduction_threshold=90.0, lifespan=1000  # Highly efficient
        )
        
        inefficient_genome = Genome(
            speed=1.0, size=1.0, aggression=0.5, perception=2.0,
            energy_efficiency=0.5, reproduction_threshold=90.0, lifespan=1000  # Inefficient
        )
        
        efficient_creature = Creature(
            id="efficient", genome=efficient_genome, species="herbivore",
            position=np.array([0.0, 0.0]), energy=100.0
        )
        
        inefficient_creature = Creature(
            id="inefficient", genome=inefficient_genome, species="herbivore",
            position=np.array([0.0, 0.0]), energy=100.0
        )
        
        # Apply same base energy cost
        base_cost = 10.0
        
        efficient_creature.consume_energy(
            base_cost / efficient_creature.genome.energy_efficiency,
            config.min_energy, config.max_energy
        )
        
        inefficient_creature.consume_energy(
            base_cost / inefficient_creature.genome.energy_efficiency,
            config.min_energy, config.max_energy
        )
        
        # Efficient creature should have more energy remaining
        assert efficient_creature.energy > inefficient_creature.energy
        
        # Calculate actual energy consumed
        efficient_consumed = 100.0 - efficient_creature.energy
        inefficient_consumed = 100.0 - inefficient_creature.energy
        
        # Verify the efficiency ratio
        efficiency_ratio = inefficient_consumed / efficient_consumed
        expected_ratio = efficient_genome.energy_efficiency / inefficient_genome.energy_efficiency
        
        assert abs(efficiency_ratio - expected_ratio) < 0.1
