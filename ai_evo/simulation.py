"""Main simulation controller with batch processing and safety checks."""

import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from .config import Config
from .rng import RNG
from .world import Environment
from .creatures import Creature
from .brain import CreatureBrain
from .evolution import EvolutionEngine
from .spatial import SpatialHash
from .stats import Statistics

class Simulation:
    """Main simulation controller managing creatures, environment, and evolution."""
    
    def __init__(self, cfg: Config):
        """Initialize simulation with configuration."""
        self.cfg = cfg
        self.rng = RNG(cfg.seed)
        self.environment = Environment(cfg, self.rng)
        self.evolution_engine = EvolutionEngine(cfg, self.rng)
        self.spatial_hash = SpatialHash(cfg.grid_cell, cfg.width, cfg.height)
        self.statistics = Statistics()
        
        # Creature management
        self.creatures: List[Creature] = []
        self.next_creature_id = 0
        
        # Performance tracking
        self.step_count = 0
        self.total_births = 0
        self.total_deaths = 0
        
        # Initialize starting population
        self._create_initial_population()
        
        # Record initial statistics
        self._record_statistics()
    
    def _create_initial_population(self) -> None:
        """Create initial random population of creatures."""
        # Create herbivores
        for _ in range(self.cfg.init_herbivores):
            genome = self.evolution_engine.create_random_genome("herbivore")
            creature = self._create_creature(genome, "herbivore", generation=0)
            self.creatures.append(creature)
        
        # Create carnivores
        for _ in range(self.cfg.init_carnivores):
            genome = self.evolution_engine.create_random_genome("carnivore")
            creature = self._create_creature(genome, "carnivore", generation=0)
            self.creatures.append(creature)
    
    def _create_creature(self, genome, species: str, generation: int = 0, 
                        parent_ids: List[str] = None, position: Optional[np.ndarray] = None) -> Creature:
        """Create a new creature with unique ID and random position."""
        if position is None:
            position = np.array([
                self.rng.rand() * self.cfg.width,
                self.rng.rand() * self.cfg.height
            ], dtype=np.float32)
        
        creature = Creature(
            id=f"c_{self.next_creature_id}_{self.step_count}",
            genome=genome,
            species=species,
            position=position,
            energy=50.0 + self.rng.rand() * 50.0,  # Random starting energy
            generation=generation,
            parent_ids=parent_ids or []
        )
        
        self.next_creature_id += 1
        return creature
    
    def step(self) -> bool:
        """Execute one simulation time step. Returns False if simulation should end."""
        self.step_count += 1
        
        # Update environment
        self.environment.step()
        
        # Rebuild spatial hash for efficient neighbor queries
        self.spatial_hash.rebuild(self.creatures)
        
        # Process each creature
        new_creatures = []
        creatures_to_remove = set()
        
        for creature in self.creatures:
            # Age the creature
            creature.age += 1
            
            # Check for natural death
            if not creature.is_alive():
                creatures_to_remove.add(creature.id)
                self.total_deaths += 1
                continue
            
            # Get nearby creatures for decision making
            nearby_creatures = self.spatial_hash.get_neighbors(
                self.creatures, creature, min(creature.genome.perception, self.cfg.perception_max)
            )
            
            # Creature brain processes sensory input and decides actions
            brain = CreatureBrain(creature.genome, self.rng)
            sensory_input = brain.get_sensory_input(creature, self.environment, nearby_creatures)
            actions = brain.forward(sensory_input)
            
            # Execute creature actions
            offspring = self._execute_actions(creature, actions, nearby_creatures, creatures_to_remove)
            if offspring:
                new_creatures.append(offspring)
                self.total_births += 1
            
            # Apply movement cost and energy consumption
            self._apply_energy_costs(creature, actions)
            
            # Check for starvation
            if creature.energy <= self.cfg.min_energy:
                creatures_to_remove.add(creature.id)
                self.total_deaths += 1
        
        # Batch remove dead creatures
        if creatures_to_remove:
            self.creatures = [c for c in self.creatures if c.id not in creatures_to_remove]
        
        # Add newborn creatures
        if new_creatures:
            self.creatures.extend(new_creatures)
        
        # Record statistics
        if self.step_count % self.cfg.snapshot_every == 0:
            self._record_statistics()
        
        # Check termination conditions
        return self._should_continue()
    
    def _execute_actions(self, creature: Creature, actions: Dict[str, float], 
                        nearby_creatures: List[Creature], dead_creatures: set) -> Optional[Creature]:
        """Execute creature's chosen actions and return potential offspring."""
        
        # Movement with toroidal wrapping
        dx = actions["move_x"] * creature.genome.speed
        dy = actions["move_y"] * creature.genome.speed
        new_x = creature.position[0] + dx
        new_y = creature.position[1] + dy
        creature.update_position(new_x, new_y, self.cfg.width, self.cfg.height)
        
        # Feeding behavior
        if actions["eat"] > 0.5:
            if creature.species == "herbivore":
                self._herbivore_feeding(creature)
            elif creature.species == "carnivore":
                self._carnivore_hunting(creature, nearby_creatures, dead_creatures)
        
        # Reproduction
        offspring = None
        if actions["reproduce"] > 0.7 and creature.can_reproduce():
            offspring = self._attempt_reproduction(creature, nearby_creatures)
        
        return offspring
    
    def _herbivore_feeding(self, creature: Creature) -> None:
        """Handle herbivore plant consumption."""
        x, y = creature.position[0], creature.position[1]
        food_consumed = self.environment.consume_food_at(x, y, self.cfg.herbivore_bite_cap)
        
        if food_consumed > 0:
            energy_gained = food_consumed * 8.0  # Energy conversion efficiency
            creature.gain_energy(energy_gained, self.cfg.max_energy)
    
    def _carnivore_hunting(self, predator: Creature, nearby_creatures: List[Creature], 
                          dead_creatures: set) -> None:
        """Handle carnivore hunting of herbivores."""
        # Find closest herbivore within attack range
        target = None
        min_distance = 2.0  # Attack range
        
        for prey in nearby_creatures:
            if (prey.species == "herbivore" and 
                prey.id not in dead_creatures and
                predator.distance_to(prey) < min_distance):
                target = prey
                min_distance = predator.distance_to(prey)
        
        if target:
            # Combat resolution based on size and aggression
            attack_power = predator.genome.aggression * predator.genome.size
            defense_power = target.genome.size * 0.5
            
            if attack_power > defense_power:
                # Successful hunt
                energy_gained = target.energy * self.cfg.carnivore_gain_eff
                predator.gain_energy(energy_gained, self.cfg.max_energy)
                dead_creatures.add(target.id)
    
    def _attempt_reproduction(self, creature: Creature, nearby_creatures: List[Creature]) -> Optional[Creature]:
        """Handle asexual reproduction (sexual reproduction can be added later)."""
        # For MVP: simple asexual reproduction with mutation
        # Future: find mate of same species for sexual reproduction
        
        # Create offspring genome through mutation
        child_genome = self.evolution_engine.mutate(creature.genome)
        
        # Spawn near parent with some randomness
        offset_x = self.rng.normal(0, 2.0)
        offset_y = self.rng.normal(0, 2.0)
        child_position = np.array([
            creature.position[0] + offset_x,
            creature.position[1] + offset_y
        ], dtype=np.float32)
        
        child = self._create_creature(
            genome=child_genome,
            species=creature.species,
            generation=creature.generation + 1,
            parent_ids=[creature.id],
            position=child_position
        )
        
        # Reproduction cost
        reproduction_cost = creature.genome.reproduction_threshold * self.cfg.reproduce_cost_frac
        creature.consume_energy(reproduction_cost, self.cfg.min_energy, self.cfg.max_energy)
        
        return child
    
    def _apply_energy_costs(self, creature: Creature, actions: Dict[str, float]) -> None:
        """Apply energy costs for movement and metabolism."""
        # Movement cost based on speed and size
        movement_intensity = abs(actions["move_x"]) + abs(actions["move_y"])
        move_cost = (self.cfg.move_cost_base * creature.genome.size * 
                    (0.5 + 0.5 * movement_intensity))
        
        # Metabolic cost based on size and age
        metabolic_cost = creature.genome.size * 0.1
        
        # Total cost adjusted by energy efficiency
        total_cost = (move_cost + metabolic_cost) / creature.genome.energy_efficiency
        
        creature.consume_energy(total_cost, self.cfg.min_energy, self.cfg.max_energy)
    
    def _record_statistics(self) -> None:
        """Record current simulation state for analysis."""
        herbivore_count = sum(1 for c in self.creatures if c.species == "herbivore")
        carnivore_count = sum(1 for c in self.creatures if c.species == "carnivore")
        
        # Population statistics
        pop_stats = {
            "step": self.step_count,
            "total_population": len(self.creatures),
            "herbivores": herbivore_count,
            "carnivores": carnivore_count,
            "births": self.total_births,
            "deaths": self.total_deaths
        }
        
        # Trait statistics for each species
        trait_stats = {}
        for species in ["herbivore", "carnivore"]:
            species_creatures = [c for c in self.creatures if c.species == species]
            if species_creatures:
                trait_stats[species] = self._calculate_trait_statistics(species_creatures)
            else:
                trait_stats[species] = None
        
        # Environment statistics
        env_stats = self.environment.get_statistics()
        
        # Spatial hash performance
        spatial_stats = self.spatial_hash.get_stats()
        
        self.statistics.record_step(pop_stats, trait_stats, env_stats, spatial_stats)
    
    def _calculate_trait_statistics(self, creatures: List[Creature]) -> Dict[str, float]:
        """Calculate average trait values for a group of creatures."""
        if not creatures:
            return {}
        
        traits = ["speed", "size", "aggression", "perception", "energy_efficiency", 
                 "reproduction_threshold", "lifespan"]
        
        stats = {}
        for trait in traits:
            values = [getattr(c.genome, trait) for c in creatures]
            stats[f"{trait}_mean"] = float(np.mean(values))
            stats[f"{trait}_std"] = float(np.std(values))
            stats[f"{trait}_min"] = float(np.min(values))
            stats[f"{trait}_max"] = float(np.max(values))
        
        # Additional statistics
        stats["avg_energy"] = float(np.mean([c.energy for c in creatures]))
        stats["avg_age"] = float(np.mean([c.age for c in creatures]))
        stats["avg_generation"] = float(np.mean([c.generation for c in creatures]))
        
        return stats
    
    def _should_continue(self) -> bool:
        """Check if simulation should continue running."""
        # Stop if no creatures left
        if len(self.creatures) == 0:
            return False
        
        # Stop if max steps reached
        if self.step_count >= self.cfg.max_steps:
            return False
        
        # Stop if only one species remains for too long
        herbivores = sum(1 for c in self.creatures if c.species == "herbivore")
        carnivores = sum(1 for c in self.creatures if c.species == "carnivore")
        
        if (herbivores == 0 or carnivores == 0) and self.step_count > 1000:
            return False
        
        return True
    
    def get_creature_data(self) -> List[Dict[str, Any]]:
        """Get current creature data for visualization."""
        return [creature.as_dict() for creature in self.creatures]
    
    def get_environment_data(self) -> Dict[str, Any]:
        """Get environment data for visualization."""
        return {
            "food_grid": self.environment.food.copy(),
            "temperature": self.environment.temperature,
            "time_step": self.environment.time_step,
            "width": self.environment.width,
            "height": self.environment.height
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the simulation."""
        return {
            "step_count": self.step_count,
            "total_creatures": len(self.creatures),
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "herbivore_count": sum(1 for c in self.creatures if c.species == "herbivore"),
            "carnivore_count": sum(1 for c in self.creatures if c.species == "carnivore"),
            "avg_generation": np.mean([c.generation for c in self.creatures]) if self.creatures else 0,
            "environment_stats": self.environment.get_statistics()
        }
    
    def reset(self, new_seed: Optional[int] = None) -> None:
        """Reset simulation to initial state with optional new seed."""
        if new_seed is not None:
            self.cfg.seed = new_seed
        
        self.rng = RNG(self.cfg.seed)
        self.environment = Environment(self.cfg, self.rng)
        self.evolution_engine = EvolutionEngine(self.cfg, self.rng)
        self.spatial_hash = SpatialHash(self.cfg.grid_cell, self.cfg.width, self.cfg.height)
        self.statistics = Statistics()
        
        self.creatures = []
        self.next_creature_id = 0
        self.step_count = 0
        self.total_births = 0
        self.total_deaths = 0
        
        self._create_initial_population()
        self._record_statistics()
