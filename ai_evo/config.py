"""Configuration management for AI Evolution Environment."""

import argparse
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class Config:
    """Configuration for the evolution simulation."""
    
    # Random seed for deterministic runs
    seed: int = 42
    
    # World dimensions
    width: int = 100
    height: int = 100
    
    # Initial population
    init_herbivores: int = 50
    init_carnivores: int = 20
    
    # Simulation control
    max_steps: int = 1000
    snapshot_every: int = 50
    
    # Spatial hashing
    grid_cell: int = 10
    
    # Energy system
    min_energy: float = 0.0
    max_energy: float = 200.0
    move_cost_base: float = 0.1
    reproduce_cost_frac: float = 0.3
    
    # Environment
    plant_growth_rate: float = 0.1
    plant_max_density: float = 5.0
    temperature: float = 20.0
    
    # Creature limits
    perception_max: float = 10.0
    max_age: int = 1000
    
    # Feeding parameters
    herbivore_bite_size: float = 2.0
    carnivore_attack_damage: float = 20.0
    carnivore_energy_gain: float = 0.7
    
    # Evolution parameters
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2
    
    # Performance settings
    enable_profiling: bool = False
    max_population: int = 1000
    
    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser for configuration."""
        parser = argparse.ArgumentParser(
            description="AI Animal Evolution Environment",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Add arguments for each configuration field
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--width', type=int, default=100, help='World width')
        parser.add_argument('--height', type=int, default=100, help='World height')
        parser.add_argument('--init-herbivores', type=int, default=50, help='Initial herbivore count')
        parser.add_argument('--init-carnivores', type=int, default=20, help='Initial carnivore count')
        parser.add_argument('--steps', '--max-steps', type=int, default=1000, help='Maximum simulation steps')
        parser.add_argument('--plant-growth-rate', type=float, default=0.1, help='Plant growth rate')
        parser.add_argument('--enable-profiling', action='store_true', help='Enable performance profiling')
        
        return parser
    
    @classmethod
    def from_args(cls, args):
        """Create configuration from parsed arguments."""
        return cls(
            seed=args.seed,
            width=args.width,
            height=args.height,
            init_herbivores=args.init_herbivores,
            init_carnivores=args.init_carnivores,
            max_steps=args.steps,
            plant_growth_rate=args.plant_growth_rate,
            enable_profiling=args.enable_profiling
        )
