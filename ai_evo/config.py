"""Configuration management for AI Evolution Environment."""

from dataclasses import dataclass
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
    
    # Evolution parameters
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2
    
    # Performance settings
    enable_profiling: bool = False
    max_population: int = 1000
