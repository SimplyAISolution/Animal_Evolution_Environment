"""Statistics tracking and analysis for evolutionary metrics."""

import numpy as np
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass

@dataclass
class StepData:
    """Data recorded for a single simulation step."""
    step: int
    population_stats: Dict[str, Any]
    trait_stats: Dict[str, Any]
    environment_stats: Dict[str, Any]
    spatial_stats: Dict[str, Any]

class Statistics:
    """Manages collection and analysis of simulation statistics."""
    
    def __init__(self):
        """Initialize statistics collector."""
        self.step_data: List[StepData] = []
        self.summary_stats: Dict[str, List[float]] = {
            "total_population": [],
            "herbivore_population": [],
            "carnivore_population": [],
            "total_food": [],
            "average_generation": [],
            "births_per_step": [],
            "deaths_per_step": []
        }
    
    def record_step(self, population_stats: Dict[str, Any], trait_stats: Dict[str, Any], 
                   environment_stats: Dict[str, Any], spatial_stats: Dict[str, Any]) -> None:
        """Record statistics for current simulation step."""
        step_data = StepData(
            step=population_stats["step"],
            population_stats=population_stats,
            trait_stats=trait_stats,
            environment_stats=environment_stats,
            spatial_stats=spatial_stats
        )
        
        self.step_data.append(step_data)
        
        # Update summary statistics
        self.summary_stats["total_population"].append(population_stats["total_population"])
        self.summary_stats["herbivore_population"].append(population_stats["herbivores"])
        self.summary_stats["carnivore_population"].append(population_stats["carnivores"])
        self.summary_stats["total_food"].append(environment_stats["total_food"])
        
        # Calculate births/deaths per step
        prev_births = self.summary_stats["births_per_step"][-1] if self.summary_stats["births_per_step"] else 0
        prev_deaths = self.summary_stats["deaths_per_step"][-1] if self.summary_stats["deaths_per_step"] else 0
        
        self.summary_stats["births_per_step"].append(population_stats["births"] - prev_births)
        self.summary_stats["deaths_per_step"].append(population_stats["deaths"] - prev_deaths)
    
    def get_population_history(self) -> Dict[str, List[float]]:
        """Get population history over time."""
        return {
            "steps": [data.step for data in self.step_data],
            "total": [data.population_stats["total_population"] for data in self.step_data],
            "herbivores": [data.population_stats["herbivores"] for data in self.step_data],
            "carnivores": [data.population_stats["carnivores"] for data in self.step_data]
        }
    
    def get_trait_evolution(self, species: str, trait: str) -> Dict[str, List[float]]:
        """Get trait evolution over time for a species."""
        steps = []
        means = []
        stds = []
        
        for data in self.step_data:
            if (species in data.trait_stats and 
                data.trait_stats[species] is not None and
                f"{trait}_mean" in data.trait_stats[species]):
                
                steps.append(data.step)
                means.append(data.trait_stats[species][f"{trait}_mean"])
                stds.append(data.trait_stats[species][f"{trait}_std"])
        
        return {
            "steps": steps,
            "mean": means,
            "std": stds
        }
    
    def get_environment_history(self) -> Dict[str, List[float]]:
        """Get environment statistics over time."""
        return {
            "steps": [data.step for data in self.step_data],
            "total_food": [data.environment_stats["total_food"] for data in self.step_data],
            "avg_food": [data.environment_stats["avg_food"] for data in self.step_data],
            "temperature": [data.environment_stats["temperature"] for data in self.step_data]
        }
    
    def calculate_selection_pressure(self, species: str, trait: str, window: int = 100) -> float:
        """Calculate selection pressure for a trait (rate of change)."""
        trait_data = self.get_trait_evolution(species, trait)
        
        if len(trait_data["mean"]) < window * 2:
            return 0.0
        
        # Compare first window to last window
        early_mean = np.mean(trait_data["mean"][:window])
        late_mean = np.mean(trait_data["mean"][-window:])
        
        return (late_mean - early_mean) / early_mean if early_mean != 0 else 0.0
    
    def get_diversity_metrics(self, species: str) -> Dict[str, float]:
        """Calculate genetic diversity metrics for a species."""
        if not self.step_data:
            return {}
        
        latest_data = self.step_data[-1]
        if (species not in latest_data.trait_stats or 
            latest_data.trait_stats[species] is None):
            return {}
        
        traits = ["speed", "size", "aggression", "perception", "energy_efficiency"]
        diversity_metrics = {}
        
        for trait in traits:
            std_key = f"{trait}_std"
            if std_key in latest_data.trait_stats[species]:
                diversity_metrics[f"{trait}_diversity"] = latest_data.trait_stats[species][std_key]
        
        return diversity_metrics
    
    def get_extinction_risk(self, species: str, threshold: int = 10) -> float:
        """Calculate extinction risk based on recent population trends."""
        if len(self.step_data) < 50:
            return 0.0
        
        recent_populations = []
        for data in self.step_data[-50:]:  # Last 50 steps
            if species == "herbivore":
                recent_populations.append(data.population_stats["herbivores"])
            elif species == "carnivore":
                recent_populations.append(data.population_stats["carnivores"])
        
        if not recent_populations:
            return 1.0
        
        avg_population = np.mean(recent_populations)
        trend = np.polyfit(range(len(recent_populations)), recent_populations, 1)[0]
        
        # Risk increases if population is low or declining
        risk = 0.0
        if avg_population < threshold:
            risk += (threshold - avg_population) / threshold
        
        if trend < 0:
            risk += abs(trend) / max(1, avg_population)
        
        return min(1.0, risk)
    
    def export_to_json(self, filename: str) -> None:
        """Export statistics to JSON file."""
        export_data = {
            "summary_stats": self.summary_stats,
            "population_history": self.get_population_history(),
            "environment_history": self.get_environment_history(),
            "step_count": len(self.step_data)
        }
        
        # Add trait evolution for both species
        for species in ["herbivore", "carnivore"]:
            export_data[f"{species}_traits"] = {}
            for trait in ["speed", "size", "aggression", "perception", "energy_efficiency"]:
                export_data[f"{species}_traits"][trait] = self.get_trait_evolution(species, trait)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate a text summary report of the simulation."""
        if not self.step_data:
            return "No data available"
        
        latest = self.step_data[-1]
        report_lines = [
            "=== AI Evolution Simulation Report ===",
            f"Total Steps: {latest.step}",
            f"Final Population: {latest.population_stats['total_population']}",
            f"  - Herbivores: {latest.population_stats['herbivores']}",
            f"  - Carnivores: {latest.population_stats['carnivores']}",
            f"Total Births: {latest.population_stats['births']}",
            f"Total Deaths: {latest.population_stats['deaths']}",
            "",
            "=== Environment ===",
            f"Total Food: {latest.environment_stats['total_food']:.1f}",
            f"Average Food: {latest.environment_stats['avg_food']:.3f}",
            f"Temperature: {latest.environment_stats['temperature']:.1f}°C",
            "",
            "=== Evolution Metrics ==="
        ]
        
        # Add trait evolution summary
        for species in ["herbivore", "carnivore"]:
            if (species in latest.trait_stats and 
                latest.trait_stats[species] is not None):
                
                report_lines.append(f"\n{species.title()} Traits:")
                traits = latest.trait_stats[species]
                
                for trait in ["speed", "size", "aggression", "perception"]:
                    mean_key = f"{trait}_mean"
                    std_key = f"{trait}_std"
                    if mean_key in traits:
                        selection_pressure = self.calculate_selection_pressure(species, trait)
                        report_lines.append(
                            f"  {trait.title()}: {traits[mean_key]:.3f} ± {traits[std_key]:.3f} "
                            f"(selection: {selection_pressure:+.3f})"
                        )
                
                # Extinction risk
                risk = self.get_extinction_risk(species)
                report_lines.append(f"  Extinction Risk: {risk:.1%}")
        
        return "\n".join(report_lines)
