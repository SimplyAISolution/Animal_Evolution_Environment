"""Visualization utilities for the AI Evolution Environment."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any

def create_world_plot(creatures: List[Dict[str, Any]], env_data: Dict[str, Any]) -> go.Figure:
    """Create a 2D world visualization with creatures and food."""
    
    # Create subplot with secondary y-axis for overlaying
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=["🌍 AI Evolution World"],
        specs=[[{"secondary_y": False}]]
    )
    
    # Food heatmap background
    food_grid = env_data["food_grid"]
    
    fig.add_trace(
        go.Heatmap(
            z=food_grid,
            colorscale="Greens",
            opacity=0.6,
            showscale=True,
            colorbar=dict(title="Food Density", x=1.02),
            hovertemplate="Food: %{z:.2f}<extra></extra>"
        )
    )
    
    if creatures:
        # Separate creatures by species
        herbivores = [c for c in creatures if c["species"] == "herbivore"]
        carnivores = [c for c in creatures if c["species"] == "carnivore"]
        
        # Plot herbivores
        if herbivores:
            fig.add_trace(
                go.Scatter(
                    x=[c["x"] for c in herbivores],
                    y=[c["y"] for c in herbivores],
                    mode="markers",
                    marker=dict(
                        size=[8 + c["size"] * 4 for c in herbivores],
                        color=[c["energy"] for c in herbivores],
                        colorscale="Viridis",
                        opacity=0.8,
                        symbol="circle",
                        line=dict(width=1, color="darkgreen")
                    ),
                    name="🌿 Herbivores",
                    hovertemplate=(
                        "Herbivore<br>"
                        "Position: (%{x:.1f}, %{y:.1f})<br>"
                        "Energy: %{marker.color:.1f}<br>"
                        "<extra></extra>"
                    ),
                    customdata=[[c["age"], c["generation"], c["speed"]] for c in herbivores],
                )
            )
        
        # Plot carnivores
        if carnivores:
            fig.add_trace(
                go.Scatter(
                    x=[c["x"] for c in carnivores],
                    y=[c["y"] for c in carnivores],
                    mode="markers",
                    marker=dict(
                        size=[10 + c["size"] * 5 for c in carnivores],
                        color=[c["energy"] for c in carnivores],
                        colorscale="Reds",
                        opacity=0.9,
                        symbol="diamond",
                        line=dict(width=1, color="darkred")
                    ),
                    name="🥩 Carnivores",
                    hovertemplate=(
                        "Carnivore<br>"
                        "Position: (%{x:.1f}, %{y:.1f})<br>"
                        "Energy: %{marker.color:.1f}<br>"
                        "<extra></extra>"
                    ),
                    customdata=[[c["age"], c["generation"], c["speed"]] for c in carnivores],
                )
            )
    
    # Update layout
    fig.update_layout(
        width=800,
        height=600,
        title=f"Step {env_data['time_step']} | Temperature: {env_data['temperature']:.1f}°C",
        xaxis=dict(
            title="X Position",
            range=[0, env_data["width"]],
            constrain="domain"
        ),
        yaxis=dict(
            title="Y Position", 
            range=[0, env_data["height"]],
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=True,
        legend=dict(x=1.05, y=1)
    )
    
    return fig

def create_population_chart(pop_history: Dict[str, List[float]]) -> go.Figure:
    """Create population history chart."""
    
    fig = go.Figure()
    
    # Total population
    fig.add_trace(
        go.Scatter(
            x=pop_history["steps"],
            y=pop_history["total"],
            mode="lines",
            name="Total Population",
            line=dict(color="blue", width=3)
        )
    )
    
    # Herbivores
    fig.add_trace(
        go.Scatter(
            x=pop_history["steps"],
            y=pop_history["herbivores"],
            mode="lines",
            name="🌿 Herbivores",
            line=dict(color="green", width=2),
            fill="tonexty"
        )
    )
    
    # Carnivores
    fig.add_trace(
        go.Scatter(
            x=pop_history["steps"],
            y=pop_history["carnivores"],
            mode="lines",
            name="🥩 Carnivores",
            line=dict(color="red", width=2),
            fill="tonexty"
        )
    )
    
    fig.update_layout(
        title="📈 Population History",
        xaxis_title="Simulation Step",
        yaxis_title="Population Count",
        height=400,
        hovermode="x unified"
    )
    
    return fig

def create_trait_evolution_chart(trait_data: Dict[str, List[float]], 
                                species: str, trait: str) -> go.Figure:
    """Create trait evolution chart with confidence bands."""
    
    fig = go.Figure()
    
    steps = trait_data["steps"]
    means = trait_data["mean"]
    stds = trait_data["std"]
    
    # Calculate confidence bands
    upper_band = [m + s for m, s in zip(means, stds)]
    lower_band = [m - s for m, s in zip(means, stds)]
    
    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=steps + steps[::-1],
            y=upper_band + lower_band[::-1],
            fill="toself",
            fillcolor="rgba(128,128,128,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±1 Std Dev",
            showlegend=True
        )
    )
    
    # Mean trait value
    color = "green" if species == "herbivore" else "red"
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=means,
            mode="lines+markers",
            name=f"Mean {trait.title()}",
            line=dict(color=color, width=3),
            marker=dict(size=4)
        )
    )
    
    fig.update_layout(
        title=f"🧬 {species.title()} {trait.title()} Evolution",
        xaxis_title="Simulation Step",
        yaxis_title=f"{trait.title()} Value",
        height=400,
        hovermode="x unified"
    )
    
    return fig

def create_fitness_landscape(creatures: List[Dict[str, Any]], 
                           trait1: str, trait2: str) -> go.Figure:
    """Create a 2D fitness landscape plot."""
    
    if not creatures:
        return go.Figure()
    
    # Separate by species
    herbivores = [c for c in creatures if c["species"] == "herbivore"]
    carnivores = [c for c in creatures if c["species"] == "carnivore"]
    
    fig = go.Figure()
    
    # Plot herbivores
    if herbivores:
        fig.add_trace(
            go.Scatter(
                x=[c[trait1] for c in herbivores],
                y=[c[trait2] for c in herbivores],
                mode="markers",
                marker=dict(
                    size=[c["energy"] / 5 for c in herbivores],
                    color="green",
                    opacity=0.6
                ),
                name="🌿 Herbivores",
                hovertemplate=f"{trait1}: %{{x:.2f}}<br>{trait2}: %{{y:.2f}}<br>Energy: %{{marker.size*5:.1f}}<extra></extra>"
            )
        )
    
    # Plot carnivores
    if carnivores:
        fig.add_trace(
            go.Scatter(
                x=[c[trait1] for c in carnivores],
                y=[c[trait2] for c in carnivores],
                mode="markers",
                marker=dict(
                    size=[c["energy"] / 5 for c in carnivores],
                    color="red",
                    opacity=0.6
                ),
                name="🥩 Carnivores",
                hovertemplate=f"{trait1}: %{{x:.2f}}<br>{trait2}: %{{y:.2f}}<br>Energy: %{{marker.size*5:.1f}}<extra></extra>"
            )
        )
    
    fig.update_layout(
        title=f"🎯 Fitness Landscape: {trait1.title()} vs {trait2.title()}",
        xaxis_title=trait1.title(),
        yaxis_title=trait2.title(),
        height=500
    )
    
    return fig

def create_phylogenetic_tree(creatures: List[Dict[str, Any]]) -> go.Figure:
    """Create a simple phylogenetic tree visualization."""
    # This is a simplified version - a full phylogenetic tree would require
    # tracking actual lineage relationships
    
    fig = go.Figure()
    
    if not creatures:
        return fig
    
    # Group by generation and species
    generations = {}
    for creature in creatures:
        gen = creature["generation"]
        species = creature["species"]
        key = (gen, species)
        
        if key not in generations:
            generations[key] = []
        generations[key].append(creature)
    
    # Plot generation progression
    for (gen, species), gen_creatures in generations.items():
        color = "green" if species == "herbivore" else "red"
        symbol = "circle" if species == "herbivore" else "diamond"
        
        avg_traits = {
            "speed": np.mean([c["speed"] for c in gen_creatures]),
            "size": np.mean([c["size"] for c in gen_creatures]),
            "aggression": np.mean([c["aggression"] for c in gen_creatures])
        }
        
        fig.add_trace(
            go.Scatter(
                x=[gen],
                y=[len(gen_creatures)],
                mode="markers",
                marker=dict(
                    size=avg_traits["size"] * 10,
                    color=color,
                    symbol=symbol,
                    opacity=0.7
                ),
                name=f"{species.title()} Gen {gen}",
                showlegend=False,
                hovertemplate=(
                    f"Generation: {gen}<br>"
                    f"Count: {len(gen_creatures)}<br>"
                    f"Avg Speed: {avg_traits['speed']:.2f}<br>"
                    f"Avg Size: {avg_traits['size']:.2f}<br>"
                    f"Avg Aggression: {avg_traits['aggression']:.2f}<br>"
                    "<extra></extra>"
                )
            )
        )
    
    fig.update_layout(
        title="🌳 Evolutionary Tree (Generation vs Population)",
        xaxis_title="Generation",
        yaxis_title="Population Count",
        height=400
    )
    
    return fig

def export_world_image(creatures: List[Dict[str, Any]], 
                      env_data: Dict[str, Any], filename: str) -> None:
    """Export world visualization as image."""
    fig = create_world_plot(creatures, env_data)
    fig.write_image(filename, width=1200, height=900, scale=2)

def create_summary_dashboard(simulation_stats: Dict[str, Any]) -> go.Figure:
    """Create a summary dashboard with key metrics."""
    
    # Create subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Population Over Time",
            "Energy Distribution", 
            "Trait Diversity",
            "Environmental Conditions"
        ],
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # This would be populated with actual data from simulation_stats
    # Implementation depends on the specific structure of simulation_stats
    
    fig.update_layout(
        height=800,
        title_text="📊 AI Evolution Dashboard",
        showlegend=True
    )
    
    return fig
