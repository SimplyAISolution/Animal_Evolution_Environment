"""Real-time Streamlit observer interface for the AI Evolution simulation."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import time
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_evo.simulation import Simulation
from ai_evo.config import Config
from ui.visualization import create_world_plot, create_population_chart, create_trait_evolution_chart

# Configure Streamlit page
st.set_page_config(
    page_title="AI Animal Evolution Environment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitApp:
    """Main Streamlit application for observing the AI evolution simulation."""
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.simulation = None
        self.is_running = False
        self.auto_run = False
        
        # Initialize session state
        if 'step_count' not in st.session_state:
            st.session_state.step_count = 0
        if 'simulation_data' not in st.session_state:
            st.session_state.simulation_data = None
        if 'config' not in st.session_state:
            st.session_state.config = Config()
    
    def run(self):
        """Main application loop."""
        st.title("🧬 AI Animal Evolution Environment")
        st.markdown("**Observe emergent evolution in a fully AI-driven ecosystem**")
        
        # Sidebar controls
        self.render_sidebar()
        
        # Main content area
        if self.simulation is None:
            self.render_welcome_screen()
        else:
            self.render_simulation_view()
    
    def render_sidebar(self):
        """Render the control sidebar."""
        st.sidebar.header("🎮 Simulation Controls")
        
        # Configuration section
        with st.sidebar.expander("⚙️ Configuration", expanded=False):
            self.render_config_controls()
        
        # Simulation control buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🚀 Start", disabled=self.simulation is not None):
                self.start_simulation()
        
        with col2:
            if st.button("🔄 Reset", disabled=self.simulation is None):
                self.reset_simulation()
        
        # Runtime controls
        if self.simulation is not None:
            st.sidebar.divider()
            
            # Single step and auto-run controls
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if st.button("⏭️ Step", disabled=self.auto_run):
                    self.step_simulation()
            
            with col2:
                if st.button("⏸️ Pause" if self.auto_run else "▶️ Play"):
                    self.auto_run = not self.auto_run
            
            # Auto-run speed control
            if self.auto_run:
                speed = st.sidebar.slider("🏃 Speed (steps/sec)", 0.1, 10.0, 2.0, 0.1)
                st.session_state.auto_speed = speed
            
            # Environmental controls
            with st.sidebar.expander("🌍 Environment Controls", expanded=True):
                self.render_environment_controls()
        
        # Statistics and export
        if self.simulation is not None:
            st.sidebar.divider()
            self.render_statistics_panel()
    
    def render_config_controls(self):
        """Render configuration controls."""
        cfg = st.session_state.config
        
        # World parameters
        st.subheader("🗺️ World")
        cfg.width = st.slider("Width", 50, 300, cfg.width, 10)
        cfg.height = st.slider("Height", 50, 300, cfg.height, 10)
        cfg.seed = st.number_input("Random Seed", 1, 999999, cfg.seed)
        
        # Population parameters
        st.subheader("🐾 Population")
        cfg.init_herbivores = st.slider("Initial Herbivores", 10, 200, cfg.init_herbivores, 5)
        cfg.init_carnivores = st.slider("Initial Carnivores", 5, 100, cfg.init_carnivores, 5)
        
        # Evolution parameters
        st.subheader("🧬 Evolution")
        cfg.mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, cfg.mutation_rate, 0.01)
        cfg.mutation_strength = st.slider("Mutation Strength", 0.01, 0.3, cfg.mutation_strength, 0.01)
        
        # Environment parameters
        st.subheader("🌱 Environment")
        cfg.plant_growth_rate = st.slider("Plant Growth Rate", 0.01, 0.5, cfg.plant_growth_rate, 0.01)
        cfg.plant_cap = st.slider("Plant Capacity", 1.0, 50.0, cfg.plant_cap, 1.0)
    
    def render_environment_controls(self):
        """Render real-time environment controls."""
        if self.simulation is None:
            return
        
        # Temperature control
        new_temp = st.slider(
            "🌡️ Temperature", 
            -10.0, 50.0, 
            self.simulation.environment.temperature, 
            0.5
        )
        if abs(new_temp - self.simulation.environment.temperature) > 0.1:
            self.simulation.environment.temperature = new_temp
        
        # Food growth rate adjustment
        new_growth = st.slider(
            "🌱 Food Growth Rate",
            0.0, 0.5,
            self.simulation.cfg.plant_growth_rate,
            0.01
        )
        if abs(new_growth - self.simulation.cfg.plant_growth_rate) > 0.001:
            self.simulation.cfg.plant_growth_rate = new_growth
        
        # Disaster controls
        st.subheader("💥 Disasters")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔥 Fire"):
                self.trigger_fire_disaster()
        
        with col2:
            if st.button("❄️ Frost"):
                self.trigger_frost_disaster()
    
    def render_statistics_panel(self):
        """Render statistics and export options."""
        if self.simulation is None:
            return
        
        st.subheader("📊 Statistics")
        
        stats = self.simulation.get_summary_stats()
        
        # Key metrics
        st.metric("Step", stats["step_count"])
        st.metric("Total Population", stats["total_creatures"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌿 Herbivores", stats["herbivore_count"])
        with col2:
            st.metric("🥩 Carnivores", stats["carnivore_count"])
        
        st.metric("Avg Generation", f"{stats['avg_generation']:.1f}")
        
        # Export data
        if st.button("💾 Export Data"):
            self.export_simulation_data()
    
    def render_welcome_screen(self):
        """Render the welcome screen with instructions."""
        st.markdown("""
        ## Welcome to the AI Animal Evolution Environment! 🦎🧬
        
        This simulation features **fully autonomous AI creatures** that evolve and adapt through:
        
        - **🧠 Neural Network Brains**: Each creature makes decisions using a neural network
        - **🧬 Genetic Evolution**: Traits mutate and evolve through natural selection
        - **🌍 Emergent Ecosystems**: Food webs and population dynamics emerge naturally
        - **🔬 Scientific Accuracy**: Based on real evolutionary principles
        
        ### Getting Started:
        1. **Configure** your world in the sidebar (or use defaults)
        2. **Click "Start"** to begin the simulation
        3. **Observe** as creatures adapt and evolve over generations
        4. **Experiment** with environmental controls to see how evolution responds
        
        ### What You'll See:
        - **World View**: Real-time visualization of creatures and food
        - **Population Charts**: Species counts over time
        - **Trait Evolution**: How genetic traits change across generations
        - **Environmental Data**: Food distribution and climate conditions
        """)
        
        # Quick start with example configurations
        st.subheader("🚀 Quick Start Scenarios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🌱 Garden Paradise"):
                self.load_preset_config("garden")
                self.start_simulation()
        
        with col2:
            if st.button("🏜️ Harsh Desert"):
                self.load_preset_config("desert")
                self.start_simulation()
        
        with col3:
            if st.button("⚡ Rapid Evolution"):
                self.load_preset_config("rapid")
                self.start_simulation()
    
    def render_simulation_view(self):
        """Render the main simulation visualization."""
        # Auto-run logic
        if self.auto_run:
            speed = getattr(st.session_state, 'auto_speed', 2.0)
            time.sleep(1.0 / speed)
            self.step_simulation()
            st.rerun()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌍 World View", 
            "📈 Population", 
            "🧬 Evolution", 
            "🌱 Environment"
        ])
        
        with tab1:
            self.render_world_view()
        
        with tab2:
            self.render_population_charts()
        
        with tab3:
            self.render_evolution_charts()
        
        with tab4:
            self.render_environment_view()
    
    def render_world_view(self):
        """Render the main world visualization."""
        if self.simulation is None:
            st.warning("No simulation running")
            return
        
        # Get current simulation data
        creatures = self.simulation.get_creature_data()
        env_data = self.simulation.get_environment_data()
        
        if not creatures:
            st.warning("No creatures remaining in simulation")
            return
        
        # Create world plot
        fig = create_world_plot(creatures, env_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Current step info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Step", self.simulation.step_count)
        with col2:
            st.metric("Population", len(creatures))
        with col3:
            herbivores = sum(1 for c in creatures if c["species"] == "herbivore")
            st.metric("🌿 Herbivores", herbivores)
        with col4:
            carnivores = sum(1 for c in creatures if c["species"] == "carnivore")
            st.metric("🥩 Carnivores", carnivores)
    
    def render_population_charts(self):
        """Render population history charts."""
        if self.simulation is None or not self.simulation.statistics.step_data:
            st.info("Run simulation to see population data")
            return
        
        # Population history
        pop_history = self.simulation.statistics.get_population_history()
        fig = create_population_chart(pop_history)
        st.plotly_chart(fig, use_container_width=True)
        
        # Population statistics table
        if pop_history["steps"]:
            st.subheader("📊 Population Statistics")
            
            recent_data = {
                "Metric": ["Total Population", "Herbivores", "Carnivores"],
                "Current": [
                    pop_history["total"][-1],
                    pop_history["herbivores"][-1], 
                    pop_history["carnivores"][-1]
                ],
                "Peak": [
                    max(pop_history["total"]),
                    max(pop_history["herbivores"]),
                    max(pop_history["carnivores"])
                ]
            }
            
            df = pd.DataFrame(recent_data)
            st.dataframe(df, use_container_width=True)
    
    def render_evolution_charts(self):
        """Render trait evolution charts."""
        if self.simulation is None or not self.simulation.statistics.step_data:
            st.info("Run simulation to see evolution data")
            return
        
        # Trait selection
        col1, col2 = st.columns(2)
        
        with col1:
            species = st.selectbox("Species", ["herbivore", "carnivore"])
        
        with col2:
            trait = st.selectbox("Trait", [
                "speed", "size", "aggression", "perception", "energy_efficiency"
            ])
        
        # Trait evolution chart
        trait_data = self.simulation.statistics.get_trait_evolution(species, trait)
        
        if trait_data["steps"]:
            fig = create_trait_evolution_chart(trait_data, species, trait)
            st.plotly_chart(fig, use_container_width=True)
            
            # Selection pressure analysis
            pressure = self.simulation.statistics.calculate_selection_pressure(species, trait)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selection Pressure", f"{pressure:+.3f}")
            with col2:
                current_mean = trait_data["mean"][-1] if trait_data["mean"] else 0
                st.metric("Current Mean", f"{current_mean:.3f}")
            with col3:
                current_std = trait_data["std"][-1] if trait_data["std"] else 0
                st.metric("Current Diversity", f"{current_std:.3f}")
        else:
            st.warning(f"No {species} data available yet")
    
    def render_environment_view(self):
        """Render environment statistics and controls."""
        if self.simulation is None:
            st.info("Start simulation to see environment data")
            return
        
        env_stats = self.simulation.environment.get_statistics()
        
        # Environment metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌡️ Temperature", f"{env_stats['temperature']:.1f}°C")
        with col2:
            st.metric("🌱 Total Food", f"{env_stats['total_food']:.0f}")
        with col3:
            st.metric("📊 Avg Food Density", f"{env_stats['avg_food']:.3f}")
        with col4:
            st.metric("🌿 Food Patches", env_stats['food_patches'])
        
        # Environment history
        env_history = self.simulation.statistics.get_environment_history()
        
        if env_history["steps"]:
            # Food over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=env_history["steps"],
                y=env_history["total_food"],
                mode='lines',
                name='Total Food',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title="🌱 Food Supply Over Time",
                xaxis_title="Simulation Step",
                yaxis_title="Total Food",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def start_simulation(self):
        """Start a new simulation."""
        try:
            self.simulation = Simulation(st.session_state.config)
            st.session_state.simulation_data = self.simulation
            st.success("✅ Simulation started successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to start simulation: {str(e)}")
    
    def reset_simulation(self):
        """Reset the current simulation."""
        self.simulation = None
        self.auto_run = False
        st.session_state.simulation_data = None
        st.session_state.step_count = 0
        st.success("🔄 Simulation reset")
        st.rerun()
    
    def step_simulation(self):
        """Execute one simulation step."""
        if self.simulation is None:
            return
        
        try:
            should_continue = self.simulation.step()
            st.session_state.step_count = self.simulation.step_count
            
            if not should_continue:
                st.warning("⚠️ Simulation ended (extinction or max steps reached)")
                self.auto_run = False
                
        except Exception as e:
            st.error(f"❌ Simulation step failed: {str(e)}")
            self.auto_run = False
    
    def load_preset_config(self, preset: str):
        """Load a preset configuration."""
        cfg = Config()
        
        if preset == "garden":
            cfg.plant_growth_rate = 0.2
            cfg.init_herbivores = 80
            cfg.init_carnivores = 20
            cfg.mutation_rate = 0.08
            
        elif preset == "desert":
            cfg.plant_growth_rate = 0.05
            cfg.init_herbivores = 40
            cfg.init_carnivores = 15
            cfg.plant_cap = 5.0
            
        elif preset == "rapid":
            cfg.mutation_rate = 0.25
            cfg.mutation_strength = 0.15
            cfg.reproduce_threshold = 60.0
        
        st.session_state.config = cfg
    
    def trigger_fire_disaster(self):
        """Trigger a fire disaster that reduces food."""
        if self.simulation is None:
            return
        
        # Reduce food by 70%
        self.simulation.environment.food *= 0.3
        st.warning("🔥 Fire disaster triggered! Food reduced by 70%")
    
    def trigger_frost_disaster(self):
        """Trigger a frost disaster that affects temperature and food growth."""
        if self.simulation is None:
            return
        
        # Lower temperature and reduce food growth
        self.simulation.environment.temperature -= 15
        self.simulation.cfg.plant_growth_rate *= 0.1
        st.warning("❄️ Frost disaster triggered! Temperature dropped and growth slowed")
    
    def export_simulation_data(self):
        """Export simulation data to downloadable file."""
        if self.simulation is None:
            return
        
        try:
            # Export statistics to JSON
            filename = f"ai_evolution_data_step_{self.simulation.step_count}.json"
            self.simulation.statistics.export_to_json(filename)
            
            # Generate report
            report = self.simulation.statistics.generate_report()
            
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"evolution_report_step_{self.simulation.step_count}.txt",
                mime="text/plain"
            )
            
            st.success(f"✅ Data exported successfully!")
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

# Main app entry point
def main():
    """Main entry point for Streamlit app."""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
