"""Main entry point for AI Animal Evolution Environment CLI."""

import argparse
import sys
import time
from typing import Optional
import signal

from ai_evo.config import Config
from ai_evo.simulation import Simulation

class SimulationRunner:
    """CLI runner for the AI evolution simulation."""
    
    def __init__(self):
        """Initialize the simulation runner."""
        self.simulation: Optional[Simulation] = None
        self.running = False
        self.paused = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
        
        if self.simulation:
            print(f"📊 Final simulation state:")
            self._print_summary_stats()
            
        sys.exit(0)
    
    def run(self, config: Config, verbose: bool = True, 
            interactive: bool = False, export_data: bool = False) -> None:
        """Run the simulation with specified configuration."""
        
        print("🧬 AI Animal Evolution Environment")
        print("=" * 50)
        
        if verbose:
            self._print_config_summary(config)
        
        # Initialize simulation
        print("🚀 Initializing simulation...")
        try:
            self.simulation = Simulation(config)
            print(f"✅ Simulation initialized with {len(self.simulation.creatures)} creatures")
        except Exception as e:
            print(f"❌ Failed to initialize simulation: {e}")
            return
        
        # Run simulation
        self.running = True
        start_time = time.time()
        
        if interactive:
            self._run_interactive_mode(verbose)
        else:
            self._run_batch_mode(config, verbose)
        
        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Simulation completed in {elapsed_time:.2f} seconds")
        
        if self.simulation:
            self._print_final_summary()
            
            if export_data:
                self._export_simulation_data()
    
    def _run_batch_mode(self, config: Config, verbose: bool) -> None:
        """Run simulation in batch mode (non-interactive)."""
        print(f"▶️ Running simulation for {config.max_steps} steps...")
        
        last_report_time = time.time()
        report_interval = 5.0  # Report every 5 seconds
        
        while self.running and self.simulation.step_count < config.max_steps:
            try:
                should_continue = self.simulation.step()
                
                # Periodic progress reports
                current_time = time.time()
                if verbose and (current_time - last_report_time > report_interval):
                    self._print_progress_report()
                    last_report_time = current_time
                
                if not should_continue:
                    print("\n⚠️ Simulation ended (termination condition met)")
                    break
                    
            except KeyboardInterrupt:
                print("\n⏸️ Simulation interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Simulation error at step {self.simulation.step_count}: {e}")
                break
    
    def _run_interactive_mode(self, verbose: bool) -> None:
        """Run simulation in interactive mode with user controls."""
        print("\n🎮 Interactive Mode Commands:")
        print("  [Enter] - Single step")
        print("  'c' + [Enter] - Continue automatically")
        print("  'p' + [Enter] - Pause automatic mode")
        print("  's' + [Enter] - Show statistics")
        print("  'r' + [Enter] - Reset simulation")
        print("  'q' + [Enter] - Quit")
        print()
        
        auto_mode = False
        
        while self.running:
            try:
                if auto_mode and not self.paused:
                    # Automatic mode
                    should_continue = self.simulation.step()
                    
                    if verbose and self.simulation.step_count % 50 == 0:
                        self._print_progress_report()
                    
                    if not should_continue:
                        print("\n⚠️ Simulation ended (termination condition met)")
                        auto_mode = False
                    
                    time.sleep(0.1)  # Small delay for readability
                    
                else:
                    # Interactive mode - wait for user input
                    try:
                        command = input(f"Step {self.simulation.step_count} > ").strip().lower()
                    except EOFError:
                        break
                    
                    if command == '':
                        # Single step
                        should_continue = self.simulation.step()
                        self._print_step_summary()
                        
                        if not should_continue:
                            print("⚠️ Simulation ended (termination condition met)")
                    
                    elif command == 'c':
                        auto_mode = True
                        self.paused = False
                        print("▶️ Continuing automatically...")
                    
                    elif command == 'p':
                        self.paused = True
                        print("⏸️ Paused automatic mode")
                    
                    elif command == 's':
                        self._print_detailed_stats()
                    
                    elif command == 'r':
                        self.simulation.reset()
                        print("🔄 Simulation reset")
                    
                    elif command == 'q':
                        break
                    
                    else:
                        print("❓ Unknown command. Try 'q' to quit.")
                        
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted. Use 'q' to quit or continue with commands.")
                auto_mode = False
                self.paused = True
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _print_config_summary(self, config: Config) -> None:
        """Print configuration summary."""
        print(f"🌍 World: {config.width}x{config.height}")
        print(f"🐾 Initial Population: {config.init_herbivores} herbivores, {config.init_carnivores} carnivores")
        print(f"🧬 Evolution: {config.mutation_rate:.1%} mutation rate, {config.mutation_strength:.1%} strength")
        print(f"🌱 Environment: {config.plant_growth_rate:.3f} growth rate, {config.plant_cap:.1f} capacity")
        print(f"🎲 Seed: {config.seed}")
        print()
    
    def _print_progress_report(self) -> None:
        """Print current simulation progress."""
        if not self.simulation:
            return
        
        stats = self.simulation.get_summary_stats()
        
        print(f"📈 Step {stats['step_count']:,} | "
              f"Pop: {stats['total_creatures']} "
              f"(🌿{stats['herbivore_count']} 🥩{stats['carnivore_count']}) | "
              f"Gen: {stats['avg_generation']:.1f} | "
              f"Food: {stats['environment_stats']['total_food']:.0f}")
    
    def _print_step_summary(self) -> None:
        """Print summary for current step."""
        if not self.simulation:
            return
        
        stats = self.simulation.get_summary_stats()
        
        print(f"Population: {stats['total_creatures']} "
              f"(Herbivores: {stats['herbivore_count']}, "
              f"Carnivores: {stats['carnivore_count']}) | "
              f"Generation: {stats['avg_generation']:.1f}")
    
    def _print_detailed_stats(self) -> None:
        """Print detailed simulation statistics."""
        if not self.simulation:
            return
        
        print("\n📊 Detailed Statistics")
        print("-" * 40)
        
        stats = self.simulation.get_summary_stats()
        
        print(f"Step: {stats['step_count']:,}")
        print(f"Total Births: {stats['total_births']:,}")
        print(f"Total Deaths: {stats['total_deaths']:,}")
        print(f"Current Population: {stats['total_creatures']}")
        print(f"  🌿 Herbivores: {stats['herbivore_count']}")
        print(f"  🥩 Carnivores: {stats['carnivore_count']}")
        print(f"Average Generation: {stats['avg_generation']:.2f}")
        
        env_stats = stats['environment_stats']
        print(f"\nEnvironment:")
        print(f"  🌱 Total Food: {env_stats['total_food']:.0f}")
        print(f"  📊 Average Density: {env_stats['avg_food']:.3f}")
        print(f"  🌡️ Temperature: {env_stats['temperature']:.1f}°C")
        
        # Trait statistics for each species
        for species in ['herbivore', 'carnivore']:
            creatures = [c for c in self.simulation.creatures if c.species == species]
            if creatures:
                print(f"\n{species.title()} Traits:")
                traits = ['speed', 'size', 'aggression', 'perception', 'energy_efficiency']
                
                for trait in traits:
                    values = [getattr(c.genome, trait) for c in creatures]
                    print(f"  {trait.title()}: {np.mean(values):.3f} ± {np.std(values):.3f}")
        
        print()
    
    def _print_summary_stats(self) -> None:
        """Print basic summary statistics."""
        if not self.simulation:
            return
        
        stats = self.simulation.get_summary_stats()
        print(f"Final step: {stats['step_count']:,}")
        print(f"Final population: {stats['total_creatures']}")
        print(f"Total births: {stats['total_births']:,}")
        print(f"Total deaths: {stats['total_deaths']:,}")
    
    def _print_final_summary(self) -> None:
        """Print comprehensive final summary."""
        print("\n🏁 Final Simulation Summary")
        print("=" * 50)
        
        # Generate and print the statistics report
        if self.simulation and self.simulation.statistics.step_data:
            report = self.simulation.statistics.generate_report()
            print(report)
        else:
            self._print_summary_stats()
    
    def _export_simulation_data(self) -> None:
        """Export simulation data to files."""
        if not self.simulation:
            return
        
        try:
            timestamp = int(time.time())
            filename = f"ai_evolution_data_{timestamp}.json"
            
            self.simulation.statistics.export_to_json(filename)
            print(f"📁 Data exported to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to export data: {e}")


def main():
    """Main entry point for the CLI application."""
    
    # Create argument parser
    parser = Config.create_parser()
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true", 
        help="Run in interactive mode with step-by-step control"
    )
    parser.add_argument(
        "--export", "-e",
        action="store_true",
        help="Export simulation data at the end"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create configuration
    config = Config.from_args(args)
    
    # Enable profiling if requested
    if args.profile:
        import cProfile
        import pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
    
    try:
        # Run simulation
        runner = SimulationRunner()
        runner.run(
            config=config,
            verbose=args.verbose,
            interactive=args.interactive,
            export_data=args.export
        )
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Save profiling results if enabled
        if args.profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            print("\n🔍 Performance Profile (Top 20 functions):")
            print("-" * 60)
            stats.print_stats(20)


if __name__ == "__main__":
    main()
