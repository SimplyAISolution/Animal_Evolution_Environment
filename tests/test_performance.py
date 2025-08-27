"""Performance tests for spatial hashing and simulation efficiency."""

import pytest
import time
import numpy as np
from ai_evo.simulation import Simulation
from ai_evo.config import Config
from ai_evo.spatial import SpatialHash
from ai_evo.creatures import Creature
from ai_evo.genome import Genome

class TestPerformance:
    """Test suite for performance optimization and scalability."""
    
    def test_spatial_hash_performance(self):
        """Test that spatial hashing provides performance benefits."""
        from ai_evo.rng import RNG
        
        # Create test agents
        rng = RNG(42)
        agents = []
        
        for i in range(1000):
            genome = Genome(
                speed=1.0, size=1.0, aggression=0.5, perception=2.0,
                energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
            )
            position = np.array([rng.rand() * 100, rng.rand() * 100])
            agent = Creature(
                id=f"test_{i}", genome=genome, species="herbivore",
                position=position, energy=50.0
            )
            agents.append(agent)
        
        # Test spatial hash performance
        spatial_hash = SpatialHash(cell_size=10, world_width=100, world_height=100)
        
        # Time spatial hash neighbor finding
        start_time = time.time()
        
        for _ in range(100):  # Multiple queries
            spatial_hash.rebuild(agents)
            query_agent = agents[0]
            neighbors = spatial_hash.get_neighbors(agents, query_agent, radius=5.0)
        
        spatial_time = time.time() - start_time
        
        # Time naive neighbor finding (O(N²) approach)
        start_time = time.time()
        
        for _ in range(100):
            query_agent = agents[0]
            naive_neighbors = []
            
            for agent in agents[1:]:  # Skip self
                distance = np.linalg.norm(query_agent.position - agent.position)
                if distance <= 5.0:
                    naive_neighbors.append(agent)
        
        naive_time = time.time() - start_time
        
        # Spatial hash should be faster for large datasets
        print(f"Spatial hash time: {spatial_time:.4f}s")
        print(f"Naive approach time: {naive_time:.4f}s")
        print(f"Speedup: {naive_time / spatial_time:.2f}x")
        
        # For 1000 agents, spatial hashing should provide speedup
        assert spatial_time < naive_time
    
    def test_simulation_step_performance(self):
        """Test simulation step performance under various loads."""
        configs = [
            Config(init_herbivores=50, init_carnivores=20, width=60, height=60),
            Config(init_herbivores=100, init_carnivores=40, width=80, height=80),
            Config(init_herbivores=200, init_carnivores=80, width=100, height=100),
        ]
        
        performance_results = []
        
        for i, config in enumerate(configs):
            config.seed = 42  # Consistent seed for fair comparison
            config.max_steps = 50
            
            sim = Simulation(config)
            initial_population = len(sim.creatures)
            
            # Time simulation steps
            start_time = time.time()
            
            for _ in range(50):
                sim.step()
                if len(sim.creatures) == 0:
                    break
            
            elapsed_time = time.time() - start_time
            steps_per_second = 50 / elapsed_time
            
            performance_results.append({
                'initial_population': initial_population,
                'steps_per_second': steps_per_second,
                'time_per_step': elapsed_time / 50
            })
            
            print(f"Config {i+1}: {initial_population} agents, "
                  f"{steps_per_second:.2f} steps/sec, "
                  f"{elapsed_time/50:.4f}s per step")
        
        # Performance shouldn't degrade dramatically with population
        # (though some degradation is expected)
        for result in performance_results:
            assert result['steps_per_second'] > 1.0  # At least 1 step per second
            assert result['time_per_step'] < 1.0     # Less than 1 second per step
    
    def test_memory_efficiency(self):
        """Test that memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large simulation
        config = Config(
            seed=42,
            init_herbivores=300,
            init_carnivores=100,
            width=150,
            height=150,
            max_steps=200
        )
        
        sim = Simulation(config)
        
        # Run simulation and monitor memory
        memory_samples = []
        
        for step in range(200):
            sim.step()
            
            if step % 20 == 0:  # Sample every 20 steps
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory - initial_memory)
        
        # Memory shouldn't grow excessively
        max_memory_growth = max(memory_samples)
        print(f"Maximum memory growth: {max_memory_growth:.1f} MB")
        
        # Should not exceed 500MB additional memory for this test
        assert max_memory_growth < 500
        
        # Memory shouldn't have large leaks (no continuous growth)
        if len(memory_samples) > 5:
            recent_growth = memory_samples[-1] - memory_samples[-5]
            assert recent_growth < 100  # Less than 100MB growth in last 100 steps
    
    def test_spatial_hash_accuracy(self):
        """Test that spatial hash returns correct neighbors."""
        from ai_evo.rng import RNG
        
        rng = RNG(123)
        agents = []
        
        # Create test agents in a grid pattern
        for x in range(10):
            for y in range(10):
                genome = Genome(
                    speed=1.0, size=1.0, aggression=0.5, perception=2.0,
                    energy_efficiency=1.0, reproduction_threshold=90.0, lifespan=1000
                )
                position = np.array([x * 5.0, y * 5.0])  # 5 unit spacing
                agent = Creature(
                    id=f"agent_{x}_{y}", genome=genome, species="herbivore",
                    position=position, energy=50.0
                )
                agents.append(agent)
        
        spatial_hash = SpatialHash(cell_size=6, world_width=50, world_height=50)
        spatial_hash.rebuild(agents)
        
        # Test query from center
        query_agent = agents[55]  # Agent at (5, 5) * 5 = (25, 25)
        radius = 10.0
        
        # Get neighbors using spatial hash
        spatial_neighbors = spatial_hash.get_neighbors(agents, query_agent, radius)
        
        # Get neighbors using brute force
        brute_force_neighbors = []
        for agent in agents:
            if agent.id != query_agent.id:
                distance = np.linalg.norm(query_agent.position - agent.position)
                if distance <= radius:
                    brute_force_neighbors.append(agent)
        
        # Convert to sets of IDs for comparison
        spatial_ids = set(agent.id for agent in spatial_neighbors)
        brute_force_ids = set(agent.id for agent in brute_force_neighbors)
        
        # Results should be identical
        assert spatial_ids == brute_force_ids
        assert len(spatial_neighbors) == len(brute_force_neighbors)
    
    def test_large_population_stability(self):
        """Test simulation stability with large populations."""
        config = Config(
            seed=999,
            init_herbivores=500,
            init_carnivores=200,
            width=200,
            height=200,
            max_steps=300,
            plant_growth_rate=0.15,  # Higher food to support larger population
            grid_cell=8  # Larger cells for better performance
        )
        
        sim = Simulation(config)
        
        population_history = []
        step_times = []
        
        for step in range(300):
            start_time = time.time()
            
            success = sim.step()
            
            step_time = time.time() - start_time
            step_times.append(step_time)
            
            population_history.append(len(sim.creatures))
            
            # Check for simulation crashes
            assert sim.step_count == step + 1
            
            # Population should not go negative or exceed reasonable bounds
            assert len(sim.creatures) >= 0
            assert len(sim.creatures) <= 2000  # Reasonable upper bound
            
            if not success or len(sim.creatures) == 0:
                break
        
        # Should maintain reasonable performance even with large populations
        avg_step_time = np.mean(step_times)
        max_step_time = max(step_times)
        
        print(f"Average step time: {avg_step_time:.4f}s")
        print(f"Maximum step time: {max_step_time:.4f}s")
        print(f"Final population: {len(sim.creatures)}")
        
        # Performance requirements
        assert avg_step_time < 0.1  # Average less than 100ms per step
        assert max_step_time < 0.5  # No single step takes more than 500ms
        
        # Should run for reasonable number of steps
        assert len(population_history) > 50
    
    def test_concurrent_access_safety(self):
        """Test thread safety of core components (if needed for future extensions)."""
        import threading
        import queue
        
        config = Config(seed=42, init_herbivores=50, init_carnivores=20)
        sim = Simulation(config)
        
        results_queue = queue.Queue()
        errors = []
        
        def worker_thread(thread_id):
            """Worker thread that performs read operations."""
            try:
                for _ in range(100):
                    # Read-only operations that should be safe
                    creature_data = sim.get_creature_data()
                    env_data = sim.get_environment_data()
                    stats = sim.get_summary_stats()
                    
                    # Verify data consistency
                    assert len(creature_data) >= 0
                    assert env_data['width'] == config.width
                    assert env_data['height'] == config.height
                    assert stats['step_count'] >= 0
                
                results_queue.put(f"Thread {thread_id} completed successfully")
                
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {str(e)}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Run simulation steps in main thread
        for _ in range(50):
            sim.step()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # All threads should have completed
        completed_threads = 0
        while not results_queue.empty():
            result = results_queue.get()
            assert "completed successfully" in result
            completed_threads += 1
        
        assert completed_threads == 5
