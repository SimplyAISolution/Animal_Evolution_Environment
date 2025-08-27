#!/usr/bin/env python3
"""Performance testing and profiling script for the Animal Evolution Environment."""

import sys
import time
from ai_evo import Config, Simulation
from ai_evo.profiler import global_profiler


def run_performance_test():
    """Run a comprehensive performance test with profiling."""
    print("🚀 Animal Evolution Environment - Performance Test")
    print("=" * 50)
    
    # Test configurations - from small to large
    test_configs = [
        {
            "name": "Small Population",
            "config": Config(
                seed=42,
                init_herbivores=25,
                init_carnivores=10,
                width=60,
                height=60,
                max_steps=100,
                enable_profiling=True
            )
        },
        {
            "name": "Medium Population",
            "config": Config(
                seed=42,
                init_herbivores=100,
                init_carnivores=40,
                width=100,
                height=100,
                max_steps=100,
                enable_profiling=True
            )
        },
        {
            "name": "Large Population",
            "config": Config(
                seed=42,
                init_herbivores=200,
                init_carnivores=80,
                width=150,
                height=150,
                max_steps=100,
                enable_profiling=True
            )
        }
    ]
    
    results = []
    
    for test_config in test_configs:
        print(f"\n🧪 Testing: {test_config['name']}")
        print("-" * 30)
        
        config = test_config["config"]
        
        # Reset profiler for each test
        global_profiler.disable()
        global_profiler.__init__()
        
        # Create and run simulation
        start_time = time.time()
        sim = Simulation(config)
        
        print(f"Initial population: {len(sim.creatures)} creatures")
        print(f"World size: {config.width}x{config.height}")
        
        # Run simulation
        step_count = 0
        for step in range(config.max_steps):
            sim.step()
            step_count += 1
            
            # Progress indicator
            if step % 20 == 0:
                print(f"  Step {step}: {len(sim.creatures)} creatures alive")
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Results for {test_config['name']}:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Steps completed: {step_count}")
        print(f"  Final population: {len(sim.creatures)}")
        print(f"  Average step time: {total_time/step_count:.4f}s")
        print(f"  Steps per second: {step_count/total_time:.2f}")
        
        # Get performance report
        if global_profiler.enabled:
            print("\n📈 Detailed Performance Analysis:")
            sim.print_performance_summary()
        
        results.append({
            "name": test_config["name"],
            "initial_population": config.init_herbivores + config.init_carnivores,
            "final_population": len(sim.creatures),
            "total_time": total_time,
            "avg_step_time": total_time / step_count,
            "steps_per_second": step_count / total_time,
            "performance_report": sim.get_performance_report() if global_profiler.enabled else None
        })
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("📋 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Test Name':<20} {'Population':<12} {'Step Time':<12} {'Steps/sec':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<20} "
              f"{result['initial_population']:>3}->{result['final_population']:<3} "
              f"{result['avg_step_time']:>8.4f}s   "
              f"{result['steps_per_second']:>6.1f}")
    
    # Performance benchmarks
    print("\n🎯 Performance Benchmarks:")
    
    for result in results:
        step_time = result['avg_step_time']
        population = result['initial_population']
        
        if step_time < 0.01:
            grade = "🌟 Excellent"
        elif step_time < 0.05:
            grade = "✅ Good"
        elif step_time < 0.1:
            grade = "⚠️  Acceptable"
        else:
            grade = "🔴 Needs Optimization"
            
        print(f"  {result['name']}: {grade} ({step_time:.4f}s per step)")
    
    # Recommendations
    print("\n💡 Optimization Recommendations:")
    large_result = results[-1]  # Largest test
    
    if large_result['avg_step_time'] > 0.1:
        print("  • Consider implementing additional optimizations for large populations")
        print("  • Spatial hash cell size may need tuning")
        print("  • Brain processing could be optimized with vectorization")
    
    if any(r['avg_step_time'] > r['initial_population'] * 0.0001 for r in results):
        print("  • Step time scaling may be suboptimal - check O(n²) operations")
    
    print("\n✅ Performance testing completed!")
    return results


def run_spatial_hash_benchmark():
    """Run specific benchmark for spatial hashing performance."""
    print("\n🔍 Spatial Hash Performance Benchmark")
    print("-" * 40)
    
    from ai_evo.spatial import SpatialHash
    from ai_evo.creatures import Creature
    from ai_evo.genome import Genome
    from ai_evo.rng import RNG
    import numpy as np
    
    # Test different cell sizes
    cell_sizes = [5, 10, 15, 20]
    agent_counts = [100, 500, 1000]
    
    for agent_count in agent_counts:
        print(f"\n📏 Testing with {agent_count} agents:")
        
        # Create test agents
        rng = RNG(42)
        agents = []
        for i in range(agent_count):
            genome = Genome()
            position = np.array([rng.rand() * 100, rng.rand() * 100])
            agent = Creature(f"test_{i}", genome, "herbivore", position, 50.0)
            agents.append(agent)
        
        for cell_size in cell_sizes:
            # Test spatial hash
            spatial_hash = SpatialHash(cell_size, 100, 100)
            
            start_time = time.time()
            for _ in range(100):  # Multiple queries for timing accuracy
                spatial_hash.rebuild(agents)
                query_agent = agents[0]
                neighbors = spatial_hash.get_neighbors(agents, query_agent, 10.0)
            query_time = time.time() - start_time
            
            print(f"  Cell size {cell_size:2d}: {query_time*1000:.1f}ms for 100 queries "
                  f"({query_time*10:.3f}ms per query)")


if __name__ == "__main__":
    print("🧬 Animal Evolution Environment - Performance Optimization Suite")
    print("================================================================")
    
    try:
        # Run main performance test
        results = run_performance_test()
        
        # Run spatial hash benchmark
        run_spatial_hash_benchmark()
        
        print(f"\n🎉 All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Performance testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during performance testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)