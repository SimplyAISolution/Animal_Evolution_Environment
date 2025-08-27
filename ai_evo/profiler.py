"""Performance profiling and monitoring tools for simulation optimization."""

import time
import psutil
import os
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class PerformanceProfiler:
    """Monitor and analyze simulation performance metrics."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.enabled = False
        self.process = psutil.Process(os.getpid())
        
        # Timing data
        self.step_times = []
        self.method_times = defaultdict(list)
        self.current_timers = {}
        
        # Memory tracking
        self.memory_samples = []
        self.initial_memory = None
        
        # Performance counters
        self.creature_counts = []
        self.spatial_hash_stats = []
        
    def enable(self):
        """Enable profiling."""
        self.enabled = True
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def disable(self):
        """Disable profiling."""
        self.enabled = False
        
    def start_timer(self, name: str):
        """Start timing a code section."""
        if not self.enabled:
            return
        self.current_timers[name] = time.time()
        
    def end_timer(self, name: str):
        """End timing a code section."""
        if not self.enabled or name not in self.current_timers:
            return
        elapsed = time.time() - self.current_timers[name]
        self.method_times[name].append(elapsed)
        del self.current_timers[name]
        
    def record_step(self, step_time: float, creature_count: int, spatial_stats: Optional[Dict] = None):
        """Record data for a simulation step."""
        if not self.enabled:
            return
            
        self.step_times.append(step_time)
        self.creature_counts.append(creature_count)
        
        # Memory sample
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = current_memory - self.initial_memory
        self.memory_samples.append(memory_growth)
        
        # Spatial hash performance
        if spatial_stats:
            self.spatial_hash_stats.append(spatial_stats)
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self.step_times:
            return {"error": "No performance data collected"}
            
        report = {
            "simulation_performance": {
                "total_steps": len(self.step_times),
                "avg_step_time": np.mean(self.step_times),
                "max_step_time": np.max(self.step_times),
                "min_step_time": np.min(self.step_times),
                "steps_per_second": 1.0 / np.mean(self.step_times),
                "total_simulation_time": np.sum(self.step_times)
            },
            
            "memory_usage": {
                "peak_memory_growth": np.max(self.memory_samples) if self.memory_samples else 0,
                "avg_memory_growth": np.mean(self.memory_samples) if self.memory_samples else 0,
                "memory_leak_detected": self._detect_memory_leak()
            },
            
            "population_dynamics": {
                "initial_population": self.creature_counts[0] if self.creature_counts else 0,
                "final_population": self.creature_counts[-1] if self.creature_counts else 0,
                "peak_population": np.max(self.creature_counts) if self.creature_counts else 0,
                "avg_population": np.mean(self.creature_counts) if self.creature_counts else 0
            },
            
            "method_performance": {}
        }
        
        # Method-level performance
        for method, times in self.method_times.items():
            report["method_performance"][method] = {
                "total_time": np.sum(times),
                "avg_time": np.mean(times),
                "call_count": len(times),
                "time_percentage": np.sum(times) / np.sum(self.step_times) * 100 if self.step_times else 0
            }
        
        # Spatial hash performance
        if self.spatial_hash_stats:
            report["spatial_hash"] = self._analyze_spatial_performance()
            
        return report
    
    def _detect_memory_leak(self) -> bool:
        """Detect potential memory leaks."""
        if len(self.memory_samples) < 10:
            return False
            
        # Check if memory consistently grows over time
        recent_samples = self.memory_samples[-10:]
        early_samples = self.memory_samples[:10]
        
        recent_avg = np.mean(recent_samples)
        early_avg = np.mean(early_samples)
        
        # If memory grew by more than 50MB and keeps growing
        return (recent_avg - early_avg > 50 and 
                np.polyfit(range(len(recent_samples)), recent_samples, 1)[0] > 1.0)
    
    def _analyze_spatial_performance(self) -> Dict:
        """Analyze spatial hash performance."""
        if not self.spatial_hash_stats:
            return {}
            
        total_queries = sum(stats.get('query_count', 0) for stats in self.spatial_hash_stats)
        total_neighbors_checked = sum(stats.get('total_neighbors_checked', 0) for stats in self.spatial_hash_stats)
        
        return {
            "total_queries": total_queries,
            "total_neighbors_checked": total_neighbors_checked,
            "avg_neighbors_per_query": total_neighbors_checked / max(total_queries, 1),
            "efficiency_ratio": total_queries / max(total_neighbors_checked, 1)  # Higher is better
        }
    
    def print_performance_summary(self):
        """Print a readable performance summary."""
        report = self.get_performance_report()
        
        if "error" in report:
            print(report["error"])
            return
            
        print("\n=== Performance Analysis ===")
        
        sim_perf = report["simulation_performance"]
        print(f"Total Steps: {sim_perf['total_steps']}")
        print(f"Average Step Time: {sim_perf['avg_step_time']:.4f}s")
        print(f"Steps per Second: {sim_perf['steps_per_second']:.2f}")
        print(f"Total Simulation Time: {sim_perf['total_simulation_time']:.2f}s")
        
        mem_usage = report["memory_usage"]
        print(f"\nMemory Growth: {mem_usage['avg_memory_growth']:.1f} MB (peak: {mem_usage['peak_memory_growth']:.1f} MB)")
        if mem_usage["memory_leak_detected"]:
            print("⚠️  Potential memory leak detected!")
        
        pop_dynamics = report["population_dynamics"]
        print(f"\nPopulation: {pop_dynamics['initial_population']} → {pop_dynamics['final_population']} (peak: {pop_dynamics['peak_population']})")
        
        # Top time-consuming methods
        method_perf = report["method_performance"]
        if method_perf:
            print("\nTop Time-Consuming Methods:")
            sorted_methods = sorted(method_perf.items(), key=lambda x: x[1]['total_time'], reverse=True)
            for method, stats in sorted_methods[:5]:
                print(f"  {method}: {stats['total_time']:.3f}s ({stats['time_percentage']:.1f}%)")
        
        # Spatial hash efficiency
        if "spatial_hash" in report:
            spatial = report["spatial_hash"]
            print(f"\nSpatial Hash Efficiency: {spatial['efficiency_ratio']:.3f}")
            print(f"Average Neighbors per Query: {spatial['avg_neighbors_per_query']:.1f}")


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name
        
    def __enter__(self):
        self.profiler.start_timer(self.name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timer(self.name)


# Global profiler instance
global_profiler = PerformanceProfiler()


def profile_method(method_name: str):
    """Decorator to profile method execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global_profiler.start_timer(method_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                global_profiler.end_timer(method_name)
        return wrapper
    return decorator


def timing_context(name: str) -> TimingContext:
    """Create a timing context for profiling code blocks."""
    return TimingContext(global_profiler, name)