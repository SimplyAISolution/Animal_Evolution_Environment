from collections import defaultdict
import math

class SpatialHash:
    """Grid-based spatial hashing for approximate neighbor retrieval."""
    def __init__(self, cell_size=None, world_width=None, world_height=None, cell=None, W=None, H=None):
        # Support both new and old parameter styles
        if cell_size is not None:
            self.cell = cell_size
            self.W = world_width
            self.H = world_height
        else:
            self.cell = cell
            self.W = W
            self.H = H
        self.grid = defaultdict(list)

    def _key(self, x, y):
        return (int(x) // self.cell, int(y) // self.cell)

    def rebuild(self, agents):
        self.grid.clear()
        for idx, a in enumerate(agents):
            self.grid[self._key(a.position[0], a.position[1])].append(idx)

    def neighbors(self, agents, a, radius):
        cx, cy = self._key(a.position[0], a.position[1])
        r = int(math.ceil(radius / self.cell))
        result = []
        
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                key = (cx + dx, cy + dy)
                if key in self.grid:
                    for idx in self.grid[key]:
                        agent = agents[idx]
                        dist = math.sqrt((a.position[0] - agent.position[0])**2 + 
                                       (a.position[1] - agent.position[1])**2)
                        if dist <= radius and agent != a:
                            result.append(agent)
        return result
        
    def get_neighbors(self, agents, query_agent, radius):
        """Get neighbors within radius of query agent (alias for neighbors)."""
        return self.neighbors(agents, query_agent, radius)
        
    def get_stats(self) -> dict:
        """Get spatial hash statistics."""
        total_cells = len(self.grid)
        total_agents = sum(len(agents) for agents in self.grid.values())
        avg_agents_per_cell = total_agents / max(1, total_cells)
        
        return {
            "total_cells": total_cells,
            "total_agents": total_agents,
            "avg_agents_per_cell": avg_agents_per_cell
        }
