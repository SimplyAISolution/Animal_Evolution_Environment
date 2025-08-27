"""Spatial hashing for efficient neighbor queries in 2D space."""

from collections import defaultdict
import math
import numpy as np
from typing import List, Tuple, Any


class SpatialHash:
    """Grid-based spatial hashing for approximate neighbor retrieval."""
    
    def __init__(self, cell_size: float, world_width: float, world_height: float):
        """Initialize spatial hash with given cell size and world dimensions.
        
        Args:
            cell_size: Size of each grid cell
            world_width: Width of the simulation world
            world_height: Height of the simulation world
        """
        self.cell_size = cell_size
        self.world_width = world_width
        self.world_height = world_height
        self.grid = defaultdict(list)
        self.query_count = 0
        self.total_neighbors_checked = 0

    def _get_cell_key(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell key."""
        # Handle toroidal wrapping
        x = x % self.world_width
        y = y % self.world_height
        return (int(x // self.cell_size), int(y // self.cell_size))

    def rebuild(self, agents: List[Any]) -> None:
        """Rebuild the spatial hash with current agent positions.
        
        Args:
            agents: List of agents with position attribute
        """
        self.grid.clear()
        for idx, agent in enumerate(agents):
            cell_key = self._get_cell_key(agent.position[0], agent.position[1])
            self.grid[cell_key].append(idx)

    def get_neighbors(self, agents: List[Any], query_agent: Any, radius: float) -> List[Any]:
        """Find all agents within radius of the query agent.
        
        Args:
            agents: List of all agents
            query_agent: Agent to find neighbors for
            radius: Search radius
            
        Returns:
            List of neighboring agents
        """
        self.query_count += 1
        neighbors = []
        
        # Get query agent's cell
        qx, qy = query_agent.position[0], query_agent.position[1]
        query_cell = self._get_cell_key(qx, qy)
        
        # Calculate how many cells to check in each direction
        cells_to_check = int(math.ceil(radius / self.cell_size))
        
        # Check all cells within the radius
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                # Handle toroidal wrapping for cell coordinates
                cell_x = (query_cell[0] + dx) % int(math.ceil(self.world_width / self.cell_size))
                cell_y = (query_cell[1] + dy) % int(math.ceil(self.world_height / self.cell_size))
                
                cell_key = (cell_x, cell_y)
                
                if cell_key in self.grid:
                    for agent_idx in self.grid[cell_key]:
                        if agent_idx < len(agents):
                            agent = agents[agent_idx]
                            
                            # Skip self
                            if agent.id == query_agent.id:
                                continue
                            
                            # Calculate distance with toroidal wrapping
                            distance = self._toroidal_distance(
                                query_agent.position, agent.position
                            )
                            
                            self.total_neighbors_checked += 1
                            
                            if distance <= radius:
                                neighbors.append(agent)
        
        return neighbors
    
    def _toroidal_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate distance between two points on a toroidal surface.
        
        Args:
            pos1: First position [x, y]
            pos2: Second position [x, y]
            
        Returns:
            Distance between the points
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        # Handle toroidal wrapping - take shortest path
        dx = min(dx, self.world_width - dx)
        dy = min(dy, self.world_height - dy)
        
        return math.sqrt(dx * dx + dy * dy)
    
    def get_stats(self) -> dict:
        """Get performance statistics for the spatial hash.
        
        Returns:
            Dictionary with performance metrics
        """
        total_cells = len(self.grid)
        occupied_cells = sum(1 for cell in self.grid.values() if len(cell) > 0)
        avg_agents_per_cell = (
            sum(len(cell) for cell in self.grid.values()) / max(occupied_cells, 1)
        )
        
        return {
            'total_cells': total_cells,
            'occupied_cells': occupied_cells,
            'avg_agents_per_cell': avg_agents_per_cell,
            'query_count': self.query_count,
            'total_neighbors_checked': self.total_neighbors_checked,
            'avg_neighbors_per_query': (
                self.total_neighbors_checked / max(self.query_count, 1)
            )
        }
