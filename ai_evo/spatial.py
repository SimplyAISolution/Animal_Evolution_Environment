from collections import defaultdict
import math

class SpatialHash:
    """Grid-based spatial hashing for approximate neighbor retrieval."""
    def __init__(self, cell, W, H):
        self.cell, self.W, self.H = cell, W, H
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
        out
