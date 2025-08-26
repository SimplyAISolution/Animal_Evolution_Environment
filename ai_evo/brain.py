import numpy as np

class RNG:
    """Central deterministic RNG wrapper."""
    def __init__(self, seed: int):
        self.seed = seed
        self.rs = np.random.RandomState(seed)

    # Basic distributions
    def normal(self, loc=0.0, scale=1.0, size=None):
        return self.rs.normal(loc, scale, size)

    def rand(self, *shape):
        return self.rs.rand(*shape)

    def randint(self, low, high=None, size=None):
        return self.rs.randint(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        return self.rs.choice(a, size, replace, p)

    def random_bool(self, p=0.5, size=None):
        return self.rs.rand(*(size if isinstance(size, tuple) else (() if size is None else (size,)))) < p
