"""Neural network brain for creatures."""

import numpy as np
from typing import Tuple, Optional
from .genome import Genome


class CreatureBrain:
    """Simple neural network brain for creature decision making."""
    
    def __init__(self, genome: Genome):
        """Initialize brain with genome weights."""
        self.weights = genome.brain_weights
        self.input_size = 8  # Basic sensory inputs
        self.output_size = 4  # movement directions
        
    def process(self, inputs: np.ndarray) -> np.ndarray:
        """Process sensory inputs to generate actions."""
        # Simple feedforward network
        if len(inputs) != self.input_size:
            inputs = np.pad(inputs, (0, max(0, self.input_size - len(inputs))))[:self.input_size]
            
        # Simple linear transformation
        if len(self.weights) >= self.input_size * self.output_size:
            weight_matrix = self.weights[:self.input_size * self.output_size].reshape(self.input_size, self.output_size)
            outputs = np.dot(inputs, weight_matrix)
            return np.tanh(outputs)  # Activation function
        else:
            # Fallback for insufficient weights
            return np.random.randn(self.output_size)
            
    def get_movement(self, sensory_data: np.ndarray) -> Tuple[float, float]:
        """Get movement direction from sensory input."""
        outputs = self.process(sensory_data)
        dx = outputs[0] - outputs[1]  # left-right
        dy = outputs[2] - outputs[3]  # up-down
        return dx, dy
