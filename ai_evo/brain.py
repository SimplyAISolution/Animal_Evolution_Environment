"""Neural network brain for creatures."""

import numpy as np
from typing import Tuple, Optional
from .genome import Genome


class CreatureBrain:
    """Simple neural network brain for creature decision making."""
    
    def __init__(self, genome: Genome, config=None):
        """Initialize brain with genome weights."""
        self.genome = genome
        self.weights = genome.brain_weights
        self.input_size = 8  # Basic sensory inputs
        self.output_size = 4  # movement directions
        
    def process(self, inputs: np.ndarray) -> np.ndarray:
        """Process sensory inputs to generate actions."""
        # Simple feedforward network
        if len(inputs) != self.input_size:
            inputs = np.pad(inputs, (0, max(0, self.input_size - len(inputs))))[:self.input_size]
            
        if isinstance(self.weights, tuple) and len(self.weights) == 4:
            # Handle tuple format (W1, b1, W2, b2)
            W1, b1, W2, b2 = self.weights
            hidden = np.tanh(np.dot(inputs, W1) + b1)
            outputs = np.tanh(np.dot(hidden, W2) + b2)
            return outputs
        else:
            # Handle simple array format - fallback
            if hasattr(self.weights, 'shape') and len(self.weights) >= self.input_size * self.output_size:
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
