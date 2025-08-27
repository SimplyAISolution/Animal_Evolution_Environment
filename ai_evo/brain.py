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
        
    def get_sensory_input(self, creature, environment, nearby_creatures) -> np.ndarray:
        """Generate sensory input array for the creature."""
        # Basic sensory inputs
        inputs = np.zeros(self.input_size)
        
        # Position relative to world center
        inputs[0] = creature.position[0] / environment.width - 0.5
        inputs[1] = creature.position[1] / environment.height - 0.5
        
        # Energy level (normalized)
        inputs[2] = creature.energy / 200.0  # Assume max energy is ~200
        
        # Food in current location
        x, y = int(creature.position[0]), int(creature.position[1])
        x, y = environment.wrap(x, y)
        inputs[3] = environment.food[y, x] / 10.0  # Assume max food is ~10
        
        # Nearby creature information
        if nearby_creatures:
            # Count of nearby herbivores and carnivores
            herbivore_count = sum(1 for c in nearby_creatures if c.species == "herbivore")
            carnivore_count = sum(1 for c in nearby_creatures if c.species == "carnivore")
            inputs[4] = min(herbivore_count / 5.0, 1.0)  # Normalize
            inputs[5] = min(carnivore_count / 5.0, 1.0)  # Normalize
            
            # Distance to nearest threat/prey
            if creature.species == "herbivore":
                carnivores = [c for c in nearby_creatures if c.species == "carnivore"]
                if carnivores:
                    nearest_distance = min(creature.distance_to(c) for c in carnivores)
                    inputs[6] = max(0, 1.0 - nearest_distance / creature.genome.perception)
            else:  # carnivore
                herbivores = [c for c in nearby_creatures if c.species == "herbivore"]
                if herbivores:
                    nearest_distance = min(creature.distance_to(c) for c in herbivores)
                    inputs[7] = max(0, 1.0 - nearest_distance / creature.genome.perception)
        
        return inputs
        
    def forward(self, sensory_input: np.ndarray) -> dict:
        """Forward pass through the neural network, return action dictionary."""
        outputs = self.process(sensory_input)
        
        # Convert neural network outputs to action dictionary
        return {
            "move_x": float(outputs[0]) if len(outputs) > 0 else 0.0,
            "move_y": float(outputs[1]) if len(outputs) > 1 else 0.0,
            "eat": float(outputs[2]) if len(outputs) > 2 else 0.0,
            "reproduce": float(outputs[3]) if len(outputs) > 3 else 0.0
        }
