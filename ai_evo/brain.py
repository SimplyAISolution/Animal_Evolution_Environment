"""Neural network brain for creature decision making."""

import numpy as np
from typing import Dict, List, Any
from .genome import Genome


class CreatureBrain:
    """Simple neural network for creature decision making."""
    
    def __init__(self, genome: Genome, rng):
        """Initialize brain with genome-based architecture.
        
        Args:
            genome: Creature's genome defining brain parameters
            rng: Random number generator for weight initialization
        """
        self.genome = genome
        self.rng = rng
        
        # Network architecture - simple feedforward
        self.input_size = 8  # [energy, food_density, nearest_food_x, nearest_food_y, 
                            # nearest_creature_x, nearest_creature_y, nearest_creature_type, danger_level]
        self.hidden_size = 6
        self.output_size = 4  # [move_x, move_y, attack, reproduce]
        
        # Initialize weights based on genome
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize neural network weights."""
        # Input to hidden weights
        self.w1 = self.rng.normal(0, 0.5, (self.input_size, self.hidden_size))
        self.b1 = self.rng.normal(0, 0.1, self.hidden_size)
        
        # Hidden to output weights  
        self.w2 = self.rng.normal(0, 0.5, (self.hidden_size, self.output_size))
        self.b2 = self.rng.normal(0, 0.1, self.output_size)
        
        # Scale weights by genome traits for personality
        aggression_scale = 0.5 + self.genome.aggression
        self.w2[:, 2] *= aggression_scale  # Attack output
        
        speed_scale = 0.5 + self.genome.speed / 2.0
        self.w2[:, :2] *= speed_scale  # Movement outputs
    
    def get_sensory_input(self, creature, environment, nearby_creatures: List) -> np.ndarray:
        """Generate sensory input vector for creature.
        
        Args:
            creature: The creature this brain belongs to
            environment: World environment
            nearby_creatures: List of nearby creatures
            
        Returns:
            Sensory input vector
        """
        inputs = np.zeros(self.input_size)
        
        # Energy level (normalized)
        inputs[0] = creature.energy / 200.0
        
        # Get local environment information
        x, y = int(creature.position[0]), int(creature.position[1])
        local_food = environment.get_food_at(x, y)
        inputs[1] = min(local_food / 5.0, 1.0)  # Normalized food density
        
        # Find nearest food source
        nearest_food_dist = float('inf')
        nearest_food_pos = [0, 0]
        
        # Sample nearby food sources
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                check_x = (x + dx) % environment.width
                check_y = (y + dy) % environment.height
                food_amount = environment.get_food_at(check_x, check_y)
                
                if food_amount > 0.1:
                    dist = dx*dx + dy*dy
                    if dist < nearest_food_dist:
                        nearest_food_dist = dist
                        nearest_food_pos = [dx, dy]
        
        # Normalize nearest food direction
        if nearest_food_dist < float('inf'):
            inputs[2] = nearest_food_pos[0] / 5.0
            inputs[3] = nearest_food_pos[1] / 5.0
        
        # Find nearest creature
        nearest_creature_dist = float('inf')
        nearest_creature_pos = [0, 0]
        nearest_creature_type = 0
        
        for other in nearby_creatures:
            dx = other.position[0] - creature.position[0]
            dy = other.position[1] - creature.position[1]
            
            # Handle toroidal wrapping
            if abs(dx) > environment.width / 2:
                dx = -np.sign(dx) * (environment.width - abs(dx))
            if abs(dy) > environment.height / 2:
                dy = -np.sign(dy) * (environment.height - abs(dy))
            
            dist = dx*dx + dy*dy
            if dist < nearest_creature_dist:
                nearest_creature_dist = dist
                nearest_creature_pos = [dx, dy]
                # Encode creature type: 1.0 for same species, -1.0 for different
                nearest_creature_type = 1.0 if other.species == creature.species else -1.0
        
        if nearest_creature_dist < float('inf'):
            # Normalize to perception range
            max_dist = self.genome.perception
            inputs[4] = np.clip(nearest_creature_pos[0] / max_dist, -1, 1)
            inputs[5] = np.clip(nearest_creature_pos[1] / max_dist, -1, 1)
            inputs[6] = nearest_creature_type
            
            # Danger level - high if different species and close
            if nearest_creature_type < 0 and nearest_creature_dist < 4:
                inputs[7] = 1.0
        
        return inputs
    
    def forward(self, inputs: np.ndarray) -> Dict[str, float]:
        """Forward pass through neural network.
        
        Args:
            inputs: Sensory input vector
            
        Returns:
            Dictionary of action outputs
        """
        # Hidden layer with ReLU activation
        hidden = np.maximum(0, np.dot(inputs, self.w1) + self.b1)
        
        # Output layer with tanh activation
        outputs = np.tanh(np.dot(hidden, self.w2) + self.b2)
        
        return {
            'move_x': outputs[0],
            'move_y': outputs[1], 
            'attack': max(0, outputs[2]),  # Attack only if positive
            'reproduce': max(0, outputs[3])  # Reproduce only if positive
        }
