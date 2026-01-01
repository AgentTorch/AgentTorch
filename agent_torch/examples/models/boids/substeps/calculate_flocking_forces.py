from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_var

@Registry.register_substep("calculate_flocking_forces", "policy")
class CalculateFlockingForces(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize learnable parameters from arguments
        self.speed_range = self.learnable_args.get("speed_range", [0.5, 1.5])
        self.position_margin = self.learnable_args.get("position_margin", 50.0)
        self.separation_weight = self.learnable_args.get("separation_weight", 1.5)
        self.alignment_weight = self.learnable_args.get("alignment_weight", 1.0)
        self.cohesion_weight = self.learnable_args.get("cohesion_weight", 1.0)
        
        # Initialization flag
        self.initialized = False
    
    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        """
        Initialize boids (first run) then calculate flocking forces.
        
        Args:
            state: Current state of the simulation
            observations: Agent observations
            
        Returns:
            Dict containing: steering_force
        """
        # Initialize on first run
        if not self.initialized:
            self._initialize_boids(state)
            self.initialized = True
        # Get input variables from state
        position = get_var(state, self.input_variables["position"])
        velocity = get_var(state, self.input_variables["velocity"])
        perception_radius = get_var(state, self.input_variables["perception_radius"])
        separation_distance = get_var(state, self.input_variables["separation_distance"])
        
        num_agents = position.shape[0]
        device = position.device
        
        # Calculate pairwise distances between all agents
        # positions: [num_agents, 2] -> distances: [num_agents, num_agents]
        distances = torch.cdist(position, position)
        
        # Create masks for neighbors within perception radius and separation distance
        neighbors_mask = (distances < perception_radius) & (distances > 0)  # exclude self
        separation_mask = (distances < separation_distance) & (distances > 0)
        
        # Initialize force accumulators
        separation_force = torch.zeros_like(position)
        alignment_force = torch.zeros_like(position)
        cohesion_force = torch.zeros_like(position)
        
        # 1. SEPARATION: Steer away from nearby neighbors
        for i in range(num_agents):
            separating_neighbors = separation_mask[i]
            if separating_neighbors.any():
                # Calculate vectors away from separating neighbors
                diff_vectors = position[i:i+1] - position[separating_neighbors]
                # Weight by inverse distance (closer = stronger repulsion)
                sep_distances = distances[i, separating_neighbors].unsqueeze(1)
                weighted_diff = diff_vectors / (sep_distances + 1e-8)  # avoid division by zero
                separation_force[i] = weighted_diff.mean(dim=0)
        
        # 2. ALIGNMENT: Steer towards average heading of neighbors
        for i in range(num_agents):
            nearby_neighbors = neighbors_mask[i]
            if nearby_neighbors.any():
                # Average velocity of neighbors
                neighbor_velocities = velocity[nearby_neighbors]
                avg_velocity = neighbor_velocities.mean(dim=0)
                # Desired velocity is the average, force is difference from current
                alignment_force[i] = avg_velocity - velocity[i]
        
        # 3. COHESION: Steer towards average position of neighbors
        for i in range(num_agents):
            nearby_neighbors = neighbors_mask[i]
            if nearby_neighbors.any():
                # Average position of neighbors (center of mass)
                neighbor_positions = position[nearby_neighbors]
                center_of_mass = neighbor_positions.mean(dim=0)
                # Desired direction towards center of mass
                cohesion_force[i] = center_of_mass - position[i]
        
        # Normalize forces to unit vectors (direction only)
        separation_force = self._normalize_force(separation_force)
        alignment_force = self._normalize_force(alignment_force)
        cohesion_force = self._normalize_force(cohesion_force)
        
        # Combine forces with learnable weights
        total_steering_force = (
            self.separation_weight * separation_force +
            self.alignment_weight * alignment_force + 
            self.cohesion_weight * cohesion_force
        )
        
        return {
            self.output_variables[0]: total_steering_force  # steering_force
        }
    
    def _initialize_boids(self, state: Dict[str, Any]) -> None:
        """Initialize random positions and velocities for boids."""
        position = get_var(state, self.input_variables["position"])
        bounds = get_var(state, "environment/bounds")
        
        num_agents = position.shape[0]
        device = position.device
        
        # Convert learnable parameters to tensors on correct device
        speed_range = torch.as_tensor(self.speed_range, device=device)
        margin = torch.as_tensor(self.position_margin, device=device)
        bounds_tensor = bounds.to(device)
        
        # Random positions within bounds with margin
        random_positions = torch.rand(num_agents, 2, device=device) * (bounds_tensor - 2*margin) + margin
        
        # Random velocities with speeds in range
        random_angles = torch.rand(num_agents, device=device) * 2 * torch.pi
        speeds = speed_range[0] + torch.rand(num_agents, device=device) * (speed_range[1] - speed_range[0])
        random_velocities = torch.stack([
            speeds * torch.cos(random_angles),
            speeds * torch.sin(random_angles)
        ], dim=1)
        
        # Update state in-place (this directly modifies the simulation state)
        # Use data assignment instead of copy_ to avoid memory sharing issues
        state["agents"]["boids"]["position"].data = random_positions
        state["agents"]["boids"]["velocity"].data = random_velocities
        
        print(f"Initialized {num_agents} boids with random positions and velocities")
    
    def _normalize_force(self, force: torch.Tensor) -> torch.Tensor:
        """Normalize force vectors to unit length, handling zero vectors."""
        magnitude = torch.norm(force, dim=1, keepdim=True)
        # Only normalize non-zero vectors
        normalized = torch.where(magnitude > 1e-8, force / magnitude, force)
        return normalized 