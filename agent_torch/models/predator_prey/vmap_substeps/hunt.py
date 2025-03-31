# agent_torch/models/predator_prey/vmap_substeps/hunt.py
import torch

def hunt_prey(pred_pos, prey_pos, pred_energy, prey_energy, nutrition):
    """
    Predators hunt prey using vectorized operations.
    
    Args:
        pred_pos: Positions of all predators
        prey_pos: Positions of all prey
        pred_energy: Energy levels of all predators
        prey_energy: Energy levels of all prey
        nutrition: Energy gained from hunting prey
        
    Returns:
        Tuple of (new_prey_energy, new_pred_energy)
    """
    # Find which prey and predators interact
    hunted_prey = torch.zeros(len(prey_pos), dtype=torch.bool)
    hunter_preds = torch.zeros(len(pred_pos), dtype=torch.bool)
    
    for i, pred_position in enumerate(pred_pos):
        # Use vmap to check all prey at once
        check_prey_matches = torch.vmap(
            lambda prey_p: torch.all(prey_p == pred_position),
            in_dims=0,
            out_dims=0
        )
        prey_matches = check_prey_matches(prey_pos)
        
        if torch.any(prey_matches):
            hunted_prey = hunted_prey | prey_matches
            hunter_preds[i] = True
    
    # Update prey energy using tensor operations (not conditionals)
    prey_energy_flat = prey_energy.squeeze(-1)
    # Zero out energy for hunted prey
    hunted_mask = (~hunted_prey).float()
    new_prey_energy = prey_energy_flat * hunted_mask
    
    # Update predator energy
    pred_energy_flat = pred_energy.squeeze(-1)
    # Add nutrition to hunters
    nutrition_gain = hunter_preds.float() * nutrition
    new_pred_energy = pred_energy_flat + nutrition_gain
    
    return new_prey_energy.view(-1, 1), new_pred_energy.view(-1, 1)