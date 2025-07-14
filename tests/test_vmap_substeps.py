#!/usr/bin/env python
"""
Unit tests for vectorized substeps in the predator-prey model.

This module contains unit tests for the vectorized implementations
of observation, policy, and transition functions in the predator-prey model.
"""
import os
import sys
import unittest
import torch
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the modules to test
from agent_torch.models.predator_prey.vmap_substeps.vmap_move import (
    FindNeighborsVmap,
    DecideMovementVmap,
    UpdatePositionsVmap,
)
from agent_torch.models.predator_prey.vmap_substeps.vmap_eat import (
    FindEatableGrassVmap,
    EatGrassVmap,
)
from agent_torch.models.predator_prey.vmap_substeps.vmap_hunt import (
    FindTargetsVmap,
    HuntPreyVmap,
)
from agent_torch.models.predator_prey.vmap_substeps.vmap_grow import GrowGrassVmap


class TestVmapMove(unittest.TestCase):
    """Tests for vectorized movement substep implementation."""

    def setUp(self):
        """Set up test data for movement tests."""
        # Create a simple adjacency matrix for a 3x3 grid
        self.adj_matrix = torch.zeros(9, 9)
        # Connect cells in a grid pattern
        grid_connections = [
            (0, 1),
            (0, 3),
            (1, 0),
            (1, 2),
            (1, 4),
            (2, 1),
            (2, 5),
            (3, 0),
            (3, 4),
            (3, 6),
            (4, 1),
            (4, 3),
            (4, 5),
            (4, 7),
            (5, 2),
            (5, 4),
            (5, 8),
            (6, 3),
            (6, 7),
            (7, 4),
            (7, 6),
            (7, 8),
            (8, 5),
            (8, 7),
        ]
        for i, j in grid_connections:
            self.adj_matrix[i, j] = 1

        # Create test state
        self.state = {
            "environment": {"bounds": torch.tensor([3, 3])},  # 3x3 grid
            "network": {
                "agent_agent": {"predator_prey": {"adjacency_matrix": self.adj_matrix}}
            },
            "agents": {
                "predator": {
                    "coordinates": torch.tensor([[0, 0], [1, 1], [2, 2]]),
                    "energy": torch.tensor([[10.0], [5.0], [0.0]]),
                },
                "prey": {
                    "coordinates": torch.tensor([[0, 1], [1, 0], [2, 1]]),
                    "energy": torch.tensor([[8.0], [6.0], [4.0]]),
                },
            },
        }

        # Initialize test instances
        input_vars = {
            "bounds": "environment/bounds",
            "adj_grid": "network/agent_agent/predator_prey/adjacency_matrix",
            "positions": "agents/predator/coordinates",
        }
        self.find_neighbors = FindNeighborsVmap(
            None, input_vars, ["possible_neighbors"], {"learnable": {}, "fixed": {}}
        )

        input_vars = {
            "positions": "agents/predator/coordinates",
            "energy": "agents/predator/energy",
        }
        self.decide_movement = DecideMovementVmap(
            None, input_vars, ["next_positions"], {"learnable": {}, "fixed": {}}
        )

        input_vars = {
            "prey_pos": "agents/prey/coordinates",
            "prey_energy": "agents/prey/energy",
            "pred_pos": "agents/predator/coordinates",
            "pred_energy": "agents/predator/energy",
            "prey_work": "agents/prey/stride_work",
            "pred_work": "agents/predator/stride_work",
        }
        self.update_positions = UpdatePositionsVmap(
            None,
            input_vars,
            ["prey_pos", "prey_energy", "pred_pos", "pred_energy"],
            {"learnable": {}, "fixed": {}},
        )

        # Add stride_work for movement test
        self.state["agents"]["prey"]["stride_work"] = torch.tensor([1.0])
        self.state["agents"]["predator"]["stride_work"] = torch.tensor([1.0])

    def test_find_neighbors(self):
        """Test the FindNeighbors function."""
        # Run the function
        result = self.find_neighbors(self.state)

        # Check that the result contains the expected key
        self.assertIn("possible_neighbors", result)

        # Check that there are three lists of neighbors (one for each predator)
        self.assertEqual(len(result["possible_neighbors"]), 3)

        # First predator is at (0,0), should have neighbors at (0,1) and (1,0)
        neighbors0 = result["possible_neighbors"][0]
        self.assertEqual(len(neighbors0), 2)
        self.assertTrue(
            torch.equal(neighbors0[0], torch.tensor([0, 1]))
            or torch.equal(neighbors0[0], torch.tensor([1, 0]))
        )

        # Third predator has no energy, should still get neighbors
        neighbors2 = result["possible_neighbors"][2]
        self.assertGreater(len(neighbors2), 0)

    def test_decide_movement(self):
        """Test the DecideMovement function."""
        # Create mock observation
        observations = {
            "possible_neighbors": [
                torch.tensor([[0, 1], [1, 0]]),  # Neighbors for predator 0
                torch.tensor(
                    [[0, 1], [1, 0], [1, 2], [2, 1]]
                ),  # Neighbors for predator 1
                torch.tensor([[1, 2], [2, 1]]),  # Neighbors for predator 2
            ]
        }

        # Set random seed for reproducible test
        torch.manual_seed(42)

        # Run the function
        result = self.decide_movement(self.state, observations)

        # Check that the result contains the expected key
        self.assertIn("next_positions", result)

        # Check that there are three positions (one for each predator)
        self.assertEqual(result["next_positions"].shape, torch.Size([3, 2]))

        # Predators 0 and 1 should move, predator 2 should stay in place
        self.assertFalse(torch.equal(result["next_positions"][0], torch.tensor([0, 0])))
        self.assertFalse(torch.equal(result["next_positions"][1], torch.tensor([1, 1])))
        self.assertTrue(torch.equal(result["next_positions"][2], torch.tensor([2, 2])))

    def test_update_positions(self):
        """Test the UpdatePositions function."""
        # Create mock action
        action = {
            "prey": {"next_positions": torch.tensor([[0, 2], [1, 1], [2, 0]])},
            "predator": {"next_positions": torch.tensor([[0, 1], [1, 2], [2, 2]])},
        }

        # Run the function
        result = self.update_positions(self.state, action)

        # Check that the result contains the expected keys
        self.assertIn("prey_pos", result)
        self.assertIn("prey_energy", result)
        self.assertIn("pred_pos", result)
        self.assertIn("pred_energy", result)

        # Check that the positions were updated
        self.assertTrue(
            torch.equal(result["prey_pos"], torch.tensor([[0, 2], [1, 1], [2, 0]]))
        )
        self.assertTrue(
            torch.equal(result["pred_pos"], torch.tensor([[0, 1], [1, 2], [2, 2]]))
        )

        # Check that energy was reduced by stride_work
        self.assertTrue(
            torch.equal(result["prey_energy"], torch.tensor([[7.0], [5.0], [3.0]]))
        )
        self.assertTrue(
            torch.equal(result["pred_energy"], torch.tensor([[9.0], [4.0], [0.0]]))
        )


class TestVmapEat(unittest.TestCase):
    """Tests for vectorized eating substep implementation."""

    def setUp(self):
        """Set up test data for eating tests."""
        # Create test state
        self.state = {
            "environment": {"bounds": torch.tensor([3, 3])},  # 3x3 grid
            "agents": {
                "prey": {
                    "coordinates": torch.tensor([[0, 0], [1, 1], [2, 2]]),
                    "energy": torch.tensor([[10.0], [5.0], [2.0]]),
                }
            },
            "objects": {
                "grass": {
                    "coordinates": torch.tensor(
                        [
                            [0, 0],
                            [0, 1],
                            [0, 2],
                            [1, 0],
                            [1, 1],
                            [1, 2],
                            [2, 0],
                            [2, 1],
                            [2, 2],
                        ]
                    ),
                    "growth_stage": torch.tensor(
                        [[1], [0], [1], [0], [1], [0], [1], [0], [1]]
                    ),
                    "growth_countdown": torch.tensor(
                        [[0], [5], [0], [10], [0], [15], [0], [8], [0]]
                    ),
                    "regrowth_time": torch.tensor([20.0]),
                    "nutritional_value": torch.tensor([5.0]),
                }
            },
        }

        # Initialize test instances
        input_vars = {
            "bounds": "environment/bounds",
            "positions": "agents/prey/coordinates",
            "grass_growth": "objects/grass/growth_stage",
        }
        self.find_eatable_grass = FindEatableGrassVmap(
            None,
            input_vars,
            ["eatable_grass_positions"],
            {"learnable": {}, "fixed": {}},
        )

        input_vars = {
            "bounds": "environment/bounds",
            "prey_pos": "agents/prey/coordinates",
            "energy": "agents/prey/energy",
            "nutrition": "objects/grass/nutritional_value",
            "grass_growth": "objects/grass/growth_stage",
            "growth_countdown": "objects/grass/growth_countdown",
            "regrowth_time": "objects/grass/regrowth_time",
        }
        self.eat_grass = EatGrassVmap(
            None,
            input_vars,
            ["energy", "grass_growth", "growth_countdown"],
            {"learnable": {}, "fixed": {}},
        )

    def test_find_eatable_grass(self):
        """Test the FindEatableGrass function."""
        # Run the function
        result = self.find_eatable_grass(self.state, {})

        # Check that the result contains the expected key
        self.assertIn("eatable_grass_positions", result)

        # Check that fully grown grass positions were identified
        eatable_positions = result["eatable_grass_positions"]
        self.assertEqual(len(eatable_positions), 5)  # 5 grass patches are fully grown

        # Check that all returned positions have growth_stage = 1
        for pos in eatable_positions:
            x, y = pos
            node = y + 3 * x  # Convert to index in the flattened grid
            self.assertEqual(
                self.state["objects"]["grass"]["growth_stage"][node].item(), 1
            )

    def test_eat_grass(self):
        """Test the EatGrass function."""
        # Create mock action with eatable grass positions
        action = {
            "prey": {
                "eatable_grass_positions": [
                    torch.tensor([0, 0]),  # Prey 0 is here
                    torch.tensor([1, 1]),  # Prey 1 is here
                    torch.tensor([2, 2]),  # Prey 2 is here
                ]
            }
        }

        # Run the function
        result = self.eat_grass(self.state, action)

        # Check that the result contains the expected keys
        self.assertIn("energy", result)
        self.assertIn("grass_growth", result)
        self.assertIn("growth_countdown", result)

        # Check that prey energy increased by nutrition value
        self.assertTrue(
            torch.equal(result["energy"], torch.tensor([[15.0], [10.0], [7.0]]))
        )

        # Check that grass growth was reset to 0 where prey ate
        self.assertEqual(result["grass_growth"][0].item(), 0)  # (0,0) was eaten
        self.assertEqual(result["grass_growth"][4].item(), 0)  # (1,1) was eaten
        self.assertEqual(result["grass_growth"][8].item(), 0)  # (2,2) was eaten

        # Check that growth countdown was reset for eaten grass
        self.assertEqual(
            result["growth_countdown"][0].item(), 20
        )  # Reset to regrowth_time
        self.assertEqual(result["growth_countdown"][4].item(), 20)
        self.assertEqual(result["growth_countdown"][8].item(), 20)


class TestVmapHunt(unittest.TestCase):
    """Tests for vectorized hunting substep implementation."""

    def setUp(self):
        """Set up test data for hunting tests."""
        # Create test state
        self.state = {
            "agents": {
                "predator": {
                    "coordinates": torch.tensor([[0, 0], [1, 1], [2, 2]]),
                    "energy": torch.tensor([[10.0], [5.0], [2.0]]),
                },
                "prey": {
                    "coordinates": torch.tensor([[0, 0], [1, 1], [0, 1], [2, 1]]),
                    "energy": torch.tensor([[8.0], [6.0], [4.0], [3.0]]),
                    "nutritional_value": torch.tensor([15.0]),
                },
            }
        }

        # Initialize test instances
        input_vars = {
            "prey_pos": "agents/prey/coordinates",
            "pred_pos": "agents/predator/coordinates",
        }
        self.find_targets = FindTargetsVmap(
            None, input_vars, ["target_positions"], {"learnable": {}, "fixed": {}}
        )

        input_vars = {
            "prey_pos": "agents/prey/coordinates",
            "prey_energy": "agents/prey/energy",
            "pred_pos": "agents/predator/coordinates",
            "pred_energy": "agents/predator/energy",
            "nutritional_value": "agents/prey/nutritional_value",
        }
        self.hunt_prey = HuntPreyVmap(
            None,
            input_vars,
            ["prey_energy", "pred_energy"],
            {"learnable": {}, "fixed": {}},
        )

    def test_find_targets(self):
        """Test the FindTargets function."""
        # Run the function
        result = self.find_targets(self.state, {})

        # Check that the result contains the expected key
        self.assertIn("target_positions", result)

        # Check that prey at predator positions were identified
        target_positions = result["target_positions"]

        # Should find 2 targets - at (0,0) and (1,1)
        self.assertEqual(len(target_positions), 2)
        self.assertTrue(
            any(torch.equal(pos, torch.tensor([0, 0])) for pos in target_positions)
        )
        self.assertTrue(
            any(torch.equal(pos, torch.tensor([1, 1])) for pos in target_positions)
        )

    def test_hunt_prey(self):
        """Test the HuntPrey function."""
        # Create mock action with target positions
        action = {
            "predator": {
                "target_positions": [
                    torch.tensor([0, 0]),  # Predator 0 is here, Prey 0 is here
                    torch.tensor([1, 1]),  # Predator 1 is here, Prey 1 is here
                ]
            }
        }

        # Run the function
        result = self.hunt_prey(self.state, action)

        # Check that the result contains the expected keys
        self.assertIn("prey_energy", result)
        self.assertIn("pred_energy", result)

        # Check that prey energy was set to 0 for eaten prey
        self.assertEqual(
            result["prey_energy"][0].item(), 0
        )  # Prey 0 at (0,0) was eaten
        self.assertEqual(
            result["prey_energy"][1].item(), 0
        )  # Prey 1 at (1,1) was eaten
        self.assertEqual(
            result["prey_energy"][2].item(), 4
        )  # Prey 2 at (0,1) was not eaten

        # Check that predator energy increased by prey nutritional value
        self.assertEqual(result["pred_energy"][0].item(), 25)  # 10 + 15
        self.assertEqual(result["pred_energy"][1].item(), 20)  # 5 + 15
        self.assertEqual(result["pred_energy"][2].item(), 2)  # Unchanged


class TestVmapGrow(unittest.TestCase):
    """Tests for vectorized grass growth substep implementation."""

    def setUp(self):
        """Set up test data for growth tests."""
        # Create test state
        self.state = {
            "objects": {
                "grass": {
                    "growth_stage": torch.tensor(
                        [[0], [0], [0], [0], [1], [0], [1], [0], [1]]
                    ),
                    "growth_countdown": torch.tensor(
                        [[5], [3], [1], [0], [0], [10], [0], [7], [0]]
                    ),
                }
            }
        }

        # Initialize test instance
        input_vars = {
            "grass_growth": "objects/grass/growth_stage",
            "growth_countdown": "objects/grass/growth_countdown",
        }
        self.grow_grass = GrowGrassVmap(
            None,
            input_vars,
            ["grass_growth", "growth_countdown"],
            {"learnable": {}, "fixed": {}},
        )

    def test_grow_grass(self):
        """Test the GrowGrass function."""
        # Run the function
        result = self.grow_grass(self.state, {})

        # Check that the result contains the expected keys
        self.assertIn("grass_growth", result)
        self.assertIn("growth_countdown", result)

        # Check that countdown was decremented
        expected_countdown = torch.tensor(
            [[4], [2], [0], [-1], [-1], [9], [-1], [6], [-1]]
        )
        self.assertTrue(torch.equal(result["growth_countdown"], expected_countdown))

        # Check that growth stage is 1 when countdown <= 0
        expected_growth = torch.tensor([[0], [0], [1], [1], [1], [0], [1], [0], [1]])
        self.assertTrue(torch.equal(result["grass_growth"], expected_growth))


if __name__ == "__main__":
    unittest.main()
