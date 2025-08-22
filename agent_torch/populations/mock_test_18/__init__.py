"""
Mock Test Population - 18 Agents
================================

A small test population with 18 agents for fast P3O testing.
Contains all the same structure as full populations (NYC, astoria) 
but with minimal data size for rapid iteration.

Population Properties:
- Size: 18 agents
- Age range: 18-60 (mapped to age groups 0-8)
- SOC codes: Real O*NET codes from job_data
- Areas: Numeric encoded (0-2)
- Gender/Ethnicity: Numeric encoded
- Disease stages: Compatible with COVID model
- Mobility networks: 2-layer network structure

Usage:
    from agent_torch.populations import mock_test_18
    
    # Use with LoadPopulation
    loader = LoadPopulation(mock_test_18)
    
    # Use with enhanced GPU loader
    gpu_loader = GPUPopulationLoader(mock_test_18)
"""

# This file makes mock_test_18 importable as a module
# The directory path serves as the module's __path__ attribute
import os
__path__ = [os.path.dirname(__file__)] 