from functools import wraps
from typing import List, Any, Type, Dict, Optional

from .executor import Executor
from .dataloader import LoadPopulation
from .llm.behavior import Behavior
from .distributed_runner import launch_distributed_simulation


class DistributedRunnerWrapper:
    """
    Wrapper that provides the same API as Runner but uses distributed execution.
    This allows existing code to work with minimal changes.
    """
    
    def __init__(self, config, registry, world_size, parameters=None, archetypes=None, population=None):
        self.config = config
        self.registry = registry
        self.world_size = world_size
        self.parameters = parameters
        self.archetypes = archetypes
        self.population = population
        self.state = None
        self._initialized = False
    
    def init(self):
        """Initialize the distributed simulation (no-op, actual init happens in step)."""
        self._initialized = True
        print(f"Distributed runner initialized for {self.world_size} GPUs")
    
    def step(self, num_steps=None):
        """Run distributed simulation steps."""
        if not self._initialized:
            raise RuntimeError("Must call init() before step()")
        
        print(f"Running distributed simulation: {num_steps} steps on {self.world_size} GPUs")
        
        # Launch distributed simulation
        self.state = launch_distributed_simulation(
            self.config, 
            self.registry, 
            world_size=self.world_size, 
            num_steps=num_steps
        )
        
        print("Distributed simulation completed")
        return self.state
    
    def reset(self):
        """Reset the simulation state."""
        # For distributed execution, reset is handled internally
        print("Resetting distributed simulation")
        pass
    
    def forward(self):
        """Run the full simulation as defined in config."""
        num_episodes = self.config["simulation_metadata"]["num_episodes"]
        num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]
        
        for episode in range(num_episodes):
            print(f"Distributed Episode {episode + 1}/{num_episodes}")
            if episode > 0:
                self.reset()
            self.step(num_steps)
    
    def _set_parameters(self, params):
        """Set parameters (handled during distributed execution)."""
        self.parameters = params
        # Parameters will be applied during distributed execution


class envs:
    @staticmethod
    def create(
        model: Type,
        population: Any,
        parameters: Optional[Dict[str, Any]] = None,
        archetypes: Optional[Dict[str, Any]] = None,
        distributed: bool = False,
        world_size: Optional[int] = None,
        distributed_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        try:
            # Check if distributed execution is requested
            if distributed:
                # For distributed execution, we need to handle this differently
                # since launch_distributed_simulation manages the full process
                import torch
                
                # Set up distributed configuration
                if world_size is None:
                    world_size = torch.cuda.device_count()
                
                if world_size <= 1:
                    print("Warning: Only 1 GPU available, falling back to single GPU")
                    distributed = False
                else:
                    print(f"Setting up distributed simulation on {world_size} GPUs")
                    
                    # Create loader to get config
                    loader = LoadPopulation(population)
                    simulation = Executor(model=model, pop_loader=loader)
                    config = simulation.config.copy()
                    
                    # Add distributed configuration
                    if distributed_config:
                        config["distributed"] = distributed_config
                    else:
                        config["distributed"] = {
                            "strategy": "data_parallel",
                            "sync_frequency": 5
                        }
                    
                    # Store model registry for distributed execution
                    # We'll return a special distributed runner wrapper
                    return DistributedRunnerWrapper(
                        config=config,
                        registry=simulation.runner.registry,
                        world_size=world_size,
                        parameters=parameters,
                        archetypes=archetypes,
                        population=population
                    )
            
            # Standard single-GPU execution
            # 1. Create loader and simulation without initialization
            loader = LoadPopulation(population)
            simulation = Executor(model=model, pop_loader=loader)
            runner = simulation.runner

            # 2. Set behaviors BEFORE initialization
            if archetypes is not None:
                runner_dict = runner.registry.helpers
                substep_func_dict = {
                    func_name: func_obj
                    for category in runner_dict.values()
                    for func_name, func_obj in category.items()
                    if callable(func_obj) or isinstance(func_obj, type)
                }

                for substep_name, archetype in archetypes.items():
                    if substep_name in substep_func_dict:
                        substep_class = substep_func_dict[substep_name]
                        if hasattr(substep_class, "set_behavior"):
                            behavior = Behavior(archetype=archetype, region=population)
                            substep_class.set_behavior(behavior)

            # 3. Now initialize runner after behaviors are set
            runner.init()

            # 4. Set parameters if needed
            if parameters is not None:
                runner._set_parameters(parameters)

            return runner
        except Exception as e:
            print(f"Error in envs.create: {str(e)}")
            raise
