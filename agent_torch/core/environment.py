from functools import wraps
from typing import List, Any, Dict, Optional

from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

from agent_torch.core.llm.behavior import Behavior

class envs:
    @staticmethod
    def create(model, population, parameters = None, archetypes = None):

        loader = LoadPopulation(population)
        simulation = Executor(model=model, pop_loader=loader)
        runner = simulation.runner

        # Set the parameters
        if parameters is not None:
            runner._set_parameters(parameters)

        if archetypes is not None:
            # Assign archetypes to specific substeps
            runner_dict = runner.registry.helpers

            # assign function to archetypes
            substep_func_dict = {
                func_name: func_obj
                for category in runner_dict.values()
                for func_name, func_obj in category.items()
                if callable(func_obj) or isinstance(func_obj, type)
            }

            for substep_name, archetype in archetypes.items():
                    print(f"Substep Name: {substep_name}, Archetype: {archetype}")
                    if substep_name in substep_func_dict and hasattr(substep_func_dict[substep_name], 'set_behavior'):
                        behavior = Behavior(archetype=archetype, region=population)
                        substep_func_dict[substep_name].set_behavior(behavior)
                    else:
                        print(f"Warning: Could not set behavior for {substep_name}")
    
        return runner
