from functools import wraps
from typing import List, Any, Type, Dict, Optional

from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
from agent_torch.core.llm.behavior import Behavior

class envs:
    @staticmethod
    def create(
        model: Type,
        population: Any,
        parameters: Optional[Dict[str, Any]] = None,
        archetypes: Optional[Dict[str, Any]] = None
    ) -> Any:
        try:
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
                        if hasattr(substep_class, 'set_behavior'):
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

# class envs:
#     @staticmethod
#     def create(
#         model: Type,
#         population: Any,
#         parameters: Optional[Dict[str, Any]] = None,
#         archetypes: Optional[Dict[str, Any]] = None
#     ) -> Any:
#         try:
#             if archetypes is not None:
#                 runner_dict = runner.registry.helpers

#                 substep_func_dict = {
#                     func_name: func_obj
#                     for category in runner_dict.values()
#                     for func_name, func_obj in category.items()
#                     if callable(func_obj) or isinstance(func_obj, type)
#                 }

#                 for substep_name, archetype in archetypes.items():
#                     print(f"Substep Name: {substep_name}, Archetype: {archetype}")
#                     if substep_name in substep_func_dict:
#                         substep_class = substep_func_dict[substep_name]
#                         if hasattr(substep_class, 'set_behavior'):
#                             behavior = Behavior(archetype=archetype, region=population)
#                             substep_class.set_behavior(behavior)
#                             print(f"Class behavior after setting: {substep_class._class_behavior}")
#                         else:
#                             print(f"Warning: {substep_name} does not have set_behavior method")
#                     else:
#                         print(f"Warning: Could not find {substep_name} in substep_func_dict")

#             loader = LoadPopulation(population)
#             simulation = Executor(model=model, pop_loader=loader)
#             runner = simulation.runner

#             runner.init()

#             if parameters is not None:
#                 runner._set_parameters(parameters)

#             return runner
#         except Exception as e:
#             print(f"Error in envs.create: {str(e)}")
#             raise

'''
class envs:
    @staticmethod
    def create(model, population, parameters = None, archetypes = None):

        loader = LoadPopulation(population)
        simulation = Executor(model=model, pop_loader=loader)
        runner = simulation.runner

        runner.init()

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
'''
