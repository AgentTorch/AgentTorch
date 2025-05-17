from functools import wraps
from typing import List, Any, Type, Dict, Optional

from .executor import Executor
from .dataloader import LoadPopulation
from .llm.behavior import Behavior

class envs:
    @staticmethod
    def create(
        model: Type,
        population: Any,
        parameters: Optional[Dict[str, Any]] = None,
        archetypes: Optional[Dict[str, Any]] = None,
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
