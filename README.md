Tasks to be done include:
1. Reorganize the code into folders.
    - Config
        - config_file.yaml
    - Controller [not touched by the user]
        - initializer.py
        - controller.py
        - registry.py
    - Runners
        - base_runner.py
        - runner_file.py
    - Trainer
        - base_trainer.py
        - trainer_experiment.py
    - Substeps
        - Observation
        - Policy
        - Transition
2. Write base classes
3. Unit Tests.
4. Helper Functions.
5. Config python wrapper.

Pip Release deadline: July 20. Next 4 weeks.
- Release as a pip package
- Refactor code + complete config wrapper + write test cases + release on PyPi