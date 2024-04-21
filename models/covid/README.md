- Steps to run a simulation

1. Start calibration process: python trainer.py --c config_opt_llm.yaml

2. Change population and mobility network
    - Go to simulator_data (follow ayush notes)
    - Update num_agents variable in config
    - Regenerate disease_stages.csv (see: disease_stage_file.ipynb) to update initial infections

3. Change initial infections
    - Run disease_stages_file.ipynb (gives you a new disease_stages.csv)