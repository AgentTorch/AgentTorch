AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'
MODEL_PATH = '/u/ayushc/projects/GradABM/MacroEcon/models'

import torch
import numpy as np
import json
import sys
import re
import pdb

sys.path.append(MODEL_PATH)
sys.path.insert(0, AGENT_TORCH_PATH)

from AgentTorch.helpers import get_by_path
from AgentTorch.substep import SubstepAction
from AgentTorch.LLM.llm_agent import LLMAgent
from prompt import prompt_template_var, system_prompt

class MakeIsolationDecision(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        OPENAI_API_KEY = self.config['simulation_metadata']['OPENAI_API_KEY']
        self.agent = LLMAgent(agent_profile = system_prompt, openai_api_key=OPENAI_API_KEY)
        self.device = self.config['simulation_metadata']['device']
        
        self.num_agents = self.config['simulation_metadata']['num_agents']
                
    async def forward(self, state, observation):
        t = state['current_step']
        input_variables = self.input_variables
        
        print("Substep Action: IsolationDecision")
        
        agent_ages = get_by_path(state, re.split("/", input_variables['age']))
        
        will_isolate = torch.rand(self.num_agents, 1)
        
        return {self.output_variables[0]: will_isolate}