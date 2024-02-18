from abc import ABC
from typing import Any
from langchain_openai import ChatOpenAI
import torch
import torch.nn as nn
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser

class LLMAgent():
    def __init__(self,config,profile=None,memory_module=None,planning_module=None, search_provider = None):
        self.config = config
        #search provider like Duckduckgo, wikipedia, etc
        # self.search = search_provider
        # #llm agent initialisation
        # self.llm = ChatOpenAI(openai_api_key=self.config['openai_api_key'])

        # #set the intial characteristic of the agent
        # self.profile = profile
        # self.memory_module = memory_module
        # self.planning_module = planning_module
        
    def prompt(self, input=None,memory_module=None,planning_module=None):
        # TODO: Implement prediction logic
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.profile),
            ("user", "{input}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        # chain = LLMChain(  
        #             llm=self.llm,  
        #             prompt=prompt 
        #         )
        prompt_out = chain.invoke({"input": input})
        return prompt_out
    
    def __call__(self, *arg: Any, **kwargs: Any):
        if len(arg) > 0: kwargs = {**arg[0], **kwargs}
        return torch.rand()
        # return self.prompt(**kwargs)
        
    
# # memory 
# #planning
# #initialise LLM model and save its refernece
# #initialise memory and planning modules and save their references
# #observation prompt for observation step
# #action prompt for action step
# #transition prompt for transition step
# #one memory module for each agent -> current state memory, past state memory(variable), 
# #planning module will remain same
# #Reflection step can be different for each agent as in at different time steps
# #Reflection step can be made to be the action step for the agent
# #one initialisation step for each agent, with different or same initialisation prompts
# #Step by step
# #1. Create LLM model
# #2. Create memory module and planning module
# #3. Create agent
# #4. Create environment

# #Action
# Reflect on the past memories and current state
# Use memory to calculate values of interest

# #Observation
# Observe the current state

# #Transition
# Use memory to transition to the next state



