from abc import ABC
from typing import Any
from langchain_openai import ChatOpenAI
import torch
import torch.nn as nn
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser

class LLMAgent():
    def __init__(self,id,llm = None, config = None,profile=None,memory_module=None,planning_module=None, search_provider = None):
        self.config = config
        self.id = id
        #search provider like Duckduckgo, wikipedia, etc
        if search_provider is not None:
            self.search = search_provider
        #llm agent initialisation
        if llm is not None:
            llm = llm
        self.llm = ChatOpenAI(openai_api_key=self.config['openai_api_key'])
        #set the intial characteristic of the agent
        if profile is not None:
            self.profile = profile
        if memory_module is not None:
            self.memory_module = memory_module
        else:
            memory_module = []
        
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
    
    def add_memory(self, memory):
        self.memory_module.append(memory)
    
    def retrieve_memory(self):
        return self.memory_module
    
    def __call__(self, *arg: Any, **kwargs: Any):
        if len(arg) > 0: kwargs = {**arg[0], **kwargs}
        # return torch.rand()
        return self.prompt(**kwargs)
        
