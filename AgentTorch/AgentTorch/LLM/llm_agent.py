from abc import ABC
from typing import Any
from langchain_openai import ChatOpenAI
import torch
import torch.nn as nn
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

class LLMAgentOpenAI():
    def __init__(self,template = None,llm = None) -> None:
        if template is None:
            self.template = """You are a chatbot having a conversation with a human.
                            {chat_history}
                            Human: {input}
                            Chatbot:"""
        else:
            self.template = template
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.profile),
            ("user", "{input}")
        ])
        # self.prompt = PromptTemplate(
        #                 input_variables=["chat_history", "human_input"], template=self.template
                    # )
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        if llm is None:
            llm = self.llm = ChatOpenAI(openai_api_key="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")

        
        self.llm_chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory,
        )
    
    def __call__(self, *arg: Any, **kwargs: Any):
        return "I am an LLM"
    
    def invoke(self,input):
        self.llm_chain.predict(input = input)
        
        
class LLMAgent():
    def __init__(self,id,llm = None, config = None,profile=None,memory_module=None,planning_module=None, search_provider = None):
        self.config = config
        self.id = id
        #search provider like Duckduckgo, wikipedia, etc
        # if search_provider is not None:
        #     self.search = search_provider
        # #llm agent initialisation
        # if llm is not None:
        #     self.llm = llm
        # else:
        #     self.llm = ChatOpenAI(openai_api_key=self.config['openai_api_key'])
        # #set the intial characteristic of the agent
        # if profile is not None:
        #     self.profile = profile
        # if memory_module is not None:
        #     self.memory_module = memory_module
        # else:
        #     memory_module = []
        
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
        return "I am an LLM agent"
        # return self.prompt(**kwargs)
        
