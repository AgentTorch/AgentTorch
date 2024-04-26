MACRO_ECON_PATH  = '/Users/shashankkumar/Documents/GitHub/MacroEcon/models'
OPENAI_API_KEY = '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'
import functools
import json
import os
from langchain_openai import ChatOpenAI
import torch
import torch.nn as nn
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
import dspy
import concurrent.futures
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
class BasicQA(dspy.Signature):
    """
    You are an individual living in New York City (NYC) during the COVID-19 pandemic. You need to decide your willingness to work each month and portion of your assests you are willing to spend to meet your consumption demands, based on the current situation of NYC.
    """
    history = dspy.InputField(desc="may contain your decision in the previous months",format = list)
    question = dspy.InputField(desc="will contain the number of COVID cases in NYC, your age and other information about the economy and your identity, to help you decide your willingness to work and consumption demands")
    answer = dspy.OutputField(desc="will contain a list only two int values between 0 and 1 representing realistic probability of your willingness to work and consumption demands. No reasoning or any other information is required")
class COT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(BasicQA)
    
    def forward(self, question,history):
        prediction = self.generate_answer(question=question,history=history)
        return dspy.Prediction(answer=prediction)

def is_json_string(answer:str):
    try:
        json.loads(answer)
    except:
        return False
    return True

class LLMAgent():
    def __init__(self,agent_profile=None, memory = None,llm = None,openai_api_key = None,num_agents = 1) -> None:
        
        
        self.llm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=openai_api_key,temperature=0.0)
        dspy.settings.configure(lm=self.llm)
        # cot_with_assertions = assert_transform_module(COT(), 
        #                                         functools.partial(backtrack_handler, max_backtracks=1))
        self.predictor = COT()
        
        if memory is not None:
            self.agent_memory = memory
        else:
            self.agent_memory = [ConversationBufferMemory(memory_key="chat_history", return_messages=True) for _ in range(num_agents)]
    
    def __call__(self,prompt_list,last_k = 12):
        last_k = 2*last_k + 8 # get last 24 messages 12 for each AI and Human and 8 for reflection prompts
        prompt_inputs = [{'agent_query': prompt, 'chat_history': self.get_memory(last_k,agent_id=agent_id)['chat_history']} for agent_id,prompt in enumerate(prompt_list)]
        # print(prompt_inputs)
        agent_outputs = []
        try:
            # for prompt_input in prompt_inputs:
            #     agent_output = self.query_agent(prompt_input['agent_query'], prompt_input['chat_history'])
            #     agent_outputs.append(agent_output.answer)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                agent_outputs = list(executor.map(self.query_and_get_answer, prompt_inputs))
        except Exception as e:
            print(e)
        
        for id,(prompt_input,agent_output) in enumerate(zip(prompt_inputs,agent_outputs)):
            self.save_memory(prompt_input,agent_output,agent_id=id)
        # print(self.llm.inspect_history(len(prompt_list)))
        return agent_outputs
    
    def query_and_get_answer(self,prompt_input):
        agent_output = self.query_agent(prompt_input['agent_query'], prompt_input['chat_history'])
        return agent_output.answer
    
    def query_agent(self,query,history):
        pred = self.predictor(question=query,history=history)
        return pred.answer

    def clear_memory(self,agent_id = 0):
        self.agent_memory[agent_id].clear()
    
    def get_memory(self,last_k = None,agent_id = 0):
        if last_k is not None:
            last_k_memory = {'chat_history': self.agent_memory[agent_id].load_memory_variables({})['chat_history'][-last_k:]}
            return last_k_memory
        else:
            return self.agent_memory[agent_id].load_memory_variables({})

    def reflect(self,reflection_prompt,last_k = 3,agent_id = 0):
        last_k = 2*last_k #get last 6 messages for each AI and Human
        memory = self.get_memory(last_k=last_k,agent_id=agent_id)
        return self.__call__(prompt_list=[reflection_prompt],last_k=last_k,agent_id=agent_id)
    
    def save_memory(self,context_in,context_out,agent_id = 0):
        self.agent_memory[agent_id].save_context({"input":context_in['agent_query']}, {"output":context_out})
    
    def export_memory_to_file(self, file_dir):
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
            
        for id in range(len(self.agent_memory)):
            file_name = "output_mem"+"_"+str(id)+".md"
            file_path = os.path.join(file_dir, file_name)
            memory = self.get_memory(agent_id=id)
            with open(file_path, 'w') as f:
                f.write(str(memory))
        
class LLMAgentLangchain():
    def __init__(self,agent_profile = None, memory = None,llm = None,openai_api_key = None,num_agents = 1) -> None:
        assert agent_profile is not None, "Agent profile is required"
        
        if llm is None:
            llm = self.llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=openai_api_key, temperature=0)
        
        self.prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(
                            content=agent_profile
                        ),  
                        MessagesPlaceholder(
                            variable_name="chat_history"
                        ),  
                        HumanMessagePromptTemplate.from_template(
                            "{agent_query}"
                        ), 
                    ]
                )
        
        if memory is not None:
            self.agent_memory = memory
        else:
            self.agent_memory = [ConversationBufferMemory(memory_key="chat_history", return_messages=True) for _ in range(num_agents)]

        self.llm_chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
            verbose=False,
        )
    
    async def __call__(self,prompt_list,last_k = 12):
        last_k = 2*last_k + 8 # get last 24 messages 12 for each AI and Human and 8 for reflection prompts
        prompt_inputs = [{'agent_query': prompt, 'chat_history': self.get_memory(last_k,agent_id=agent_id)['chat_history']} for agent_id,prompt in enumerate(prompt_list)]
        agent_outputs = await self.llm_chain.aapply(prompt_inputs)
        for id,(prompt_input,agent_output) in enumerate(zip(prompt_inputs,agent_outputs)):
            self.save_memory(prompt_input,agent_output,agent_id=id)
        return agent_outputs
    
    def clear_memory(self,agent_id = 0):
        self.agent_memory[agent_id].clear()
    
    def get_memory(self,last_k = None,agent_id = 0):
        if last_k is not None:
            last_k_memory = {'chat_history': self.agent_memory[agent_id].load_memory_variables({})['chat_history'][-last_k:]}
            return last_k_memory
        else:
            return self.agent_memory[agent_id].load_memory_variables({})

    async def reflect(self,reflection_prompt,last_k = 3,agent_id = 0):
        last_k = 2*last_k #get last 6 messages for each AI and Human
        memory = self.get_memory(last_k=last_k,agent_id=agent_id)
        return await self.__call__(prompt_list=[reflection_prompt],last_k=last_k,agent_id=agent_id)
    
    def save_memory(self,context_in,context_out,agent_id = 0):
        self.agent_memory[agent_id].save_context({"input":context_in['agent_query']}, {"output":context_out['text']})
    
    def export_memory_to_file(self, file_dir):
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
            
        for id in range(len(self.agent_memory)):
            file_name = "output_mem"+"_"+str(id)+".md"
            file_path = os.path.join(file_dir, file_name)
            memory = self.get_memory(agent_id=id)
            with open(file_path, 'w') as f:
                f.write(str(memory))
        


if __name__ == "__main__":
    
    import sys
    sys.path.append(MACRO_ECON_PATH)
    from macro_economics.prompt import complete_final_report_prompt
    import asyncio
    # define open AI key
    query1 = "Your age is 61, current covid cases is 5000"
    query2 = "Your age is 97, current covid cases is 5000"
    # define open AI key
    agent = LLMAgent(openai_api_key=OPENAI_API_KEY,num_agents=2)
    # results = asyncio.run(agent(prompt_list=[query,query]))
    results = agent(prompt_list=[query1,query2])
    results = agent(prompt_list=[query1,query2])
    results = agent(prompt_list=[query1,query2])
    print(agent.get_memory())
    print(results)