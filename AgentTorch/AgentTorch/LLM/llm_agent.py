MACRO_ECON_PATH  = '/Users/shashankkumar/Documents/GitHub/MacroEcon/models'

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
class LLMAgent():
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
        # self.llm_chain = [LLMChain(
        #     llm=llm,
        #     prompt=self.prompt,
        #     verbose=False,
        #     memory=self.agent_memory[id],
        # ) for id in range(num_agents)]
        self.llm_chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
            verbose=False,
        )
    
    async def __call__(self,prompt_list,last_k = 12,agent_id = 0):
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

    def format_financial_report(config):
        month = config.get("month")
        year = config.get("year")
        profession = config.get("profession")
        income = config.get("income")
        consumption = config.get("consumption")
        tax_deduction = config.get("tax_deduction")
        tax_credit = config.get("tax_credit")
        tax_brackets = config.get("tax_brackets")
        tax_rates = config.get("tax_rates")
        essential_price = config.get("essential_price")
        savings_balance = config.get("savings_balance")
        interest_rate = config.get("interest_rate")
        report = complete_final_report_prompt.format(
            month=month,
            year=year,
            profession=profession,
            income=income,
            consumption=consumption,
            tax_deduction=tax_deduction,
            tax_credit=tax_credit,
            tax_brackets=tax_brackets,
            tax_rates=tax_rates,
            essential_price=essential_price,
            savings_balance=savings_balance,
            interest_rate=interest_rate,
        )
        return report
    
    agent_profile = """
                    You’re Adam Mills, 
                    a 40-year-old individual living in New York City, New York. As with all Americans, 
                    a portion of your monthly income is taxed by the federal government. This tax-ation system is 
                    tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: 
                    after collection, the government evenly redistributes the tax revenue back to all citizens, 
                    irrespective of their earnings. 
                    """
    
    config = {
    "name": "Adam Mills",
    "age": 40,
    "city": "New York City",
    "state": "New York",
    "month": "2001.01",
    "year": 2001,
    "profession": "Professional Athlete",
    "income": 84144.58,
    "consumption": 49825.69,
    "tax_deduction": 28216.98,
    "tax_credit": 6351.29,
    "tax_brackets": [0.00, 808.33, 3289.58, 7016.67, 13393.75, 17008.33, 42525.00],
    "tax_rates": [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
    "essential_price": 135.82,
    "savings_balance": 12456.42,
    "interest_rate": 3.00
    }   
    query = format_financial_report(config)
    # define open AI key
    agent = LLMAgent(agent_profile = agent_profile,openai_api_key=OPENAI_API_KEY,num_agents=2)
    results = asyncio.run(agent(prompt_list=[query,query]))
    print(results)