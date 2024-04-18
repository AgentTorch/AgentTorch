MACRO_ECON_PATH  = '/u/ayushc/projects/GradABM/MacroEcon/models'

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
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.callbacks import get_openai_callback
import time

class LLMAgent():
    def __init__(self,agent_profile = None, memory = None,llm = None,openai_api_key = None) -> None:
        assert agent_profile is not None, "Agent profile is required"

        if llm is None:
            llm = self.llm = ChatOpenAI(
                model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0
            )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(agent_profile),
                # MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{user_prompt}"),
            ]
        )

        if memory is not None:
            self.agent_memory = memory
        else:
            self.agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.llm_chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
            verbose=False,
            memory=self.agent_memory,
        )

    async def __call__(self,prompt_list):
        # memory = self.get_memory()
        agent_output = await self.llm_chain.aapply(prompt_list)
        # self.save_memory(prompt_list,agent_output)
        return agent_output

    def get_memory(self):
        return self.agent_memory.load_memory_variables({})

    def reflect(self,reflection_prompt):
        return self.llm_chain.predict(agent_query=reflection_prompt)

    def save_memory(self,context_in,context_out):
        for query, response in zip(context_in, context_out):
            self.agent_memory.save_context({"input": query["user_prompt"]}, {"output": response['text']})


if __name__ == "__main__":
    
    import sys
    sys.path.append(MACRO_ECON_PATH)
    from macro_economics.prompt import complete_final_report_prompt
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
    agent = LLMAgent(agent_profile = agent_profile,open_api_key=OPENAI_API_KEY)
    print(agent(query))
