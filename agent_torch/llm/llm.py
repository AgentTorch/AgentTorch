import os
import sys
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import dspy
import concurrent.futures
import io
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage

class BasicQAEcon(dspy.Signature):
    """
    You are an individual living during the COVID-19 pandemic. You need to decide your willingness to work each month and portion of your assests you are willing to spend to meet your consumption demands, based on the current situation of NYC.
    """
    history = dspy.InputField(desc="may contain your decision in the previous months",format = list)
    question = dspy.InputField(desc="will contain the number of COVID cases in NYC, your age and other information about the economy and your identity, to help you decide your willingness to work and consumption demands")
    answer = dspy.OutputField(desc="will contain single float value, between 0 and 1, representing realistic probability of your willingness to work. No other information should be there.")

class BasicQACovid(dspy.Signature):
    """Consider a random person with the following attributes:
    * age: {age}
    * location: {location}

    There is a novel disease. It spreads through contact. It is more dangerous to older people.
    People have the option to isolate at home or continue their usual recreational activities outside.
    Given this scenario, you must estimate the person's actions based on
        1) the information you are given,
        2) what you know about the general population with these attributes.

    "There isn't enough information" and "It is unclear" are not acceptable answers.
    Give a "Yes" or "No" answer, followed by a period. Give one sentence explaining your choice.
    """
    history = dspy.InputField(desc="may contain your decision in the previous months",format = list)
    question = dspy.InputField(desc="will contain the number of weeks since a disease started (if specified), the number of new cases this week, the percentage change from the past month's average, and asks if the person chooses to isolate at home. It may have other information also.")
    answer = dspy.OutputField(desc="Give a 'Yes' or 'No' answer, followed by a period. No other information should be there in the answer")

class Reflect(dspy.Signature):
    """
    You are an Economic Analyst.
    """
    history = dspy.InputField(desc="may contain data on previous months",format = list)
    question = dspy.InputField(desc="may contain the question you are being asked")
    answer = dspy.OutputField(desc="may contain your analysis of the question asked based on the data in the history")

class COT(dspy.Module):
    def __init__(self, qa):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(qa)
    
    def forward(self, question,history):
        prediction = self.generate_answer(question=question,history=history)
        return dspy.Prediction(answer=prediction)

class COTReflect(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(Reflect)
    
    def forward(self, question,history):
        prediction = self.generate_answer(question=question,history=history)
        return dspy.Prediction(answer=prediction)
    
    
class LLMInitializer:
    def __init__(self, backend, openai_api_key, model='gpt-3.5-turbo', qa = None, cot = None,  agent_profile=None):
        self.backend = backend
        if self.backend == 'dspy':
            assert qa is not None and cot is not None, "qa and cot are required for dspy backend"
        
        self.qa = qa
        self.cot = cot
        
        self.model = model
        self.openai_api_key = openai_api_key
        self.agent_profile = agent_profile

    def initialize_llm(self):
        if self.backend == 'dspy':
            self.llm = dspy.OpenAI(model=self.model, api_key=self.openai_api_key, temperature=0.0)
            dspy.settings.configure(lm=self.llm)
            self.predictor = self.cot(self.qa)
            # self.reflect_predictor = COTReflect()
        
        elif self.backend == 'langchain':
            assert self.agent_profile is not None, "Agent profile is required for Langchain backend"
            self.llm = ChatOpenAI(model=self.model, openai_api_key=self.openai_api_key, temperature=1)
            self.prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(self.agent_profile),
                HumanMessagePromptTemplate.from_template("{user_prompt}"),
            ])
            self.predictor = LLMChain(llm=self.llm, prompt=self.prompt, verbose=False)
        
        return self.predictor
    
    def prompt(self,prompt_list):

        if self.backend == 'dspy':
            agent_outputs = self.call_dspy_agent(prompt_list)
        elif self.backend == 'langchain':
            agent_outputs = self.call_langchain_agent(prompt_list)

        return agent_outputs
    
    def call_langchain_agent(self, prompt_inputs):
        agent_outputs = []
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                agent_outputs = list(executor.map(self.langchain_query_and_get_answer, prompt_inputs))
        except Exception as e:
            print(e)
        return agent_outputs

    def langchain_query_and_get_answer(self, prompt_input):
        agent_output = self.predictor.apply(prompt_input)
        return agent_output.answer
    
    def call_dspy_agent(self, prompt_inputs):
        agent_outputs = []
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                agent_outputs = list(executor.map(self.dspy_query_and_get_answer, prompt_inputs))
        except Exception as e:
            print(e)
        return agent_outputs
    
    def inspect_history(self, last_k,file_dir):
        if self.backend == 'dspy':
            buffer = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = buffer
            self.llm.inspect_history(last_k)
            printed_data = buffer.getvalue()
            if file_dir is not None:
                save_path = os.path.join(file_dir, "inspect_history.md")
                with open(save_path, 'w') as f:
                    f.write(printed_data)
            sys.stdout = original_stdout
        elif self.backend == 'langchain':
            raise NotImplementedError("inspect_history method is not applicable for Langchain backend")

    def dspy_query_and_get_answer(self, prompt_input):
        if type(prompt_input) is str:
            agent_output = self.query_agent(prompt_input, [])
        else:
            agent_output = self.query_agent(prompt_input['agent_query'], prompt_input['chat_history'])
        return agent_output.answer
    
    def query_agent(self, query, history):
        pred = self.predictor(question=query, history=history)
        return pred.answer
