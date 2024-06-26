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
    SystemMessagePromptTemplate,
)
from abc import ABC, abstractmethod


class LLM(ABC):
    def __init__(self):
        pass

    def initialize_llm(self):
        raise NotImplementedError

    @abstractmethod
    def prompt(self, prompt_list):
        pass

    def inspect_history(self, last_k, file_dir):
        raise NotImplementedError


class DspyLLM(LLM):
    def __init__(self, openai_api_key, qa, cot, model="gpt-3.5-turbo"):
        super().__init__()
        self.qa = qa
        self.cot = cot
        self.backend = "dspy"
        self.openai_api_key = openai_api_key
        self.model = model

    def initialize_llm(self):
        self.llm = dspy.OpenAI(
            model=self.model, api_key=self.openai_api_key, temperature=0.0
        )
        dspy.settings.configure(lm=self.llm)
        self.predictor = self.cot(self.qa)
        return self.predictor

    def prompt(self, prompt_list):
        agent_outputs = self.call_dspy_agent(prompt_list)
        return agent_outputs

    def call_dspy_agent(self, prompt_inputs):
        agent_outputs = []
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                agent_outputs = list(
                    executor.map(self.dspy_query_and_get_answer, prompt_inputs)
                )
        except Exception as e:
            print(e)
        return agent_outputs

    def dspy_query_and_get_answer(self, prompt_input):
        if type(prompt_input) is str:
            agent_output = self.query_agent(prompt_input, [])
        else:
            agent_output = self.query_agent(
                prompt_input["agent_query"], prompt_input["chat_history"]
            )
        return agent_output.answer

    def query_agent(self, query, history):
        pred = self.predictor(question=query, history=history)
        return pred.answer

    def inspect_history(self, last_k, file_dir):
        buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer
        self.llm.inspect_history(last_k)
        printed_data = buffer.getvalue()
        if file_dir is not None:
            save_path = os.path.join(file_dir, "inspect_history.md")
            with open(save_path, "w") as f:
                f.write(printed_data)
        sys.stdout = original_stdout


class LangchainLLM(LLM):
    def __init__(
        self,
        openai_api_key,
        agent_profile,
        model="gpt-3.5-turbo",
    ):
        super().__init__()
        self.backend = "langchain"
        self.llm = ChatOpenAI(
            model=self.model, openai_api_key=self.openai_api_key, temperature=1
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.agent_profile),
                HumanMessagePromptTemplate.from_template("{user_prompt}"),
            ]
        )

    def initialize_llm(self):
        self.predictor = LLMChain(llm=self.llm, prompt=self.prompt, verbose=False)
        return self.predictor

    def prompt(self, prompt_list):
        agent_outputs = self.call_langchain_agent(prompt_list)
        return agent_outputs

    def call_langchain_agent(self, prompt_inputs):
        agent_outputs = []
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                agent_outputs = list(
                    executor.map(self.langchain_query_and_get_answer, prompt_inputs)
                )
        except Exception as e:
            print(e)
        return agent_outputs

    def langchain_query_and_get_answer(self, prompt_input):
        agent_output = self.predictor.apply(prompt_input)
        return agent_output

    def inspect_history(self, last_k, file_dir):
        raise NotImplementedError(
            "inspect_history method is not applicable for Langchain backend"
        )
