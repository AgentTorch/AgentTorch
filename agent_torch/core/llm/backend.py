import os
import sys
import io
import concurrent.futures
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import dspy
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)


class LLM(ABC):
    @abstractmethod
    def initialize_llm(self):
        pass

    @abstractmethod
    def prompt(self, prompt_list):
        pass

    @abstractmethod
    def inspect_history(self, last_k, file_dir):
        pass


class DspyLLM(LLM):
    def __init__(self, openai_api_key, qa, cot, model="gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.qa = qa
        self.cot = cot
        self.model = model
        self.llm = None
        self.predictor = None

    def initialize_llm(self):
        self.llm = dspy.OpenAI(model=self.model, api_key=self.openai_api_key, temperature=0.0)
        dspy.settings.configure(lm=self.llm)
        self.predictor = self.cot(self.qa)
        return self.predictor

    def prompt(self, prompt_list):
        return self.call_dspy_agent(prompt_list)

    def call_dspy_agent(self, prompt_inputs):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.dspy_query_and_get_answer, prompt_inputs))

    def dspy_query_and_get_answer(self, prompt_input):
        if isinstance(prompt_input, str):
            return self.query_agent(prompt_input, []).answer
        return self.query_agent(prompt_input["agent_query"], prompt_input["chat_history"]).answer

    def query_agent(self, query, history):
        return self.predictor(question=query, history=history)

    def inspect_history(self, last_k, file_dir):
        buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer
        try:
            self.llm.inspect_history(last_k)
            printed_data = buffer.getvalue()
            if file_dir:
                save_path = os.path.join(file_dir, "inspect_history.md")
                with open(save_path, "w") as f:
                    f.write(printed_data)
        finally:
            sys.stdout = original_stdout


class LangchainLLM(LLM):
    def __init__(self, openai_api_key, agent_profile, model="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model, openai_api_key=openai_api_key, temperature=1)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(agent_profile),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{user_prompt}"),
            ]
        )
        self.predictor = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=False)

    def initialize_llm(self):
        return self.predictor

    def prompt(self, prompt_list):
        return self.call_langchain_agent(prompt_list)

    def call_langchain_agent(self, prompt_inputs):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.langchain_query_and_get_answer, prompt_inputs))

    def langchain_query_and_get_answer(self, prompt_input):
        if isinstance(prompt_input, str):
            return self.predictor.apply({"user_prompt": prompt_input, "chat_history": []})
        return self.predictor.apply({
            "user_prompt": prompt_input["agent_query"],
            "chat_history": prompt_input["chat_history"],
        })

    def inspect_history(self, last_k, file_dir):
        raise NotImplementedError("inspect_history method is not applicable for Langchain backend")
