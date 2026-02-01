"""
LLM Backend Abstraction for AgentTorch
======================================

Provides abstract base class and implementations for LLM integration.

Supported backends:
- MockLLM: For testing without API calls (see mock_llm.py)
- DspyLLM: DSPy-based LLM integration (requires dspy package)

Usage:
    from agent_torch.core.llm.mock_llm import MockLLM
    llm = MockLLM(low=0.1, high=0.9)
    
    # Or with DSPy:
    from agent_torch.core.llm.backend import DspyLLM
    llm = DspyLLM(openai_api_key="...", qa=MyQA, cot=MyCOT)
"""
import os
import sys
import io
import concurrent.futures
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.
    
    All LLM backends must implement the `prompt()` method which takes
    a list of prompts and returns a list of outputs.
    """
    
    def __init__(self):
        pass

    def initialize_llm(self):
        """Initialize the LLM. Override in subclasses if needed."""
        raise NotImplementedError

    @abstractmethod
    def prompt(self, prompt_list):
        """
        Send prompts to the LLM and get responses.
        
        Args:
            prompt_list: List of prompts. Each can be:
                - str: Simple prompt string
                - dict: {"agent_query": str, "chat_history": list}
                
        Returns:
            List of outputs, one per input prompt.
            Each output should be a dict with at least {"text": str}
        """
        pass

    def inspect_history(self, last_k, file_dir):
        """Inspect LLM call history. Override in subclasses if supported."""
        raise NotImplementedError


class DspyLLM(LLMBackend):
    """
    DSPy-based LLM backend.
    
    Uses DSPy's chain-of-thought reasoning for structured prompting.
    
    Args:
        openai_api_key: OpenAI API key
        qa: Question-answering signature class
        cot: Chain-of-thought module class
        model: Model name (default: "gpt-4o-mini")
    """
    
    def __init__(self, openai_api_key, qa, cot, model="gpt-4o-mini"):
        super().__init__()
        self.qa = qa
        self.cot = cot
        self.backend = "dspy"
        self.openai_api_key = openai_api_key
        self.model = model

    def initialize_llm(self):
        import dspy
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
        return {"text": agent_output}

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
