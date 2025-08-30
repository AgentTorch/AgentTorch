import os
import sys
import re
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import dspy
import concurrent.futures
import io
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dotenv import load_dotenv
from anthropic import Anthropic


class LLMBackend(ABC):
    def __init__(self):
        pass

    def initialize_llm(self):
        raise NotImplementedError

    @abstractmethod
    def prompt(self, prompt_list):
        pass

    def inspect_history(self, last_k, file_dir):
        raise NotImplementedError
    
    def _batch_query(self, func, prompts):
        """
        Extract thread-pool dispatch logic to base class.
        
        Args:
            func: Function to apply to each prompt
            prompts: List of prompts to process
            
        Returns:
            List of results from processing each prompt
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(func, prompts))
    
    def _unpack_prompt(self, prompt):
        """
        Extract prompt-format handling logic to base class.
        
        Args:
            prompt: Prompt in string or dict format
            
        Returns:
            Tuple of (agent_query, chat_history)
        """
        if isinstance(prompt, dict):
            return prompt.get("agent_query", str(prompt)), prompt.get("chat_history", [])
        else:
            return str(prompt), []


class DspyLLM(LLMBackend):
    def __init__(self, openai_api_key, qa, cot, model="gpt-4o-mini"):
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
        """Process multiple prompts using base class batch query."""
        return self._batch_query(self._process_single_prompt, prompt_list)

    def _process_single_prompt(self, prompt_input):
        """Process a single prompt using base class unpack helper."""
        agent_query, chat_history = self._unpack_prompt(prompt_input)
        agent_output = self.query_agent(agent_query, chat_history)
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


class LangchainLLM(LLMBackend):
    def __init__(
        self,
        openai_api_key,
        agent_profile,
        model="gpt-4o-mini",
    ):
        super().__init__()
        self.backend = "langchain"
        self.llm = ChatOpenAI(model=model, openai_api_key=openai_api_key, temperature=1)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(agent_profile),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{user_prompt}"),
            ]
        )

    def initialize_llm(self):
        self.predictor = LLMChain(
            llm=self.llm, prompt=self.prompt_template, verbose=False
        )
        return self.predictor

    def prompt(self, prompt_list):
        """Process multiple prompts using base class batch query."""
        return self._batch_query(self._process_single_prompt, prompt_list)

    def _process_single_prompt(self, prompt_input):
        """Process a single prompt using base class unpack helper."""
        agent_query, chat_history = self._unpack_prompt(prompt_input)
        agent_output = self.predictor.invoke({
            "user_prompt": agent_query,
            "chat_history": chat_history
        })
        return agent_output["text"]

    def inspect_history(self, last_k, file_dir):
        raise NotImplementedError(
            "inspect_history method is not applicable for Langchain backend"
        )


class ClaudeLLM(LLMBackend):
    """
    Claude backend extending base LLMBackend class.
    Follows the same interface as DspyLLM and LangchainLLM.
    """

    def __init__(self, model_name="claude-3-haiku-20240307", temperature=0.1, max_tokens=1000, system_prompt=None, output_format: Optional[Dict] = None):
        """Initialize Claude backend following base class pattern."""
        super().__init__()

        self.backend = "claude"
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = None
        self.predictor = None

        # Parsing configuration provided by the user (strict)
        self.output_format = output_format

    def initialize_llm(self):
        """Initialize the Claude client following base class pattern."""
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.client = Anthropic(api_key=api_key)
        self.predictor = self
        print(f"Claude backend initialized: {self.model_name}")
        return self.predictor

    def prompt(self, prompt_list: List[str]) -> List[str]:
        if not prompt_list:
            return []
        if self.client is None:
            raise RuntimeError("Claude client not initialized. Call initialize_llm() first.")
        try:
            return self._batch_query(self._process_single_prompt, prompt_list)
        except Exception as e:
            print(f"Error in Claude backend: {e}")
            return [f"Error: {str(e)}" for _ in prompt_list]

    def _process_single_prompt(self, prompt_input) -> str:
        prompt_text, chat_history = self._unpack_prompt(prompt_input)

        message_params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
        }
        if self.system_prompt:
            message_params["system"] = self.system_prompt

        message = self.client.messages.create(**message_params)
        raw_response = message.content[0].text if message.content else ""

        structured_response = self._extract_structured_response(raw_response, self.output_format)
        return {"text": structured_response}

    def inspect_history(self, last_k, file_dir):
        print("⚠️ inspect_history not implemented for Claude backend")
        if file_dir is not None:
            os.makedirs(file_dir, exist_ok=True)
            history_path = os.path.join(file_dir, "claude_history.md")
            with open(history_path, 'w') as f:
                f.write("Claude history inspection not implemented\n")

    def _extract_structured_response(self, raw_response: str, output_format: Optional[Dict] = None) -> str:
        """
        Strict, template-driven parsing. No fallbacks.
        Supported output_format:
          - {"type": "float", "regex": "...", "range": [min,max]}  # regex optional
          - {"type": "json"}                                           # entire response must be JSON
        """
        if not output_format:
            raise ValueError("Claude backend requires output_format; none provided.")

        if isinstance(output_format, str):
            output_format = {"type": output_format}

        response_type = output_format.get("type")
        if not response_type:
            raise ValueError("output_format must include 'type'")

        if response_type == "float":
            text = raw_response.strip()
            pattern = output_format.get("regex")
            if pattern:
                m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if not m:
                    raise ValueError("float extraction failed: regex did not match response")
                group = 1 if m.lastindex else 0
                value_str = m.group(group)
            else:
                value_str = text

            try:
                val = float(value_str)
            except Exception as e:
                raise ValueError(f"float extraction failed: {e}")

            rng = output_format.get("range")
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                min_val, max_val = float(rng[0]), float(rng[1])
                val = max(min_val, min(max_val, val))
            return str(val)

        if response_type == "json":
            import json as _json
            text = raw_response.strip()
            try:
                obj = _json.loads(text)
                return _json.dumps(obj, ensure_ascii=False)
            except Exception as e:
                raise ValueError(f"json extraction failed: {e}")

        raise ValueError(f"Unsupported output_format type: {response_type}")


# Backward compatibility alias
ClaudeLocal = ClaudeLLM
