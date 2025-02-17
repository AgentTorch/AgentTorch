from agent_torch.core.llm.backend import LLMBackend


class MockLLMBackend(LLMBackend):
    def __init__(self):
        super().__init__()
        self.backend = "langchain"
        self.predictor = None

    def initialize_llm(self):
        """Mock the LLM initialization without external dependencies"""
        self.predictor = True  # Just need a non-None value
        return self.predictor

    def prompt(self, prompt_list):
        """Mock the prompt method with simplified logic"""
        if isinstance(prompt_list, list):
            return self.call_langchain_agent(prompt_list)
        return self.langchain_query_and_get_answer(prompt_list)

    def call_langchain_agent(self, prompt_inputs):
        """Mock batch processing"""
        return [self.langchain_query_and_get_answer(prompt) for prompt in prompt_inputs]

    def langchain_query_and_get_answer(self, prompt_input):
        """Mock single query processing"""
        if isinstance(prompt_input, str):
            return "0.5"  # Return string to match the real implementation
        elif isinstance(prompt_input, dict):
            return "0.5"  # Handle dict input case
        return "0.5"

    def inspect_history(self, last_k, file_dir):
        """Keep the same behavior as original"""
        raise NotImplementedError(
            "inspect_history method is not applicable for Langchain backend"
        )
