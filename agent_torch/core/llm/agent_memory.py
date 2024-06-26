import abc
import os


class MemoryHandler(abc.ABC):
    """Abstract base class for handling memory operations."""

    @abc.abstractmethod
    def save_memory(self, context_in, context_out, agent_id):
        """Save conversation context to memory."""
        pass

    @abc.abstractmethod
    def get_memory(self, last_k, agent_id):
        """Retrieve memory for the specified agent and last_k messages."""
        pass

    @abc.abstractmethod
    def clear_memory(self, agent_id):
        """Clear memory for the specified agent."""
        pass

    @abc.abstractmethod
    def export_memory_to_file(self, file_dir, last_k):
        """Export memory to a file."""
        pass


class DSPYMemoryHandler(MemoryHandler):
    """Concrete implementation of MemoryHandler for dspy backend."""

    def __init__(self, agent_memory, llm):
        self.agent_memory = agent_memory
        self.llm = llm

    def save_memory(self, query, output, agent_id):
        self.agent_memory[agent_id].save_context(
            {"input": query["agent_query"]}, {"output": output}
        )

    def get_memory(self, last_k, agent_id):
        last_k_memory = {
            "chat_history": self.agent_memory[agent_id].load_memory_variables({})[
                "chat_history"
            ][-last_k:]
        }
        return last_k_memory

    def clear_memory(self, agent_id):
        self.agent_memory[agent_id].clear()

    def export_memory_to_file(self, file_dir, last_k):
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        for id in range(len(self.agent_memory)):
            file_name = f"output_mem_{id}.md"
            file_path = os.path.join(file_dir, file_name)
            memory = self.get_memory(agent_id=id, last_k=last_k)
            with open(file_path, "w") as f:
                f.write(str(memory))
        self.llm.inspect_history(file_dir=file_dir, last_k=last_k)


class LangchainMemoryHandler(MemoryHandler):
    """Concrete implementation of MemoryHandler for langchain backend."""

    def __init__(self, agent_memory):
        self.agent_memory = agent_memory

    def save_memory(self, query, output, agent_id):
        self.agent_memory[agent_id].save_context(
            {"input": query["agent_query"]}, {"output": output["text"]}
        )

    def get_memory(self, last_k, agent_id):
        last_k_memory = {
            "chat_history": self.agent_memory[agent_id].load_memory_variables({})[
                "chat_history"
            ][-last_k:]
        }
        return last_k_memory

    def clear_memory(self, agent_id):
        self.agent_memory[agent_id].clear()

    def export_memory_to_file(self, file_dir, last_k):
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        for id in range(len(self.agent_memory)):
            file_name = f"output_mem_{id}.md"
            file_path = os.path.join(file_dir, file_name)
            memory = self.get_memory(agent_id=id, last_k=last_k)
            with open(file_path, "w") as f:
                f.write(str(memory))
