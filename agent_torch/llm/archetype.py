import sys
import os

from agent_torch.llm.agent_memory import DSPYMemoryHandler, LangchainMemoryHandler

os.environ["DSP_CACHEBOOL"] = "False"
from langchain.memory import ConversationBufferMemory


class Archetype:
    def __init__(self, n_arch=1):
        self.n_arch = n_arch

    def llm(self, llm, user_prompt, num_agents):
        try:
            llm.initialize_llm()
            llm_archetype = LLMArchetype(llm, user_prompt, num_agents)
            return llm_archetype
        except Exception as e:
            print(
                " 'initialize_llm' Not Implemented, make sure if it's the intended behaviour"
            )
            llm_archetype = LLMArchetype(llm, user_prompt, num_agents)
            return llm_archetype

    def rule_based(self):
        raise NotImplementedError


class LLMArchetype:
    def __init__(self, llm, user_prompt, num_agents):
        self.num_agents = num_agents
        self.llm = llm
        # self.predictor = self.llm.initialize_llm()
        self.backend = llm.backend
        self.user_prompt = user_prompt
        self.agent_memory = [
            ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            for _ in range(num_agents)
        ]
        if self.backend == "dspy":
            self.memory_handler = DSPYMemoryHandler(
                agent_memory=self.agent_memory, llm=self.llm
            )
        elif self.backend == "langchain":
            self.memory_handler = LangchainMemoryHandler(agent_memory=self.agent_memory)
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

    def __call__(self, prompt_list, last_k):
        last_k = 2 * last_k + 8

        prompt_inputs = self.preprocess_prompts(prompt_list, last_k)
        agent_outputs = self.llm.prompt(prompt_inputs)

        # Save conversation history
        for id, (prompt_input, agent_output) in enumerate(
            zip(prompt_inputs, agent_outputs)
        ):
            self.save_memory(prompt_input, agent_output, agent_id=id)

        return agent_outputs

    def preprocess_prompts(self, prompt_list, last_k):
        prompt_inputs = [
            {
                "agent_query": prompt,
                "chat_history": self.get_memory(last_k, agent_id=agent_id)[
                    "chat_history"
                ],
            }
            for agent_id, prompt in enumerate(prompt_list)
        ]
        return prompt_inputs

    def reflect(self, reflection_prompt, agent_id, last_k=3):
        last_k = 2 * last_k  # get last 6 messages for each AI and Human
        return self.__call__(prompt_list=[reflection_prompt], last_k=last_k)

    def save_memory(self, query, output, agent_id):
        self.memory_handler.save_memory(query, output, agent_id)

    def export_memory_to_file(self, file_dir, last_k):
        self.memory_handler.export_memory_to_file(file_dir, last_k)

    def get_memory(self, last_k, agent_id):
        return self.memory_handler.get_memory(last_k=last_k, agent_id=agent_id)
