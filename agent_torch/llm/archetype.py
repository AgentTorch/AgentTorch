import sys
import os
os.environ['DSP_CACHEBOOL'] = 'False'
from langchain.memory import ConversationBufferMemory

class Archetype():
    def __init__(self, llm, user_prompt, num_agents):
        self.llm = LLMArchetype(llm)
        self.rule_based = RuleBasedArchetype()

class LLMArchetype():
    def __init__(self, llm, user_prompt, num_agents):
        self.num_agents = num_agents
        self.llm = llm
        self.predictor = self.llm.initialize_llm()
        self.backend = llm.backend
        self.user_prompt = user_prompt
        self.agent_memory = [ConversationBufferMemory(memory_key="chat_history", return_messages=True) for _ in range(num_agents)]

    def __call__(self, prompt_list, last_k=12):
        last_k = 2 * last_k + 8
        
        prompt_inputs = self.preprocess_prompts(prompt_list, last_k)
        agent_outputs = self.llm.prompt(prompt_inputs)

        # Save conversation history
        for id, (prompt_input, agent_output) in enumerate(zip(prompt_inputs, agent_outputs)):
            self.save_memory(prompt_input, agent_output, agent_id=id)

        return agent_outputs

    def preprocess_prompts(self, prompt_list, last_k):
        prompt_inputs = [{'agent_query': prompt, 'chat_history': self.get_memory(last_k, agent_id=agent_id)['chat_history']} for agent_id, prompt in enumerate(prompt_list)]
        return prompt_inputs
    
    def clear_memory(self, agent_id=0):
        self.agent_memory[agent_id].clear()

    def get_memory(self, last_k=None, agent_id=0):
        if last_k is not None:
            last_k_memory = {'chat_history': self.agent_memory[agent_id].load_memory_variables({})['chat_history'][-last_k:]}
            return last_k_memory
        else:
            return self.agent_memory[agent_id].load_memory_variables({})

    def reflect(self, reflection_prompt, agent_id, last_k=3):
        last_k = 2 * last_k  # get last 6 messages for each AI and Human
        return self.__call__(prompt_list=[reflection_prompt], last_k=last_k)

    def save_memory(self, context_in, context_out, agent_id):
        if self.backend == 'dspy':
            self.agent_memory[agent_id].save_context({"input": context_in['agent_query']}, {"output": context_out})
        
        elif self.backend == 'langchain':
            self.agent_memory[agent_id].save_context({"input": context_in['agent_query']}, {"output": context_out['text']})

    def export_memory_to_file(self, file_dir,last_k):
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        for id in range(len(self.agent_memory)):
            file_name = f"output_mem_{id}.md"
            file_path = os.path.join(file_dir, file_name)
            memory = self.get_memory(agent_id=id)
            with open(file_path, 'w') as f:
                f.write(str(memory))
        
        if self.backend == 'dspy':
            self.llm.inspect_history(file_dir=file_dir, last_k=last_k)

