from AgentTorch.AgentTorch.LLM.llm_agent import LLMAgent

class GenerateAgents:
    def __init__(self, params, agents_num):
        self.params = params
        self.agents_num = agents_num
        self.agent_list = []
        

    def generate_agents(self):
        for agent_id in self.agents_num:
            agent = LLMAgent(self.params[agent_id]) # params[agent_id] is the config for the agent
            self.agent_list.append(agent)
        return self.agent_list
            