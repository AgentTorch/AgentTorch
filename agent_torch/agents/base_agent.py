# agent_torch/agents/base_agent.py

class BaseAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
    
    def step(self):
        """
        This method defines the agent's behavior at each step in the simulation.
        It should be overridden by subclasses.
        """
        pass
