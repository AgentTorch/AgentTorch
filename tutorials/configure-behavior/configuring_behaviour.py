from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior
from agent_torch.core.llm.backend import DspyLLM, LangchainLLM
from agent_torch.populations import NYC
from dspy_modules import COT, BasicQAWillToWork

OPENAI_API_KEY = None
user_prompt_template = "Your age is {age} {gender},{unemployment_rate} the number of COVID cases is {covid_cases}."

kwargs = {
    "month": "January",
    "year": "2020",
    "covid_cases": 1200,
    "device": "cpu",
    "current_memory_dir": "/path-to-save-memory",
    "unemployment_rate": 0.05,
}

# Using Dspy to build LLM Agents
llm_dspy = DspyLLM(qa=BasicQAWillToWork, cot=COT, openai_api_key=OPENAI_API_KEY)

# Using Langchain to build LLM Agents
agent_profile = "You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0 and 1, only."
llm_langchian = LangchainLLM(
    openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-3.5-turbo"
)

llm_dspy.initialize_llm()
output_dspy = llm_dspy.prompt(
    [
        "You are an individual living during the COVID-19 pandemic. You need to decide your willingness to work each month and portion of your assests you are willing to spend to meet your consumption demands, based on the current situation of NYC."
    ]
)

llm_langchian.initialize_llm()
output_langchain = llm_langchian.prompt(
    [
        "You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0 and 100, only."
    ]
)

# Create an object of the Archetype class
# n_arch is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.
archetype = Archetype(n_arch=7)

# Create an object of the Behavior class
# You have options to pass any of the above created llm objects to the behavior class
# Specify the region for which the behavior is to be generated. This should be the name of any of the regions available in the populations folder.
earning_behavior = Behavior(
    archetype=archetype.llm(llm=llm_dspy, user_prompt=user_prompt_template), region=NYC
)

# Sample the behavior for the population, the size of the output will be equal to the population size
output = earning_behavior.sample(kwargs)
print("Output:", output)
