import warnings
warnings.filterwarnings('ignore')

from agent_torch.core.environment import envs

from agent_torch.models import covid
from agent_torch.populations import sample2

from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.backend import LangchainLLM

user_prompt_template = "Your age is {age} {gender},{unemployment_rate} the number of COVID cases is {covid_cases}."

OPENAI_API_KEY = "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"

# Using Langchain to build LLM Agents
agent_profile = "You are a person living in NYC. Given some info about you and your surroundings, decide your willingness to work. Give answer as a single number between 0 and 1, only."
llm_langchain = LangchainLLM(
    openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-3.5-turbo"
)

print("Creating archetype")

# Create an object of the Archetype class
# n_arch is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.
archetype = Archetype(n_arch=7)
isolation_archetype = archetype.llm(llm=llm_langchain, user_prompt=user_prompt_template)

print("Creating Runner")

runner = envs.create(
        model=covid, 
        population=sample2,
        archetypes={'make_isolation_decision': isolation_archetype})

breakpoint()

runner.step(1)

breakpoint()