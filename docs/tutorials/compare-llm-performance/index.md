## Exploring LLM Influence

#### Introduction
AgentTorch is a framework that scales ABM simulations to real-world problems. This tutorial will guide you through the process of experimenting with different LLMs in AgentTorch to understand their impact on the framework's effectiveness.

#### Step 1: Setup
First, let's set up our environment and import the necessary libraries:


```python
import sys
from dspy_modules import COT, BasicQAWillToWork
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior
from agent_torch.populations import NYC
from agent_torch.core.llm.backend import DspyLLM, LangchainLLM

import torch
OPENAI_API_KEY = None
```

Setup : Covid Cases Data and Unemployment Rate


```python
from utils import get_covid_cases_data
csv_path = 'agent_torch/models/covid/data/county_data.csv'
monthly_cases_kings = get_covid_cases_data(csv_path=csv_path,county_name='Kings County')
monthly_cases_queens = get_covid_cases_data(csv_path=csv_path,county_name='Queens County')
monthly_cases_bronx = get_covid_cases_data(csv_path=csv_path,county_name='Bronx County')
monthly_cases_new_york = get_covid_cases_data(csv_path=csv_path,county_name='New York County')
monthly_cases_richmond = get_covid_cases_data(csv_path=csv_path,county_name='Richmond County')

```

#### Step 2: Initialise LLM Instance

We can use either of the Langchain and Dspy backends to initialise a LLM instance. While these are the frameworks we are supporting currently, you may choose to use your own framework of choice by extending the LLMBackend class provided with AgentTorch.

Let's see how we can use Langchain to initialise an LLM instance

GPT 3.5 Turbo


```python
agent_profile = "You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0 and 1, only."
llm_langchain_35 = LangchainLLM(
    openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-3.5-turbo"
)
```

GPT 4-0 Mini


```python
agent_profile = "You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0 and 1, only."
llm_langchain_4o_mini = LangchainLLM(
    openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-4o-mini"
) 
```

Similarly if we wanted to use Dspy backend, we can instantiate the DspyLLM object.
We can pass the desired model name as argument just like we did with Langchain.


```python
# Agent profile is decided by the QA module and the COT module enforces Chain of Thought reasoning
llm_dspy = DspyLLM(qa=BasicQAWillToWork, cot=COT, openai_api_key=OPENAI_API_KEY)
```

#### Step 3: Define agent Behavior

Create an object of the Behavior class
You have options to pass any of the above created llm objects to the behavior class
Specify the region for which the behavior is to be generated. This should be the name of any of the regions available in the populations folder.


```python
# Create an object of the Archetype class
# n_arch is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.
archetype = Archetype(n_arch=2) 

# Define a prompt template
# Age,Gender and other attributes which are part of the population data, will be replaced by the actual values of specified region, during the simulation.
# Other variables like Unemployment Rate and COVID cases should be passed as kwargs to the behavior model.
user_prompt_template = "Your age is {age}, gender is {gender}, ethnicity is {ethnicity}, and the number of COVID cases is {covid_cases}.Current month is {month} and year is {year}."

# Create a behavior model
# You have options to pass any of the above created llm objects to the behavior class
# Specify the region for which the behavior is to be sampled. This should be the name of any of the regions available in the populations folder.
earning_behavior_4o_mini = Behavior(
    archetype=archetype.llm(llm=llm_langchain_4o_mini, user_prompt=user_prompt_template),
    region=NYC
)
earning_behavior_35 = Behavior(
    archetype=archetype.llm(llm=llm_langchain_35, user_prompt=user_prompt_template),
    region=NYC
)
```


```python
# Define arguments to be used for creating a query for the LLM instance
kwargs = {
    "month": "January",
    "year": "2020",
    "covid_cases": 1200,
    "device": "cpu",
    "current_memory_dir": "/populations/astoria/conversation_history",
    "unemployment_rate": 0.05,
}
```

#### Step 4: Compare performance between different LLM models


```python
from utils import get_labor_data, get_labor_force_correlation

labor_force_df_4o_mini, observed_labor_force_4o_mini, correlation_4o_mini = get_labor_force_correlation(
    monthly_cases_kings, 
    earning_behavior_4o_mini, 
    'agent_torch/models/macro_economics/data/unemployment_rate_csvs/Brooklyn-Table.csv',
    kwargs
)
labor_force_df_35, observed_labor_force_35, correlation_35 = get_labor_force_correlation(
    monthly_cases_kings, 
    earning_behavior_35, 
    'agent_torch/models/macro_economics/data/unemployment_rate_csvs/Brooklyn-Table.csv',
    kwargs
)
print(f"Correlation with GPT 3.5 is {correlation_35} and with GPT 4o Mini is {correlation_4o_mini}")
```
