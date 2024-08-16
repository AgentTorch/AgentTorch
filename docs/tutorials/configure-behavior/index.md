# Guide to Prompting LLM as ABM Agents

Welcome to this comprehensive tutorial on AI agent behavior generation! This
guide is designed for newcomers who want to learn how to create AI agents and
simulate population behaviors using a custom framework. We'll walk you through
each step, explaining concepts as we go.

## Table of Contents

1. [Introduction to the Framework](#introduction)
2. [Setting Up Your Environment](#setup)
3. [Understanding the Core Components](#components)
4. [Creating Your First AI Agent](#first-agent)
5. [Generating Population Behaviors](#behaviors)
6. [Putting It All Together](#all-together)
7. [Next Steps and Advanced Topics](#next-steps)

<a name="introduction"></a>

## 1. Introduction to the Framework

Our framework is designed to simulate population behaviors using AI agents. It
combines several key components:

- **LLM Agents**: We use Large Language Models (LLMs) to create intelligent
  agents that can make decisions based on given scenarios.
- **Archetypes**: These represent different types of individuals in a
  population.
- **Behaviors**: These simulate how individuals might act in various situations.

This framework is particularly useful for modeling complex social or economic
scenarios, such as population responses during a pandemic.

<a name="setup"></a>

## 2. Setting Up Your Environment

Now, let's set up your OpenAI API key (you'll need an OpenAI account):

```python
OPENAI_API_KEY = None # Replace with your actual API key
```

<a name="components"></a>

## 3. Understanding the Core Components

Let's break down the main components of our framework:

### DspyLLM and LangchainLLM

These are wrappers around language models that allow us to create AI agents.
They can process prompts and generate responses based on given scenarios.

### Archetype

This component helps create different "types" of individuals in our simulated
population. Like Male under 19, Female from 20 to 29 years of age.

### Behavior

The Behavior component simulates how individuals (or groups) might act in
various situations. It uses the AI agents to generate these behaviors.

Now you have two AI agents ready to process prompts!

## 4. Creating LLM Agents

##### We support using Langchain and Dspy backends to initialize LLM instances - for agent and archetypes. Using our LLMBackend class, you can integrate any framework of your choice.

### Using DSPy

```python
from dspy_modules import COT, BasicQAWillToWork
from agent_torch.core.llm.llm import DspyLLM

llm_dspy = DspyLLM(qa=BasicQAWillToWork, cot=COT, openai_api_key=OPENAI_API_KEY)
llm_dspy.initialize_llm()

output_dspy = llm_dspy.prompt(["You are an individual living during the COVID-19 pandemic. You need to decide your willingness to work each month and portion of your assets you are willing to spend to meet your consumption demands, based on the current situation of NYC."])
print("DSPy Output:", output_dspy)
```

### Using Langchain

```python
from agent_torch.core.llm.llm import LangchainLLM

agent_profile = "You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0 and 1, only."

llm_langchian = LangchainLLM(openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-3.5-turbo")
llm_langchian.initialize_llm()

output_langchain = llm_langchian.prompt(["You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0.0 and 1.0, only."])
print("Langchain Output:", output_langchain)
```

<a name="behaviors"></a>

## 5. Generating Population Behaviors

To simulate population behaviors, we'll use the Archetype and Behavior classes:

```python
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior
from agent_torch.populations import NYC

# Create an object of the Archetype class
# n_arch is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.
archetype = Archetype(n_arch=7)

# Define a prompt template
# Age,Gender and other attributes which are part of the population data, will be replaced by the actual values of specified region, during the simulation.
# Other variables like Unemployment Rate and COVID cases should be passed as kwargs to the behavior model.
user_prompt_template = "Your age is {age} {gender}, unemployment rate is {unemployment_rate}, and the number of COVID cases is {covid_cases}.Current month is {month} and year is {year}."

# Create a behavior model
# You have options to pass any of the above created llm objects to the behavior class
# Specify the region for which the behavior is to be sampled. This should be the name of any of the regions available in the populations folder.
earning_behavior = Behavior(
    archetype=archetype.llm(llm=llm_dspy, user_prompt=user_prompt_template, num_agents=12),
    region=NYC
)

print("Behavior model created successfully!")
```

This sets up a behavior model that can simulate how 12 different agents might
behave in NYC during the COVID-19 pandemic.

<a name="all-together"></a>

## 6. Putting It All Together

Now, let's use our behavior model to generate some population behaviors:

```python
# Define scenario parameters
# Names of the parameters should match the placeholders in the user_prompt template
scenario_params = {
    'month': 'January',
    'year': '2020',
    'covid_cases': 1200,
    'device': 'cpu',
    'current_memory_dir': '/path-to-save-memory',
    'unemployment_rate': 0.05,
}

# Generate behaviors
population_behaviors = earning_behavior.sample(scenario_params)
print("Population Behaviors:")
print(population_behaviors)
```

```python
# Define another scenario parameters
scenario_params = {
    'month': 'February',
    'year': '2020',
    'covid_cases': 900,
    'device': 'cpu',
    'current_memory_dir': '/path-to-save-memory',
    'unemployment_rate': 0.1,
}

# Generate behaviors
population_behaviors = earning_behavior.sample(scenario_params)
print("Population Behaviors:")
print(population_behaviors)
```

```python
# Define yet another scenario parameters
scenario_params = {
    'month': 'March',
    'year': '2020',
    'covid_cases': 200,
    'device': 'cpu',
    'current_memory_dir': '/path-to-save-memory',
    'unemployment_rate': 0.11,
}

# Generate behaviors
population_behaviors = earning_behavior.sample(scenario_params)
print("Population Behaviors:")
print(population_behaviors)
```

And so on...

This will output a set of behaviors for our simulated population based on the
given scenario.

<a name="next-steps"></a>

## 7. Next Steps and Advanced Topics

You've just created your first AI agents and simulated population behaviors.
Here are some advanced topics you might want to explore next:

- Customizing archetypes for specific populations
- Creating more complex behavior models
