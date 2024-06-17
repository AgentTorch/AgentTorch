from agent_torch.llm.archetype import LLMArchetype
from agent_torch.llm.behavior import Behavior
from agent_torch.llm.llm import COT, BasicQAEcon, LLMInitializer


def test_behavior_dspy_NYC():
    from populations import NYC
    
    user_prompt = "Your age is {age} {gender},{unemployment_rate} the number of COVID cases is {covid_cases}."
    kwargs = {
        'month': 'January',
        'year': '2020',
        'covid_cases': 1200,
        'device': 'cpu',
        'current_memory_dir': None,
        'unemployment_rate': 0.05,
    }
    
    llm = LLMInitializer(backend='dspy', qa=BasicQAEcon, cot=COT, openai_api_key=OPENAI_API_KEY)
    llm.initialize_llm()
    
    # Num agents is based on the unique combinations of the prompt variables which correspond to the population attributes
    archetype = LLMArchetype(llm = llm, user_prompt = user_prompt,num_agents=12) 
    earning_behavior = Behavior(archetype=archetype,region=NYC)
    output = earning_behavior.sample(kwargs)
    assert output is not None
    assert len(output) == 12
    assert output[0] is not None
    
def test_behavior_langchain_NYC():
    from populations import NYC
    
    user_prompt = "Your age is {age} {gender},{unemployment_rate} the number of COVID cases is {covid_cases}."
    kwargs = {
        'month': 'January',
        'year': '2020',
        'covid_cases': 1200,
        'device': 'cpu',
        'current_memory_dir': None,
        'unemployment_rate': 0.05,
    }
    
    llm = LLMInitializer(backend='langchain',openai_api_key=OPENAI_API_KEY)
    llm.initialize_llm()
    
    # Num agents is based on the unique combinations of the prompt variables which correspond to the population attributes
    archetype = LLMArchetype(llm = llm, user_prompt = user_prompt,num_agents=12) 
    earning_behavior = Behavior(archetype=archetype,region=NYC)
    output = earning_behavior.sample(kwargs)
    assert output is not None
    assert len(output) == 12
    assert output[0] is not None