def debug():
    import os
    import sys
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    package_root_directory = os.path.dirname(os.path.dirname(current_directory))
    sys.path.append(package_root_directory)
    
debug()



from agent_torch.llm.archetype import Archetype, LLMArchetype
from agent_torch.llm.behavior import Behavior
from agent_torch.llm.llm import COT, BasicQAEcon, DspyLLM, LLMInitializer


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
    
if __name__ == "__main__":
    
    from populations import NYC
    from docs.examples.dspy_modules import COT, BasicQAEcon
    
    OPENAI_API_KEY = None
    user_prompt = "Your age is {age} {gender},{unemployment_rate} the number of COVID cases is {covid_cases}."
    
    kwargs = {
        'month': 'January',
        'year': '2020',
        'population_attributes_mapping_path': '/Users/shashankkumar/Documents/GitHub/MacroEcon/populations/NYC/population_attributes_mapping.json',
        'covid_cases': 1200,
        'device': 'cpu',
        'current_memory_dir': '/Users/shashankkumar/Documents/GitHub/MacroEcon/populations/NYC/memory',
        'unemployment_rate': 0.05,
    }
    llm = DspyLLM( qa=BasicQAEcon, cot=COT, openai_api_key=OPENAI_API_KEY)
    
    archetype = Archetype(n_arch=7)
    earning_behavior = Behavior(archetype=archetype.llm(llm=llm,user_prompt=user_prompt,num_agents=12),region=NYC)
    output = earning_behavior.sample(kwargs)
    print("Output:",output)