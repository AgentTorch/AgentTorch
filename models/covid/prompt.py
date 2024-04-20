system_prompt = "There is a novel disease. It spreads through contact. It is more dangerous to older people. People have the option to isolate at home or continue their usual recreational activities outside. Given this scenario, you must estimate the person's actions based on 1) the information you are given, 2) what you know about the general population with these attributes."

prompt_template_var = """Consider a random person with the following attributes:
* age: {age} 
* location: lives in Astoria, New York, a neighborhood with a population of 35000
This week, there are {week_i_num} new cases in the neighborhood, which is {change_text}. Does this person choose to isolate at home? "There isn't enough information to definitely determine" is not an acceptable answer.
Give a "True" or "False" answer. Give one sentence explaining your answer.
"""