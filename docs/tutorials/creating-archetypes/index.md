# Creating Archetypes

This page shows how a user can build a useful archetype in minutes.

## What is an archetype?

An archetype is a plug‑in decision module that turns agent context into a single numeric decision each step. It has a basic surface: `configure(...)`, `broadcast(...)`, `sample(...)`. It’s backend‑agnostic: you can use rules, extend the skeleton class `MockLLM`, or bring in your own LLM to evaluate decisions.

Setup:
```python
import agent_torch.core.llm.template as lm
from agent_torch.core.llm.archetype import Archetype
```
## Defining your Archetype:

Archetype is created with a prompt, LLM object, and an n_arch parameter which is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.

Example:
```python
llm = MockLLM()
archetype_n_2  = Archetype(prompt="Your age is {age}...", llm=llm, n_arch=2)
archetype_n_12 = Archetype(prompt="Your age is {age}...", llm=llm, n_arch=12)
```

### Step 1: Setting up your LLM

First, let us set up the LLM. You can use any LLM of your choosing, provided you implement a __call__ hook. This is the ony hook that is __REQUIRED__. 

```python
class MyLLM:
    # optional:
    def initialize_llm(self):
        return self 
    def __call__(self, prompt_inputs):
        # prompt_inputs: [{"agent_query": str, "chat_history": [...]}, ...]
        # Return float-convertible outputs; you own parsing/validation
        return ["..." for _ in prompt_inputs]
```

Optional:

- If your LLM exposes `initialize_llm(self)`, Archetype will call it on construction. Use it to configure your environment/API keys if needed.
- Use the __call__ hook when your LLM wrapper is directly callable, or modify the __prompt__ hook for SDKs that expose a named method instead.

Additionally, you can extend the `MockLLM` class to overload these hooks and implement your LLM like so:

```python
from agent_torch.core.llm.mock_llm import MockLLM

class MyLLM(MockLLM):
    # Optional: run any one-time setup (keys, clients, etc.)
    def initialize_llm(self):
        self.client = SomeSDK(api_key=os.getenv("API_KEY"))
        return self

    # Preferred entrypoint: make the wrapper callable
    # prompt_list: list[{"agent_query": str, "chat_history": list}]
    # return: list[str] or list[{"text": str}] of same length
    def __call__(self, prompt_list):
        outputs = []
        for item in prompt_list:
            query = item.get("agent_query", "")
            history = item["chat_history"] 
            # Replace this with your SDK call and parsing
            score = self._rng.uniform(self.low, self.high)
            outputs.append({"text": f"{score:.3f}"})
        return outputs

    # If your SDK exposes a named method instead of __call__, implement prompt(...)
    # Archetype will fall back to this if __call__ is missing.
    # def prompt(self, prompt_list):
    #     return self.__call__(prompt_list)
```

I/O contract:

Input: A list of items, one per prompt in the form: {"agent_query": str, "chat_history": list}. Chat history can be [] if you do not wish to use the memory handler.

Output: a list of the same length, one result per input, in the same order.
Each result must be float-convertible (you manage formatting/validation).
Preferred: return strings like "0.12" (or floats like 1.23).
Also accepted: dicts with a "text" key, e.g., {"text": "321.0"}.
Archetype parses to floats and returns a torch tensor.

### Step 2: Creating a prompt for Archetype through a Template object

The prompt field for archetype will accept either a simple string template or Template object. 

String template example:

```python
arch = Archetype(prompt="Your age is {age}, gender is {gender}, and you are deciding whether to isolate or not.", llm=MyLLM(), n_arch=1)
```

However, when you have external data and you want to optimize on prompts, consider extending the Template class. The Template class defines how to turn agent and external context data into a single prompt for the LLM.

Here is an example dataframe:

```python
import pandas as pd

df = pd.DataFrame({
    "Salary": [65000, 43000, 120000],
    "Work_Week_Hours": [60, 40, 80],
    "Satisfaction_Level": ["Not satisfied", "Partially satisfied", "Very satisfied"]
})
```

To create a template class extension that utilizes this dataframe, first create Variable objects to encapsulate them.
Variable objects mark placeholders and allow you to make them learnable for further optimization down the line (see [Optimizing Prompt Variables with P3O](../optimizing-on-prompts/index.md)).

The "desc" field is used for metadata regarding the slot. Set learnable to True when considering prompt optimizations, else set it to False. Additionally, if you are uncertain if data for a particular row/attribute exists, you may use the default field to specify what the default placeholder value should be (OPTIONAL).

Ex.

```python
# lm is an alias for template
salary = lm.Variable(desc="salary") # # by default, sets learnable to False, default param is set to an empty string
work_week = lm.Variable(desc="hours worked in a week", learnable=True)
satisfaction = lm.Variable(desc="categorical satisfaction levels", learnable=False, default="unknown satisfaction")
```
--------------------------------------------------------------------------------------------
//todo: population helpers are not implemented yet.

Note: Variable objects are not limited to exclusively external data; Variable objects may reference population attributes as well, where Template will infer the data source. To view what population attributes exist, load a population like so, and then use the __.view()__ utility:

```python
from agent_torch.populations import astoria
astoria.view() # prints all attributes relevant attributes with sample values
```

You can then register Variable objects to the attributes you determine are relevant.
```python
age = lm.Variable(desc = age) # sets learnable = False, default to ''
```
--------------------------------------------------------------------------------------------

Then when extending template, reference the Variable objects in the __prompt__ hook.

```python
class MyTemplate(lm.Template):

    def __prompt__(self):
        #set prompt_string field
        self.prompt_string = "I work for {work_week}, and get paid {salary}. I am {satisfaction} with my current job."

```

When considering higher level instructions such as a system prompt and output prompt, reference __system_prompt__ and the __output_prompt__ hook. These are purely optional and will default to empty strings if not used. 

```python
#inside MyTemplate
def __system_prompt__(self):
    return "Consider the following variables and determine your willingness to isolate."
def __output__(self):
    return "Answer on a scale from 0 - 1.0."
```

The Template object assembles them in order of: system_prompt -> prompt -> output_prompt for the LLM.
Once this is set up, you can integrate your newly configured Template into Archetype.

Here is an example of it put together:

```python
import agent_torch.core.llm.template as lm

class MyTemplate(lm.Template):
    salary = lm.Variable(desc="salary")
    work_week = lm.Variable(desc="hours worked in a week", learnable=True)
    satisfaction = lm.Variable(desc="categorical satisfaction levels", learnable=False, default="unknown satisfaction")

    def __system_prompt__(self):
        return "Consider the following variables and determine your willingness to isolate."

    def __prompt__(self):
        self.prompt_string = "I work for {work_week} hours, and get paid ${salary}. I am {satisfaction} with my current job."

    def __output__(self):
        return "Answer on a scale from 0 - 1.0."


arch = Archetype(prompt=MyTemplate(), llm=MockLLM(), n_arch=3)

# Example of a filled LLM prompt:
# -------------------------------
'''
"Consider the following variables and determine your willingness to isolate.
I work for 60 hours and get paid $65000. I am Not Satisfied with my current job.
Answer on a scale from 0-1.0."
'''
```

### Step 3: Configuring your data

This step is crucial in ensuring your data is fully compatible and wired to the Archetype object. 
Utilize the .configure method to set your external data to the archetype. If you want extra flexibility for testing, use the split parameter to selectively choose how many rows to feed from.

```python
arch.configure(external_df=df)
arch.configure(external_df=df, split=2)  # Only takes the first 2 rows of the df
```
### Step 4: Using Archetype

Archetype contains sampling and broadcasting functionalities. Use .sample when you want a quick preview of how your external data prompts might look, when you may not necessarily need per-agent outputs yet. Use .broadcast when you want to bind to a population for one value per agent in a given population. 

#### Using .sample() pre-broadcast vs post-broadcast

It is advisable to call .sample before calling .broadcast to determine if prompts are correctly configured for your external data source.

When sampling from Archetype pre-broadcast call (if your data is configured correctly), it will render one prompt per dataframe row and produce a tensor in the shape of (n_rows,). If data is configured incorrectly or .configure() was not called, .sample() will return a shape of (1,). If split=k was called during configure, it will return a tensor in the shape of (k,).

When using .sample(), use `verbose = True` to carefully inspect prompts. Note that when calling sample before binding to a population, population fields (such as age, gender, supplied by the population module) will remain as placeholders as archetype has not been bounded to a population yet. Only after broadcast is called, will it fill in the full prompt.

```python
arch.sample()  # no prompt printing
#OR
arch.configure(df, split = k)
arch.sample(verbose=True)  # prints k prompts
```

After you have used .broadcast(), you can call sample again to receive per-agent outputs in the shape of a (n_agents,) tensor.

#### Using .broadcast()

If you would like to broadcast the prompts to a wider population, use the .broadcast() method.
It expects a population object to broadcast to, a match_on argument to determine the proper key from the external dataframe for each agent, and a group-on argument which determines which batching key to use. By default, if the group-on is not specified, the group_on key will default to match_on. 

First, import your population.

```python
from agent_torch.populations import astoria
```
Then determine what your external data should "match_on" to. If your external data contains n_rows for each n_agents in a population on a per-agent basis, disregard match_on. By default, if a match_on key is not provided, .broadcast() will index on a per-agent basis (one row in external data assigned to one agent). 

However, __if your dataframe does not cover a per-agent basis__, ensure that:

The population chosen has a .pkl file which references the key you wish to map to. Template will fill the external data based on that key for each agent, so ensure your dataframe references the exact key. 

Assume we have created a Job_Number.pkl file in the "astoria" population which contain the numbers 1, 2, 3 and matches them across n_agents.

In your dataframe, add a Job_Number column:

```python
df = pd.DataFrame({
    "Job_Number": [1,2,3], # Job 1 matches to an agent with the attributes: (salary, 65000), (Work_Week_Hours, 60), (Satisfaction_Level, "Not Satisfied")
    "Salary": [65000, 43000, 120000],
    "Work_Week_Hours": [60, 40, 80],
    "Satisfaction_Level": ["Not satisfied", "Partially satisfied", "Very satisfied"]
})
'''
call broadcast on astoria, mapping external data to Job_Number attribute in agent data. Broadcast binds the population and joins external data by Job_Number. 
'''
arch.broadcast(astoria, match_on="Job_Number")
```

Lastly, consider how to batch/group your prompts with an explicit group_on argument. The group_on argument creates buckets with the given keys, so that all agents in the same bucket share one prompt and receive the same score. Use a single key (e.g., "Job_Number") or a list (e.g., ["Job_Number", "Age"]). This reduces duplicate LLM calls, speeds up runs, and keeps decisions consistent for identical contexts.

Examples of broadcast calls:

```python
arch.broadcast(population=astoria, match_on="Job_Number")  # groups based on match_on argument aka Job_Number
arch.broadcast(population=astoria, match_on="Job_Number", group_on = "Age")         # any agent with the same age is put into a bucket
arch.broadcast(population=astoria, match_on="Job_Number", group_on = ["Age", "Job_Number"])  # same age AND job_number share a bucket
```

Call .sample after calling .broadcast to execute on the buckets created. This step is necessary to execute if you want to receive an (n_agents,) decision tensor.
