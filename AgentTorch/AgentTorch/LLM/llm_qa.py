import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain_community.document_loaders import TextLoader
import os
import glob
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from typing import List, Union, Optional
import dspy
from dsp.utils import deduplicate
from pandasai import Agent
import openai
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.llms import OpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain import hub

class SimulationAnalysisAgent:
    def __init__(
        self,
        openai_api_key,
        state_trace_path,
        model_name="gpt-4",
        temperature=0,
        document_retriever=None,
        prompt=None,
    ):
        self.openai_chat_model = OpenAI(
            api_key=openai_api_key, model_name=model_name, temperature=temperature
        )
        self.state_trace = load_state_trace(state_trace_path)
        self.pandas_agent = get_pandas_agent(
            agent_prop_df_list=self.state_trace, llm=self.openai_chat_model
        )
        self.setup_tools(document_retriever, prompt)

    @tool
    def run_analysis_on_simulation_state(self, query: str) -> str:
        """
        Run the analysis on the simulation state.

        Use this tool to generate Pandas code for data retrieval.

        :param query: The query to run
        :return: The data analysis result
        """
        return self.pandas_agent.chat(query)

    def setup_tools(self, document_retriever, prompt):
        self.retriever_tool = create_retriever_tool(
            document_retriever.retriever,
            "simulation_memory_retriever",
            "You must use this tool to retrieve context for each user query!",
        )
        self.tools = [self.run_analysis_on_simulation_state, self.retriever_tool]
        self.llm_agent = create_tool_calling_agent(
            self.openai_chat_model, self.tools, prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.llm_agent, tools=self.tools, verbose=True
        )

    def run(self, query):
        return self.agent_executor.run(query)

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_state_trace(sim_data_path = '/Users/shashankkumar/Documents/GitHub/MacroEcon/state_data_dict.pkl'):
    with open(sim_data_path, 'rb') as handle:
        sim_data_dict = pickle.load(handle)
    agent_prop_df_list = []
    # Loop through each episode in the simulation data dictionary
    for episode in sim_data_dict.keys():
        agent_prop_dict = sim_data_dict[episode]['agents']

        # Loop through each step and agent properties in the episode
        for step, agent_prop in agent_prop_dict.items():
            processed_data = {'consumers': {}}

            # Extract consumer data from the agent properties
            for key, value in agent_prop['consumers'].items():
                value = value.flatten().squeeze()
                processed_data['consumers'][key] = value.numpy()

            # Limit the 'assets' column to the first 541516 entries
            processed_data['consumers']['assets'] = processed_data['consumers']['assets'][:541516]

            # Create a DataFrame from the processed consumer data
            consumer_df = pd.DataFrame(processed_data['consumers'])

            # Explode (flatten) the nested columns
            consumer_df = consumer_df.explode(['assets', 'consumption_propensity', 'monthly_income', 'post_tax_income', 'will_work'])

            # Ensure consistent data types for the DataFrame columns
            consumer_df = consumer_df.astype({'ID': int, 'age': float, 'area': float, 'assets': float, 'consumption_propensity': float,
                                                'ethnicity': float, 'gender': float, 'monthly_income': float, 'post_tax_income': float,
                                                'will_work': float, 'household_id': int})

            # Remove unnecessary columns
            consumer_df = consumer_df.drop(['work_propensity', 'monthly_consumption'], axis=1)

            # Add month and year columns based on the current step and episode
            consumer_df['month'] = step
            consumer_df['year'] = episode

            # Mapping for categorical variables
            mapping = {
                "age": ["20t29", "30t39", "40t49", "50t64", "65A", "U19"],
                "gender": ["male", "female"],
                "ethnicity": ["hispanic", "asian", "black", "white", "other", "native"],
                "county": ["BK", "BX", "MN", "QN", "SI"]
            }

            # Reverse the mapping for replacement
            reverse_mapping = {col: {i: val for i, val in enumerate(vals)} for col, vals in mapping.items()}

            # Replace numerical values with categorical labels in the DataFrame
            consumer_df.replace(reverse_mapping, inplace=True)

            # Append the current consumer DataFrame to the list
            agent_prop_df_list.append(consumer_df)

    return agent_prop_df_list

def get_pandas_agent(agent_prop_df_list, llm):
    # Create and return the PandasAI Agent instance
    return Agent(agent_prop_df_list, config={"llm": llm})

class DocumentRetriever:
    def __init__(self, model_name, model_kwargs, encode_kwargs, directory,search_kwargs):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.directory = directory
        self.hf = HuggingFaceBgeEmbeddings(
            model_name=self.model_name, 
            model_kwargs=self.model_kwargs, 
            encode_kwargs=self.encode_kwargs
        )
        self.docs = self.load_documents()
        self.docs = self.split_documents(self.docs)
        self.vectorstore = self.create_vectorstore(self.docs)
        self.retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def load_documents(self):
        md_files =  glob.glob(self.directory + '/**/*.md', recursive=True)
        docs = []
        for file in md_files:
            loader = TextLoader(file)
            docs.append(loader.load()[0])
        return docs

    def split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        return text_splitter.split_documents(documents=docs)

    def create_vectorstore(self, docs):
        embeddings = self.hf
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    
    def save_vectorstore(self, store_name):
        self.vectorstore.save_local(store_name)
    
    def load_vectorstore(self, store_name):
        return FAISS.load_local(store_name, self.hf, allow_dangerous_deserialization=True)

    def get_documents(self,query,k):
        return self.retriever.get_relevant_documents(query,k=k)

class QueryRunnerUsingLangChain:
    def __init__(self, retriever, openai_api_key):
        self.qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key = openai_api_key), chain_type="stuff", retriever=retriever)

    def run_query(self, query):
        result = self.qa.invoke(query)
        print(result)

    def __call__(self):
        query_in = input("Type in your query: \n")
        while query_in != "exit":
            self.run_query(query_in)
            query_in = input("Type in your query: \n")
    
    def __call__(self, query):
        self.run_query(query)

class DSPythonicRMClient(dspy.Retrieve):
    def __init__(self, model_name,model_kwargs, encode_kwargs, directory,search_kwargs):
        super().__init__(k=search_kwargs['k'])
        self.retriever = DocumentRetriever(model_name, model_kwargs, encode_kwargs, directory,search_kwargs=search_kwargs)
    
    def forward(self, query_or_queries:str, k:Optional[int]) -> dspy.Prediction:
        documents = self.retriever.get_documents(query_or_queries, k=k)
        # Convert each document to a DotDict
        passages = [DotDict(long_text=doc.page_content) for doc in documents]
        # print(psg.long_text for psg in passages)
        return passages
        # List of top k passages
        # return dspy.Prediction(
        #     passages=passages
        # )

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
    
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()
    
class QueryRunnerUsingDspy(dspy.Module):
    def __init__(self, passages_per_hop=5, max_hops=4):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
    
if __name__ == "__main__":
    OPENAI_API_KEY = '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'
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    directory = "/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/census_populations/NYC/simulation_memory_output"
    search_kwargs = {"k": 5}
    rm = DSPythonicRMClient(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs,directory=directory,search_kwargs=search_kwargs)
    turbo = dspy.OpenAI(model='gpt-3.5-turbo', api_key=OPENAI_API_KEY)
    dspy.settings.configure(lm=turbo, rm=rm)
    # Ask any question you like to this simple RAG program.
    my_question = "How many age groups are there in the population?"

    # Get the prediction. This contains `pred.context` and `pred.answer`.
    query_runner = QueryRunnerUsingDspy()  # uncompiled (i.e., zero-shot) program
    pred = query_runner(my_question)

    # Print the contexts and the answer.
    print(f"Question: {my_question}")
    print(f"Predicted Answer: {pred.answer}")
    print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")