import dspy
from langchain.agents import AgentExecutor,create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_openai import ChatOpenAI
from pandasai.llm import OpenAI

from agent_torch.core.analyzer.retriever import DSPythonicRMClient, DocumentRetriever
from agent_torch.core.analyzer.tools import PandasAIClarificationTool, PandasAITool
from agent_torch.core.analyzer.utils import get_pandas_agent, load_state_trace

class SimulationAnalysisAgent:
    def __init__(
        self,
        openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0,
        document_retriever=None # this retriever is responsible for getting context from simulation memory
    ):
        self.llm_pandas = OpenAI(
            api_token=openai_api_key, model_name=model_name, temperature=temperature
        )
        self.pandas_agent = get_pandas_agent(
            agent_prop_df_list=self.state_trace, llm=self.llm_pandas,
            
        )
        
        self.langchain_llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)
        self.prompt = hub.pull("hwchase17/react")
        self.setup_tools(document_retriever, self.prompt)

    def setup_tools(self, document_retriever, prompt):
        self.retriever_tool = self.get_state_trace_retriever(document_retriever)
        self.tools = [PandasAITool(metadata={'pandas_agent': self.pandas_agent}),PandasAIClarificationTool(metadata={'pandas_agent': self.pandas_agent}), PandasAIExplanationTool(metadata={'pandas_agent': self.pandas_agent}),self.retriever_tool]
        self.llm_agent = create_react_agent(self.langchain_llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.llm_agent, tools=self.tools, verbose=True
        )

    def get_state_trace_retriever(self, document_retriever):
        return create_retriever_tool(
            document_retriever.retriever,
            "simulation_memory_retriever",
            "You must always use this tool to retrieve relevant context for each query.",
        )

    def query(self, query):
        return self.agent_executor.invoke({"input": query})
    
if __name__ == "__main__":
    conversations_memory_directory = ROOT_DIR + 'populations/NYC/simulation_memory_output/2020'

    conversations_memory_retriever = DocumentRetriever(directory=conversations_memory_directory)
    analyzer = SimulationAnalysisAgent(openai_api_key=OPENAI_API_KEY, document_retriever=conversations_memory_retriever, temperature=0)
    analyzer.query("Which age group has lowest median income, how much is it?")