from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
class SearchInput(BaseModel):
    query: str = Field(description="should be a query")
    
class PandasAIExplanationTool(BaseTool):
    name = "get explaination for 'run_analysis_on_simulation_state' tool"
    description = """
        Use this tool when to validate the reasoning of "run_analysis_on_simulation_state" tool.
        Helps to validate the results better.
        If any doubt in the results, use this tool to get explaination.

        :param query: None
        :return: Explanation for the results
        """
    args_schema: Type[BaseModel] = SearchInput

    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # External variable return when tool runs
        return self.metadata['pandas_agent'].explain()
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")

class PandasAIClarificationTool(BaseTool):
    name = "get clarification question for simulation analysis"
    description = """
        Use this tool when "run_analysis_on_simulation_state" does not understand the query or gives error.
        Get clarification questions for the simulation analysis.
        Helps to understand the query better

        :param query: The original query string sent to "run_analysis_on_simulation_state"
        :return: Questions to ask for clarification
        """
    args_schema: Type[BaseModel] = SearchInput

class PandasAITool(BaseTool):
    name = "run_analysis_on_simulation_state"
    description = """
        Run the analysis on the simulation state trace.

        Use this tool for data retrieval and analysis. 
        Data is saved for each episode of simulation.

        :param query: The query string
        :return: The data analysis result
        """
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # External variable return when tool runs
        return self.metadata['pandas_agent'].chat(query)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")

