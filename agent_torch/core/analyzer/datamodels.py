from typing import List
from typing import Literal

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

from typing import Dict
from langchain_core.pydantic_v1 import BaseModel, Field


class RouteQueryStateTrace(BaseModel):
    """Route a user query to one or more relevant datasources."""

    reasoning: str = Field(description="Reasoning for the score")


def create_route_query_state_trace(datasource_options):
    """Factory function to create a RouteQueryStateTrace class with dynamic options."""

    class DynamicRouteQueryStateTrace(RouteQueryStateTrace):
        datasource: List[Literal[tuple(datasource_options.keys())]] = Field(
            ..., description="List of datasources to route the query to"
        )

    return DynamicRouteQueryStateTrace


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal[
        "retrieve_from_conversation_history", "retrieve_from_state_trace"
    ] = Field(
        ...,
        description="""Given a user question choose to route it to retrieve_from_conversation_history retriever or a retrieve_from_state_trace retriever.
                        Simulation retrieve_from_state_trace retriever has dataframes consisting of data about simulation variables.
                        Simulation retrieve_from_conversation_history retriever has reasoning that agents give for their actions during the simulation.""",
    )
    reasoning: str = Field(description="Reasoning for the score")


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    reasoning: str = Field(description="Reasoning for the score")


class ModifySimulationConfig(BaseModel):
    """Modify a simulation configuration based on a user query."""

    response: Dict = Field(..., description="Modified simulation configuration as dict")


class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: List = Field(
        description="Only A dictionary of configuration options and their new values based on the user's query."
    )


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    reasoning: str = Field(description="Reasoning for the score")


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
    reasoning: str = Field(description="Reasoning for the score")


class RouteWorkflow(BaseModel):
    """Route a user query to the most relevant node."""

    workflow: Literal["run_simulation", "continue"] = Field(
        ...,
        description="""Given a user question choose to route it to run_simulation or transform_query.
                        run_simulation is used if the user intent is speculative, it runs the simulation with the given hypothesis.
                        continue transforms the user query into a structured query and returns answer to the query.""",
    )
    reasoning: str = Field(description="Reasoning for the decision")
