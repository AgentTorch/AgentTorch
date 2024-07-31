
import sys
from typing import Any, Dict
sys.path.append("/Users/shashankkumar/Documents/GitHub/AgentTorchLLM")
from agent_torch.core.analyzer.datamodels import (
    ConversationalResponse,
    GradeAnswer,
    GradeDocuments,
    GradeHallucinations,
    RouteQuery,
    RouteQueryStateTrace,
    RouteWorkflow,
    create_route_query_state_trace,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

def rewrite_prompt(llm) -> Any:
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. \n
        Then rewrite the question keeping in mind that retrieval is being done on information about simulations."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    return re_write_prompt | llm | StrOutputParser()

def create_answer_grader(llm) -> Any:
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    return answer_prompt | structured_llm_grader

def create_hallucination_grader(llm) -> Any:
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
            ),
        ]
    )

    return hallucination_prompt | structured_llm_grader

def create_simulation_config(llm) -> Any:
    """Creates and returns a simulation configuration chain."""
    system = """
    Given a query that speculates how a simulation would change if certain 
    variables are modified, and a dictionary of configuration options for 
    the simulation, return only modified configuration as a dict with 
    key-value pairs.

    Instructions:
    1. Analyze the user's query to identify which simulation parameters 
       they want to modify.
    2. Select relevant configuration options from the provided dictionary 
       that best match the user's intentions.
    3. For each selected option, determine an appropriate new value based 
       on the user's speculation.
    4. Return a dictionary with key-value pairs, where:
       - Keys are the selected configuration options (must be from the 
         provided dictionary)
       - Values are the new float values for these options
    """
    llm_modify_simulation_config = llm.with_structured_output(
        ConversationalResponse
    )
    modify_simulation_config_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "User query is : {question}. Configuration options are: "
                "{config_options}",
            ),
        ]
    )
    return modify_simulation_config_prompt | llm_modify_simulation_config

def create_retrieval_grader(llm) -> Any:
    """Creates and returns a retrieval grader chain."""
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    system = """
    You are a grader assessing relevance of a retrieved document to a user 
    question. 
    If the document contains information that might help answer the user 
    question, grade it as relevant. 
    It does not need to be a stringent test. The goal is to filter out 
    erroneous retrievals. 
    Give a binary score 'yes' or 'no' score to indicate whether the document 
    is relevant to the question.
    """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: "
                "{question}",
            ),
        ]
    )
    return grade_prompt | structured_llm_grader

def create_question_router(llm) -> Any:
    """Creates and returns a question router chain."""
    structured_llm_router = llm.with_structured_output(RouteQuery)
    system = """
    You are an expert at routing a user question to a 
    retrieve_from_conversation_history or retrieve_from_state_trace. 
    Simulation retrieve_from_state_trace retriever has dataframes consisting 
    of data about simulation variables. 
    Simulation retrieve_from_conversation_history retriever has reasoning 
    that agents give for their actions during the simulation.
    """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    return route_prompt | structured_llm_router

def generate_router_state_trace(llm, agents_dict: Dict[str, str]) -> Any:
    """
    Generates and returns a state trace router chain.

    Args:
        agents_dict: A dictionary of agent names and their descriptions.

    Returns:
        A chain for routing state trace queries.
    """
    datasource_options = {
        key: key.replace("_", " ") for key in list(agents_dict.keys())
    }
    description = (
        "Given a user question choose to route it to one or more "
        "of the following retrievers:\n"
        + "\n".join(
            [f"{key}: {value}" for key, value in datasource_options.items()]
        )
    )
    system_prompt = (
        f"You are an expert at routing a user question to one or more of "
        f"the following retrievers:\n"
        + "\n".join(
            [f"{key}: {value}" for key, value in datasource_options.items()]
        )
        + "\nThe retrievers provides numerical data about variables such as "
        "Employment Rate, Monthly Income, Virus Spread, etc. for their "
        "respective simulations.\n"
    )
    structured_llm_router_state_trace = llm.with_structured_output(
        create_route_query_state_trace(datasource_options)
    )
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )
    return route_prompt | structured_llm_router_state_trace

def create_rag_chain(llm):
    prompt = hub.pull("rlm/rag-prompt")
    return  prompt | llm | StrOutputParser()

def create_workflow_router(llm):
    # LLM with function call
    structured_llm_workflow = llm.with_structured_output(RouteWorkflow)
    route_workflow_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at routing a user question to run_simulation or to continue the flow."),
            ("human", "{question}"),
        ]
    )
    return route_workflow_prompt | structured_llm_workflow