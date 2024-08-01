
from agent_torch.core.analyzer.utils import initialize_agents_dict
from agent_torch.core.analyzer.agent_helpers import create_hallucination_grader, create_question_router, create_rag_chain, create_retrieval_grader, create_simulation_config, create_workflow_router, generate_router_state_trace, rewrite_prompt
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph, START
from pprint import pprint
from IPython.display import Image, display
from langchain_core.documents.base import Document

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        retries_limit: number of retries
        messages: list of messages
        datasource: datasource
        status: status
    """

    question: str
    generation: str
    documents: List[str]
    retries_limit: int
    messages: Annotated[List[AnyMessage], add_messages]
    datasource: str
    status: str
    
class AgentRunner:
    def __init__(self,llm,simulation,retriever,thread_id):
        self.simulation_graph = SimulationGraph()
        self.app = None
        self.config = {"configurable": {"thread_id": thread_id,"simulation_object": simulation,"llm": llm,"retriever": retriever}}

    def setup_and_compile(self,visualize=True):
        self.app = self.simulation_graph.setup_graph()
        if visualize:
            self.simulation_graph.visualize_graph()

    def query(self, query: str):
        if not self.app:
            raise ValueError("Workflow not compiled. Call setup_and_compile() first.")

        
        inputs = {"question": query}

        for output in self.app.stream(inputs, self.config, stream_mode="values"):
            # pprint(output)
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint(value["keys"], indent=2, width=80, depth=None)
                pprint("\n---\n")

        # Final generation
        if output is not None:
            return output.get("generation", "No generation in output.")
        else:
            return ("No output generated.")


class SimulationGraph:
    def __init__(self):
        self.workflow = StateGraph(GraphState)

    def setup_graph(self):
        # Define the nodes
        self.workflow.add_node("retrieve_from_conversation_history", retrieve_from_conversation_history)
        self.workflow.add_node("retrieve_from_state_trace", retrieve_from_state_trace)
        self.workflow.add_node("grade_documents", grade_documents)
        self.workflow.add_node("generate",generate)
        self.workflow.add_node("transform_query", transform_query)
        self.workflow.add_node("retry", retry)
        self.workflow.add_node("select_datasource", select_datasource)
        self.workflow.add_node("decide_to_generate", decide_to_generate)
        self.workflow.add_node("run_simulation", run_simulation)
        
        # Define the
        self.workflow.add_edge("transform_query", "select_datasource")
        self.workflow.add_edge("retrieve_from_conversation_history", "grade_documents")
        self.workflow.add_edge("retrieve_from_state_trace", "grade_documents")
        self.workflow.add_edge("grade_documents", "decide_to_generate")
        self.workflow.add_edge("run_simulation", END)
        
        
        self.workflow.add_conditional_edges(
            "select_datasource",
            route_question,
            {
                "retrieve_from_conversation_history": "retrieve_from_conversation_history",
                "retrieve_from_state_trace": "retrieve_from_state_trace",
            },
        )
        self.workflow.add_conditional_edges(
            START,
            route_workflow,
            {
                "continue": "transform_query",
                "run_simulation": "run_simulation",
            },
            
        )
        self.workflow.add_conditional_edges(
            "decide_to_generate",
            route_after_decide_to_generate,
            {
                "not_relevant": "retry",
                "generate": "generate",
            },
        )
        self.workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "retry",
                "useful": END,
                "not useful": "retry",
            },
        )
        self.workflow.add_conditional_edges(
            "retry",
            control_edge,
            {
                "transform_query": "transform_query",
                "end": END,
            },
        )
        # Compile
        self.memory = SqliteSaver.from_conn_string(":memory:")
        # clear_memory(memory, "2")
        self.app = self.workflow.compile(checkpointer=self.memory)
        return self.app

    def visualize_graph(self):
        # Visualize
        try:
            display(Image(self.app.get_graph(xray=True).draw_mermaid_png()))
        except Exception:
            # This requires some extra dependencies and is optional
            pass


def generate(state,config):
    """
    Generate Answer.

    Args:
        state (dict): Current state of the graph.

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("Generating answer...")
    question = state["question"]
    documents = state["documents"]

    # RAG
    rag_chain = create_rag_chain(config['configurable']['llm'])
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def retrieve_from_state_trace(state,config):
    """
    Retrieve relevant documents.

    Args:
        state (dict): Current state of the graph.

    Returns:
        state (dict): New key added to state, documents, that contains relevant documents
    """

    print("Retrieving documents...")
    question = state["question"]
    print(question)
    documents = []
    agents_dict = initialize_agents_dict(config["configurable"]["retriever"].state_trace, config['configurable']['llm'])
    question_router_state_trace = generate_router_state_trace( config['configurable']['llm'], agents_dict)
    question += "Don't display any plots, just give me the analysis."
    retrievers_list = question_router_state_trace.invoke({"question": question})
    for retriever in retrievers_list.datasource:
        documents.append(str(agents_dict[retriever].chat(question)))
    # documents = Document(page_content=documents)
    pprint(documents)
    # Retrieve
    # documents = pandas_agent.chat(question)
    return {"documents": documents, "question": question}

def retrieve_from_conversation_history(state,config):
    """
    Retrieve relevant documents.

    Args:
        state (dict): Current state of the graph.

    Returns:
        state (dict): New key added to state, documents, that contains relevant documents
    """

    print("Retrieving documents...")
    question = state["question"]

    # Retrieve
    documents = config["configurable"]["retriever"].retriever.invoke(question)
    pprint(documents)
    return {"documents": documents, "question": question}

def grade_documents(state,config):
    """
    Grade Retrieved Documents.
    Determines if the retrieved documents are relevant to the question.

    Args:
        state (dict): Current state of the graph.

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    message = ""
    retrieval_grader = create_retrieval_grader(config["configurable"]["llm"])
    try:
        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": documents}
            )
            grade = score.binary_score
            grade = "yes"
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
            reasoning = score.reasoning
            message += f""" 
                Reasoning: {reasoning} \n
            """
    except Exception as e:
        print(e)
        filtered_docs = documents
    message = AIMessage(content=message)
    return {"documents": filtered_docs, "question": question, "messages": [message]}

def retry(state):
    """
    Retry the process.

    Args:
        state (dict): Current state of the graph.

    Returns:
        state (dict): Updates retries_limit key with the number of retries
    """

    print("---RETRY---")
    retries = state["retries_limit"]
    if retries is None:
        retries = 0
    retries += 1
    message = f"""
            Retrying the process... Number of retries: {retries}
            """
    message = AIMessage(content=message)
    return {"retries_limit": retries, "messages": [message]}

def control_edge(state):
    """
    Checks if the number of retries has reached the limit.
    
    Args:
        state (dict): Current state of the graph.
    
    Returns:
        str: Next node to call, it points to the either END node or Transform Query node
    """
    if state["retries_limit"] > 3:
        print("---DECISION: RE-TRIES LIMIT REACHED---")
        return "end"
    else:
        print("---DECISION: CONTINUE---")
        return "transform_query"
    
    
def transform_query(state,config):
    """
    Transform Query to produce a better question

    Args:
        state (dict): Current state of the graph.

    Returns:
        state (dict): Updates question key with the better question
    """

    print("Transforming query...")
    question = state["question"]
    documents = state["documents"]
    # Re-write
    question_rewriter = rewrite_prompt(config['configurable']['llm'])
    better_question = question_rewriter.invoke({"question": question})
    message = f"""
            Original Question was: {question}
            Transformed Question is: {better_question}
            """
    message = AIMessage(content=message)
    return {"documents": documents, "question": better_question, "messages": [message]}

def route_question(state):
    """
    Route question to the most relevant datasource.

    Args:
        state (dict): Current state of the graph.

    Returns:
        str: Next node to call, it points to the datasource to use
    """
    print("---ROUTE QUESTION---")
    datasource = state["datasource"]
    if datasource == "retrieve_from_conversation_history":
        print("---ROUTE QUESTION TO CONVERSATION HISTORY---")
        return "retrieve_from_conversation_history"
    elif datasource == "retrieve_from_state_trace":
        print("---ROUTE QUESTION TO STATE TRACE---")
        return "retrieve_from_state_trace"

def select_datasource(state,config):
    """
    Route question to the most relevant datasource.

    Args:
        state (dict): Current state of the graph.

    Returns:
        state (dict): Updates datasource key with the datasource to use. Updates messages key with reasoning for the selection.
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    question_router = create_question_router(config['configurable']['llm'])
    source = question_router.invoke({"question": question})
    datasource = source.datasource
    reasoning = source.reasoning
    message = f""" Reason for selecting {datasource} datasource: {reasoning} """
    message = AIMessage(content=message)
    return {"datasource": datasource, "messages": [message]}


def decide_to_generate(state):
    """
    Decide whether to generate an answer or re-generate a question.

    Args:
        state (dict): Current state of the graph.

    Returns:
        state (dict): Updates status key with the decision. Updates messages key with reasoning for the decision.
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        message = "The Documents are not relevant to the question. Transform the query so that we have better results."
        return {"status":"not_relevant", "messages": [message]}
    else:
        # We have relevant documents, so generate answer
        message = "We have relevant documents, so generate answer"
        print("---DECISION: GENERATE---")
        return {"status":"generate", "messages": [message]}

def route_after_decide_to_generate(state):
    """
    Route after deciding to generate an answer.

    Args:
        state (dict): Current state of the graph.

    Returns:
        str: Next node to call, it points to the datasource to use
    """
    print("---ROUTE AFTER DECIDE TO GENERATE---")
    status = state["status"]
    if status == "generate":
        print("---ROUTE TO GENERATE---")
        return "generate"
    else:
        print("---ROUTE TO TRANSFORM QUERY---")
        return "transform_query"

def grade_generation_v_documents_and_question(state,config):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    hallucination_grader = create_hallucination_grader(config['configurable']['llm'])
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def run_simulation(state,config):
    """
    Run the simulation using the provided state.

    Parameters:
    state (dict): The state of the simulation.

    Returns:
    dict: The state of the simulation after running.
    """

    question = state["question"]
    simulation_config = create_simulation_config(config['configurable']['llm'])
    config_options = config['configurable']['simulation_object'].config
    response = simulation_config.invoke({"question": question, "config_options": config_options})
    for item in response.response:
        for key, value in item.items():
            config['configurable']['simulation_object'].data_loader.set_config_attribute(key, value)
    
    try:
        print("Running simulation...")
        # pop_loader = config['configurable']['simulation_object'].pop_loader
        # model = config['configurable']['simulation_object'].model
        # simulation = Executor(model=model, pop_loader=pop_loader)
        # simulation.init()
        config["configurable"]["simulation_object"].init()
        print("Simulation ran successfully.")
        return {"status": "success", "message": "Simulation ran successfully."}
    except Exception as e:
        print(f"Error running simulation: {e}")
        return {"status": "error", "message": str(e)}


def route_workflow(state,config):
    """
    Route the workflow based on the decision.

    Args:
        state (dict): Current state of the graph.

    Returns:
        str: Next node to call, it points to the either run_simulation node or Transform Query node
    """
    print("---ROUTE WORKFLOW---")
    question = state["question"]
    workflow_router = create_workflow_router(config['configurable']['llm'])
    decision = workflow_router.invoke({"question": question})
    if decision.workflow == "run_simulation":
        print("---ROUTE TO RUN SIMULATION---")
        return "run_simulation"
    else :
        print("---CONTINUE WORKFLOW---")
        return "continue"