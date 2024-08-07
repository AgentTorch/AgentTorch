
import asyncio
from langchain_openai import ChatOpenAI
import yaml
from agent_torch.core.analyzer.agent_graph import AgentRunner
from agent_torch.core.analyzer.retriever import DocumentRetriever
from agent_torch.core.analyzer.utils import generate_attribute_info_list
from agent_torch.core.executor import Executor
from langchain.chains.query_constructor.base import AttributeInfo


class SimulationAnalysisAgent:
    def __init__(
        self,
        openai_api_key: str,
        simulation: Executor,
        document_retriever: DocumentRetriever,
        model_name: str = "gpt-4o-mini",
        thread_id: int = 7,
        temperature: float = 0,
    ):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.temperature = temperature
        self.document_retriever = document_retriever
        self.simulation_id = 0
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model_name=self.model_name,
            temperature=self.temperature,
        )
        self.simulation = simulation
        self.thread_id = thread_id

    def query(self, query: str):
        return self.agent.query(query=query)

    def speculate(self, query: str):
        pass

    def append_simulation_id_if_not_present(self,metadata_field_info):
        # Check if 'simulation_id' is already present
        if not any(attr.name == "simulation_id" for attr in metadata_field_info):
            # Append 'simulation_id' if not present
            metadata_field_info.append(
                AttributeInfo(
                    name="simulation_id",
                    description="The unique identifier for the simulation",
                    type="integer",
                )
            )

    async def add_new_run(self, metadata: dict, metadata_field_info):
        self.simulation_id += 1

        metadata.update(
            {
                "simulation_id": self.simulation_id,
            }
        )

        self.append_simulation_id_if_not_present(metadata_field_info)
        await self.document_retriever.initialise_vectorstore(
            metadata=metadata, metadata_field_info=metadata_field_info, llm=self.llm
        )
        self.agent = AgentRunner(
            llm=self.llm,
            simulation=self.simulation,
            thread_id=self.thread_id,
            retriever=self.document_retriever,
        )
        self.agent.setup_and_compile()
