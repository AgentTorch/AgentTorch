import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agent_torch.core.analyzer.utils import DotDict, load_state_trace
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


class DocumentRetriever:
    def __init__(self, region, vectorstore_type, embedding=None):

        if not embedding:
            self.embedding = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            self.embedding = embedding

        self.directory = region.__path__[0]
        self.search_kwargs = {"k": 5}
        self.docs = []
        self.state_trace = {}
        self.vectorstore_type = vectorstore_type
        self.retriever_description = (
            "Collection of LLM Queries and Responses, for Agent Based Simulations."
        )

    async def initialise_vectorstore(self, metadata, metadata_field_info, llm):
        self.extend_state_trace(description=metadata["description"])
        self.docs.extend(self.load_documents(metadata))
        self.vectorstore = await self.create_vectorstore(
            self.docs, self.vectorstore_type
        )
        self.retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=self.vectorstore,
            document_contents=self.retriever_description,
            metadata_field_info=metadata_field_info,
            verbose=True,
        )
        # self.retriever =  self.vectorstore.as_retriever(search_kwargs=self.search_kwargs)

    def extend_state_trace(self, description):
        state_trace_path = self.directory + "/state_data_dict.pkl"
        state_trace, trace_name = load_state_trace(
            state_trace_path, description=description
        )
        self.state_trace[trace_name] = state_trace

    def load_documents(self, metadata):
        md_files = glob.glob(
            self.directory + "/conversation_history" + "/**/*.md", recursive=True
        )
        docs = []
        for file in md_files:
            loader = TextLoader(file)
            docs.append(loader.load()[0])
        split_docs = self.split_documents(docs)
        return self.add_metadata(split_docs, metadata=metadata)

    def split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=30
        )
        return text_splitter.split_documents(documents=docs)

    def add_metadata(self, docs, metadata):
        for doc in docs:
            doc.metadata.update(metadata)

        return docs

    async def create_vectorstore(self, docs, vectorstore):
        embeddings = self.embedding
        vectorstore = await vectorstore.afrom_documents(docs, embeddings)
        return vectorstore

    def save_vectorstore(self, store_name):
        self.vectorstore.save_local(store_name)

    def load_vectorstore(self, store_name):
        return FAISS.load_local(
            store_name, self.embedding, allow_dangerous_deserialization=True
        )

    def get_documents(self, query, k):
        return self.retriever.get_relevant_documents(query, k=k)
