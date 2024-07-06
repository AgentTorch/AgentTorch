import glob
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional
import dspy
from agent_torch.core.analyzer.utils import DotDict, load_state_trace
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import asyncio

class DocumentRetriever:
    def __init__(self, region):
        self.model_name = "BAAI/bge-small-en"
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        self.search_kwargs = {"k": 5}
        self.directory = region
        self.hf = HuggingFaceBgeEmbeddings(
            model_name=self.model_name, 
            model_kwargs=self.model_kwargs, 
            encode_kwargs=self.encode_kwargs
        )
        self.docs = self.load_documents()
        self.docs = self.split_documents(self.docs)
        self.vectorstore = asyncio.run(self.create_vectorstore(self.docs))
        self.retriever =  self.vectorstore.as_retriever(search_kwargs=self.search_kwargs)
        state_trace_path = self.directory + '/state_data_dict.pkl'
        self.state_trace = load_state_trace(state_trace_path)
    
    def load_documents(self):
        md_files =  glob.glob(self.directory + '/conversation_history' + '/**/*.md', recursive=True)
        docs = []
        for file in md_files:
            loader = TextLoader(file)
            docs.append(loader.load()[0])
        return docs

    def split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
        return text_splitter.split_documents(documents=docs)

    async def create_vectorstore(self, docs):
        embeddings = self.hf
        vectorstore = await FAISS.afrom_documents(docs, embeddings)
        return vectorstore
    
    def save_vectorstore(self, store_name):
        self.vectorstore.save_local(store_name)
    
    def load_vectorstore(self, store_name):
        return FAISS.load_local(store_name, self.hf, allow_dangerous_deserialization=True)

    def get_documents(self,query,k):
        return self.retriever.get_relevant_documents(query,k=k)

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
