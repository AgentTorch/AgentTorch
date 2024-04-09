from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
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
# class QueryProcessor:
#     def __init__(self, model_name, model_kwargs, encode_kwargs, directory,openai_api_key):
#         self.model_name = model_name
#         self.model_kwargs = model_kwargs
#         self.encode_kwargs = encode_kwargs
#         self.directory = directory
#         self.hf = HuggingFaceBgeEmbeddings(
#             model_name=self.model_name, 
#             model_kwargs=self.model_kwargs, 
#             encode_kwargs=self.encode_kwargs
#         )
#         docs = self.load_documents()
#         docs = self.split_documents(docs)
#         vectorstore = self.create_vectorstore(docs)
#         self.retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
#         # persisted_vectorstore = self.load_vectorstore(self.hf)
#         self.qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key = openai_api_key), chain_type="stuff", retriever=self.retriever)

#     def load_documents(self):
#         md_files =  glob.glob(self.directory + '/**/*.md', recursive=True)
#         docs = []
#         for file in md_files:
#             loader = TextLoader(file)
#             docs.append(loader.load()[0])
#         return docs

#     def split_documents(self, docs):
#         text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=30, separator="\n")
#         return text_splitter.split_documents(documents=docs)

#     def create_vectorstore(self, docs):
#         embeddings = self.hf
#         vectorstore = FAISS.from_documents(docs, embeddings)
#         return vectorstore
    
#     def save_vectorstore(self, vectorstore, store_name):
#         vectorstore.save_local(store_name)
    
#     def load_vectorstore(self, embeddings, store_name ):
#         return FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)

#     def run_query(self, query):
#         result = self.qa.invoke(query)
#         print(result)

#     def get_documents(self,query):
#         return self.retriever.get_documents(query)
    
#     def __call__(self):
#         query_in = input("Type in your query: \n")
#         while query_in != "exit":
#             self.run_query(query_in)
#             query_in = input("Type in your query: \n")

class DocumentRetriever:
    def __init__(self, model_name, model_kwargs, encode_kwargs, directory):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.directory = directory
        self.hf = HuggingFaceBgeEmbeddings(
            model_name=self.model_name, 
            model_kwargs=self.model_kwargs, 
            encode_kwargs=self.encode_kwargs
        )
        docs = self.load_documents()
        docs = self.split_documents(docs)
        self.vectorstore = self.create_vectorstore(docs)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1})

    def load_documents(self):
        md_files =  glob.glob(self.directory + '/**/*.md', recursive=True)
        docs = []
        for file in md_files:
            loader = TextLoader(file)
            docs.append(loader.load()[0])
        return docs

    def split_documents(self, docs):
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=30, separator="\n")
        return text_splitter.split_documents(documents=docs)

    def create_vectorstore(self, docs):
        embeddings = self.hf
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    
    def save_vectorstore(self, store_name):
        self.vectorstore.save_local(store_name)
    
    def load_vectorstore(self, store_name):
        return FAISS.load_local(store_name, self.hf, allow_dangerous_deserialization=True)

    def get_documents(self,query):
        return self.retriever.get_documents(query)
    


class QueryRunner:
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
    def __init__(self, url: str, port:int = None, k:int = 3):
        super().__init__(k=k)

        self.retriever = requests.get(url)

    def forward(self, query_or_queries:str, k:Optional[int]) -> dspy.Prediction:
        params = {"query": query_or_queries, "k": k if k else self.k}
        response = requests.get(self.url, params=params)

        response = response.json()["retrieved_passages"]    # List of top k passages
        return dspy.Prediction(
            passages=response
        )
    
if __name__ == "__main__":
    OPENAI_API_KEY = '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'
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    directory = "/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/census_populations/NYC/simulation_input/simulation_memory_output/1/12"
    query_processor = QueryProcessor(model_name, model_kwargs, encode_kwargs, directory, OPENAI_API_KEY)
    query_processor()