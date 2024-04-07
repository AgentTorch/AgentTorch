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

class QueryProcessor:
    def __init__(self, model_name, model_kwargs, encode_kwargs, directory,openai_api_key):
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
        vectorstore = self.create_vectorstore(docs)
        # persisted_vectorstore = self.load_vectorstore(self.hf)
        self.qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key = openai_api_key), chain_type="stuff", retriever=vectorstore.as_retriever())

    def load_documents(self):
        md_files =  glob.glob(self.directory + '/**/*.md', recursive=True)
        docs = []
        for file in md_files:
            loader = TextLoader(file)
            docs.append(loader.load()[0])
        return docs

    def split_documents(self, docs):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        return text_splitter.split_documents(documents=docs)

    def create_vectorstore(self, docs):
        embeddings = self.hf
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    
    def save_vectorstore(self, vectorstore, store_name):
        vectorstore.save_local(store_name)
    
    def load_vectorstore(self, embeddings, store_name ):
        return FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)

    def run_query(self, query):
        result = self.qa.invoke(query)
        print(result)

    def __call__(self):
        query_in = input("Type in your query: \n")
        while query_in != "exit":
            self.run_query(query_in)
            query_in = input("Type in your query: \n")
            
if __name__ == "__main__":
    OPENAI_API_KEY = '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'
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    directory = "/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/census_populations/NYC/simulation_input/simulation_memory_output/1/0/"
    query_processor = QueryProcessor(model_name, model_kwargs, encode_kwargs, directory, OPENAI_API_KEY)
    query_processor()