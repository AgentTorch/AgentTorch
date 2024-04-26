from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
from dsp.utils import deduplicate

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class DocumentRetriever:
    def __init__(self, model_name, model_kwargs, encode_kwargs, directory,search_kwargs):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.directory = directory
        self.hf = HuggingFaceBgeEmbeddings(
            model_name=self.model_name, 
            model_kwargs=self.model_kwargs, 
            encode_kwargs=self.encode_kwargs
        )
        self.docs = self.load_documents()
        self.docs = self.split_documents(self.docs)
        self.vectorstore = self.create_vectorstore(self.docs)
        self.retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def load_documents(self):
        md_files =  glob.glob(self.directory + '/**/*.md', recursive=True)
        docs = []
        for file in md_files:
            loader = TextLoader(file)
            docs.append(loader.load()[0])
        return docs

    def split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        return text_splitter.split_documents(documents=docs)

    def create_vectorstore(self, docs):
        embeddings = self.hf
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    
    def save_vectorstore(self, store_name):
        self.vectorstore.save_local(store_name)
    
    def load_vectorstore(self, store_name):
        return FAISS.load_local(store_name, self.hf, allow_dangerous_deserialization=True)

    def get_documents(self,query,k):
        return self.retriever.get_relevant_documents(query,k=k)

class QueryRunnerUsingLangChain:
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
    def __init__(self, model_name,model_kwargs, encode_kwargs, directory,search_kwargs):
        super().__init__(k=search_kwargs['k'])
        self.retriever = DocumentRetriever(model_name, model_kwargs, encode_kwargs, directory,search_kwargs=search_kwargs)
    
    def forward(self, query_or_queries:str, k:Optional[int]) -> dspy.Prediction:
        documents = self.retriever.get_documents(query_or_queries, k=k)
        # Convert each document to a DotDict
        passages = [DotDict(long_text=doc.page_content) for doc in documents]
        # print(psg.long_text for psg in passages)
        return passages
        # List of top k passages
        # return dspy.Prediction(
        #     passages=passages
        # )

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
    
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()
    
class QueryRunnerUsingDspy(dspy.Module):
    def __init__(self, passages_per_hop=5, max_hops=4):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
    
if __name__ == "__main__":
    OPENAI_API_KEY = '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'
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    directory = "/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/census_populations/NYC/simulation_memory_output"
    search_kwargs = {"k": 5}
    rm = DSPythonicRMClient(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs,directory=directory,search_kwargs=search_kwargs)
    turbo = dspy.OpenAI(model='gpt-3.5-turbo', api_key=OPENAI_API_KEY)
    dspy.settings.configure(lm=turbo, rm=rm)
    # Ask any question you like to this simple RAG program.
    my_question = "How many age groups are there in the population?"

    # Get the prediction. This contains `pred.context` and `pred.answer`.
    query_runner = QueryRunnerUsingDspy()  # uncompiled (i.e., zero-shot) program
    pred = query_runner(my_question)

    # Print the contexts and the answer.
    print(f"Question: {my_question}")
    print(f"Predicted Answer: {pred.answer}")
    print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")