
from agent_torch.core.analyzer.dspy_modules import GenerateAnswer, GenerateSearchQuery
import dspy
from dsp.utils import deduplicate
from langchain.chains import RetrievalQA
from pandasai.llm import OpenAI

class QueryRunnerDspy(dspy.Module):
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


class QueryRunnerLangChain:
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