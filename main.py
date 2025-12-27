from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3.2")

template = """
Your are expert in answering questions about pizza restraunt.
You have analysed all the reviews data and you know the best pizza restraunt in the city.
You are given a question, you need to answer the question based on the reviews data.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Ask them if they want to know more about the restraunt.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
