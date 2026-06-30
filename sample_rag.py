"""
RAG-style application:
  - OpenAI embeddings
  - Chroma vector database
  - LangChain retrieval chain
"""

import os

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def create_embeddings(texts):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_texts(texts, embedding=embeddings)
    return vectorstore


def ask_question(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    db = create_embeddings(
        [
            "Cisco builds networking products.",
            "AIBOM inventories AI components.",
        ]
    )

    docs = db.similarity_search("What is AIBOM?")
    print(docs)
    print(ask_question("What is retrieval augmented generation?"))
