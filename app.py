import streamlit as st
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)

# response = llm.complete("Explain the importance of low latency LLMs")

loader = SimpleDirectoryReader(
    input_dir="./test/",
    recursive=True,
    required_exts=[".epub"],
)

documents = loader.load_data()



embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")



index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embedding_model,
    show_progress=True
)

# from llama_index.llms.ollama import Ollama

# llama = Ollama(
#     model="llama2",
#     request_timeout=40.0,
# )

query_engine = index.as_query_engine(llm=llm)

print(query_engine.query("What are the titles of all the books available? Show me the context used to derive your answer."))