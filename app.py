import streamlit as st
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
openai_api_key = os.environ.get("OPENAI_API_KEY")

st.title("Testing Chat")

models_dict = {
    "gemma2_9b": {"model_name": "gemma2-9b-it", "context_window": "8,192"},
    "gemma7_7b": {"model_name": "gemma-7b-it", "context_window": "8,192"},
    "llama3_70b": {"model_name": "llama-3.3-70b-versatile", "context_window": "128k"},
    "llama3_8b_instant": {"model_name": "llama-3.1-8b-instant", "context_window": "128k"},
    "llama70b_8192": {"model_name": "llama3-70b-8192", "context_window": "8,192"},
    "llama8b_8192": {"model_name": "llama3-8b-8192", "context_window": "8,192"},
    "mixtral8x7b": {"model_name": "mixtral-8x7b-32768", "context_window": "32,768"},
}

@st.cache_data
def get_embeddings(embedding_type):
    if embedding_type == "HuggingFace":
        return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    elif embedding_type == "OpenAI":
        return OpenAIEmbedding()


sidebar = st.sidebar
sidebar.title("Configure")
with sidebar:

    st.subheader("Choose your model")
    selected_model = st.selectbox(
        "Model",
        index=None,
        options=models_dict.keys(),
        placeholder="Selected Model"
    )

    st.subheader("Choose your embeddings")
    selected_embeddings = st.selectbox(
        "Embeddings",
        index=None,
        options=("HuggingFace", "OpenAI"),
    )
    if selected_embeddings == "HuggingFace":
        selected_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    elif selected_embeddings == "OpenAI":
        selected_embeddings = OpenAIEmbedding()
    submit = st.button("Submit")
    if submit:
        st.session_state.embeddings = get_embeddings(selected_embeddings)
        st.session_state.model = selected_model
        # st.session_state.embeddings = selected_embeddings
        st.markdown(f"Model selected: `{st.session_state.model}`")
        st.markdown(f"Context window: `{models_dict[st.session_state.model]['context_window']}`")
        st.markdown(f"Embeddings selected: `{st.session_state.embeddings.model_name}`")
        st.session_state.llm = Groq(model=st.session_state.model, api_key=groq_api_key)

        initialized = True
    else:
        initialized = False

def default_initialization():
    st.session_state.model = "llama3-8b-8192"
    st.session_state.embeddings = OpenAIEmbedding()
    st.session_state.llm = Groq(model=st.session_state.model, api_key=groq_api_key)

def initialize_vectorstore():
    loader = SimpleDirectoryReader(
        input_dir="./test/",
        recursive=True,
        required_exts=[".epub"],
    )
    st.session_state.loader = loader

    documents = loader.load_data()
    st.session_state.documents = documents

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=st.session_state.embeddings,
        show_progress=True
    )
    st.session_state.index = index

    query_engine = index.as_chat_engine(llm=st.session_state.llm)
    st.session_state.query_engine = query_engine

    return query_engine



if initialized == False:
    default_initialization()
    chat_engine = initialize_vectorstore()

    question = st.chat_input("Enter your question")

    streaming_response = chat_engine.stream_chat(question)
    for token in streaming_response.response_gen:
        st.write_stream(token, end="")