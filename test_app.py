import streamlit as st
from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()

# Retrieve API keys with error handling
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()

openai_api_key = os.getenv("OPENAI_API_KEY")  # Optional, used by OpenAIEmbedding

# Streamlit App Title
st.title("Chat with LLM")

# Define available models
MODELS_DICT = {
    "gemma2_9b": {"model_name": "gemma2-9b-it", "context_window": "8,192"},
    "gemma7_7b": {"model_name": "gemma-7b-it", "context_window": "8,192"},
    "llama3_70b": {"model_name": "llama-3.3-70b-versatile", "context_window": "128k"},
    "llama3_8b_instant": {"model_name": "llama-3.1-8b-instant", "context_window": "128k"},
    "llama70b_8192": {"model_name": "llama3-70b-8192", "context_window": "8,192"},
    "llama8b_8192": {"model_name": "llama3-8b-8192", "context_window": "8,192"},
    "mixtral8x7b": {"model_name": "mixtral-8x7b-32768", "context_window": "32,768"},
}

# Initialize Session State
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Caching embedding models to avoid reloading
@st.cache_data(show_spinner=False)
def get_embedding_model(embedding_type):
    if embedding_type == "HuggingFace":
        return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    elif embedding_type == "OpenAI":
        return OpenAIEmbedding(api_key=openai_api_key)
    else:
        raise ValueError("Unsupported embedding type selected.")

# Caching document loading to improve performance
@st.cache_data(show_spinner=True)
def load_documents(input_dir="./test/", required_exts=[".epub"]):
    loader = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
        required_exts=required_exts,
    )
    return loader.load_data()

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")

    # Model Selection
    st.subheader("Choose Your Model")
    selected_model_key = st.selectbox(
        "Model",
        options=list(MODELS_DICT.keys()),
        index=0,
        format_func=lambda x: f"{x} ({MODELS_DICT[x]['context_window']})"
    )

    # Embedding Selection
    st.subheader("Choose Your Embeddings")
    embedding_type = st.selectbox(
        "Embeddings",
        options=["HuggingFace", "OpenAI"],
        index=0
    )

    # Submit Button
    if st.button("Submit"):
        try:
            # Retrieve embedding model
            embedding_model = get_embedding_model(embedding_type)
            
            # Update session state
            st.session_state.embeddings = embedding_model
            st.session_state.model = MODELS_DICT[selected_model_key]['model_name']
            st.session_state.context_window = MODELS_DICT[selected_model_key]['context_window']
            st.session_state.llm = Groq(model=st.session_state.model, api_key=groq_api_key)
            st.session_state.initialized = True

            st.success("Configuration updated successfully!")
            st.markdown(f"**Model selected:** `{st.session_state.model}`")
            st.markdown(f"**Context window:** `{st.session_state.context_window}`")
            st.markdown(f"**Embeddings selected:** `{embedding_type}`")
        except Exception as e:
            st.error(f"Error during configuration: {e}")

# Function to initialize the vector store
def initialize_vector_store():
    try:
        documents = load_documents()
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=st.session_state.embeddings,
            show_progress=True
        )
        query_engine = index.as_chat_engine(llm=st.session_state.llm)
        return query_engine
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return None

# Main Interaction
if st.session_state.initialized:
    if 'query_engine' not in st.session_state:
        with st.spinner("Initializing vector store..."):
            st.session_state.query_engine = initialize_vector_store()

    if st.session_state.query_engine:
        question = st.chat_input("Enter your question")

        if question:
            with st.spinner("Generating response..."):
                try:
                    streaming_response = st.session_state.query_engine.stream_chat(question)
                    response = ""
                    for token in streaming_response.response_gen:
                        response += token
                        st.markdown(response + "▌")  # Simulate typing indicator
                    st.markdown(response)  # Final response without indicator
                except Exception as e:
                    st.error(f"Error during chat: {e}")
else:
    # Default Initialization (Optional: Can be removed if you want user to configure first)
    if not st.session_state.initialized:
        with st.spinner("Setting default configuration..."):
            try:
                st.session_state.embeddings = OpenAIEmbedding(api_key=openai_api_key)
                st.session_state.model = MODELS_DICT["llama8b_8192"]['model_name']
                st.session_state.context_window = MODELS_DICT["llama8b_8192"]['context_window']
                st.session_state.llm = Groq(model=st.session_state.model, api_key=groq_api_key)
                st.session_state.query_engine = initialize_vector_store()
                st.session_state.initialized = True
                st.success("Default configuration initialized. Please adjust settings in the sidebar if needed.")
            except Exception as e:
                st.error(f"Error during default initialization: {e}")
