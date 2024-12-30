import streamlit as st
import os
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain

def load_documents():
    loader = DirectoryLoader("./data", glob="**/*.pdf", loader_cls=PDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def initialize_vector_store():
    try:
        documents = load_documents()
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return None

if st.session_state.initialized:
    if 'qa_chain' not in st.session_state:
        with st.spinner("Initializing vector store..."):
            st.session_state.qa_chain = initialize_vector_store()

    if st.session_state.qa_chain:
        question = st.chat_input("Enter your question")

        if question:
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.qa_chain({"question": question})
                    st.write(response['answer'])
                    
                    with st.expander("View source documents"):
                        for doc in response['source_documents']:
                            st.write(doc.page_content)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


prompt1 = st.text_input("Enter Your question from the documents")
if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("Vector Store DB is ready")


if prompt1:
    documents_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt1})
    print("Response time : ", time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------")
