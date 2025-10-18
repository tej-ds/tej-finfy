import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
import logging

# --- Configuration ---
# IMPORTANT: This path should point to the directory containing your pre-built FAISS index.
# This directory MUST contain 'index.faiss' and 'index.pkl' files.
FAISS_INDEX_DIR = './faiss_index_finfy'  # Change this to your index directory name
GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # The Groq model you want to use
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # The embedding model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Groq API Key from Streamlit Secrets ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    logger.info("GROQ_API_KEY loaded from Streamlit secrets.")
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please set it.")
    logger.critical("GROQ_API_KEY not found. Stopping app.")
    st.stop()

# --- Helper Functions (Cached for Performance) ---
@st.cache_resource
def get_embeddings_model():
    """Load the embedding model once."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_vectorstore(_embeddings, faiss_index_dir):
    """Load a pre-built FAISS index."""
    faiss_index_path = os.path.join(faiss_index_dir, "index.faiss")
    faiss_pickle_path = os.path.join(faiss_index_dir, "index.pkl")

    if os.path.exists(faiss_index_path) and os.path.exists(faiss_pickle_path):
        logger.info(f"Loading pre-built FAISS index from {faiss_index_dir}...")
        try:
            vectorstore = FAISS.load_local(faiss_index_dir, _embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded successfully.")
            return vectorstore
        except Exception as e:
            st.error(f"Error loading FAISS index from '{faiss_index_dir}': {e}.")
            logger.critical(f"FAISS load error: {e}", exc_info=True)
            st.stop()
    else:
        st.error(f"Pre-built FAISS index not found at '{faiss_index_dir}'.")
        logger.critical(f"FAISS index directory missing or incomplete: {faiss_index_dir}.")
        st.stop()

@st.cache_resource
def get_llm_model():
    """Load the Groq LLM model once."""
    logger.info(f"Loading Groq LLM: {GROQ_MODEL_NAME}...")
    return ChatGroq(model_name=GROQ_MODEL_NAME, temperature=0.1, groq_api_key=GROQ_API_KEY)

@st.cache_resource
def get_retrieval_chain(_vectorstore, _llm):
    """Create the LangChain retrieval chain."""
    logger.info("Creating LangChain document and retrieval chains...")
    
    # --- CHANGE SYSTEM PROMPT HERE ---
    system_prompt = (
        "You are an AI assistant designed to teach youth about personal finance. "
        "Your task is to answer user questions using simple, clear, and engaging language, "
        "drawing information ONLY from the provided context. The context contains documents "
        "about topics such as saving, investing, budgeting, and credit. "
        "Explain complex topics with simple examples that are relatable to a young audience. "
        "If the exact answer or specific information is not available in the context, "
        "clearly state that the information is not in your knowledge base and encourage the user to ask a different question. "
        "Do not make up information or provide financial advice outside the given context.\n\n"
        "Context:\n"
        "{context}"
    )
    # --- END SYSTEM PROMPT CHANGE ---

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(_llm, prompt)
    retrieval_chain = create_retrieval_chain(_vectorstore.as_retriever(k=5), document_chain)
    logger.info("LangChain chains created.")
    return retrieval_chain

# --- Streamlit UI ---
st.set_page_config(page_title="Finance for Youth AI Assistant", page_icon="💰")
st.title("💰 Finance for Youth AI Assistant")
st.markdown("Ask me anything about saving, budgeting, and other money topics!")

# --- Initialize components ---
embeddings = get_embeddings_model()
vectorstore = get_vectorstore(embeddings, FAISS_INDEX_DIR)
llm = get_llm_model()
retrieval_chain = get_retrieval_chain(vectorstore, llm)

# --- Session State Initialization for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Response Generation ---
if prompt := st.chat_input("What do you want to learn about today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = retrieval_chain.invoke({"input": prompt})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.markdown("I'm sorry, I encountered an error while processing your request. Please try again.")
                logger.error(f"Chatbot error during invocation: {e}", exc_info=True)