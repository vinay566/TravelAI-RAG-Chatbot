
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Load local .env file
load_dotenv(BASE_DIR / ".env")


# ==========================================
# 2. GET GROQ API KEY
# ==========================================

def get_api_key():

    # First check local environment / .env
    api_key = os.getenv("GROQ_API_KEY")

    # Then check Streamlit Cloud Secrets
    try:
        api_key = st.secrets.get("GROQ_API_KEY", api_key)
    except Exception:
        pass

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to your .env file or Streamlit Cloud Secrets."
        )

    return api_key


# ==========================================
# 3. LOAD HUGGINGFACE EMBEDDINGS
# ==========================================

@st.cache_resource
def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# ==========================================
# 4. CREATE VECTOR DATABASE IF MISSING
# ==========================================

def create_vectorstore():

    print("Creating TravelAI vector database...")

    documents = []

    for file_path in DATA_DIR.glob("*.txt"):

        content = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        if content.strip():

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path)
                    }
                )
            )

    if not documents:
        raise FileNotFoundError(
            "No travel text documents found in the data folder."
        )

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    print("Documents loaded:", len(documents))
    print("Chunks created:", len(chunks))

    # Generate embeddings
    embeddings = get_embeddings()

    # Create FAISS database
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save database
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print("TravelAI vector database created!")

    return vectorstore


# ==========================================
# 5. LOAD FAISS VECTOR DATABASE
# ==========================================

@st.cache_resource
def get_vectorstore():

    embeddings = get_embeddings()

    index_file = VECTORSTORE_DIR / "index.faiss"
    metadata_file = VECTORSTORE_DIR / "index.pkl"

    if index_file.exists() and metadata_file.exists():

        print("Loading existing FAISS database...")

        vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    else:

        print("FAISS database not found. Creating new database...")

        return create_vectorstore()


# ==========================================
# 6. INITIALIZE GROQ LLM
# ==========================================

def get_llm():

    api_key = get_api_key()

    llm = ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-20b",
        temperature=0.2
    )

    return llm


# ==========================================
# 7. MAIN RAG FUNCTION
# ==========================================

def get_answer(question):

    if not question or not question.strip():

        return "Please enter a question.", []

    # Load vector database
    vectorstore = get_vectorstore()

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:

        return (
            "I could not find this information "
            "in my travel knowledge base.",
            []
        )

    # Prepare context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Collect source filenames
    sources = list(dict.fromkeys(
        Path(doc.metadata.get("source", "Unknown")).name
        for doc in retrieved_docs
    ))

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            You are TravelAI, an intelligent travel assistant.

            Answer the user's question using ONLY the
            provided travel knowledge base context.

            Instructions:
            1. Give clear and helpful answers.
            2. Use simple English.
            3. Organize answers using bullet points when useful.
            4. Do not invent travel information.
            5. If the answer is not available in the
               context, say:
               "I could not find this information
               in my travel knowledge base."
            """
        ),

        (
            "human",
            """
            Travel Knowledge Base:

            {context}

            User Question:

            {question}

            Answer:
            """
        )

    ])

    # Generate response
    llm = get_llm()

    chain_input = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(chain_input)

    # Return answer and source files
    return response.content, sources
