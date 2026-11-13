
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Project paths
DATA_PATH = "data"
VECTORSTORE_PATH = "vectorstore"

print("Starting TravelAI document ingestion...")

# 1. Load documents
loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()

if not documents:
    raise ValueError("No documents found in the data folder!")

print(f"Documents loaded: {len(documents)}")

# 2. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)

chunks = text_splitter.split_documents(documents)

print(f"Text chunks created: {len(chunks)}")

# 3. Generate embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Generating embeddings...")

# 4. Store embeddings in FAISS
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 5. Save vector database
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

vectorstore.save_local(VECTORSTORE_PATH)

print("===================================")
print("TravelAI ingestion completed!")
print(f"Total documents: {len(documents)}")
print(f"Total chunks: {len(chunks)}")
print("FAISS vector database saved!")
print("===================================")
