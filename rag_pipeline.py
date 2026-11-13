
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize Groq API
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=api_key,
    temperature=0.2
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Create prompt template
prompt = ChatPromptTemplate.from_template("""
You are TravelAI, an AI travel assistant.

Answer the user's question using ONLY the
provided travel context.

If the answer is not available in the context,
say:
"I could not find this information in my
travel knowledge base."

Do not invent travel information.

Travel Context:
{context}

User Question:
{question}

Answer:
""")

# RAG function
def get_answer(question):

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Combine retrieved content
    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    # Generate prompt
    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    # Generate response using Groq
    response = llm.invoke(messages)

    # Extract source documents
    sources = list(set(
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in documents
    ))

    return response.content, sources


# Test the RAG pipeline
if __name__ == "__main__":

    print("\nWelcome to TravelAI RAG Chatbot!")

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            print("TravelAI: Goodbye!")
            break

        try:
            answer, sources = get_answer(question)

            print("\nTravelAI:", answer)

            print("\nSources:")
            for source in sources:
                print("-", source)

        except Exception as e:
            print("\nError:", e)
