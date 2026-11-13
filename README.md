\# ✈️ TravelAI – RAG-Based Travel Assistant



\## 📌 Project Overview



TravelAI is an AI-powered travel information chatbot built using Retrieval-Augmented Generation (RAG).



It helps users find travel-related information about destinations, transportation, travel safety, and frequently asked questions.



The chatbot retrieves relevant information from a custom travel knowledge base and uses the Groq LLM to generate contextual answers.



\## 🎯 Project Objectives



\- Provide accurate travel-related information.

\- Answer questions using a custom knowledge base.

\- Retrieve relevant documents using semantic search.

\- Reduce hallucinations by grounding answers in retrieved context.

\- Display source documents for transparency.



\## 🛠️ Technologies Used



\- Python

\- Streamlit

\- LangChain

\- Groq LLM API

\- HuggingFace Sentence Transformers

\- FAISS Vector Database

\- Retrieval-Augmented Generation (RAG)



\## 🏗️ RAG Architecture



The project follows this workflow:



1\. Document Loading

2\. Text Splitting

3\. Embedding Generation

4\. Vector Database Storage

5\. Retriever

6\. Groq LLM Integration

7\. Final Answer Generation



\## 📂 Project Structure



```text

TravelAI-RAG-Chatbot/

│

├── data/

│   ├── destinations.txt

│   ├── travel\_faqs.txt

│   ├── travel\_safety.txt

│   └── transportation.txt

│

├── vectorstore/

│

├── app.py

├── ingest.py

├── rag\_pipeline.py

├── requirements.txt

├── .gitignore

├── .env

└── README.md

```



\## ⚙️ Installation and Setup



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd TravelAI-RAG-Chatbot

```



\### 2. Install dependencies



```bash

pip install -r requirements.txt

```



\### 3. Configure API Key



Create a `.env` file in the project directory.



```env

GROQ\_API\_KEY=your\_groq\_api\_key

```



\### 4. Build the Vector Database



```bash

python ingest.py

```



\### 5. Run the Chatbot



```bash

python -m streamlit run app.py

```



Open the local URL displayed in the terminal.



\## 💬 Example Questions



\- What are the popular tourist attractions in Goa?

\- What are the popular destinations in Kerala?

\- What should I do if I lose my passport?

\- What are the popular destinations in Rajasthan?



\## 🔍 Key Features



\- AI-powered travel question answering.

\- Semantic document retrieval.

\- FAISS-based vector search.

\- Context-aware responses.

\- Source document references.

\- Interactive Streamlit interface.

\- Knowledge-base fallback for unavailable information.



\## 🔐 Security



API keys are stored in environment variables and should never be uploaded to GitHub.



\## 🚀 Future Enhancements



\- Add more travel destinations.

\- Integrate live weather information.

\- Add hotel and flight information.

\- Include multilingual support.

\- Add Google Maps integration.



\## 👨‍💻 Project Type



Python | Generative AI | Retrieval-Augmented Generation | Chatbot



\---



\*\*TravelAI – Your AI-Powered Travel Assistant!\*\* ✈️

