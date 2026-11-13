
import streamlit as st
from rag_pipeline import get_answer

# Page configuration
st.set_page_config(
    page_title="TravelAI Assistant",
    page_icon="✈️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    color: #2563eb;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    color: #64748b;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="title">✈️ TravelAI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your AI-powered travel information assistant</div>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
with st.sidebar:
    st.title("🌍 TravelAI")
    st.write("Explore travel information using AI.")

    st.subheader("Suggested Questions")

    if st.button("Best places in Goa"):
        st.session_state.question = "What are the best places to visit in Goa?"

    if st.button("Kerala destinations"):
        st.session_state.question = "What are popular destinations in Kerala?"

    if st.button("Travel safety"):
        st.session_state.question = "What should I do if I lose my passport?"

    if st.button("Rajasthan tourism"):
        st.session_state.question = "What are popular destinations in Rajasthan?"

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write("•", source)

# Chat input
question = st.chat_input("Ask me anything about travel...")

# Sidebar suggested question
if "question" in st.session_state:
    question = st.session_state.pop("question")

# Generate response
if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Searching travel knowledge..."):

            try:
                answer, sources = get_answer(question)

                st.markdown(answer)

                # Hide sources if knowledge is unavailable
                if "could not find this information" in answer.lower():
                    sources = []

                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write("•", source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
