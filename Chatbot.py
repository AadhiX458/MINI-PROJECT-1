import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# =========================
# Page Configuration
# =========================
st.set_page_config(page_title="Aadhi's AI Buddy", page_icon="🤖")
st.title("🤖 Chat with Aadhi's AI Buddy")

# =========================
# Load External CSS
# =========================
try:
    with open("style.css", "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ style.css not found. Using default styles.")

# =========================
# Initialize Session State
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# Display Previous Messages
# =========================
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# Setup LangChain Prompt & LLM
# =========================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly assistant. Respond clearly and naturally without mentioning that you are an AI or language model."),
    ("user", "User query: {query}")
])

llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# =========================
# Handle User Input
# =========================
if user_input := st.chat_input("Ask me anything..."):
    # Store & display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display bot response
    with st.spinner("Thinking... 🤔"):
        bot_response = chain.invoke({"query": user_input})

    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_response)

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    "<footer>Made with ❤️ by Aadhi | Powered by Ollama + LangChain + Streamlit</footer>",
    unsafe_allow_html=True
)
