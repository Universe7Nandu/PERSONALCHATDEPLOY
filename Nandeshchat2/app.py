import sys
import os
import asyncio
import nest_asyncio
import streamlit as st
from langchain_groq import ChatGroq

# 1. CONFIGURATION
GROQ_API_KEY = "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"

# 2. SYSTEM PROMPT FOR THE MATHEMATICS ASSISTANT
DEFAULT_SYSTEM_PROMPT = """
## Enterprise-Grade Mathematics Assistant
- You are a professional mathematics assistant designed to solve complex mathematical problems.
- Provide correct solutions with detailed, step-by-step explanations.
- Include clear and accurate visualizations (e.g., graphs, diagrams) when appropriate.
- Utilize advanced LaTeX integration for precise mathematical expressions.
- Offer interactive guidance to help users understand underlying concepts.
- Your tone should be professional, clear, and supportive.
Question: {user_query}
"""

# 3. APPLY ASYNC PATCH
nest_asyncio.apply()

# 4. MAIN APPLICATION FUNCTION
def main():
    st.set_page_config(page_title="Mathematics Chatbot", layout="wide")
    
    # Advanced CSS for a clean, modern look
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    body {
        background: linear-gradient(135deg, #2c3e50, #bdc3c7);
        margin: 0; padding: 0;
    }
    header, footer { display: none; }
    .chat-container {
        max-width: 900px;
        margin: 40px auto;
        background: rgba(255,255,255,0.9);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .chat-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.4rem;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .chat-subtitle {
        text-align: center;
        color: #34495e;
        margin-top: 0;
        margin-bottom: 20px;
        font-size: 1.1rem;
    }
    .stChatInput {
        position: sticky;
        bottom: 0;
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(6px);
        padding: 10px;
        margin-top: 20px;
        border-radius: 12px;
    }
    .stChatInput>div>div>input {
        color: #2c3e50;
        font-weight: 500;
        border-radius: 8px;
        border: 1px solid #bdc3c7;
    }
    .stChatInput>div>div>input:focus {
        outline: 2px solid #2980b9;
    }
    [data-testid="stSidebar"] {
        background: #2c3e50 !important;
        color: #ecf0f1 !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #e74c3c !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #ecf0f1 !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        background: #e74c3c !important;
        color: #fff !important;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        transition: background 0.3s;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background: #c0392b !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # -------- SIDEBAR --------
    with st.sidebar:
        st.header("About")
        st.markdown("""
**Enterprise-Grade Mathematics Assistant**  
Provides step-by-step solutions, interactive visualizations, and LaTeX-enhanced outputs for complex math problems.
        """)
        st.markdown("---")
        st.header("Usage Instructions")
        st.markdown("""
1. **Type** your mathematical question or problem in the chat box below.  
2. **Receive** a detailed, step-by-step explanation along with relevant visualizations.  
3. **Interact** for further clarification or follow-up questions.
        """)
        st.markdown("---")
        st.header("Conversation History")
        if st.button("New Chat"):
            st.session_state.pop("chat_history", None)
            st.success("New conversation started! 🆕")
        if "chat_history" in st.session_state and st.session_state["chat_history"]:
            for i, item in enumerate(st.session_state["chat_history"], 1):
                st.markdown(f"{i}. **You**: {item['question']}")
        else:
            st.info("No conversation history yet.")

    # -------- MAIN CHAT AREA --------
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='chat-title'>Mathematics Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='chat-subtitle'>Ask your math problems and get step-by-step solutions with visualizations.</p>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display existing conversation
    for msg in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.markdown(msg["question"])
        with st.chat_message("assistant"):
            st.markdown(msg["answer"])
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- CHAT INPUT --------
    user_query = st.chat_input("Type your math question here... (Press Enter)")
    if user_query:
        st.session_state["chat_history"].append({"question": user_query, "answer": ""})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.spinner("Solving..."):
            # Build the prompt for the mathematics assistant
            prompt = DEFAULT_SYSTEM_PROMPT.format(user_query=user_query)
            llm = ChatGroq(
                temperature=0.7,
                groq_api_key=GROQ_API_KEY,
                model_name="mixtral-8x7b-32768"
            )
            # Invoke the LLM asynchronously
            response = asyncio.run(llm.ainvoke([{"role": "user", "content": prompt}]))
            bot_answer = response.content
        st.session_state["chat_history"][-1]["answer"] = bot_answer
        with st.chat_message("assistant"):
            st.markdown(bot_answer)

if __name__ == "__main__":
    main()
