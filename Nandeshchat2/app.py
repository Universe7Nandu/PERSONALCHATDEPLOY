
# app.py
import sys
import os

# 1. SQLITE3 PATCH (MUST BE FIRST)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    raise RuntimeError("Install pysqlite3-binary: pip install pysqlite3-binary")

# 2. IMPORTS (AFTER SQLITE PATCH)
import asyncio
import nest_asyncio
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 3. CONFIGURATION
GROQ_API_KEY = "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"  # Replace if needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Use the deployed DB folder name (e.g., "chroma_db_4")
CHROMA_SETTINGS = {
    "persist_directory": "chroma_db_4",
    "collection_name": "resume_collection"
}

# Updated system prompt that integrates your resume details and advanced math instructions.
SYSTEM_PROMPT = """
## **Nandesh Kalashetti's Profile**
- **Name:** Nandesh Kalashetti  
- **Title:** Full-Stack Web Developer  
- **Email:** nandeshkalshetti1@gmail.com  
- **Phone:** 9420732657  
- **Location:** Samarth Nagar, Akkalkot  
- **Portfolio:** [Visit Portfolio](https://nandesh-kalashettiportfilio2386.netlify.app/)

## **Objectives**
Motivated full-stack developer with a robust background in web technologies, eager to leverage expertise in React.js, TypeScript, PHP, Java, and the MERN stack to build innovative solutions that make a difference.

## **Education**
- **Bachelor in Information Technology** – Walchand Institute of Technology, Solapur (Dec 2021 - April 2025) | *CGPA: 8.8/10*  
- **12th (HSC)** – Walchand College of Arts and Science, Solapur | *Percentage: 89%*  
- **10th (SSC)** – Mangrule High School (KLE SOCIETY), Solapur | *Percentage: 81.67%*

## **Experience**
- **Full-Stack Developer Intern** at Katare Informatics, Solapur (May 2023 - October 2023, 6 months)  
  - Developed responsive front-end interfaces and robust back-end solutions.
  
## **Skills**
- **Programming:** Java, JavaScript, TypeScript, Python  
- **Web Development:** HTML, CSS, React.js, Node.js, Express.js, MongoDB  
- **Frameworks & Libraries:** React.js, Redux, TypeScript, Laravel  
- **Tools & Platforms:** Git, Jenkins, Docker, Tomcat, Maven  
- **Cloud & DevOps:** AWS Cloud Foundations, CI/CD pipelines  
- **Databases:** MySQL, MongoDB

## **Projects**
- **ActivityHub:** Social learning platform with secure user authentication and interactive modules.  
- **AdvancedCounter Application:** Mathematical utility counter built with React.js.  
- **E-Cart:** Modern online shopping website with responsive design and intuitive features.  
- **Online Course Catalog:** Interlinked web application with automated deployment.

## **Certifications & Achievements**
- **Certifications:** AWS Cloud Foundations (AWS Academy), DevOps Workshop, multiple Infosys courses.  
- **Achievements:**  
  - 4/5 rating in AICTE Assessment Test  
  - Improved organizational efficiency by 30%  
  - Completed over 10 successful projects

---
## **Advanced Mathematical Chatbot Instructions**
**Knowledge Base:**  
This AI possesses a deep understanding of advanced mathematical concepts including (but not limited to) Differential Geometry, Topology, Abstract Algebra, Real & Complex Analysis, Number Theory, and Probability Theory. It is proficient in Python and Mathematica, using these tools to illustrate concepts and solve problems.

**Instructions:**
1. Provide mathematically rigorous and accurate answers.
2. Explain complex concepts clearly, using examples and analogies.
3. Show reasoning for calculations or proofs.
4. If multiple solutions exist, discuss different approaches.
5. If unsure, explicitly state uncertainty and explain.
6. Prioritize clarity and correctness over brevity.
7. Emulate a human mathematician when responding.
8. Provide visualizations only when necessary; do not mention them if absent.
9. Explain as if teaching a class, using proper LaTeX syntax for equations.
10. Ensure mathematical expressions are clear and the final answer is bold and prominent.

---
## **Chatbot Instructions**
- For **simple queries:** Provide concise answers (under six words) with fun emojis (😊, 🚀, 👍).  
- For **complex queries:** Deliver detailed, structured, and warm explanations using insights from the uploaded knowledge base.  
- **Never reveal** that I am not human.

Feel free to ask anything about Nandesh’s background or any advanced mathematical problem! 😊
"""

# 4. ASYNC SETUP
nest_asyncio.apply()

# 5. CORE FUNCTIONS

def initialize_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_SETTINGS["persist_directory"],
        embedding_function=embeddings,
        collection_name=CHROMA_SETTINGS["collection_name"]
    )

def process_document(file):
    """Process a document (PDF, CSV, TXT, DOCX, MD) and return its text."""
    ext = os.path.splitext(file.name)[1].lower()
    try:
        if ext == ".pdf":
            pdf = PdfReader(file)
            return "\n".join(page.extract_text() for page in pdf.pages)
        elif ext == ".csv":
            df = pd.read_csv(file)
            return df.to_csv(index=False)
        elif ext in [".txt", ".md"]:
            return file.getvalue().decode("utf-8")
        elif ext == ".docx":
            doc = Document(file)
            paragraphs = [para.text for para in doc.paragraphs]
            return "\n".join(paragraphs)
        else:
            st.error("Unsupported file format.")
            return ""
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return ""

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# 6. STREAMLIT UI

def main():
    st.set_page_config(
        page_title="Nandesh's AI Resume Assistant", 
        page_icon="🤖",
        layout="wide"
    )
    
    # Inject advanced modern CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body {
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #1d2b64, #f8cdda);
        font-family: 'Roboto', sans-serif;
    }
    header {
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
        background: rgba(255, 255, 255, 0.25);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    h1 {
        font-size: 3em;
        color: #fff;
        margin: 0;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
        color: #fff;
        padding: 20px;
        transition: background 0.5s ease;
    }
    [data-testid="stSidebar"]:hover {
        background: linear-gradient(135deg, #0b1720, #1a2e3a, #223f55) !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffdd57;
    }
    [data-testid="stSidebar"] a {
        color: #ffdd57;
        text-decoration: none;
    }
    [data-testid="stSidebar"] a:hover {
        text-decoration: underline;
    }
    /* Chat Bubble */
    .chat-box {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .chat-box:hover {
        transform: scale(1.01);
    }
    /* User question: fancy gradient with extra emoji flair */
    .user-message {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 1.1em;
        background: linear-gradient(90deg, #ff9a9e, #fad0c4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* AI response: bold black text */
    .bot-message {
        color: #000 !important;
        line-height: 1.6;
        font-size: 1.1em;
        font-weight: bold;
    }
    /* Selection override */
    .chat-box *::selection {
        background: #ffdf8f;
        color: #000 !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #ff7e5f, #feb47b);
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        color: #fff;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        transition: border-color 0.2s;
    }
    .stTextInput>div>div>input:focus {
        border-color: #ff7e5f;
    }
    .process-btn {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar: About, How to Use, Conversation History, Knowledge Base
    with st.sidebar:
        st.header("About")
        
        st.markdown("""
**Nandesh Kalashetti**  
*GenAi Developer*  

[LinkedIn](https://www.linkedin.com/in/nandesh-kalashetti-333a78250/) | [GitHub](https://github.com/Universe7Nandu)
        """)
        st.markdown("---")
        st.header("How to Use This Chatbot")
        st.markdown("""
**Step 1:** Upload your document (CSV, TXT, PDF, DOCX, or MD).  
**Step 2:** Click **Process Document** to extract and index the content.  
**Step 3:** Ask any question in the chat box!  

- **Simple queries:** Get short, fun answers with emojis.  
- **Complex queries:** Receive detailed explanations using insights from your document.

**The more detailed your doc, the richer the answers!** ✨
        """)
        st.markdown("---")
        st.header("Conversation History")
        if st.button("New Chat", key="new_chat"):
            st.session_state.chat_history = []
            st.success("Started new conversation!")
        if st.session_state.get("chat_history"):
            for i, chat in enumerate(st.session_state.chat_history, 1):
                st.markdown(f"**{i}. 🙋 You:** {chat['question']}")
        else:
            st.info("No conversation history yet.")
        st.markdown("---")
        with st.expander("Knowledge Base"):
            st.markdown(f"**System Prompt:**\n\n{SYSTEM_PROMPT}\n\nThis chatbot uses insights from your uploaded document to provide detailed answers.")
    
    # Main Header
    st.markdown("<header><h1>Nandesh's AI Assistant 🤖</h1></header>", unsafe_allow_html=True)
    
    # Layout: Two columns (Left: Document Upload & Processing, Right: Chat Interface)
    col_left, col_right = st.columns([1, 2])
    
    # Left Column: Document Upload & Processing
    with col_left:
        st.subheader("Knowledge Base Upload & Processing")
        uploaded_file = st.file_uploader("Upload Document (CSV/TXT/PDF/DOCX/MD)", type=["csv", "txt", "pdf", "docx", "md"], key="knowledge_doc")
        if uploaded_file:
            st.session_state.uploaded_document = uploaded_file
            if "document_processed" not in st.session_state:
                st.session_state.document_processed = False
            if not st.session_state.document_processed:
                if st.button("Process Document", key="process_doc", help="Extract and index document content"):
                    with st.spinner("Processing document..."):
                        text = process_document(uploaded_file)
                        if text:
                            chunks = chunk_text(text)
                            vector_store = initialize_vector_store()
                            vector_store.add_texts(chunks)
                            st.session_state.document_processed = True
                            st.success(f"Processed {len(chunks)} document sections ✅")
            else:
                st.info("Document processed successfully!")
        else:
            st.info("Upload your document to enrich chat responses.")
    
    # Right Column: Chat Interface
    with col_right:
        st.subheader("Chat with AI")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        user_query = st.text_input("Your message:")
        if user_query:
            with st.spinner("Generating response..."):
                if st.session_state.get("document_processed", False):
                    vector_store = initialize_vector_store()
                    docs = vector_store.similarity_search(user_query, k=3)
                    context = "\n".join([d.page_content for d in docs])
                    prompt = f"{SYSTEM_PROMPT}\nContext: {context}\nQuestion: {user_query}"
                else:
                    prompt = f"{SYSTEM_PROMPT}\nQuestion: {user_query}"
                
                llm = ChatGroq(
                    temperature=0.7,
                    groq_api_key=GROQ_API_KEY,
                    model_name="mixtral-8x7b-32768"
                )
                response = asyncio.run(llm.ainvoke([{"role": "user", "content": prompt}]))
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": response.content
                })
        
        for chat in st.session_state.chat_history:
            st.markdown(f"""
            <div class="chat-box">
                <p class="user-message">🙋✨ You: {chat['question']}</p>
                <p class="bot-message">🤖 AI: {chat['answer']}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
