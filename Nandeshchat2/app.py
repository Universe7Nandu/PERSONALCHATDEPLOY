# app.py

import sys
import os
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

# 1. SQLITE3 PATCH (MUST BE FIRST)
try:
    __import__('pysqlite3')  # Correct: use __import__
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    raise RuntimeError("Install pysqlite3-binary: pip install pysqlite3-binary")

# 2. CONFIGURATION
GROQ_API_KEY = "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"  # Updated API key
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_SETTINGS = {
    "persist_directory": "chroma_db_4",
    "collection_name": "resume_collection"
}

# --------------------------------------------------------------------------------
# TWO SEPARATE PROMPTS:
# --------------------------------------------------------------------------------

# Prompt for when NO DOCUMENT is uploaded (uses Nandesh's info).
NANDESH_SYSTEM_PROMPT = """
## *Nandesh Kalashetti's Profile*
- *Name:* Nandesh Kalashetti
- *Title:* Full-Stack Web Developer
- *Email:* nandeshkalshetti1@gmail.com
- *Phone:* 9420732657
- *Location:* Samarth Nagar, Akkalkot
- *Portfolio:* [Visit Portfolio](https://nandesh-kalashettiportfilio2386.netlify.app/)

## *Objectives*
Aspiring full-stack developer with a strong foundation in web development technologies, eager to leverage skills in React.js, TypeScript, PHP, Java, and the MERN stack to create impactful and innovative solutions.

## *Education*
- *Bachelor in Information Technology* – Walchand Institute of Technology, Solapur (Dec 2021 - April 2025) | *CGPA:* 8.8/10  
- *12th (HSC)* – Walchand College of Arts and Science, Solapur | *Percentage:* 89%  
- *10th (SSC)* – Mangrule High School (KLE SOCIETY), Solapur | *Percentage:* 81.67%

## *Experience*
- *Full-Stack Developer Intern* at Katare Informatics, Solapur (May 2023 - October 2023, 6 months)  
  - Worked on HTML, CSS, JavaScript, MySQL, XAMPP, Advanced PHP  
  - Gained hands-on experience in both front-end and back-end development

## *Skills*
- *Programming:* Java, JavaScript, TypeScript, Python  
- *Web Development:* HTML, CSS, React.js, Node.js, Express.js, MongoDB  
- *Frameworks & Libraries:* React.js, Redux, TypeScript, Laravel  
- *Tools & Platforms:* Git, Jenkins, Docker, Tomcat, Maven  
- *Cloud & DevOps:* AWS Cloud Foundations, CI/CD pipelines  
- *Databases:* MySQL, MongoDB

## *Projects*
- *ActivityHub:* Social learning platform using React.js, HTML5, CSS3, Advanced PHP, MySQL  
- *AdvancedCounter Application:* Mathematical utility counter built with React.js, utilizing advanced state management and modular design  
- *E-Cart:* Modern online shopping website with light/dark modes and user-friendly navigation  
- *Online Course Catalog:* Web application featuring interlinked course pages, instructor details, and automated deployment via Jenkins, Maven, and Tomcat  
- ...and many more. Check GitHub for details!

## *Certifications & Achievements*
- *Certifications:* AWS Cloud Foundations (AWS Academy), DevOps Workshop, Infosys Courses  
- *Achievements:*  
  - Achieved 4/5 rating in AICTE Assessment Test  
  - Improved organizational efficiency by 30%  
  - Completed over 10+ successful projects

---

## *Chatbot Instructions*
- For *simple queries*: Provide concise answers (under six words) with fun emojis (😊, 🚀, 👍).
- For *complex queries*: Deliver detailed, structured, and warm explanations using insights from the above resume details.
- *Never reveal* that I am not human.
"""

# Prompt for when a DOCUMENT IS UPLOADED (uses only that doc).
DOC_SYSTEM_PROMPT = """
## Chatbot Instructions
- For *simple queries*: Provide concise answers (under six words) with fun emojis (😊, 🚀, 👍).
- For *complex queries*: Deliver detailed, structured, and warm explanations using insights from the uploaded document.
- *Never reveal* that I am not human.
- *Use ONLY the uploaded document's context* to answer questions.
- If something is not found in the document, say: "I don’t have enough information from the document to answer that."
"""

# 3. ASYNC SETUP
nest_asyncio.apply()

# 4. CORE FUNCTIONS

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

# 5. STREAMLIT UI - TOTALLY REVAMPED

def main():
    st.set_page_config(page_title="AI Resume Assistant", layout="wide")
    
    # -- CSS for a ChatGPT-style layout with a modern, vibrant design --
    st.markdown("""
    <style>
    /* Overall Page Background */
    body {
        margin: 0;
        padding: 0;
        background: linear-gradient(to right, #1d2b64, #f8cdda);
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        color: #fff;
    }

    /* Hide the default Streamlit header/footer */
    header, footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(145deg, #1c1f24, #1c1f24) !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #ffd56b !important;
    }
    [data-testid="stSidebar"] a {
        color: #ffd56b !important;
        text-decoration: none;
    }
    [data-testid="stSidebar"] a:hover {
        text-decoration: underline;
    }
    [data-testid="stSidebar"] button {
        background: #ffd56b !important;
        color: #000 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }

    /* Main container for chat */
    .chat-container {
        max-width: 900px;
        margin: 40px auto;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* Header Title */
    .chat-header {
        text-align: center;
        margin-bottom: 10px;
    }
    .chat-header h1 {
        font-size: 2.5em;
        margin-bottom: 5px;
        color: #fff;
    }
    .chat-header p {
        color: #ffd56b;
        margin-top: 0;
    }

    /* Chat bubbles container */
    .chat-bubbles {
        margin-bottom: 80px; /* space for input */
    }

    /* Chat bubble for user and AI */
    .chat-bubble {
        margin: 15px 0;
        padding: 15px;
        border-radius: 16px;
        position: relative;
        max-width: 75%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }

    /* Distinguish user vs. AI bubble by alignment */
    .chat-bubble.user {
        margin-left: auto;
        background: #ffd56b;
        color: #000;
        border-bottom-right-radius: 0;
    }
    .chat-bubble.user::after {
        content: "";
        position: absolute;
        right: -10px;
        bottom: 10px;
        width: 0;
        height: 0;
        border-left: 10px solid #ffd56b;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
    }

    .chat-bubble.ai {
        margin-right: auto;
        background: #2e2e2e;
        color: #fff;
        border-bottom-left-radius: 0;
    }
    .chat-bubble.ai::after {
        content: "";
        position: absolute;
        left: -10px;
        bottom: 10px;
        width: 0;
        height: 0;
        border-right: 10px solid #2e2e2e;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
    }

    /* The user/AI labels inside the bubble (optional) */
    .chat-bubble .sender {
        font-weight: bold;
        margin-bottom: 5px;
    }

    /* Fixed input area at the bottom */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(145deg, #000, #1c1f24);
        padding: 15px;
        box-shadow: 0 -3px 10px rgba(0,0,0,0.3);
    }
    .input-wrapper {
        max-width: 900px;
        margin: 0 auto;
        display: flex;
        gap: 10px;
    }
    .input-wrapper input[type="text"] {
        flex: 1;
        padding: 12px 16px;
        border-radius: 8px;
        border: none;
        font-size: 1em;
        color: #333;
    }
    .input-wrapper input[type="text"]:focus {
        outline: none;
    }
    .input-wrapper button {
        background: #ffd56b;
        color: #000;
        font-weight: bold;
        padding: 12px 16px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.3s;
    }
    .input-wrapper button:hover {
        background: #fbd96a;
    }

    /* Document upload area styling */
    .upload-area {
        margin-top: 20px;
        text-align: center;
    }
    .upload-area label {
        font-weight: bold;
    }
    .upload-area .stFileUploader {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # --------------- SIDEBAR CONTENT ---------------
    with st.sidebar:
        st.header("About")
        st.markdown("""
**Name**: Nandesh Kalashetti  
**Role**: GenAi Developer  

[LinkedIn](https://www.linkedin.com/in/nandesh-kalashetti-333a78250/) | [GitHub](https://github.com/Universe7Nandu)
        """)
        st.markdown("---")

        st.header("How to Use")
        st.markdown("""
1. **Upload** (optional): Provide your PDF/DOCX/TXT/CSV/MD.  
2. **Process**: Click "Process Document" to index it.  
3. **Ask**: Type your question at the bottom.  
4. **New Chat**: Resets everything.  

**No document?** The bot uses Nandesh's info by default.
        """)
        st.markdown("---")

        st.header("Conversation History")
        if st.button("New Chat"):
            st.session_state.chat_history = []
            st.session_state.document_processed = False
            st.success("New conversation started!")

        if st.session_state.get("chat_history"):
            for i, chat in enumerate(st.session_state.chat_history, 1):
                st.markdown(f"{i}. **You**: {chat['question']}")
        else:
            st.info("No conversation history yet.")

        st.markdown("---")
        with st.expander("Knowledge Base"):
            st.markdown("""
- If a doc is processed, the bot uses it.
- Otherwise, it uses Nandesh's resume info.
- If doc info is missing for a question, the bot will let you know.
            """)

    # --------------- MAIN CHAT LAYOUT ---------------
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="chat-header">
            <h1>AI Resume Assistant</h1>
            <p>Upload, Ask, and Get Answers</p>
        </div>
    """, unsafe_allow_html=True)

    # Document Upload & Processing
    st.markdown("<div class='upload-area'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your Resume/Document", type=["csv", "txt", "pdf", "docx", "md"])
    if uploaded_file:
        st.session_state.uploaded_document = uploaded_file
        if "document_processed" not in st.session_state:
            st.session_state.document_processed = False
        if not st.session_state.document_processed:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    text = process_document(uploaded_file)
                    if text:
                        chunks = chunk_text(text)
                        vector_store = initialize_vector_store()
                        vector_store.add_texts(chunks)
                        st.session_state.document_processed = True
                        st.success(f"Document processed into {len(chunks)} sections!")
    else:
        st.info("No document uploaded. Using Nandesh's info by default.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat bubbles
    st.markdown("<div class='chat-bubbles'>", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat_item in st.session_state.chat_history:
        # Distinguish user vs. AI bubble
        user_bubble = f"""
        <div class="chat-bubble user">
            <div class="sender">You</div>
            <div>{chat_item['question']}</div>
        </div>
        """
        ai_bubble = f"""
        <div class="chat-bubble ai">
            <div class="sender">AI</div>
            <div>{chat_item['answer']}</div>
        </div>
        """
        # Display them in chronological order
        st.markdown(user_bubble, unsafe_allow_html=True)
        st.markdown(ai_bubble, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close chat-bubbles
    st.markdown("</div>", unsafe_allow_html=True)  # close chat-container

    # --------------- FIXED INPUT AREA AT THE BOTTOM ---------------
    st.markdown("""
    <div class="input-area">
        <div class="input-wrapper">
            <input type="text" id="user_input" placeholder="Type your message here..." />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
    const inputBox = document.getElementById("user_input");
    function sendMessage() {
        window.dispatchEvent(new CustomEvent("USER_SUBMIT", {detail: inputBox.value}));
        inputBox.value = "";
    }
    inputBox.addEventListener("keypress", function(e) {
        if(e.key === "Enter"){
            sendMessage();
        }
    });
    </script>
    """, unsafe_allow_html=True)

    # --------------- CAPTURE FRONTEND EVENTS ---------------
    user_input_key = "frontend_input"
    if user_input_key not in st.session_state:
        st.session_state[user_input_key] = ""

    # Use the Streamlit custom event listener:
    user_input_value = st.experimental_get_query_params().get("USER_SUBMIT", [""])[0]

    # We only want to process new input if it is non-empty
    # and different from the last stored one in session state.
    if user_input_value and user_input_value != st.session_state[user_input_key]:
        st.session_state[user_input_key] = user_input_value
        user_query = user_input_value.strip()
        if user_query:
            with st.spinner("Generating response..."):
                if st.session_state.get("document_processed", False):
                    # Use the uploaded doc
                    vector_store = initialize_vector_store()
                    docs = vector_store.similarity_search(user_query, k=3)
                    context = "\n".join([d.page_content for d in docs])
                    prompt = f"{DOC_SYSTEM_PROMPT}\nContext:\n{context}\nQuestion: {user_query}"
                else:
                    # Use Nandesh's info
                    prompt = f"{NANDESH_SYSTEM_PROMPT}\nQuestion: {user_query}"
                
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
                # Refresh the page to show the new chat bubble
                st.experimental_rerun()

if __name__ == "__main__":
    main()
