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
# Note: The Chroma class is deprecated in current langchain versions.
# For now we use it as before.
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. SQLITE3 PATCH (MUST BE FIRST)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    raise RuntimeError("Install pysqlite3-binary: pip install pysqlite3-binary")

# 2. CONFIGURATION
GROQ_API_KEY = "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"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Use a temporary persistent directory so that memory is cleared on new chat or reload
CHROMA_SETTINGS = {
    "persist_directory": "chroma_db_temp",
    "collection_name": "uploaded_doc_collection"
}

# SYSTEM PROMPTS
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
- Always remain professional, helpful, and friendly.
"""

DOC_SYSTEM_PROMPT = """
## Chatbot Instructions
- For *simple queries*: Provide concise answers (under six words) with fun emojis (😊, 🚀, 👍).
- For *complex queries*: Deliver detailed, structured, and warm explanations using only the uploaded document's context.
- *Never reveal* that I am not human.
- If information is missing from the document, say: "I don't have enough information from the document to answer that."
- Remain professional and helpful.
"""

# ASYNC SETUP
nest_asyncio.apply()

# CORE FUNCTIONS
def initialize_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_SETTINGS["persist_directory"],
        embedding_function=embeddings,
        collection_name=CHROMA_SETTINGS["collection_name"]
    )

def process_document(file):
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

# MAIN APP
def main():
    st.set_page_config(page_title="AI Resume Assistant", layout="wide")

    # --- ADVANCED CSS / UI ---
    st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap">
    <style>
    /* Main font & background */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    body {
        background: radial-gradient(circle at top left, #1d2b64, #f8cdda);
        margin: 0; padding: 0;
    }
    header, footer {visibility: hidden;}

    /* Chat container with glassy effect */
    .chat-container {
        max-width: 900px;
        margin: 40px auto 60px auto;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        animation: fadeIn 0.6s ease;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .chat-title {
        text-align: center;
        color: #fff;
        margin-bottom: 5px;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .chat-subtitle {
        text-align: center;
        color: #ffe6a7;
        margin-top: 0;
        margin-bottom: 20px;
        font-size: 1.1rem;
    }
    @keyframes fadeUp {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .element-container {
        animation: fadeUp 0.4s ease;
        margin-bottom: 20px !important;
    }
    /* Sidebar dark theme */
    [data-testid="stSidebar"] {
        background: #1c1f24 !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #ffd56b !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #fff !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        background: #ffd56b !important;
        color: #000 !important;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        transition: background 0.3s;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background: #fbd96a !important;
    }
    /* File uploader style override */
    .stFileUploader label div {
        background: #ffe6a7 !important;
        color: #000 !important;
        font-weight: 600;
        border-radius: 8px;
        cursor: pointer;
        padding: 8px 0;
        text-align: center;
        transition: background 0.3s;
    }
    .stFileUploader label div:hover {
        background: #ffd56b !important;
    }
    /* Pinned chat input with black text */
    .stChatInput {
        position: sticky;
        bottom: 0;
        background: rgba(28,31,36,0.85) !important;
        backdrop-filter: blur(6px);
        padding: 10px;
        margin-top: 20px;
        border-radius: 12px;
    }
    .stChatInput>div>div>input {
        color: #000 !important;
        font-weight: 500;
        border-radius: 8px;
        border: none;
    }
    .stChatInput>div>div>input:focus {
        outline: 2px solid #ffd56b !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ------------- SIDEBAR -------------
    with st.sidebar:
        st.header("About")
        # Replace the local image with an external placeholder image
        st.image("https://via.placeholder.com/150", width=150)
        st.markdown("""
**Name**: *Nandesh Kalashetti*  
**Role**: *GenAi Developer*  

[LinkedIn](https://www.linkedin.com/in/nandesh-kalashetti-333a78250/) | [GitHub](https://github.com/Universe7Nandu)
        """)
        st.markdown("---")
        st.header("How to Use ✨")
        st.markdown("""
1. **Upload** a document (PDF, DOCX, TXT, CSV, MD) *optional*.  
2. **Process** it with the **Process Document** button.  
3. **Ask** questions in the pinned chat box below.  
4. **New Chat** resets everything, clearing any uploaded document.

- If **no document** is processed, the bot uses **Nandesh’s** info.  
- If a document is processed, it uses *only* that document’s content.
        """)
        st.markdown("---")
        st.header("Conversation History")
        if st.button("New Chat", help="Clears conversation and uploaded document memory."):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("document_processed", None)
            st.success("New conversation started! 🆕")
        if "chat_history" in st.session_state and st.session_state["chat_history"]:
            for i, item in enumerate(st.session_state["chat_history"], start=1):
                st.markdown(f"{i}. **You**: {item['question']}")
        else:
            st.info("No conversation history yet. Start chatting!")

    # ------------- MAIN CHAT CONTAINER -------------
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='chat-title'>AI Resume Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='chat-subtitle'>Document Upload • ChatGPT‐Style Q&A • Temporary Memory</p>", unsafe_allow_html=True)

    # Document Upload & Processing
    uploaded_file = st.file_uploader("Upload Document (CSV/TXT/PDF/DOCX/MD)",
                                     type=["csv", "txt", "pdf", "docx", "md"])
    if uploaded_file:
        if "document_processed" not in st.session_state or not st.session_state["document_processed"]:
            if st.button("Process Document"):
                with st.spinner("Reading & Embedding your document..."):
                    text = process_document(uploaded_file)
                    if text:
                        chunks = chunk_text(text)
                        vector_store = initialize_vector_store()
                        vector_store.add_texts(chunks)
                        st.session_state["document_processed"] = True
                        st.success(f"Document processed into {len(chunks)} sections! ✅")
    else:
        st.info("No document uploaded. Using Nandesh's info by default.")

    # Initialize chat history if not already present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display existing chat messages
    for message_data in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.markdown(message_data["question"])
        with st.chat_message("assistant"):
            st.markdown(message_data["answer"])

    st.markdown("</div>", unsafe_allow_html=True)  # End chat container

    # -------------- Chat Input (Pinned at Bottom) --------------
    user_query = st.chat_input("Type your message here... (Press Enter)")
    if user_query:
        st.session_state["chat_history"].append({"question": user_query, "answer": ""})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Thinking... 🤔"):
            # If a document is processed, use its context; otherwise use Nandesh's info
            if st.session_state.get("document_processed"):
                vector_store = initialize_vector_store()
                docs = vector_store.similarity_search(user_query, k=3)
                context = "\n".join([d.page_content for d in docs])
                prompt = f"{DOC_SYSTEM_PROMPT}\nContext:\n{context}\nQuestion: {user_query}"
            else:
                prompt = f"{NANDESH_SYSTEM_PROMPT}\nQuestion: {user_query}"
            llm = ChatGroq(
                temperature=0.7,
                groq_api_key=GROQ_API_KEY,
                model_name="mixtral-8x7b-32768"
            )
            response = asyncio.run(llm.ainvoke([{"role": "user", "content": prompt}]))
            bot_answer = response.content

        st.session_state["chat_history"][-1]["answer"] = bot_answer
        with st.chat_message("assistant"):
            st.markdown(bot_answer)

if __name__ == "__main__":
    main()
