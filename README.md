
# 🤖 Generative AI Chatbot by Nandesh Kalashetti

[![Watch Demo](chatbot.png)](https://youtu.be/V0ffoYtm_Uk)

## Overview

Welcome to the **Generative AI Chatbot** repository! This project is a personal chatbot that showcases the expertise and innovative mindset of **Nandesh Kalashetti** – a visionary full‑stack web developer, mentor, and tech enthusiast. Leveraging state‑of‑the‑art techniques like **Retrieval Augmented Generation (RAG)**, **advanced prompt engineering**, and **optimized inference**, this chatbot dynamically integrates data from your uploaded document (e.g., resume) to deliver accurate, context‑aware, and human‑like responses. Get ready for an engaging experience! 😊🚀

## Key Features

- **📄 Multi-Format Knowledge Base:**  
  Seamlessly extract and index key details from your resume or any document (CSV, TXT, PDF, DOCX, MD) covering your education, technical skills, projects, certifications, and achievements.

- **🧠 Advanced AI Models:**  
  - Utilizes the **sentence-transformers/all‑MiniLM‑L6‑v2** model for high‑quality semantic embeddings.  
  - Powered by the **Groq Chat model (mixtral‑8x7b‑32768)** to generate responses that are both precise and natural.

- **💬 Adaptive & Context-Aware:**  
  - **Simple queries:** Enjoy concise, emoji‑enhanced answers.  
  - **Complex queries:** Receive detailed, structured explanations enriched with insights from your uploaded document.

- **📚 Intelligent Retrieval:**  
  Retrieves the most relevant context from a vector database (ChromaDB) to ensure every answer is tailored and accurate.

- **✨ Modern & Engaging UI:**  
  Built with **Streamlit** and styled with custom CSS for a sleek, glassmorphic look—complete with smooth transitions, hover effects, and a professional, user‑friendly interface.

- **🔄 Conversation Management:**  
  Keep track of your conversation history, and start a fresh chat anytime with the “New Chat” button.

## How It Works

1. **Data Ingestion:**  
   Upload your knowledge base document (resume or any related file). The chatbot extracts text, splits it into manageable chunks, and indexes it in ChromaDB using semantic embeddings.

2. **Query Processing:**  
   When you ask a question, the chatbot retrieves the most relevant context from the indexed data and combines it with your query.

3. **Response Generation:**  
   A sophisticated prompt—enriched with your personal details—guides the AI to generate responses that are precise, empathetic, and tailored to your needs, all while maintaining a warm, human tone. 👍

## Getting Started

### Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/YourRepo.git
   cd YourRepo
   ```

2. **Create & Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Chatbot:**
   ```bash
   streamlit run app.py
   ```
   Open the provided URL (usually `http://localhost:8501`) in your browser.

## Demo Video

Watch the demo video on YouTube to see the chatbot in action:

[![Watch Demo](chatbot.png)](https://youtu.be/V0ffoYtm_Uk)

## How to Use

1. **Upload Your Document:**  
   In the left panel, upload your resume or any document (CSV, TXT, PDF, DOCX, MD) containing your personal details. This enriches the chatbot’s responses with context!

2. **Process the Document:**  
   Click **Process Document** to extract and index the content from your file.

3. **Ask a Question:**  
   Type your question in the chat box.  
   - **For simple queries:** Get short, fun responses with emojis.  
   - **For complex queries:** Receive detailed, context-rich explanations!

4. **Manage Conversations:**  
   View your conversation history in the sidebar and start a new conversation anytime with the **New Chat** button.

## Why This Chatbot?

- **Innovative Integration:** Combines cutting-edge AI techniques with your personal data to deliver context-rich responses.  
- **User-Centric Design:** Offers an adaptive experience with both short and detailed answers tailored to your needs.  
- **Engaging & Modern:** Designed with a sleek UI and dynamic effects to make interactions delightful and professional.  
- **Future-Ready:** Demonstrates real-world applicability for personal branding, career growth, and professional presentations.

## Additional Information

- **Knowledge Base:** The more detailed your document, the richer the insights in the responses.  
- **Customization:** You can modify the system prompt and UI CSS to further personalize the chatbot.  
- **Support:** For any questions or enhancements, please feel free to open an issue or submit a pull request.

---



