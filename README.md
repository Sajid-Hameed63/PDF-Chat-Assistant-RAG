# 📚 PDF Chat Assistant

A Streamlit-based AI chatbot that allows users to upload PDF documents, process them into embeddings using Google Gemini AI and Pinecone, and ask questions about their documents interactively.

---

## 🚀 Features
- **Upload and Process PDFs**: Extracts text from PDF files.
- **Text Chunking**: Uses `RecursiveCharacterTextSplitter` to split text into meaningful chunks.
- **Embeddings with Google Gemini AI**: Generates vector embeddings for efficient retrieval.
- **Pinecone Vector Database**: Stores and retrieves document embeddings.
- **AI-Powered Q&A**: Answers user queries based on uploaded document content.
- **Error Handling & Logging**: Ensures smooth execution with detailed logs.

---

## 🛠️ Setup & Installation

### 1️⃣ Clone the Repository
```sh
 git clone https://github.com/Sajid-Hameed63/PDF-Chat-Assistant-RAG.git
 cd PDF-Chat-Assistant-RAG
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```sh
 python -m venv venv
 source venv/bin/activate  # On macOS/Linux
 venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
```sh
 pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the root directory and add your API keys:
```ini
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## 🏃‍♂️ Running the Application
```sh
 docker build -t rag-app-gemini .
 docker run -p 8500:8500 --env-file .env rag-app-gemini
```

---

## 🔍 How It Works
1. **Upload PDFs**: Users upload PDFs through the Streamlit interface.
2. **Text Extraction**: Extracts text using `PyPDF2`.
3. **Chunking & Embedding**:
   - Splits text into chunks.
   - Generates embeddings using Google Gemini AI.
   - Stores embeddings in Pinecone for retrieval.
4. **Query Processing**:
   - User asks a question.
   - Retrieves relevant document chunks from Pinecone.
   - Generates AI-powered responses using Gemini AI.

---

**Developed by [Sajid Hameed]** 🚀
