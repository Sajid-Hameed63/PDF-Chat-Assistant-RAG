import os
import streamlit as st # type: ignore
from PyPDF2 import PdfReader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # type: ignore
from langchain_pinecone import PineconeVectorStore # type: ignore
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain # type: ignore
from langchain.chains import create_retrieval_chain # type: ignore
from langchain.prompts import PromptTemplate #type: ignore
from dotenv import load_dotenv # type: ignore
from pinecone import Pinecone, ServerlessSpec # type: ignore
import logging

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "qa-docs-langchain-pinecone-gemini"

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs with error handling"""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
            logging.info(f"Processed PDF: {pdf.name}")
        return text
    except Exception as e:
        logging.error(f"Error processing PDFs: {str(e)}")
        st.error(f"Error reading PDF files: {str(e)}")
        return None

def get_text_chunks(text, chunk_size=2000, chunk_overlap=300):
    """Split text into manageable chunks with validation"""
    try:
        if not text:
            raise ValueError("Empty text content")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)
    except Exception as e:
        logging.error(f"Error splitting text: {str(e)}")
        st.error(f"Error processing document content: {str(e)}")
        return None

def initialize_pinecone_index():
    """Create Pinecone index with error handling"""
    try:
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logging.info("Created new Pinecone index")
        else:
            logging.info("Using existing Pinecone index")
        return True
    except Exception as e:
        logging.error(f"Pinecone index error: {str(e)}")
        st.error(f"Failed to initialize Pinecone index: {str(e)}")
        return False

def store_embeddings(text_chunks):
    """Store embeddings in Pinecone with error handling"""
    try:
        if not text_chunks:
            raise ValueError("No text chunks to process")
            
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        
        PineconeVectorStore.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        logging.info("Successfully stored embeddings")
        return True
    except Exception as e:
        logging.error(f"Embedding storage error: {str(e)}")
        st.error(f"Failed to process documents: {str(e)}")
        return False 

def get_qa_chain():
    """Create QA chain with error handling"""
    try:
        prompt_template = """
        Answer based on the context. If query is not related to the context, tell them to ask questions related to their documents.
        
        Context: {context}
        
        Question: {input}
        
        Answer:"""
        
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-002",
            temperature=0.5,
            google_api_key=GEMINI_API_KEY
        )
        
        prompt = PromptTemplate.from_template(prompt_template)
        return create_stuff_documents_chain(model, prompt)
    except Exception as e:
        logging.error(f"QA chain creation error: {str(e)}")
        st.error("Failed to initialize question answering system")
        return None

def handle_query(user_query):
    """Handle user query with comprehensive error handling"""
    try:
        if not user_query:
            raise ValueError("Empty query")
            
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        
        retriever = vector_store.as_retriever()
        document_chain = get_qa_chain()
        
        if not document_chain:
            return "System error - please try again later"
            
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_query})
        return response["answer"]
    
    except Exception as e:
        logging.error(f"Query handling error: {str(e)}")
        return f"Error processing your query: {str(e)}"

def main():
    """Main Streamlit app with error handling"""
    try:
        st.set_page_config(page_title="PDF Chat Assistant", page_icon="📚")
        st.header("Chat with Your Documents")

        # Initialize session state
        if "history" not in st.session_state:
            st.session_state.history = []

        # Sidebar controls
        with st.sidebar:
            st.subheader("Document Management")
            pdf_files = st.file_uploader(
                "Upload PDF files",
                type="pdf",
                accept_multiple_files=True
            )
            
            if st.button("Process Documents"):
                if pdf_files:
                    with st.spinner("Analyzing documents..."):
                        raw_text = get_pdf_text(pdf_files)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            if text_chunks and initialize_pinecone_index():
                                if store_embeddings(text_chunks):
                                    st.success("Documents processed successfully!")
                else:
                    st.warning("Please upload PDF files first")

            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

        # Main chat interface
        user_input = st.chat_input("Ask about your documents:")
        
        if user_input:
            with st.spinner("Generating answer..."):
                answer = handle_query(user_input)
                st.session_state.history.append({
                    "question": user_input,
                    "answer": answer
                })
                st.rerun()

        # Display chat history
        for entry in st.session_state.history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.write(entry["answer"])
                
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        st.error("A critical error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()

