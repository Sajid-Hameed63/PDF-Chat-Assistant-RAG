from langchain_google_genai import GoogleGenerativeAIEmbeddings # type: ignore
import os
from dotenv import load_dotenv # type: ignore

load_dotenv()

# Load API key
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# Initialize Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

# Get the dimension of embeddings
sample_text = "Hello, world!"
embedding_vector = embedding_model.embed_query(sample_text)

print("Embedding Dimension:", len(embedding_vector))
