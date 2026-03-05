import os
import logging
import sys
from dotenv import load_dotenv

# Set up logging to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set. Please set it in the .env file.")

from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Configure Gemini for LlamaIndex
Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")

DATA_FILE = "portfolio_data.md"
STORAGE_DIR = "./storage"

def build_graph():
    logging.info("Reading portfolio data...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Ensure the portfolio extraction step was completed.")
        
    documents = SimpleDirectoryReader(input_files=[DATA_FILE]).load_data()
    
    logging.info(f"Loaded {len(documents)} document(s). Building Property Graph Index. This might take a minute depending on API limits...")
    
    # We use PropertyGraphIndex which natively prompts the LLM to extract entities and relations 
    # and stores them alongside vector embeddings.
    index = PropertyGraphIndex.from_documents(
        documents,
        embed_model=Settings.embed_model,
        llm=Settings.llm,
        show_progress=True
    )
    
    logging.info("Graph constructed successfully. Persisting to disk...")
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    logging.info(f"Graph persisted to {STORAGE_DIR}.")

if __name__ == "__main__":
    build_graph()
