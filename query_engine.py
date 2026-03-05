import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# We need the key to initialize Gemini
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set. Please set it in the .env file.")

from llama_index.core import PropertyGraphIndex, StorageContext, Settings, load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.prompts import PromptTemplate

# Configure Gemini for LlamaIndex globally
Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.3) 
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")

STORAGE_DIR = "./storage"

# Custom prompt to ensure the LLM acts as the digital clone
QA_PROMPT_TMPL = (
    "You are the digital clone of Anupam Patil, an AI & Bioinformatics Engineer.\n"
    "Your goal is to answer questions about 'yourself' (Anupam) based strictly on the provided context information.\n"
    "Speak in the first person ('I', 'my', 'me'). Be friendly, professional, and concise.\n"
    "If you don't know the answer based on the context, politely say that you don't have that information right now.\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and no prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt = PromptTemplate(QA_PROMPT_TMPL)

def get_query_engine():
    """Loads the persisted graph index and returns a configured query engine."""
    if not os.path.exists(STORAGE_DIR):
        raise FileNotFoundError(f"Storage directory '{STORAGE_DIR}' not found. Please run build_graph.py first.")
        
    logging.info(f"Loading property graph from {STORAGE_DIR}...")
    
    # Load storage context and index
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context=storage_context)
    
    # Set up the query engine
    # We use both graph traversal and vector search for robustness
    query_engine = index.as_query_engine(
        include_text=True,
        text_qa_template=qa_prompt
    )
    
    return query_engine

if __name__ == "__main__":
    # Simple test shell
    logging.basicConfig(level=logging.INFO)
    engine = get_query_engine()
    print("\n--- Digital Clone Initialized ---")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            response = engine.query(user_input)
            print(f"\nAnupam's Clone: {response}")
        except KeyboardInterrupt:
            break
