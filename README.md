# Anupam's Digital Clone (Graph-Based RAG)

A Graph-based Retrieval-Augmented Generation (RAG) chatbot utilizing LlamaIndex, Google Gemini GenAI, and Streamlit. It acts as a digital clone, answering questions about Anupam Patil's projects, skills, courses, and educational achievements based on his extracted knowledge base.

## Features
- **Graph-Based Retrieval:** Uses LlamaIndex's `PropertyGraphIndex` to extract entities and relations alongside vector embeddings.
- **LLM Engine:** Powered by Google's `gemini-2.5-flash` model.
- **Interactive UI:** Built with Streamlit for an intuitive chatbot experience.
- **CLI Mode:** Includes a simple test shell to query the model through the terminal.

## Prerequisites
- Python 3.8+
- [Google Gemini API Key](https://aistudio.google.com/app/apikey)

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnupamPatil899/Hybrid-Graph-RAG-Personal-AI-Assistant.git
   cd Graph_Based_RAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GOOGLE_API_KEY="your-gemini-api-key-here"
   ```

## Usage

### 1. Build the Knowledge Graph
Before querying, you need to extract the entities and relationships from the dataset (`portfolio_data.md`) and persist them. Run the following command:
```bash
python build_graph.py
```
> *Note: This will read `portfolio_data.md` and create a local `storage` folder containing the indexed property graph. It may take roughly a minute depending on API limits.*

### 2. Run the Streamlit Application
Launch the chatbot web interface:
```bash
streamlit run app.py
```

### 3. Command Line Interface (CLI) Testing
If you prefer not to start the UI, you can chat directly with the clone in your terminal:
```bash
python query_engine.py
```

## Project Structure
- **`app.py`**: The Streamlit web application script.
- **`build_graph.py`**: Script to generate and persist the Property Graph Index from local data.
- **`query_engine.py`**: Handles loading the stored graph and configures the querying logic + custom QA prompts to steer the LLM.
- **`portfolio_data.md`**: The source knowledge base (markdown file extracted from the portfolio).
- **`requirements.txt`**: Required Python packages.
