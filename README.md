# RAG Q&A Agent (LangGraph + Chroma + Gemini)

A Retrieval-Augmented Generation (RAG) agent that answers questions from a small knowledge base using:
- **LangGraph** for the agent workflow (plan → retrieve → answer → reflect)
- **ChromaDB** for vector search
- **Sentence-Transformers** for embeddings
- **Gemini** (via `langchain-google-genai`) for answer generation and reflection
- **Streamlit** for a simple chat UI
- **LangSmith** for tracing 

---
## How It Works

**Workflow (LangGraph):**
1. **plan** – decides if retrieval is needed (simple keyword matching logic)
2. **retrieve** – pulls top-3 chunks from Chroma
3. **answer** – calls Gemini with question + retrieved context
4. **reflect** – evaluates answer quality based on relevance, accuracy of the retrieved context
and provides feedback and score 

**UI (Streamlit):**
- Chat interface
- Shows quality score
- Expandable panel with retrieved context snippets
---

## Project Structure
```
.
├─ agent.py              # LangGraph workflow (plan → retrieve → answer → reflect)
├─ ingest.py             # Ingestion pipeline (load → chunk → embed → store in Chroma)
├─ app.py                # Streamlit UI
├─ requirements.txt
├─ data/                 # Place your .txt / .pdf files here
└─ local_vector_store/   # Chroma persistence (created by ingest.py)
```

## Setup

1) **Initalize uv**
```bash/terminal
uv init
```

2) **Create & activate venv**
```bash/terminal
uv venv
.venv\Scripts\activate
```

3) **Install requirements**
```bash/terminal
pip install -r requirements.txt
```

4) **Environment variables** – create a `.env` in project root:
```
GOOGLE_API_KEY=your_gemini_api_key
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

---

## Ingest Documents

Place your `.txt` / `.pdf` files in `data/` (e.g., `renewable_energy.txt`), then run:
```bash
python ingest.py
```
This will create/refresh `local_vector_store/` with embeddings.

---

## Run the App
```bash
streamlit run app.py
```
Open the local URL shown in the terminal and start chatting.

---


