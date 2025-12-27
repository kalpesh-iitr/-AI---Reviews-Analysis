# Pizza Reviews Retriever

## Short overview
This repository builds a simple retrieval-based QA system over a dataset of restaurant (pizza) reviews.
It uses embeddings and a vector store to index reviews, then a language model to answer questions using the most relevant review snippets.

Key components found in the repository:
- `vector.py` — loads `realistic_restaurant_reviews.csv`, creates embeddings (OllamaEmbeddings) and a Chroma vector store, and exposes `retriever`.
- `main.py` — constructs an Ollama language model and a chat prompt template that expects review snippets. (The file appears partially incomplete — see *Notes* below.)
- `realistic_restaurant_reviews.csv` — the dataset of review records used to build the vector index.

## Local setup & dependencies

Recommended Python version: **3.10+**

Create and activate a virtual environment:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

Install the (suggested) dependencies:
```bash
pip install --upgrade pip
pip install pandas langchain-core langchain-ollama langchain-chroma chromadb ollama
```

If you prefer a `requirements.txt`, you can create one with the following content:
```
pandas
langchain-core
langchain-ollama
langchain-chroma
chromadb
ollama
```

**Notes on dependencies**
- This project uses Ollama libraries and expects access to Ollama models (local Ollama server or API). Make sure you have Ollama installed and the models referenced (`mxbai-embed-large`, `llama3.2`, or similar) available on your system.
- `langchain-chroma` / `chromadb` are used for the vector store; confirm your installed package names and versions if you encounter import errors.

## Data file
`realistic_restaurant_reviews.csv` — CSV with columns `Title,Date,Rating,Review`. Each row is a review; the project builds embeddings from the textual review content.

## How to build the vector index (what `vector.py` does)
1. `vector.py` reads `realistic_restaurant_reviews.csv` using `pandas`.
2. It instantiates Ollama embeddings and a Chroma vector store (persisted to `./chroma_langchain_db`).
3. If the DB directory does not exist, documents are added and persisted.
4. A `retriever` is exposed for use in the application; the retriever performs nearest-neighbor search (`k=7` by default).

To (re)build the index manually:
```bash
python vector.py
```

This will:
- create `./chroma_langchain_db` (if missing)
- compute embeddings for the reviews and persist them in the Chroma collection

## How to run (using `main.py`)
`main.py` sets up an `OllamaLLM` and defines a chat template that expects a `{reviews}` insertion. The file currently looks like it contains sample data and may be truncated. A minimal example to query the retriever and get an answer might look like:

```python
# example_run.py (not present in repo — create it)
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

prompt_template = """
You are an expert in answering questions about pizza restaurants.
You are given a question and some relevant reviews. Answer based on the reviews only.

Reviews:
{reviews}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# get top review docs
docs = retriever.get_relevant_documents("Which pizza has the best thin crust?")
reviews_text = "\n---\n".join([d.page_content for d in docs[:7]])

# run the LLM
response = model.generate([prompt.format_prompt(reviews=reviews_text, question="Which pizza has the best thin crust?").to_messages()])
print(response.generations)
```

Adjust to match the `langchain` API versions you have installed; APIs between `langchain_core` and `langchain` vary by version.



---

