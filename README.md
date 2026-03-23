# 🧠 RAG Assistant --- LangChain + Chroma + OpenAI

```{=html}
<p align="center">
```
`<b>`{=html}Production-ready Retrieval-Augmented Generation (RAG)
pipeline`</b>`{=html}`<br>`{=html} Built with Python, LangChain, Chroma
and OpenAI
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Python-3.12+-blue.svg"/>`{=html}
`<img src="https://img.shields.io/badge/LangChain-RAG-green"/>`{=html}
`<img src="https://img.shields.io/badge/Chroma-VectorDB-purple"/>`{=html}
`<img src="https://img.shields.io/badge/OpenAI-gpt--4o--mini-black"/>`{=html}
`<img src="https://img.shields.io/badge/Status-Active-success"/>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 🚀 Overview

This project implements a **complete RAG (Retrieval-Augmented
Generation) system**, enabling users to query PDF documents using
natural language and receive **accurate, context-aware answers**.

Unlike traditional LLM usage, this system: - Retrieves **relevant
knowledge dynamically** - Grounds responses in **real documents** -
Reduces hallucinations through **controlled prompting**

------------------------------------------------------------------------

## 🧠 Architecture

    User Question
          ↓
    Embedding (OpenAI)
          ↓
    Similarity Search (Chroma)
          ↓
    Top-K Relevant Chunks
          ↓
    Context Injection (Prompt)
          ↓
    LLM (gpt-4o-mini)
          ↓
    Final Answer

------------------------------------------------------------------------

## 🎯 Key Features

-   📄 Multi-PDF ingestion pipeline
-   ✂️ Smart chunking strategy
-   🔎 Semantic similarity search
-   🎯 Relevance filtering (score threshold)
-   🧠 Context-aware prompting
-   ❌ Hallucination control
-   💬 CLI interface
-   ⚡ Ready for API / Streamlit

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python\
-   LangChain\
-   ChromaDB\
-   OpenAI (Embeddings + Chat)\
-   python-dotenv\
-   uv (dependency manager)

------------------------------------------------------------------------

## 📦 Installation

``` bash
git clone https://github.com/MarcioLuizBR/rag-assistant-langchain.git
cd rag-assistant-langchain
uv sync
```

Create `.env` file:

    OPENAI_API_KEY=your_key_here

------------------------------------------------------------------------

## ⚙️ Usage

### 1. Ingest documents

    uv run ingest.py

### 2. Ask questions

    uv run main.py

------------------------------------------------------------------------

## 🧪 Example

    Question:
    What is the main topic of the document?

    Answer:
    The document discusses...

------------------------------------------------------------------------

## ⚠️ RAG Best Practices Applied

-   Temperature = 0 (deterministic answers)
-   Context-only answering (no external knowledge)
-   Similarity threshold filtering
-   Structured prompt engineering
-   Fallback for low-confidence retrieval

------------------------------------------------------------------------

## 📈 Roadmap

-   [ ] API with FastAPI
-   [ ] Streamlit UI
-   [ ] Source attribution in responses
-   [ ] MMR retrieval
-   [ ] LangSmith observability
-   [ ] Cloud deployment

------------------------------------------------------------------------

## 👨‍💻 Author

**Marcio Luiz**

Python \| Data \| AI \| Cloud (Azure)

🔗 https://github.com/MarcioLuizBR

------------------------------------------------------------------------

## ⭐ If you found this useful

Give it a ⭐ and help this project grow!
