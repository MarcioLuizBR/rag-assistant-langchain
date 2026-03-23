# 🧠 RAG Assistant — LangChain + Chroma + OpenAI

<p align="center">
  <strong>Production-ready Retrieval-Augmented Generation (RAG) pipeline</strong><br>
  Built with Python, LangChain, Chroma and OpenAI
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/LangChain-RAG-green" alt="LangChain RAG"/>
  <img src="https://img.shields.io/badge/Chroma-VectorDB-purple" alt="Chroma VectorDB"/>
  <img src="https://img.shields.io/badge/OpenAI-gpt--4o--mini-black" alt="OpenAI gpt-4o-mini"/>
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status Active"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License"/>
</p>

---

## 🚀 Overview

This project implements a complete **RAG (Retrieval-Augmented Generation)** pipeline that allows users to ask questions about PDF documents using natural language and receive answers grounded in the retrieved context.

Instead of relying only on the model’s internal knowledge, the system first searches a vector database for the most relevant chunks, injects that context into the prompt, and then generates a response based on the retrieved information.

This approach helps create answers that are:

- more precise
- more contextualized
- more reliable
- less prone to hallucination

---

## ✨ Highlights

- 📄 Multi-PDF ingestion pipeline
- ✂️ Text chunking with `RecursiveCharacterTextSplitter`
- 🔢 Embeddings with OpenAI
- 🗄️ Vector storage with Chroma
- 🔎 Semantic retrieval with relevance score filtering
- 🧠 Prompt-based grounded answering
- ❌ Reduced hallucination risk through context restriction
- 💬 Simple CLI interface for testing
- ⚡ Ready to evolve into API or Streamlit app

---

## 🧠 How It Works

The project is divided into two main stages:

### 1. Ingestion Pipeline

The ingestion step reads PDF files, splits them into smaller chunks, generates embeddings, and stores them in Chroma.

### 2. Query Pipeline

When the user asks a question:

1. the question is converted into an embedding
2. the system searches for the most relevant chunks in Chroma
3. low-quality matches can be filtered out by relevance score
4. the retrieved content is assembled into context
5. the LLM receives the question plus the retrieved context
6. the final answer is generated based on that context

---

## 🏗️ Architecture

```text
PDF Documents (base/)
        ↓
      ingest.py
        ↓
RecursiveCharacterTextSplitter
        ↓
OpenAIEmbeddings
        ↓
 Chroma Vector Database (db/)
        ↓
       main.py
        ↓
   User Question
        ↓
 Similarity Search
        ↓
 Relevant Chunks
        ↓
 Prompt + Context
        ↓
 ChatOpenAI
        ↓
   Final Answer
```

---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **langchain-openai**
- **langchain-community**
- **langchain-chroma**
- **langchain-text-splitters**
- **ChromaDB**
- **OpenAI**
- **python-dotenv**
- **uv**

---

## 📁 Project Structure

```text
rag-assistant-langchain/
│
├── base/                  # PDF files used as knowledge base
├── db/                    # Persisted Chroma vector database
├── ingest.py              # Ingestion pipeline
├── main.py                # Query pipeline
├── .env                   # Environment variables
├── pyproject.toml         # Project dependencies
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/MarcioLuizBR/rag-assistant-langchain.git
cd rag-assistant-langchain
```

### 2. Install dependencies

Using **uv**:

```bash
uv sync
```

If you prefer pip:

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🚀 Usage

### Step 1 — Add your PDF files

Place your documents inside the `base/` folder.

### Step 2 — Run the ingestion pipeline

```bash
uv run ingest.py
```

This step will:

- load PDF files
- split them into chunks
- generate embeddings
- create or recreate the vector database

### Step 3 — Run the query pipeline

```bash
uv run main.py
```

Then type your question in the terminal.

---

## 💬 Example Flow

```text
Escreva sua pergunta:
Qual é o tema principal do documento?

Resposta da IA:
O documento trata principalmente de ...
```

---

## 🧪 Best Practices Applied

This project already includes important RAG-oriented decisions:

- **temperature = 0** for more deterministic answers
- relevance score check before using retrieved chunks
- context-based prompt to reduce unsupported answers
- separation between ingestion and query stages
- simple structure that supports future refactoring

---

## ⚠️ Current Limitations

This is an educational and portfolio-oriented project, so there are still possible improvements such as:

- source citation in the final answer
- better retrieval strategies such as MMR
- metadata display for retrieved chunks
- conversational memory
- evaluation and observability
- UI or API layer

---

## 🛣️ Roadmap

- [ ] Show source file and page in the answer
- [ ] Add structured output for retrieved chunks
- [ ] Test MMR retrieval for more diverse context
- [ ] Build API with FastAPI
- [ ] Build interface with Streamlit
- [ ] Add LangSmith observability
- [ ] Prepare cloud deployment version

---

## 📚 Why This Project Matters

RAG is one of the most practical ways to connect LLMs with real business data.

This project demonstrates applied skills in:

- LLM application development
- vector databases
- embeddings
- semantic retrieval
- prompt engineering
- Python-based AI workflows

It is a solid foundation for future evolutions such as internal assistants, document Q&A systems, AI copilots, and domain-specific knowledge agents.

---

## 👨‍💻 Author

**Marcio Luiz**

Python • Data • AI • Cloud

- GitHub: https://github.com/MarcioLuizBR

---

## ⭐ Support

If this project helped you or inspired your own implementation, consider giving it a star.
