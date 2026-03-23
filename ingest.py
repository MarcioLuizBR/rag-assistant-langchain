# =========================
# 📦 Imports
# =========================

# Loader para ler múltiplos PDFs de um diretório
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Responsável por dividir o texto em partes menores (chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Banco vetorial (armazenamento dos embeddings)
from langchain_chroma.vectorstores import Chroma

# Modelo de embeddings (transforma texto em vetores)
from langchain_openai import OpenAIEmbeddings

# Carrega variáveis de ambiente (.env)
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (ex: OPENAI_API_KEY)
load_dotenv()


# =========================
# 📁 Configuração
# =========================

# Pasta onde estão os arquivos PDF
PASTA_BASE = "base"


# =========================
# 🚀 Pipeline principal
# =========================

def criar_db():
    """
    Função principal que executa todo o pipeline de ingestão:
    
    1. Carrega os documentos PDF
    2. Divide os documentos em chunks menores
    3. Gera embeddings e armazena no banco vetorial
    """
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)


# =========================
# 📄 Etapa 1: Carregamento
# =========================

def carregar_documentos():
    """
    Carrega todos os arquivos PDF da pasta definida em PASTA_BASE.

    Utiliza o PyPDFDirectoryLoader para ler múltiplos arquivos automaticamente.
    
    Retorna:
        lista de documentos carregados
    """
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
    documentos = carregador.load()
    return documentos


# =========================
# ✂️ Etapa 2: Chunking
# =========================

def dividir_chunks(documentos):
    """
    Divide os documentos em partes menores (chunks).

    Isso é necessário porque LLMs possuem limite de tokens e
    o RAG funciona melhor com pedaços menores de contexto.

    Parâmetros importantes:
    - chunk_size: tamanho de cada pedaço de texto
    - chunk_overlap: sobreposição entre chunks para manter contexto
    - add_start_index: adiciona índice para rastrear origem do trecho

    Retorna:
        lista de chunks (documentos fragmentados)
    """
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = separador_documentos.split_documents(documentos)
    return chunks


# =========================
# 🧠 Etapa 3: Vetorização
# =========================

def vetorizar_chunks(chunks):
    """
    Converte os chunks em embeddings e armazena no banco vetorial (Chroma).

    Fluxo:
    1. Cada chunk é transformado em vetor (embedding)
    2. Os vetores são armazenados no banco vetorial
    3. O banco é persistido no diretório "db"

    Isso permitirá buscas semânticas posteriormente.
    """
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory="db",
    )
    print("Vetorização concluída e banco de dados criado.")


# =========================
# ▶️ Execução direta
# =========================

if __name__ == "__main__":
    criar_db()