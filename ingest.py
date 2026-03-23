from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF_PATH = "dFAQ Python Video YouTube.pdf"
CHROMA_PATH = "db/chroma"

def load_documents(path: str):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vector_store

def main():
    documents = load_documents(PDF_PATH)
    print(f"Documentos carregados: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Chunks gerados: {len(chunks)}")

    create_vector_store(chunks)
    print("Banco vetorial criado com sucesso.")

if __name__ == "__main__":
    main()