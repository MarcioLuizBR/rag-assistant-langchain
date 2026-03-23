# =========================
# 📦 Imports
# =========================

# Banco vetorial usado para armazenar e consultar embeddings
from langchain_chroma.vectorstores import Chroma

# Modelo de embeddings da OpenAI
from langchain_openai import OpenAIEmbeddings

# Template de prompt para estruturar a entrada do modelo
from langchain_core.prompts import ChatPromptTemplate

# Modelo de chat da OpenAI
from langchain_openai import ChatOpenAI

# Carrega variáveis de ambiente do arquivo .env
from dotenv import load_dotenv


# =========================
# ⚙️ Configuração inicial
# =========================

# Carrega variáveis de ambiente, como OPENAI_API_KEY
load_dotenv()

# Caminho da pasta onde o banco vetorial Chroma foi persistido
CAMINHO_DB = "db"


# Prompt base usado para orientar a resposta do modelo
prompt_template = """
Você é um assistente que responde perguntas com base exclusivamente no contexto fornecido.

Regras:
- Responda apenas com base nas informações da base de conhecimento.
- Não invente fatos ou complete lacunas com suposições.
- Se a resposta não estiver clara no contexto, diga:
  "Não encontrei informação suficiente na base para responder com segurança."

Pergunta do usuário:
{pergunta}

Base de conhecimento:
{base_conhecimento}

Resposta:
"""


# =========================
# 🚀 Função principal
# =========================

def perguntar():
    """
    Executa o fluxo principal de consulta do sistema RAG.

    Etapas:
    1. Recebe a pergunta do usuário.
    2. Carrega o banco vetorial persistido no Chroma.
    3. Busca os trechos mais relevantes com base em similaridade.
    4. Monta o contexto com os chunks recuperados.
    5. Cria o prompt final.
    6. Envia o prompt para o modelo da OpenAI.
    7. Exibe a resposta gerada.
    """

    # Solicita a pergunta ao usuário
    pergunta = input("Escreva sua pergunta: ").strip()

    # Validação simples para evitar consulta vazia
    if not pergunta:
        print("Você precisa digitar uma pergunta válida.")
        return

    # =========================
    # 📚 Carregar banco vetorial
    # =========================

    # Cria a função de embeddings usada pelo Chroma para comparar a pergunta com os documentos
    funcao_embedding = OpenAIEmbeddings()

    # Carrega o banco vetorial já persistido na pasta "db"
    db = Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=funcao_embedding
    )

    # =========================
    # 🔎 Buscar trechos relevantes
    # =========================

    # Busca os 3 chunks mais relevantes para a pergunta
    # Cada item retornado contém:
    # - o documento recuperado
    # - o score de relevância
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)

    # Se não houver resultados ou se o melhor resultado for fraco, interrompe a resposta
    if len(resultados) == 0 or resultados[0][1] < 0.7:
        print("Não consegui encontrar alguma informação relevante na base de dados.")
        return

    # =========================
    # 🧩 Montar base de conhecimento
    # =========================

    # Lista que armazenará os textos recuperados
    textos_resultado = []

    # Extrai apenas o conteúdo textual de cada documento retornado
    for doc, score in resultados:
        textos_resultado.append(doc.page_content)

    # Junta os trechos em um único bloco de contexto, separado visualmente
    base_conhecimento = "\n\n-----\n\n".join(textos_resultado)

    # =========================
    # 📝 Construir prompt
    # =========================

    # Cria um template estruturado para enviar pergunta + contexto ao modelo
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Preenche o template com a pergunta do usuário e os trechos recuperados
    prompt_formatado = prompt.invoke({
        "pergunta": pergunta,
        "base_conhecimento": base_conhecimento
    })

    # =========================
    # 🤖 Chamar o modelo
    # =========================

    # Inicializa o modelo com temperatura 0 para respostas mais estáveis
    modelo = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # Envia o prompt ao modelo e obtém a resposta final
    texto_resposta = modelo.invoke(prompt_formatado).content

    # =========================
    # 📤 Exibir resposta
    # =========================

    print("\nResposta da IA:")
    print(texto_resposta)


# =========================
# ▶️ Ponto de entrada
# =========================

if __name__ == "__main__":
    perguntar()