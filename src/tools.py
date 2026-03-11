import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from src.config import ARQUIVO_CALENDARIO, DATA_DIR

def preparar_arquivos_teste():
    """Garante que a pasta 'data' exista."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ARQUIVO_CALENDARIO):
        with open(ARQUIVO_CALENDARIO, "w", encoding="utf-8") as f:
            f.write("CALENDÁRIO VAZIO")

def configurar_ferramenta_rag(api_key: str): 
    chave_limpa = api_key.strip().strip("'").strip('"')
    os.environ["GOOGLE_API_KEY"] = chave_limpa
    
    preparar_arquivos_teste()

    # --- FERRAMENTA 1: O CALENDÁRIO (Lê tudo de uma vez, 100% de precisão) ---
    @tool
    def consultar_calendario() -> str:
        """Usa esta ferramenta SEMPRE que precisarem saber qual é o livro de um mês específico, qual é a leitura atual, ou as regras do clube."""
        if os.path.exists(ARQUIVO_CALENDARIO):
            with open(ARQUIVO_CALENDARIO, 'r', encoding='utf-8') as f:
                return f.read()
        return "Calendário não encontrado."
    
    # --- FERRAMENTA 2: OS PDFs (Fatia os livros em páginas e acha citações) ---
    documentos_pdf = []
    loader_pdf = PyPDFDirectoryLoader(DATA_DIR)
    documentos_pdf.extend(loader_pdf.load())
    
    if not documentos_pdf:
        from langchain_core.documents import Document
        documentos_pdf.append(Document(page_content="Acervo de PDFs vazio.", metadata={"source": "none"}))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(documentos_pdf)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    @tool
    def consultar_livros_pdf(query: str) -> str:
        """Busca trechos DENTRO dos livros em PDF. 
        REGRA DE OURO: A sua 'query' DEVE SEMPRE conter o nome do livro ou do autor junto com o tema pesquisado para filtrar corretamente.
        Exemplo RUIM: 'definição de dinheiro'
        Exemplo BOM: 'definição de dinheiro livro O que os donos do poder não querem que você saiba'
        """
        documentos = retriever.invoke(query)
        return "\n\n".join([f"Livro/Arquivo: {doc.metadata.get('source', 'Desconhecido')} (Página {doc.metadata.get('page', 'N/A')})\nTrecho: {doc.page_content}" for doc in documentos])
    
    return [consultar_calendario, consultar_livros_pdf]
