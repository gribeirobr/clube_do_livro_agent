import os
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from src.config import ARQUIVO_CALENDARIO, DATA_DIR

def preparar_arquivos_teste():
    """Garante que a pasta 'data' exista."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ARQUIVO_CALENDARIO):
        conteudo = "CALENDÁRIO VAZIO\nAdicione as leituras aqui."
        with open(ARQUIVO_CALENDARIO, "w", encoding="utf-8") as f:
            f.write(conteudo)

def configurar_ferramenta_rag(api_key: str): 
    chave_limpa = api_key.strip().strip("'").strip('"')
    os.environ["GOOGLE_API_KEY"] = chave_limpa
    
    preparar_arquivos_teste()
    
    documentos_completos = []

    if os.path.exists(ARQUIVO_CALENDARIO):
        loader_txt = TextLoader(ARQUIVO_CALENDARIO, encoding="utf-8")
        documentos_completos.extend(loader_txt.load())
    
    loader_pdf = PyPDFDirectoryLoader(DATA_DIR)
    documentos_completos.extend(loader_pdf.load())
    
    if not documentos_completos:
        from langchain_core.documents import Document
        documentos_completos.append(Document(page_content="Acervo vazio.", metadata={"source": "none"}))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(documentos_completos)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    @tool
    def consultar_acervo_clube(query: str) -> str:
        """Busca informações no calendário do clube e dentro do texto dos livros em PDF. 
        Use isso para descobrir datas de leitura, detalhes da história, ou analisar personagens."""
        documentos = retriever.invoke(query)
        return "\n\n".join([f"Livro/Arquivo: {doc.metadata.get('source', 'Desconhecido')} (Página {doc.metadata.get('page', 'N/A')})\nTrecho encontrado:\n{doc.page_content}" for doc in documentos])
    
    return [consultar_acervo_clube]
