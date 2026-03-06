import streamlit as st
from langchain_core.messages import HumanMessage
from src.agent import criar_grafo_agente

st.set_page_config(page_title="Clube do Livro AI", page_icon="📚")

st.title("📚 Mediador do Clube do Livro")
st.markdown("Bem-vindo! Sou a IA do nosso Clube do Livro. Já li nossos PDFs e o calendário. Vamos debater?")

try:
    api_key = st.secrets["API_KEY"]
except KeyError:
    st.error("Chave de API não encontrada! Configure os Secrets no Streamlit Cloud.")
    st.stop()

with st.sidebar:
    st.header("💡 Dicas de Perguntas")
    st.markdown("- *'Qual é o livro que vamos ler este mês?'*")
    st.markdown("- *'Me faça um resumo sem spoilers do livro atual.'*")
    st.markdown("- *'Por que o protagonista tomou aquela atitude no capítulo 3?'*")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource(show_spinner=False)
def obter_agente(chave_api):
    return criar_grafo_agente(chave_api)

def extrair_texto(conteudo):
    if isinstance(conteudo, str):
        return conteudo
    elif isinstance(conteudo, list):
        textos = []
        for bloco in conteudo:
            if isinstance(bloco, dict) and 'text' in bloco:
                textos.append(bloco['text'])
            elif isinstance(bloco, str):
                textos.append(bloco)
        return "\n".join(textos)
    return str(conteudo)

for msg in st.session_state.messages:
    if msg.type in ["human", "ai"] and msg.content:
        with st.chat_message(msg.type):
            st.markdown(extrair_texto(msg.content))

if prompt := st.chat_input("Pergunte sobre a leitura atual..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
        
    graph = obter_agente(api_key)
    
    with st.chat_message("ai"):
        with st.spinner("Folheando as páginas e o calendário..."):
            inputs = {"messages": st.session_state.messages}
            config = {"configurable": {"thread_id": "sessao_clube"}}
            
            response = graph.invoke(inputs, config=config)
            
            resposta_final = response["messages"][-1]
            texto_limpo = extrair_texto(resposta_final.content)
            
            st.markdown(texto_limpo)
            st.session_state.messages = response["messages"]
