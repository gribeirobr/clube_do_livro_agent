from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from src.prompts import SYSTEM_PROMPT
from src.tools import configurar_ferramenta_rag

# Uma colinha em português para facilitar para a IA
MESES = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]

def criar_grafo_agente(api_key: str):
    tools = configurar_ferramenta_rag(api_key)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3, 
        google_api_key=api_key
    )
    
    llm_with_tools = llm.bind_tools(tools)
    
    # === AQUI ESTÁ O PULO DO GATO (Injetando um Relógio) ===
    hoje = datetime.now()
    mes_atual = MESES[hoje.month - 1]
    
    prompt_dinamico = SYSTEM_PROMPT + f"\n\n[INFORMAÇÃO DO SISTEMA]: Hoje é dia {hoje.strftime('%d/%m/%Y')} ({mes_atual} de {hoje.year}). Baseie-se APENAS nesta data exata para definir qual é a 'leitura atual' no calendário."
    
    sys_msg = SystemMessage(content=prompt_dinamico)
    
    def assistente(state: MessagesState):
        mensagens = state["messages"]
        if not mensagens or not isinstance(mensagens[0], SystemMessage):
            mensagens = [sys_msg] + mensagens
        resposta = llm_with_tools.invoke(mensagens)
        return {"messages": [resposta]}
    
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", assistente)
    workflow.add_node("tools", ToolNode(tools)) 
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition) 
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()
