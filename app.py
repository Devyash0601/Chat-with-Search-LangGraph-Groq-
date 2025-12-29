import streamlit as st
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)
from langchain_community.tools import (
    WikipediaQueryRun,
    ArxivQueryRun,
    DuckDuckGoSearchRun,
)

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent


st.set_page_config(
    page_title="Chat with Search (LangGraph + Groq)",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Chat with Web, Wikipedia & arXiv")


st.sidebar.title("Settings")

api_key = (
    st.secrets.get("GROQ_API_KEY")
    if "GROQ_API_KEY" in st.secrets
    else st.sidebar.text_input("Enter Groq API Key", type="password")
)

if not api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()


llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
)


# Tool Input Schema 

class QueryInput(BaseModel):
    query: str

# Tool

wiki_tool = StructuredTool.from_function(
    name="wikipedia",
    description="Search Wikipedia for factual information.",
    func=lambda query: WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=300,
        )
    ).run(query),
    args_schema=QueryInput,
)


arxiv_tool = StructuredTool.from_function(
    name="arxiv",
    description="Search arXiv for academic papers.",
    func=lambda query: ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=300,
        )
    ).run(query),
    args_schema=QueryInput,
)


search_tool = DuckDuckGoSearchRun(name="search")

tools = [search_tool, wiki_tool, arxiv_tool]


# LangGraph Agent

agent = create_react_agent(
    model=llm,
    tools=tools,
)


# Session State-Chat History

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can search the web, Wikipedia, and arXiv. What would you like to know?",
        }
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# User Input

prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke(
                {"messages": st.session_state.messages}
            )

            final_message = result["messages"][-1].content
            st.write(final_message)

            st.session_state.messages.append(
                {"role": "assistant", "content": final_message}
            )