import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import StructuredTool

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

from pydantic import BaseModel, Field


load_dotenv()

st.set_page_config(page_title="Chat with Search", page_icon="🔍")
st.title("🔍 Chat with Search (LangGraph + Groq)")

api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
if not api_key:
    st.stop()

# Streaming callback

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# Tool schema

class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")


#tools 

wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

def wikipedia_tool_fn(query: str) -> str:
    return wiki_api.run(query)

wikipedia_tool = StructuredTool.from_function(
    name="wikipedia",
    description="Search Wikipedia for factual information.",
    func=wikipedia_tool_fn,
    args_schema=SearchInput,
)



arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)

def arxiv_tool_fn(query: str) -> str:
    return arxiv_api.run(query)

arxiv_tool = StructuredTool.from_function(
    name="arxiv",
    description="Search arXiv for academic papers.",
    func=arxiv_tool_fn,
    args_schema=SearchInput,
)


ddg = DuckDuckGoSearchRun()

def web_search_fn(query: str) -> str:
    return ddg.run(query)

web_search_tool = StructuredTool.from_function(
    name="brave_search",
    description="Search the web for recent information.",
    func=web_search_fn,
    args_schema=SearchInput,
)

tools = [web_search_tool, arxiv_tool, wikipedia_tool]


# LLM

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    streaming=True,
)


# Agent

agent = create_react_agent(
    model=llm,
    tools=tools
)


# Session state

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "content": "Hi! Ask me anything."}
    ]

for msg in st.session_state.chat:
    st.chat_message(msg["role"]).write(msg["content"])


# input

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        container = st.empty()
        callback = StreamHandler(container)

        messages = [
            ("system", "You are a helpful assistant. Use tools only when needed."),
            ("human", prompt),
        ]

        result = agent.invoke(
            {"messages": messages},
            config={"callbacks": [callback]}
        )

        final = result["messages"][-1].content
        st.session_state.chat.append(
            {"role": "assistant", "content": final}
        )