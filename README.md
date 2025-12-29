## 🔍 Chat with Search — LangGraph + Groq

A production-ready AI search assistant built using LangGraph, Groq (LLaMA-3.1), and Streamlit.
The app can intelligently decide when to search the web, Wikipedia, or arXiv to answer user queries accurately.

🚀 Live Demo: (https://c8aucvcxhu3tutr4pjyuoy.streamlit.app)
📌 Tech Focus: LLM Agents, Tool Calling, LangGraph, Groq

⸻

## ✨ Features
	•	🤖 LLM Agent (LangGraph) — ReAct-style reasoning with tool usage
	•	🔍 Web Search — DuckDuckGo for real-time information
	•	📚 Wikipedia Search — Factual and encyclopedic answers
	•	📄 arXiv Search — Academic papers & research queries
	•	🧠 Multi-tool Decision Making — Agent chooses the right tool automatically
	•	🌐 Deployed on Streamlit Cloud
	•	🔐 Secure API Key Handling using Streamlit Secrets

⸻

## 🧠 How It Works (High Level)
	1.	User enters a query
	2.	LLaMA-3.1 (via Groq) reasons about the query
	3.	Agent decides whether to:
	•	Search the web
	•	Query Wikipedia
	•	Query arXiv
	4.	Tool results are combined into a final response
	5.	Answer is displayed in a chat interface

This uses LangGraph, the modern replacement for deprecated LangChain agents.


