import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.1-flash-lite"  # use whichever Gemini model your key has access to

emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = FAISS.load_local("src/rag_udemy/faiss_index", emb,
                      allow_dangerous_deserialization=True)

def ask(prompt: str) -> str:
    return client.models.generate_content(model=MODEL, contents=prompt).text

def standalone_question(question, history):
    if not history:
        return question
    chat = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    return ask(
        "Rewrite the last user question as a standalone question, "
        f"using the chat for context. Return only the question.\n\n{chat}\n\nQuestion: {question}"
    )

def answer(question, history):
    q = standalone_question(question, history)
    docs = db.similarity_search(q, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""You are a university assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I couldn't find that in the university documents."

Context:
{context}

Question: {q}
Answer:"""
    return ask(prompt), docs

st.title("University Assistant")
if "history" not in st.session_state:
    st.session_state.history = []
for m in st.session_state.history:
    st.chat_message(m["role"]).write(m["content"])

if question := st.chat_input("Ask about fees, exams, attendance..."):
    st.chat_message("user").write(question)
    reply, docs = answer(question, st.session_state.history)
    st.chat_message("assistant").write(reply)
    with st.expander("Sources"):
        for d in docs:
            st.write(d.metadata.get("source"), "-", d.page_content[:200])
    st.session_state.history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
    ]