import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# --- Setup (runs once per session start) ---

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = FAISS.load_local(
    "src/rag_udemy/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

prompt = ChatPromptTemplate.from_template("""You are a university student assistant.
Answer the question using only the context below.

Rules:
1. Use only the context. Do not use outside knowledge.
2. If the context does not contain the answer, say: "The provided documents don't say."
3. Never guess numbers, dates or rules.
4. Keep the answer short and simple.

Context:
{context}

Question: {question}

Answer:""")

THRESHOLD = 0.75


def search_or_refuse(question, k=3):
    results = vectorstore.similarity_search_with_score(question, k=k)
    good = [doc for doc, score in results if score <= THRESHOLD]
    return good


def ask(question):
    docs = search_or_refuse(question)
    if not docs:
        return "I couldn't find this in the university documents.", []
    context = "\n\n".join(d.page_content for d in docs)
    raw_answer = llm.invoke(prompt.format_messages(
        context=context, question=question)).content

    if isinstance(raw_answer, list):
        answer = "".join(
            block["text"] for block in raw_answer
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        answer = raw_answer

    return answer, sorted({d.metadata["source"] for d in docs})

# --- Streamlit chat interface ---


st.title("University Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

question = st.chat_input("Ask about attendance, exams, or the library")
if question:
    st.chat_message("user").write(question)
    answer, sources = ask(question)
    st.chat_message("assistant").write(answer)
    if sources:
        st.caption("Sources: " + ", ".join(sources))
    st.session_state.history.append(("user", question))
    st.session_state.history.append(("assistant", answer))
