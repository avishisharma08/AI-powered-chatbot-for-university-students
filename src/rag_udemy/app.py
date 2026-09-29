import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Must be the first Streamlit call
st.set_page_config(page_title="University Assistant",
                   page_icon="🎓", layout="centered")

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.1-flash-lite"

INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")


@st.cache_resource
def load_db():
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)


db = load_db()


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
    docs = db.similarity_search(q, k=5)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""You are a university assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I couldn't find that in the university documents."

Context:
{context}

Question: {q}
Answer:"""
    return ask(prompt), docs


# ---------- UI ----------
st.markdown("""
<style>
#MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 820px;}
.hero h1 {font-size: 2.2rem; margin-bottom: 0;}
.hero p {color: #6B7280; margin-top: 0.3rem;}
[data-testid="stChatMessage"] {
    border-radius: 14px; padding: 1rem; margin-bottom: 0.6rem;
    border: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] {background: #F3F4F8;}
</style>
""", unsafe_allow_html=True)

USER_AVATAR, BOT_AVATAR = "🧑‍🎓", "🎓"

if "history" not in st.session_state:
    st.session_state.history = []


def show_sources(files):
    if files:
        with st.expander("📄 Sources"):
            for f in files:
                st.markdown(f"- `{f}`")


with st.sidebar:
    st.header("🎓 University Assistant")
    st.caption("Answers come only from official university documents.")
    st.markdown("**Try asking**")
    for ex in [
        "What is the fee structure?",
        "What is the minimum attendance requirement?",
        "What are the exam eligibility rules?",
        "What is the uniform policy?",
        "How does library membership work?",
    ]:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.markdown(
    "<div class='hero'><h1>🎓 University Assistant</h1>"
    "<p>Ask about fees, exams, attendance, library rules and more.</p></div>",
    unsafe_allow_html=True,
)

for m in st.session_state.history:
    with st.chat_message(m["role"], avatar=USER_AVATAR if m["role"] == "user" else BOT_AVATAR):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources"))

question = st.chat_input(
    "Ask about fees, exams, attendance...") or st.session_state.pop("pending", None)

if question:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Searching university documents..."):
            reply, docs = answer(question, st.session_state.history)
        st.markdown(reply)
        files = sorted({os.path.basename(d.metadata.get("source", ""))
                       for d in docs})
        show_sources(files)
    st.session_state.history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply, "sources": files},
    ]
 