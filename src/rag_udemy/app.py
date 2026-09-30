import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Must be the first Streamlit call
st.set_page_config(page_title="University Assistant",
                   page_icon="🎓", layout="wide")

load_dotenv()


BASE_DIR = os.path.dirname(__file__)
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
DOCS_DIR = os.path.join(BASE_DIR, "data", "text_files")

# file name -> (emoji, title, colour, sample question)
TOPICS = {
    "fees_structure.txt": ("💰", "Fees", "#059669", "What is the fee structure?"),
    "attendance_policy.txt": ("📅", "Attendance", "#2563EB", "What is the minimum attendance requirement?"),
    "exam_rules.txt": ("📝", "Exams", "#D97706", "What are the exam eligibility rules?"),
    "library_rules.txt": ("📚", "Library", "#7C3AED", "How does library membership work?"),
    "uniform_rules.txt": ("👔", "Uniform", "#DB2777", "What is the uniform policy?"),
    "antiragging_policy.txt": ("🛡️", "Anti-Ragging", "#DC2626", "What is the anti-ragging policy?"),
}


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


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

prompt = ChatPromptTemplate.from_template("""You are a friendly university student assistant.
Answer the question using only the context below.

Rules:
1. Use only the context. Do not use outside knowledge.
2. If the context does not contain the answer, say: "I couldn't find that in the university documents."
3. Never guess numbers, dates or rules.
4. Use short paragraphs or bullet points.

Context:
{context}

Question: {question}

Answer:""")

THRESHOLD = 0.75

def search_or_refuse(question, k=5):
    results = db.similarity_search_with_score(question, k=k)
    return [doc for doc, score in results if score <= THRESHOLD]

def answer(question, history):
    docs = search_or_refuse(question)
    if not docs:
        return "I couldn't find that in the university documents.", []
    context = "\n\n".join(d.page_content for d in docs)
    raw_answer = llm.invoke(prompt.format_messages(context=context, question=question)).content
    if isinstance(raw_answer, list):
        text = "".join(b["text"] for b in raw_answer if isinstance(b, dict) and b.get("type") == "text")
    else:
        text = raw_answer
    return text, docs

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
#MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1050px;}

.hero {
    background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);
    border-radius: 20px; padding: 2rem 2.2rem; color: #fff; margin-bottom: 1.2rem;
}
.hero h1 {color: #fff; font-size: 2.3rem; margin: 0 0 .3rem 0; padding: 0;}
.hero p {color: #E0E7FF; font-size: 1.05rem; margin: 0;}

.stat {
    border-radius: 16px; padding: 1rem 1.2rem; color: #fff; font-weight: 600;
}
.stat span {display: block; font-size: 1.7rem; font-weight: 800;}
.s1 {background: #4F46E5;} .s2 {background: #059669;} .s3 {background: #D97706;}

.section-title {font-size: 1.25rem; font-weight: 700; margin: 1.2rem 0 .6rem 0;}

.chip {
    display: inline-block; border-radius: 999px; padding: .15rem .7rem;
    font-size: .8rem; font-weight: 600; color: #fff; margin: 0 .3rem .3rem 0;
}

[data-testid="stChatMessage"] {
    border-radius: 16px; padding: 1rem; margin-bottom: .7rem;
    border: 1px solid #E5E7EB; border-left: 5px solid #7C3AED; background: #FAFAFF;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 5px solid #DB2777; background: #FFF5FA;
}

.stTabs [data-baseweb="tab"] {font-weight: 600; font-size: 1rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #EEF2FF, #FDF2F8);}
.stButton > button {border-radius: 12px; border: 1px solid #C7D2FE; font-weight: 500;}
.stButton > button:hover {border-color: #7C3AED; color: #7C3AED;}
</style>
""", unsafe_allow_html=True)

USER_AVATAR, BOT_AVATAR = "🧑‍🎓", "🤖"

if "history" not in st.session_state:
    st.session_state.history = []


def set_pending(q):
    st.session_state.pending = q


def chips(files):
    html = ""
    for f in files:
        emoji, title, color, _ = TOPICS.get(f, ("📄", f, "#6B7280", ""))
        html += f"<span class='chip' style='background:{color}'>{emoji} {title}</span>"
    return html


def show_sources(files):
    if files:
        st.markdown("**Sources:** " + chips(files), unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🎓 Campus Helper")
    st.caption("Answers come only from official university documents.")
    st.markdown("### ⚡ Quick questions")
    for f, (emoji, title, _, q) in TOPICS.items():
        st.button(f"{emoji} {title}", key=f"side_{f}", use_container_width=True,
                  on_click=set_pending, args=(q,))
    st.divider()
    st.button("🗑️ Clear chat", use_container_width=True,
              on_click=lambda: st.session_state.update(history=[]))
    st.info("💡 For anything official, please confirm with the college office.")

# ---------------------------------------------------------------- header
st.markdown(
    "<div class='hero'><h1>🎓 University Assistant</h1>"
    "<p>Your 24/7 guide to fees, exams, attendance, library rules and more ✨</p></div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.markdown(
    f"<div class='stat s1'><span>📄 {len(TOPICS)}</span>Policy documents</div>", unsafe_allow_html=True)
c2.markdown("<div class='stat s2'><span>🤖 Gemini</span>AI-powered answers</div>",
            unsafe_allow_html=True)
c3.markdown("<div class='stat s3'><span>🔎 Cited</span>Every answer shows sources</div>",
            unsafe_allow_html=True)

# chat_input is pinned to the bottom of the page whichever tab is open
question = st.chat_input(
    "Ask about fees, exams, attendance...") or st.session_state.pop("pending", None)

tab_chat, tab_docs, tab_how = st.tabs(
    ["💬 Chat", "📚 Documents", "🧠 How it works"])

# ---------------------------------------------------------------- chat tab
with tab_chat:
    if not st.session_state.history and not question:
        st.markdown(
            "<div class='section-title'>👋 Hi! What would you like to know?</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (f, (emoji, title, color, q)) in enumerate(TOPICS.items()):
            with cols[i % 3]:
                st.button(f"{emoji}  {title}", key=f"card_{f}", use_container_width=True,
                          on_click=set_pending, args=(q,))
        st.caption("Click a topic or type your own question below 👇")

    for m in st.session_state.history:
        with st.chat_message(m["role"], avatar=USER_AVATAR if m["role"] == "user" else BOT_AVATAR):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                show_sources(m.get("sources"))

    if question:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("🔍 Searching university documents..."):
                reply, docs = answer(question, st.session_state.history)
            st.markdown(reply)
            files = sorted({os.path.basename(d.metadata.get(
                "source", "").replace("\\", "/")) for d in docs})
            show_sources(files)
        st.session_state.history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply, "sources": files},
        ]

# ---------------------------------------------------------------- documents tab
with tab_docs:
    st.markdown("<div class='section-title'>📚 Browse the official documents</div>",
                unsafe_allow_html=True)
    st.caption("This is exactly what the assistant reads to answer you.")
    for f, (emoji, title, color, _) in TOPICS.items():
        path = os.path.join(DOCS_DIR, f)
        if os.path.exists(path):
            with st.expander(f"{emoji} {title}  ·  {f}"):
                with open(path, encoding="utf-8") as fh:
                    st.text(fh.read())

# ---------------------------------------------------------------- how it works tab
with tab_how:
    st.markdown("<div class='section-title'>🧠 How the assistant answers</div>",
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.info("**1. 🔎 Search**\n\nYour question is turned into numbers (an embedding) and matched against the document chunks.")
    b.success(
        "**2. 📎 Retrieve**\n\nThe 5 most relevant chunks are picked from the FAISS index.")
    c.warning(
        "**3. ✍️ Answer**\n\nGemini writes the reply using only those chunks, and cites the files.")
    st.markdown("<div class='section-title'>❓ Good to know</div>",
                unsafe_allow_html=True)
    st.markdown(
        "- 💬 Follow-ups like *“elaborate on that”* work, because the question is rewritten using the chat.\n"
        "- 🚫 If the answer isn't in the documents, the assistant says so instead of guessing.\n"
        "- 🔄 To update the knowledge, edit the `.txt` files and rebuild the index in `dataingestion.ipynb`."
    )
