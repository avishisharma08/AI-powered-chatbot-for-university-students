# AI-Powered Chatbot for University Students

A Retrieval-Augmented Generation (RAG) chatbot that answers student
questions using real university documents, attendance policy, exam
rules, and library rules, instead of relying on a language model's
general knowledge. Built as a learning project to understand how RAG
systems work end to end.

## Why RAG?

A plain chatbot can confidently make up answers to questions about
college rules. This project instead retrieves the exact relevant text
from official documents first, then asks the language model to answer
using only that text. If nothing relevant is found, it says so instead
of guessing.

## How it works

1. **Load** — text documents are loaded from `data/text_files/`
2. **Chunk** — documents are split into smaller pieces (500 characters,
   with 50-character overlap) so each piece captures one focused idea
3. **Embed** — each chunk is converted into a vector using Gemini's
   embedding model, capturing its meaning rather than its exact words
4. **Store** — vectors are saved in a FAISS index for fast similarity
   search
5. **Retrieve** — a user's question is embedded and compared against
   the index; the closest matching chunks are returned
6. **Filter** — a similarity-score threshold discards results that
   aren't close enough to be relevant, so unrelated questions (e.g.
   "What is the capital of France?") are refused before reaching the
   language model
7. **Generate** — the retrieved chunks are passed to Gemini along with
   the question, using a prompt that instructs it to answer only from
   the given context and to say "The provided documents don't say" if
   the answer isn't there
8. **Respond** — the answer is returned along with the source
   document(s) it came from

## Two layers of defense against wrong answers

| Layer | Catches |
|---|---|
| Similarity threshold | Questions on topics the documents don't cover at all |
| Prompt instructions | Questions on covered topics where the specific fact is missing |

For example, a question about library fines retrieves the library
rules document (right topic), but the document has no fine amount, so
the model correctly says the information isn't available rather than
inventing a number.

## Tech stack

- **Python**
- **LangChain** — document loading, splitting, prompt templates
- **FAISS** — vector similarity search
- **Google Gemini API** — embeddings (`gemini-embedding-001`) and
  generation (`gemini-3.1-flash-lite`)
- **Streamlit** — chat interface *(in progress)*
- **uv** — dependency management

## Project status

- [x] Document loading and chunking
- [x] Embeddings and FAISS index
- [x] Similarity-threshold refusal for out-of-scope questions
- [x] Prompt-based hallucination guardrails
- [x] End-to-end question answering (tested in notebook)
- [ ] Streamlit chat interface
- [ ] Chat memory for follow-up questions
- [ ] Expanded document set
- [ ] Formal evaluation with a fixed test-question set

## Setup

1. Clone the repo
2. Create a `.env` file in the project root with:
4. Open `src/rag_udemy/dataingestion.ipynb` and run the cells in order

## Sample documents

The `data/text_files/` folder contains a small set of fictional sample
policies (attendance, exams, library) written for testing this
pipeline. They are not real college rules.

## What I learned building this

Notes on chunk sizing, embedding vs. keyword search, why retrieval and
generation need separate safeguards, and common pitfalls (stale FAISS
indexes after editing source documents, environment/kernel issues) are
in `learning-log.md`.
