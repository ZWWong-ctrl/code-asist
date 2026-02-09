import os
import re
import io
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

# =========================
# App Config
# =========================
load_dotenv()

st.set_page_config(
    page_title="Study Helper",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Styling (Decoration)
# =========================
st.markdown(
    """
<style>
:root {
  --bg: #0b1220;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.12);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.70);
  --accent: #6ea8fe;
  --accent2: #8b5cf6;
  --good: #22c55e;
  --warn: #f59e0b;
  --bad: #ef4444;
}

.stApp {
  background: radial-gradient(1200px 600px at 15% 10%, rgba(110,168,254,0.18), transparent 40%),
              radial-gradient(900px 500px at 85% 20%, rgba(139,92,246,0.14), transparent 40%),
              linear-gradient(180deg, #060a13, #070c18 40%, #060a13);
  color: var(--text);
}

.block-container { padding-top: 1.4rem; }

.hero {
  padding: 18px 20px;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(110,168,254,0.10), rgba(139,92,246,0.08));
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  margin-bottom: 14px;
}
.hero h1 { margin: 0; font-size: 1.55rem; }
.hero p { margin: 6px 0 0; color: var(--muted); }

.card {
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 14px;
  background: var(--card);
}

.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
  color: var(--muted);
  margin-right: 8px;
}

.small { color: var(--muted); font-size: 0.92rem; }
hr { border-color: rgba(255,255,255,0.08) !important; }

.chat-hint {
  padding: 10px 12px;
  border-left: 3px solid var(--accent);
  background: rgba(110,168,254,0.08);
  border-radius: 12px;
  color: var(--muted);
}

.source-box {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.04);
  margin-bottom: 10px;
}
.source-title { font-weight: 600; }
.source-meta { color: var(--muted); font-size: 0.85rem; margin-top: 2px; }
.codechip {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  background: rgba(139,92,246,0.12);
  border: 1px solid rgba(139,92,246,0.25);
  font-size: 0.78rem;
  color: rgba(255,255,255,0.85);
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Data Structures
# =========================
@dataclass
class Chunk:
    doc_name: str
    page: int
    text: str

# =========================
# Utility: Tokenize
# =========================
_WORD = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]

def clean_text(t: str) -> str:
    # Light cleanup for PDFs
    t = t.replace("\u00ad", "")          # soft hyphen
    t = re.sub(r"[ \t]+\n", "\n", t)     # trailing spaces before newline
    t = re.sub(r"\n{3,}", "\n\n", t)     # collapse many newlines
    return t.strip()

# =========================
# PDF Extraction
# =========================
@st.cache_data(show_spinner=False)
def extract_chunks_from_pdf_bytes(pdf_bytes: bytes, doc_name: str, min_chars_per_page: int = 60) -> List[Chunk]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: List[Chunk] = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = clean_text(txt)
        if len(txt) >= min_chars_per_page:
            chunks.append(Chunk(doc_name=doc_name, page=i, text=txt))
        else:
            # Keep placeholder; helps detect scanned PDFs
            chunks.append(Chunk(doc_name=doc_name, page=i, text=""))
    return chunks

def chunk_pages_into_windows(page_chunks: List[Chunk], window_chars: int = 2200, overlap_chars: int = 250) -> List[Chunk]:
    # Turn each page into smaller windows (better retrieval)
    out: List[Chunk] = []
    for c in page_chunks:
        t = c.text
        if not t:
            continue
        if len(t) <= window_chars:
            out.append(c)
            continue
        i = 0
        while i < len(t):
            out.append(Chunk(doc_name=c.doc_name, page=c.page, text=t[i:i+window_chars]))
            i += max(1, window_chars - overlap_chars)
    return out

# =========================
# BM25 Retrieval (No extra libraries)
# =========================
def build_bm25_index(chunks: List[Chunk]) -> Dict:
    docs_tokens = [tokenize(c.text) for c in chunks]
    N = len(docs_tokens)
    df: Dict[str, int] = {}
    doc_lens = []
    for toks in docs_tokens:
        doc_lens.append(len(toks))
        seen = set(toks)
        for w in seen:
            df[w] = df.get(w, 0) + 1
    avgdl = (sum(doc_lens) / N) if N else 0.0
    return {
        "docs_tokens": docs_tokens,
        "df": df,
        "N": N,
        "avgdl": avgdl,
        "doc_lens": doc_lens,
    }

def bm25_scores(query: str, index: Dict, k1: float = 1.5, b: float = 0.75) -> List[float]:
    q = tokenize(query)
    if not q or index["N"] == 0:
        return [0.0] * index["N"]
    df = index["df"]
    N = index["N"]
    avgdl = index["avgdl"] or 1.0
    doc_lens = index["doc_lens"]
    docs_tokens = index["docs_tokens"]

    # term frequencies per doc
    scores = [0.0] * N
    for term in q:
        n_q = df.get(term, 0)
        if n_q == 0:
            continue
        idf = math.log(1 + (N - n_q + 0.5) / (n_q + 0.5))
        for i in range(N):
            toks = docs_tokens[i]
            if not toks:
                continue
            tf = 0
            # quick count
            for w in toks:
                if w == term:
                    tf += 1
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * (doc_lens[i] / avgdl))
            scores[i] += idf * (tf * (k1 + 1)) / denom
    return scores

def retrieve_top_chunks(query: str, chunks: List[Chunk], index: Dict, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[Tuple[int, float]]:
    scores = bm25_scores(query, index, k1=k1, b=b)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(i, s) for i, s in ranked[:top_k]]

# =========================
# Prompt Builders
# =========================
def system_rules() -> str:
    return (
        "You are a study helper.\n"
        "Rules:\n"
        "- Use ONLY the provided NOTES.\n"
        '- If something is not in NOTES, say: "Not in the notes."\n'
        "- Be beginner-friendly and structured.\n"
        "- Use headings and bullets.\n"
        "- When doing math, show steps.\n"
        "- If the user asks to check work, ask them to paste their working.\n"
    )

MODE_STYLE = {
    "Explain": "Explain clearly with steps and a small example.",
    "Hint": "Give hints and steps first. Avoid giving the final answer immediately unless asked again.",
    "Step-by-step": "Give a full step-by-step solution with reasoning.",
    "Flashcards": "Create 10 flashcards (Q/A) based on the notes relevant to the request.",
    "Quiz": "Create a short quiz (5 questions). Include an answer key.",
    "Check My Work": "Ask the user to paste their working, then point out mistakes and correct them.",
}

def build_chat_prompt(mode: str, notes_block: str, user_question: str) -> str:
    return (
        f"{system_rules()}\n"
        f"MODE: {mode}\n"
        f"MODE INSTRUCTION: {MODE_STYLE.get(mode,'')}\n\n"
        "NOTES:\n"
        f"\"\"\"\n{notes_block}\n\"\"\"\n\n"
        f"USER QUESTION:\n{user_question}\n"
    )

def build_quiz_json_prompt(notes_block: str, topic: str) -> str:
    # Forces JSON so we can score
    return (
        f"{system_rules()}\n"
        "Task: Create a 5-question multiple-choice quiz.\n"
        "Return STRICT JSON ONLY. No markdown.\n"
        "JSON format:\n"
        "{\n"
        '  "title": "string",\n'
        '  "questions": [\n'
        "    {\n"
        '      "q": "string",\n'
        '      "options": {"A":"...", "B":"...", "C":"...", "D":"..."},\n'
        '      "answer": "A|B|C|D",\n'
        '      "explain": "string",\n'
        '      "sources": [{"doc":"string","page":number}]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "NOTES:\n"
        f"\"\"\"\n{notes_block}\n\"\"\"\n\n"
        f"TOPIC: {topic}\n"
    )

def format_notes_block(chunks: List[Chunk], picked: List[Tuple[int, float]]) -> str:
    blocks = []
    for rank, (idx, score) in enumerate(picked, start=1):
        c = chunks[idx]
        meta = f"[Source {rank}] doc={c.doc_name} page={c.page} score={score:.3f}"
        blocks.append(meta + "\n" + c.text)
    return "\n\n".join(blocks)

# =========================
# Sidebar Controls
# =========================
api_key = os.getenv("GEMINI_API_KEY")

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if api_key:
        st.success("API key loaded ✅")
    else:
        st.error("API key missing ❌")
        st.caption("Create .env with GEMINI_API_KEY=...")

    model_name = st.text_input("Model", value="gemini-3-flash-preview")
    mode = st.selectbox("Mode", ["Explain", "Hint", "Step-by-step", "Flashcards", "Quiz", "Check My Work"])
    top_k = st.slider("Top sources (k)", 2, 10, 5)

    st.markdown("### Retrieval (BM25)")
    k1 = st.slider("k1", 0.8, 2.2, 1.5, 0.1)
    b = st.slider("b", 0.1, 0.95, 0.75, 0.05)
    show_sources = st.checkbox("Show sources used", value=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## 📄 Notes Library")
    st.caption("Upload PDFs. The app will cite doc + page.")
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    colA, colB = st.columns(2)
    with colA:
        clear_chat = st.button("Clear chat")
    with colB:
        clear_notes = st.button("Clear notes")

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = {"docs_tokens": [], "df": {}, "N": 0, "avgdl": 0.0, "doc_lens": []}
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = set()
if "quiz_obj" not in st.session_state:
    st.session_state.quiz_obj = None

if clear_chat:
    st.session_state.messages = []

if clear_notes:
    st.session_state.chunks = []
    st.session_state.bm25_index = {"docs_tokens": [], "df": {}, "N": 0, "avgdl": 0.0, "doc_lens": []}
    st.session_state.docs_loaded = set()
    st.session_state.quiz_obj = None

# =========================
# Header
# =========================
st.markdown(
    """
<div class="hero">
  <span class="badge">📘 PDF Study Helper</span>
  <span class="badge">🔎 BM25 Retrieval</span>
  <span class="badge">🧠 Gemini</span>
  <h1>Ask from your notes, get structured answers + citations</h1>
  <p>Upload PDF notes → ask questions → the assistant answers using only the retrieved pages.</p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# Load PDFs into Knowledge Base
# =========================
if uploaded:
    new_chunks: List[Chunk] = []
    scanned_pages = 0
    total_pages = 0

    for f in uploaded:
        if f.name in st.session_state.docs_loaded:
            continue
        pdf_bytes = f.getvalue()
        pages = extract_chunks_from_pdf_bytes(pdf_bytes, doc_name=f.name)
        total_pages += len(pages)
        scanned_pages += sum(1 for c in pages if c.text.strip() == "")
        windowed = chunk_pages_into_windows(pages, window_chars=2200, overlap_chars=250)
        new_chunks.extend(windowed)
        st.session_state.docs_loaded.add(f.name)

    if new_chunks:
        st.session_state.chunks.extend(new_chunks)
        st.session_state.bm25_index = build_bm25_index(st.session_state.chunks)

    if total_pages > 0:
        if scanned_pages / total_pages > 0.4:
            st.warning("Many pages have no extractable text (likely scanned). You may need OCR for best results.")
        st.success(f"Loaded {len(st.session_state.docs_loaded)} PDF(s). Searchable chunks: {len(st.session_state.chunks)}")

# =========================
# Gemini Client
# =========================
if not api_key:
    st.stop()

client = genai.Client(api_key=api_key)

# =========================
# Tabs (More Complex UI)
# =========================
tab_chat, tab_summary, tab_flash, tab_quiz = st.tabs(["💬 Chat", "🧾 Summary", "🗂️ Flashcards", "📝 Quiz"])

# =========================
# CHAT TAB
# =========================
with tab_chat:
    left, right = st.columns([1.35, 0.85], gap="large")

    with left:
        st.markdown('<div class="chat-hint">Tip: Upload PDFs first. Then ask questions like: "Explain DTFT vs DFT" or "Derive the formula and give steps".</div>', unsafe_allow_html=True)
        st.write("")

        # Show existing messages
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask a question from your notes…")

        if user_q:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            if not st.session_state.chunks:
                with st.chat_message("assistant"):
                    st.markdown("Upload your PDF first.")
                st.stop()

            picked = retrieve_top_chunks(
                user_q,
                st.session_state.chunks,
                st.session_state.bm25_index,
                top_k=top_k,
                k1=k1,
                b=b
            )
            notes_block = format_notes_block(st.session_state.chunks, picked)
            prompt = build_chat_prompt(mode, notes_block, user_q)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    answer = resp.text or ""
                    st.markdown(answer)

                    with st.chat_message("assistant"):
    with st.spinner("Thinking…"):
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        answer = resp.text or ""
        st.markdown(answer)

        if show_sources:
            with st.expander("Sources used (doc + page + snippet)"):
                for rank, (idx, score) in enumerate(picked, start=1):
                    c = st.session_state.chunks[idx]
                    snippet = c.text[:500].replace("\n", " ")
                    st.markdown(
                        f"""
<div class="source-box">
  <div class="source-title">Source {rank} <span class="codechip">score {score:.3f}</span></div>
  <div class="source-meta">📄 {c.doc_name} — page {c.page}</div>
  <div class="small">{snippet}...</div>
</div>
""",
                        unsafe_allow_html=True,
                    )


