# app.py
import os
import re
import io
import uuid
import json
import time
from pathlib import Path
from textwrap import shorten
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Web tools (optional browsing/RAG)
import requests
import trafilatura
from tavily import TavilyClient

# ============ Setup ============
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

HF_MODELS = {
    "GPT-OSS-20B": "openai/gpt-oss-20b",
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.3",
}

# Use local folder; for cloud hosting you can switch to Path("/tmp") / "chat_data"
DATA_DIR = Path("./chat_data")
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "conversations.json"
NOTES_FILE = DATA_DIR / "notes.json"
MEM_FILE = DATA_DIR / "memory.json"

# === Local usage tracking (HF + RAG) ===
DEFAULT_COST_PER_REQ = 0.01 / 38  # ~ $0.000263 based on your snapshot
COST_PER_REQ = float(os.getenv("HF_COST_PER_REQ", DEFAULT_COST_PER_REQ))
FREE_CAP = float(os.getenv("HF_FREE_CAP", 0.10))  # HF free monthly cap in USD

USAGE_FILE = DATA_DIR / "usage.json"

def _load_usage():
    if USAGE_FILE.exists():
        try:
            u = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            u = {}
    else:
        u = {}
    month = datetime.utcnow().strftime("%Y-%m")
    u.setdefault("month", month)
    if u["month"] != month:
        u = {"month": month, "dollars": 0.0, "requests": 0, "rag_calls": 0, "rag_results": 0}
    u.setdefault("dollars", 0.0)
    u.setdefault("requests", 0)
    u.setdefault("rag_calls", 0)
    u.setdefault("rag_results", 0)
    return u

def _save_usage(u):
    USAGE_FILE.write_text(json.dumps(u, ensure_ascii=False, indent=2), encoding="utf-8")

def add_inference_usage(u, cost=COST_PER_REQ):
    u["requests"] += 1
    u["dollars"] = round(u["dollars"] + float(cost), 6)
    _save_usage(u)

def add_rag_usage(u, results_count):
    u["rag_calls"] += 1
    u["rag_results"] += int(results_count)
    _save_usage(u)

usage = _load_usage()

st.set_page_config(page_title="HF Chat", page_icon="🤖", layout="wide")

# ============ Styles ============
st.markdown("""
<style>
section.main > div { padding-top: 0rem; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

.chat-bubble {
  border-radius: 18px;
  padding: 12px 14px;
  margin: 6px 0;
  line-height: 1.5;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  border: 1px solid var(--bubble-border);
  background: var(--bubble-bg);
  color: var(--bubble-fg);
}

/* Light default */
:root {
  --bubble-bg: #ffffff;
  --bubble-fg: #111111;
  --bubble-border: rgba(0,0,0,0.06);
  --bubble-user-bg: #f6f7f9;
  --link-color: #0a58ca;
}

/* Respect OS / Streamlit dark mode */
@media (prefers-color-scheme: dark) {
  :root {
    --bubble-bg: #1f2227;
    --bubble-fg: #e8eaed;
    --bubble-border: rgba(255,255,255,0.10);
    --bubble-user-bg: #2a2e35;
    --link-color: #8ab4f8;
  }
}
.user-bubble { background: var(--bubble-user-bg); }
.assistant-bubble { background: var(--bubble-bg); }

a, .stMarkdown a { color: var(--link-color) !important; text-decoration: underline; }
.sidebar .sidebar-content { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("💬 HuggingFace Chat")

# ============ Persistence helpers ============
def _new_chat(title="New chat"):
    cid = str(uuid.uuid4())
    return cid, {"title": title, "messages": []}

def _read_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def _write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def load_conversations():
    data = _read_json(STATE_FILE, None)
    if isinstance(data, dict) and data:
        return data
    cid, chat = _new_chat("First chat")
    return {cid: chat}

def load_notes():
    data = _read_json(NOTES_FILE, {})
    return data if isinstance(data, dict) else {}

def load_memory():
    data = _read_json(MEM_FILE, {"global": "", "per_chat": {}})
    if "global" not in data: data["global"] = ""
    if "per_chat" not in data: data["per_chat"] = {}
    return data

def sanitize_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s\-\&\(\)\.\,\:]", "", text)
    return shorten(text, width=40, placeholder="…")

def auto_rename_chat(chat_obj):
    if chat_obj["title"] in {"New chat", "First chat"} and chat_obj["messages"]:
        first_user = next((m for m in chat_obj["messages"] if m["role"] == "user"), None)
        if first_user:
            guess = sanitize_title(first_user["content"])
            if guess:
                chat_obj["title"] = guess

# ============ Web research (RAG) ============
def web_research(query: str, max_results: int = 4):
    """
    Returns (snippets, citations)
    snippets: list[str] clean text snippets
    citations: list[dict] {title, url}
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [], []
    tv = TavilyClient(api_key=api_key)
    try:
        res = tv.search(query=query, max_results=max_results)
    except Exception:
        return [], []
    snippets, sources = [], []
    for r in res.get("results", []):
        url = r.get("url")
        title = r.get("title") or url
        text = r.get("content") or ""
        if len(text) < 400 and url:
            try:
                html = requests.get(url, timeout=10).text
                extracted = trafilatura.extract(html, include_links=False) or ""
                if len(extracted) > len(text):
                    text = extracted
            except Exception:
                pass
        text = (text or "").strip()
        if text:
            snippets.append(text[:1500])
            sources.append({"title": title, "url": url})
        if len(snippets) >= max_results:
            break

    # track RAG usage locally
    try:
        add_rag_usage(usage, results_count=len(snippets))
    except Exception:
        pass

    return snippets, sources

def build_history_block(chat_obj, max_turns=8):
    """
    Convert the last N turns into a readable transcript block.
    A 'turn' is one user+assistant message pair; we slice by messages instead.
    """
    msgs = chat_obj["messages"][-max_turns*2:] if max_turns else chat_obj["messages"]
    lines = []
    for m in msgs:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines).strip()

def build_rag_prompt_from_history(user_query: str, snippets: list[str], memory_text: str, history_text: str):
    mem = (memory_text.strip() + "\n\n") if memory_text and memory_text.strip() else ""
    hist = (history_text.strip() + "\n\n") if history_text and history_text.strip() else ""
    system_hint = (
        "You are a helpful assistant. Use ONLY the sources below. "
        "Cite sources inline like [1], [2]. If the sources don't contain the answer, say you don't know."
    )
    ctx = "\n\n".join([f"[Source {i+1}] {s}" for i, s in enumerate(snippets)])
    return (
        f"{mem}"
        f"{hist}"
        f"{system_hint}\n\n"
        f"=== SOURCES ===\n{ctx}\n\n"
        f"=== USER QUESTION ===\n{user_query}\n\n"
        f"Answer concisely and include citations like [1], [2] where relevant."
    )

def build_plain_prompt_from_history(user_query: str, memory_text: str, history_text: str):
    mem = (memory_text.strip() + "\n\n") if memory_text and memory_text.strip() else ""
    hist = (history_text.strip() + "\n\n") if history_text and history_text.strip() else ""
    return f"{mem}{hist}{user_query}"

# ============ App state ============
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()
if "notes" not in st.session_state:
    st.session_state.notes = load_notes()
if "memory" not in st.session_state:
    st.session_state.memory = load_memory()
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = next(iter(st.session_state.conversations.keys()))
if "max_turns" not in st.session_state:
    st.session_state.max_turns = 8  # last N turns used for context

def get_active_chat():
    return st.session_state.conversations[st.session_state.active_chat_id]

# ============ Sidebar: Settings ============
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Model", list(HF_MODELS.keys()), index=0)
repo_id = HF_MODELS[model_choice]

# Browsing controls
st.sidebar.markdown("---")
use_browse = st.sidebar.checkbox("🌐 Use Internet (RAG)", value=True)
max_results = st.sidebar.slider("Web results", 1, 8, 4)
if use_browse and not os.getenv("TAVILY_API_KEY"):
    st.sidebar.info("Set TAVILY_API_KEY in environment/secrets to enable web browsing.")
st.sidebar.markdown("Context turns")
st.session_state.max_turns = st.sidebar.slider("Include last N turns", 2, 20, st.session_state.max_turns)

# Build model client
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="conversational",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=256,
    temperature=0.7,
)
chat_model = ChatHuggingFace(llm=llm)

# === Usage panel (local estimate) ===
st.sidebar.markdown("---")
st.sidebar.header("📊 Usage (local)")
used = usage["dollars"]
cap = FREE_CAP if FREE_CAP > 0 else 0.10
st.sidebar.progress(min(used / cap, 1.0))
colA, colB, colC = st.sidebar.columns(3)
colA.metric("HF $ used", f"${used:.4f}")
colB.metric("HF requests", f"{usage['requests']}")
colC.metric("RAG calls", f"{usage['rag_calls']}")

with st.sidebar.expander("Usage settings"):
    st.caption("Local estimate based on COST_PER_REQ and your history. Set env vars HF_COST_PER_REQ / HF_FREE_CAP to override.")
    new_cpr = st.number_input("Cost per request ($)", value=float(COST_PER_REQ), step=0.0001, format="%.6f")
    if new_cpr != COST_PER_REQ:
        COST_PER_REQ = float(new_cpr)
    warn_at = st.slider("Warn at % of free cap", 50, 100, 90, step=5)

# ============ Sidebar: Chats ============
st.sidebar.markdown("---")
st.sidebar.header("🗂️ Chats")

# Select by ID (no collisions)
chat_ids = list(st.session_state.conversations.keys())
chat_ids = sorted(chat_ids, key=lambda cid: (st.session_state.conversations[cid]["title"].lower(), cid))

def _format_chat(cid: str) -> str:
    return st.session_state.conversations[cid]["title"]

selected_cid = st.sidebar.radio(
    "Select a chat",
    options=chat_ids,
    index=chat_ids.index(st.session_state.active_chat_id),
    format_func=_format_chat,
    label_visibility="collapsed",
    key="chat_selector",
)

if selected_cid != st.session_state.active_chat_id:
    st.session_state.active_chat_id = selected_cid
    st.experimental_rerun()

# New / Rename / Delete / Clear This
c1, c2, c3, c4 = st.sidebar.columns([1,1,1,1])
with c1:
    if st.button("➕ New"):
        cid, chat = _new_chat("New chat")
        st.session_state.conversations[cid] = chat
        st.session_state.active_chat_id = cid
        _write_json(STATE_FILE, st.session_state.conversations)
        st.rerun()
with c2:
    with st.popover("✏️ Rename"):
        new_name = st.text_input("New chat name", value=get_active_chat()["title"])
        if st.button("Save"):
            if new_name.strip():
                get_active_chat()["title"] = sanitize_title(new_name)
                _write_json(STATE_FILE, st.session_state.conversations)
                st.rerun()
with c3:
    if st.button("🗑️ Delete"):
        if len(st.session_state.conversations) > 1:
            del st.session_state.conversations[st.session_state.active_chat_id]
            st.session_state.active_chat_id = next(iter(st.session_state.conversations.keys()))
            _write_json(STATE_FILE, st.session_state.conversations)
            st.rerun()
        else:
            st.sidebar.warning("At least one chat is required.")
with c4:
    if st.button("🧹 Clear This"):
        get_active_chat()["messages"].clear()
        _write_json(STATE_FILE, st.session_state.conversations)
        st.rerun()

# Clear ALL (Chats + Notes + Memory)
with st.sidebar.expander("🧨 Clear ALL (Chats + Notes + Memory)"):
    st.write("This will remove all chats, notes, and memory.")
    confirm = st.checkbox("I understand")
    if st.button("Reset Everything", type="secondary", disabled=not confirm):
        cid, chat = _new_chat("First chat")
        st.session_state.conversations = {cid: chat}
        st.session_state.notes = {}
        st.session_state.memory = {"global": "", "per_chat": {}}
        st.session_state.active_chat_id = cid
        _write_json(STATE_FILE, st.session_state.conversations)
        _write_json(NOTES_FILE, st.session_state.notes)
        _write_json(MEM_FILE, st.session_state.memory)
        st.rerun()

# Preview current chat
with st.sidebar.expander("📝 Current chat (last 10)"):
    msgs = get_active_chat()["messages"][-10:]
    if not msgs:
        st.caption("No messages yet.")
    else:
        for m in msgs:
            who = "🧑‍💻" if m["role"] == "user" else "🤖"
            preview = (m["content"][:80] + "…") if len(m["content"]) > 80 else m["content"]
            st.write(f"{who} **{m['role']}**: {preview}")

# ============ Memory ============
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Memory")
global_mem = st.sidebar.text_area("Global Memory (all chats)", value=st.session_state.memory.get("global", ""), height=100)
if global_mem != st.session_state.memory.get("global", ""):
    st.session_state.memory["global"] = global_mem
    _write_json(MEM_FILE, st.session_state.memory)

chat_mem = st.sidebar.text_area("This Chat Memory", value=st.session_state.memory.get("per_chat", {}).get(st.session_state.active_chat_id, ""), height=100)
if st.session_state.memory.get("per_chat", {}).get(st.session_state.active_chat_id, "") != chat_mem:
    st.session_state.memory["per_chat"].setdefault(st.session_state.active_chat_id, "")
    st.session_state.memory["per_chat"][st.session_state.active_chat_id] = chat_mem
    _write_json(MEM_FILE, st.session_state.memory)

# ============ Notepad ============
st.sidebar.markdown("---")
st.sidebar.subheader("🗒️ Notepad (this chat)")
note_key = st.session_state.active_chat_id
if note_key not in st.session_state.notes:
    st.session_state.notes[note_key] = ""
note_text = st.sidebar.text_area("Write notes…", value=st.session_state.notes[note_key], height=160)
if note_text != st.session_state.notes[note_key]:
    st.session_state.notes[note_key] = note_text
    _write_json(NOTES_FILE, st.session_state.notes)

st.sidebar.download_button(
    "⬇️ Export Notes (TXT)",
    data=st.session_state.notes[note_key].encode("utf-8"),
    file_name=f"notes_{note_key[:8]}.txt",
    mime="text/plain"
)

# ============ Export / Import ============
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Export / Import")
export_json = json.dumps(st.session_state.conversations, ensure_ascii=False, indent=2)
st.sidebar.download_button(
    "⬇️ Export All Chats (JSON)",
    data=export_json.encode("utf-8"),
    file_name="conversations.json",
    mime="application/json"
)
uploaded = st.sidebar.file_uploader("⬆️ Import Chats (JSON)", type=["json"])
if uploaded is not None:
    try:
        imported = json.loads(uploaded.read().decode("utf-8"))
        if not isinstance(imported, dict):
            raise ValueError("Root must be an object")
        for cid, chat in imported.items():
            if cid in st.session_state.conversations:
                new_id, _ = _new_chat(chat.get("title", "Imported chat"))
                st.session_state.conversations[new_id] = chat
            else:
                st.session_state.conversations[cid] = chat
        _write_json(STATE_FILE, st.session_state.conversations)
        st.success("Imported chats successfully.")
        st.rerun()
    except Exception as e:
        st.error(f"Import failed: {e}")

# ============ Export Chat to PDF ============
def build_pdf_from_chat(chat_obj, notes_text=""):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import cm

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    body = []

    title = chat_obj["title"] or "Chat Export"
    body.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    body.append(Spacer(1, 12))

    for m in chat_obj["messages"]:
        who = "User" if m["role"] == "user" else "Assistant"
        body.append(Paragraph(f"<b>{who}:</b>", styles["Heading4"]))
        text = m["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace("\n", "<br/>")
        body.append(Paragraph(text, styles["BodyText"]))
        body.append(Spacer(1, 8))

    if notes_text.strip():
        body.append(Spacer(1, 12))
        body.append(Paragraph("<b>Notes</b>", styles["Heading3"]))
        ntext = notes_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        body.append(Paragraph(ntext, styles["BodyText"]))

    doc.build(body)
    buf.seek(0)
    return buf

if st.sidebar.button("🧾 Export Current Chat (PDF)"):
    pdf_buf = build_pdf_from_chat(get_active_chat(), st.session_state.notes.get(note_key, ""))
    st.sidebar.download_button(
        "⬇️ Download PDF",
        data=pdf_buf,
        file_name=f"{get_active_chat()['title'] or 'chat'}.pdf",
        mime="application/pdf"
    )

# ============ Main Chat Window ============
active_chat = get_active_chat()

# Render history bubbles
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        klass = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        st.markdown(f'<div class="chat-bubble {klass}">{msg["content"]}</div>', unsafe_allow_html=True)

# Typewriter animation
def typewriter(placeholder, text: str, min_delay=0.005, max_chunk=4):
    shown = []
    for i, ch in enumerate(text):
        shown.append(ch)
        if (i % max_chunk == 0) or (ch in ".!?"):
            placeholder.markdown(''.join(shown) + "▌")
            time.sleep(min_delay)
    placeholder.markdown(''.join(shown))

# Input
if prompt := st.chat_input("Type your message…"):
    # Add user message
    active_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-bubble user-bubble">{prompt}</div>', unsafe_allow_html=True)

    # Memory & history blocks
    mem_global = st.session_state.memory.get("global", "")
    mem_chat = st.session_state.memory.get("per_chat", {}).get(st.session_state.active_chat_id, "")
    memory_preface = "\n".join(x for x in [mem_global.strip(), mem_chat.strip()] if x).strip()
    history_block = build_history_block(active_chat, max_turns=st.session_state.max_turns)

    # Build model input (with optional web context)
    citations = []
    if use_browse and os.getenv("TAVILY_API_KEY"):
        with st.status("🔎 Searching the web…", expanded=False):
            snips, srcs = web_research(prompt, max_results=max_results)
        if snips:
            model_input = build_rag_prompt_from_history(prompt, snips, memory_preface, history_block)
            citations = srcs
        else:
            model_input = build_plain_prompt_from_history(prompt, memory_preface, history_block)
    else:
        model_input = build_plain_prompt_from_history(prompt, memory_preface, history_block)

    # Soft cap guard before sending paid inference (local estimate)
    soft_cap = (locals().get("warn_at", 90) / 100.0) * FREE_CAP
    can_send = True
    if usage["dollars"] + COST_PER_REQ >= soft_cap:
        with st.chat_message("assistant"):
            st.warning(
                f"Approaching free cap. Estimated after this call: ${usage['dollars'] + COST_PER_REQ:.4f} "
                f"of ${FREE_CAP:.2f}."
            )
            can_send = st.toggle("Force send anyway", value=False)
    if not can_send:
        # Save assistant warning as a message and skip the call
        active_chat["messages"].append({
            "role": "assistant",
            "content": "⏸️ Skipped model call to avoid exceeding the free cap (local estimate). Toggle 'Force send anyway' to proceed."
        })
        auto_rename_chat(active_chat)
        _write_json(STATE_FILE, st.session_state.conversations)
        _write_json(NOTES_FILE, st.session_state.notes)
        _write_json(MEM_FILE, st.session_state.memory)
        st.stop()

    # Assistant response
    with st.chat_message("assistant"):
        ph = st.empty()
        try:
            resp = chat_model.invoke(model_input)
            answer = resp.content
            # record local inference usage
            try:
                add_inference_usage(usage, cost=COST_PER_REQ)
            except Exception:
                pass
        except Exception as e:
            answer = f"⚠️ Error: {e}"

        ph.markdown('<div class="chat-bubble assistant-bubble"></div>', unsafe_allow_html=True)
        inner = st.empty()
        typewriter(inner, answer, min_delay=0.006, max_chunk=3)

    # Save assistant message
    active_chat["messages"].append({"role": "assistant", "content": answer})

    # Show citations (if any)
    if citations:
        with st.expander("🔗 Sources"):
            for i, c in enumerate(citations, 1):
                # fix bracket so link renders correctly
                st.markdown(f"[{i}] {c['title']} ({c['url']})")

    # Auto-rename & persist
    auto_rename_chat(active_chat)
    _write_json(STATE_FILE, st.session_state.conversations)
    _write_json(NOTES_FILE, st.session_state.notes)
    _write_json(MEM_FILE, st.session_state.memory)
