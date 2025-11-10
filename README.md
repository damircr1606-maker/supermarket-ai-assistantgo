# supermarket-ai-assistantgo
ИИ ассистент для супермаркетов
# supermarket-ai-assistant
ИИ ассистент для супермаркетов
supermarket-ai-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api.py
│   │   ├── models.py
│   │   ├── rag.py
│   │   ├── prompt_manager.py
│   │   ├── db.py
│   │   └── config.py
│   ├── scripts/
│   │   ├── index_docs.py
│   │   └── sample_docs/
│   │       ├── store_policies.md
│   │       ├── products.csv
│   │       └── faq.md
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── package.json
│   ├── index.html
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Chat.jsx
│   │   │   └── Message.jsx
│   │   └── api.js
│   └── Dockerfile
├── docker-compose.yml
├── README.md
└── tests/
    ├── test_api.py
    └── test_rag.py
# Supermarket AI Assistant - MVP

Стек: FastAPI + Python (RAG local: sentence-transformers + FAISS) + React (Vite).

Запуск локально (dev):
1. Создать виртуальное окружение и установить зависимости для backend:
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Индексировать документы (пример):
   python scripts/index_docs.py

3. Запустить backend:
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

4. Запустить frontend:
   cd frontend
   npm install
   npm run dev

Или:
docker-compose up --build
from pydantic import BaseSettings

class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K: int = 4
    FAISS_INDEX_PATH: str = "faiss_index.index"
    DOCS_META_PATH: str = "docs_meta.json"
    OPENAI_API_KEY: str | None = None
    MAX_CONTEXT_TOKENS: int = 1500

    class Config:
        env_file = "../.env"

settings = Settings()
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "assistant.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      history TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_session(session_id: str, history: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO sessions (id, history) VALUES (?, ?)", (session_id, history))
    conn.commit()
    conn.close()

def load_session(session_id: str) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT history FROM sessions WHERE id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

# initialize on import
init_db()
import os
import json
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from .config import settings

class RAG:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.index = None
        self.meta = []
        if os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.DOCS_META_PATH):
            self._load_index()

    def _load_index(self):
        self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
        with open(settings.DOCS_META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def save_index(self, index, meta):
        faiss.write_index(index, settings.FAISS_INDEX_PATH)
        with open(settings.DOCS_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.index = index
        self.meta = meta

    def create_index(self, texts: List[str]):
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        meta = [{"text": t} for t in texts]
        self.save_index(index, meta)

    def search(self, query: str, k: int = None) -> List[Tuple[str, float]]:
        if self.index is None:
            return []
        k = k or settings.TOP_K
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.meta):
                results.append((self.meta[idx]["text"], float(dist)))
        return results

rag = RAG()
from .config import settings

SYSTEM_PROMPT = """
You are Supermarket Assistant — concise, accurate and helpful for supermarket staff and customers.
Always answer based on the provided context documents. If the answer is not in the context, say "Не могу найти ответ в базе знаний, уточните пожалуйста" and offer to create a ticket.
Answer in Russian. Provide short actionable steps if appropriate.
"""

def build_prompt(user_question: str, retrieved_contexts: list[str], history: str | None = None) -> str:
    ctx = "\n\n---context---\n".join(retrieved_contexts) if retrieved_contexts else ""
    history_block = f"\n\n---history---\n{history}" if history else ""
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{ctx}\n\nUser: {user_question}{history_block}\n\nAssistant:"
    return prompt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .rag import rag
from .prompt_manager import build_prompt
from .db import load_session, save_session
from .config import settings
import os, subprocess, json

router = APIRouter()

class MessageRequest(BaseModel):
    session_id: str
    message: str

class MessageResponse(BaseModel):
    reply: str
    contexts: list[str] = []

@router.post("/api/message", response_model=MessageResponse)
async def chat(req: MessageRequest):
    # 1) Load session history
    history = load_session(req.session_id) or ""
    # 2) Retrieve relevant docs from RAG
    retrieved = rag.search(req.message, k=settings.TOP_K)
    contexts = [t for t, _ in retrieved]
    # 3) Build prompt
    prompt = build_prompt(req.message, contexts, history)
    # 4) Call LLM: default — local fallback via HuggingFace/transformers not included; 
    #    Here we use OpenAI if key provided, otherwise simple echo for MVP.
    reply_text = ""
    if settings.OPENAI_API_KEY:
        import openai
        openai.api_key = settings.OPENAI_API_KEY
        resp = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.2,
        )
        reply_text = resp.choices[0].text.strip()
    else:
        # MVP fallback — concise heuristic (you should replace with API)
        reply_text = "MVP: Ответ пока генерируется локально. Контекст(ы):\n" + ("\n---\n".join([c[:400] for c in contexts]))[:1500]

    # 5) Save session (append)
    new_history = (history + "\nUser: " + req.message + "\nAssistant: " + reply_text)[-2000:]
    save_session(req.session_id, new_history)

    return MessageResponse(reply=reply_text, contexts=contexts)

@router.get("/api/health")
async def health():
    return {"status": "ok"}
from fastapi import FastAPI
from .api import router as api_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Supermarket AI Assistant")
н
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # поменяй для продакшна
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
import os
from pathlib import Path
from app.rag import RAG

def load_sample_texts():
    base = Path(__file__).parent / "sample_docs"
    texts = []
    for p in base.glob("*.md"):
        texts.append(p.read_text(encoding="utf-8"))
    # products.csv -> load as short descriptions per line
    prod = base / "products.csv"
    if prod.exists():
        for line in prod.read_text(encoding="utf-8").splitlines()[1:]:
            # crude: CSV: id,name,category,description,price
            parts = line.split(",")
            if len(parts) >= 4:
                texts.append(f"Product: {parts[1]}\nCategory: {parts[2]}\nDescription: {parts[3]}")
    return texts

if __name__ == "__main__":
    texts = load_sample_texts()
    print(f"Indexing {len(texts)} documents...")
    rag = RAG()
    rag.create_index(texts)
    print("Index saved.")
fastapi
uvicorn[standard]
sentence-transformers
faiss-cpu
pydantic
openai
OPENAI_API_KEY=
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=4
FAISS_INDEX_PATH=faiss_index.index
DOCS_META_PATH=docs_meta.json
{
  "name": "supermarket-assistant-frontend",
  "version": "0.0.1",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "vite": "^5.0.0"
  }
}
export async function sendMessage(sessionId, text) {
  const res = await fetch("http://localhost:8000/api/message", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message: text })
  });
  return res.json();
}
import React, {useState} from "react";
import { sendMessage } from "../api";

export default function Chat(){
  const [sessionId] = useState(() => "sess_"+Math.random().toString(36).slice(2,9));
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState("");

  async function onSend(){
    if(!text) return;
    const userMsg = { role: "user", text };
    setMessages(m => [...m, userMsg]);
    setText("");
    const resp = await sendMessage(sessionId, text);
    setMessages(m => [...m, userMsg, { role:"assistant", text: resp.reply }]);
  }

  return (
    <div style={{maxWidth:800, margin:"20px auto", padding:20, border:"1px solid #eee", borderRadius:8}}>
      <h2>Supermarket Assistant (MVP)</h2>
      <div style={{minHeight:200, border:"1px solid #ddd", padding:10, marginBottom:10, overflowY:"auto"}}>
        {messages.map((m,i)=>(
          <div key={i} style={{marginBottom:8}}>
            <b>{m.role === "user" ? "Вы" : "Ассистент"}:</b> <div style={{whiteSpace:"pre-wrap"}}>{m.text}</div>
          </div>
        ))}
      </div>
      <div style={{display:"flex", gap:8}}>
        <input value={text} onChange={e=>setText(e.target.value)} style={{flex:1}} placeholder="Напишите вопрос..." />
        <button onClick={onSend}>Отправить</button>
      </div>
    </div>
  )
}
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM node:18-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]
version: "3.8"
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/app
    depends_on:
      - backend
