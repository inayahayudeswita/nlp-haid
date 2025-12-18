import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import random

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Chatbot Edukasi Haid",
    page_icon="🌸",
    layout="centered"
)

# =========================
# CUSTOM CSS CHAT
# =========================
st.markdown("""
<style>
.chat-container { max-width: 700px; margin: auto; }
.user-bubble {
    background: #ffd6e8;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    text-align: right;
}
.bot-bubble {
    background: #f1f3f6;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
}
.sender {
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 4px;
    color: #666;
}
.accuracy {
    font-size: 11px;
    color: #888;
    margin-top: 6px;
}
.footer {
    text-align: center;
    color: #999;
    font-size: 12px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<h1 style='text-align:center;'>🌸 Chatbot Edukasi Haid</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;'>Tanyakan apapun tentang menstruasi dan kesehatan reproduksi perempuan</p>", unsafe_allow_html=True)

# =========================
# LOAD KNOWLEDGE BASE
# =========================
@st.cache_data
def load_kb():
    kb = pd.read_csv("knowledge_base_haid.csv")
    kb["content"] = kb["question"] + " " + kb["answer"]
    return kb

kb = load_kb()
documents = kb["content"].tolist()

# =========================
# EMBEDDING & FAISS INDEX
# =========================
@st.cache_resource
def build_index(documents):
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_model.encode(documents, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embed_model

index, embed_model = build_index(documents)

# =========================
# RETRIEVAL FUNCTION + ACCURACY
# =========================
fallbacks = [
    "Maaf, saya belum memiliki jawaban untuk pertanyaan itu.",
    "Hmm, saya tidak yakin dengan jawabannya. Bisa coba pertanyaan lain?",
    "Sayangnya saya belum tahu jawabannya. Coba gunakan kata yang berbeda."
]

def retrieve_answer(query, k=1, similarity_threshold=0.5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)

    # Konversi L2 distance → similarity
    similarity = 1 / (1 + D[0][0])
    accuracy = round(similarity * 100, 2)

    if similarity < similarity_threshold:
        return random.choice(fallbacks), accuracy
    else:
        answer = kb.iloc[I[0][0]]["answer"]
        return answer, accuracy

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# CHAT DISPLAY
# =========================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, msg, acc in st.session_state.messages:
    if role == "user":
        st.markdown(f"""
        <div class="user-bubble">
            <div class="sender">User</div>
            {msg}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-bubble">
            <div class="sender">Bot</div>
            {msg}
            <div class="accuracy">Tingkat akurasi: {acc}%</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# INPUT + BUTTON
# =========================
def send():
    q = st.session_state.input.strip()
    if q:
        a, acc = retrieve_answer(q)
        st.session_state.messages.append(("user", q, None))
        st.session_state.messages.append(("bot", a, acc))
        st.session_state.input = ""

col1, col2 = st.columns([5, 1])
with col1:
    st.text_input("💬 Tulis pertanyaanmu di sini...", key="input")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Kirim", on_click=send)

# =========================
# FOOTER
# =========================
st.markdown(
    "<div class='footer'>⚠️ Chatbot ini hanya untuk edukasi, bukan pengganti konsultasi medis.</div>",
    unsafe_allow_html=True
)
