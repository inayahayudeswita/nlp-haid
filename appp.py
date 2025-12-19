import streamlit as st
import pandas as pd
import faiss
import random
import torch
import os
import zipfile
import gdown

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# MODEL PATH SETUP
# =========================
LOCAL_MODEL_PATH = "qwen_model"

# =========================
# DOWNLOAD MODEL DARI GOOGLE DRIVE
# =========================
def download_and_extract_model_from_drive():
    # URL untuk file ZIP model dari Google Drive
    drive_link = "https://drive.google.com/uc?id=1xT3eVhxpNeXVjcxJVCmtIFm56Y9GFwr-"
    
    # Path lokal untuk menyimpan file ZIP
    zip_file_path = "qwen_model.zip"
    
    # Cek apakah folder model sudah ada
    if not os.path.exists(LOCAL_MODEL_PATH):
        try:
            # Download file dari Google Drive
            gdown.download(drive_link, zip_file_path, quiet=False)
            print("Model berhasil didownload.")
            
            # Ekstrak file ZIP ke folder LOCAL_MODEL_PATH
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(LOCAL_MODEL_PATH)
                print(f"Model berhasil diekstrak ke {LOCAL_MODEL_PATH}.")
            
            # Hapus file ZIP setelah ekstraksi (optional)
            os.remove(zip_file_path)
            print(f"File ZIP {zip_file_path} telah dihapus.")
            
        except Exception as e:
            print(f"Terjadi kesalahan saat mendownload dan mengekstrak model: {str(e)}")
    else:
        print(f"Model sudah ada di {LOCAL_MODEL_PATH}, tidak perlu mengunduh ulang.")

# Pastikan model ada, jika tidak, download dan ekstrak
download_and_extract_model_from_drive()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Chatbot Edukasi Haid",
    page_icon="🌸",
    layout="centered"
)

# =========================
# CSS CHAT
# =========================
st.markdown("""<style>
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
</style>""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<h1 style='text-align:center;'>🌸 Chatbot Edukasi Haid</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;'>RAG Chatbot dengan Qwen Model</p>", unsafe_allow_html=True)

# =========================
# LOAD KB
# =========================
@st.cache_data
def load_kb():
    kb = pd.read_csv("knowledge_base_haid.csv")
    kb["content"] = kb["question"] + " " + kb["answer"]
    return kb

kb = load_kb()
documents = kb["content"].tolist()

# =========================
# EMBEDDING + FAISS
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
# LOAD QWEN MODEL
# =========================
@st.cache_resource
def load_qwen():
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return None, None

tokenizer, qwen_model = load_qwen()
if tokenizer is None or qwen_model is None:
    st.error("Terjadi kesalahan saat memuat model.")

# =========================
# RAG FUNCTION
# =========================
fallbacks = [
    "Maaf, saya belum memiliki informasi yang cukup untuk menjawab pertanyaan tersebut.",
    "Saya belum menemukan jawaban yang sesuai. Silakan coba pertanyaan lain.",
    "Informasi tersebut belum tersedia dalam basis pengetahuan saya."
]

def rag_answer(query, k=1, threshold=0.5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    similarity = 1 / (1 + D[0][0])
    accuracy = round(similarity * 100, 2)

    if similarity < threshold:
        return random.choice(fallbacks), accuracy

    context = kb.iloc[I[0][0]]["content"]
    prompt = f"""Informasi: {context}\nPertanyaan: {query}\nJawaban:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        outputs = qwen_model.generate(**inputs, max_new_tokens=120)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
        st.markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg}<div class='accuracy'>Akurasi: {acc}%</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# INPUT
# =========================
def send():
    q = st.session_state.input.strip()
    if q:
        a, acc = rag_answer(q)
        st.session_state.messages.append(("user", q, None))
        st.session_state.messages.append(("bot", a, acc))
        st.session_state.input = ""

col1, col2 = st.columns([5,1])
with col1:
    st.text_input("💬 Tulis pertanyaanmu di sini...", key="input")
with col2:
    st.button("Kirim", on_click=send)

# =========================
# FOOTER
# =========================
st.markdown("<div class='footer'>⚠️ Chatbot ini hanya untuk edukasi, bukan pengganti konsultasi medis.</div>", unsafe_allow_html=True)
