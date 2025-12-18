import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# LOAD KNOWLEDGE BASE CSV
# =========================
print("Loading knowledge base...")
kb = pd.read_csv("knowledge_base_haid.csv")

# Gabungkan Q&A jadi satu konteks
kb["content"] = kb["question"] + " " + kb["answer"]

# Ambil beberapa data untuk uji (misalnya 3 data pertama)
test_data = kb.sample(3, random_state=42)

# =========================
# LOAD QWEN MODEL
# =========================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# =========================
# UJI MODEL DENGAN CSV
# =========================
print("\n=== UJI MODEL DENGAN DATA CSV ===\n")

for i, row in test_data.iterrows():
    context = row["content"]
    question = row["question"]

    prompt = f"""
Anda adalah chatbot edukasi kesehatan reproduksi perempuan.
Gunakan informasi berikut untuk menjawab pertanyaan dengan jelas dan sopan.

Informasi:
{context}

Pertanyaan:
{question}

Jawaban:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Jawaban:")[-1].strip()

    print("Pertanyaan:", question)
    print("Jawaban Model:", answer)
    print("-" * 60)

# =========================
# SIMPAN MODEL
# =========================
SAVE_PATH = "./qwen_model"

print("\nMenyimpan model ke folder:", SAVE_PATH)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("Model berhasil diuji dan disimpan!")
