import streamlit as st
import json
import random
import re
import requests

# Load preprocessed chunks
with open("laptop_chunks.json", "r", encoding="utf-8") as f:
    pdf_chunks = json.load(f)

# ========== Helper Functions ==========
def find_relevant_chunk(question, chunks):
    question_keywords = question.lower().split()
    best_score = 0
    best_chunk = chunks[0]
    for chunk in chunks:
        score = sum(1 for word in question_keywords if word in chunk.lower())
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk

def format_response(text):
    text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n\n", text)
    replacements = {
        r"\bCPU\b": "🧠 CPU", r"\bprocessor\b": "🧠 Processor",
        r"\bRAM\b": "💾 RAM", r"\bSSD\b": "💽 SSD",
        r"\bstorage\b": "💽 Storage", r"\bdisplay\b": "🖥️ Display",
        r"\bscreen\b": "🖥️ Screen", r"\bbattery\b": "🔋 Battery",
        r"\bgraphics\b": "🎮 Graphics", r"\bprice\b": "💰 Price",
        r"\bweight\b": "⚖️ Weight",
    }
    for word, emoji in replacements.items():
        text = re.sub(word, emoji, text, flags=re.IGNORECASE)
    return text.strip()

def ask_deepseek(question, context, hf_token):
    url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    prompt = f"""[SYSTEM]
You are a friendly AI assistant who gives casual and helpful laptop advice.
ONLY use the internal knowledge you gain from the info below — but NEVER mention, refer to, or hint at it in your answers.
Avoid formal tones or sign-offs. Be friendly, clear, and conversational.
[INFO SOURCE]
{context}

[USER]
{question}

[ASSISTANT]
"""
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.3, "max_new_tokens": 512, "top_p": 0.9}
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        text = response.json()[0]["generated_text"]
        return format_response(text.split("[ASSISTANT]")[-1])
    else:
        return f"❌ Error {response.status_code}: {response.text}"

# ========== Streamlit UI ==========
st.set_page_config(page_title="Laptop Chatbot", page_icon="💻")
st.title("💻 Laptop Recommendation Chatbot")

hf_token = st.text_input("Enter your HuggingFace API Token", type="password")

if hf_token:
    question = st.text_input("Ask a question about our laptops (e.g., good for video editing?)")

    if question:
        context = find_relevant_chunk(question, pdf_chunks)
        answer = ask_deepseek(question, context, hf_token)
        st.markdown(f"**AI Assistant:**\n\n{answer}")
else:
    st.info("Please enter your HuggingFace token to start chatting.")
