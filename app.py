import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

import streamlit as st
import fitz  # PyMuPDF
import re
import requests
import random
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches

# ========== Greeting Logic ==========
lemmatizer = WordNetLemmatizer()

greeting_responses = [
    "Hi there! How can I assist you with choosing a laptop today?",
    "Hello! Looking for something for work, study, or gaming?",
    "Hey! Need help picking the right laptop for your needs?",
    "Hi! I can help you find a laptop that fits your budget and usage.",
    "Hello! What kind of tasks do you plan to use your laptop for?",
    "Hi! Would you like recommendations for student, business, or gaming laptops?"
]

greeting_keywords = [
    "hi", "hello", "hey", "heyy", "helloo", "hellooo", "helo", "hii", "yo", "hiya", "sup", "what's up",
    "howdy", "good morning", "good evening", "good afternoon", "how are you", "how's it going"
]

category_suggestion = (
    "Would you like suggestions for laptops used in:\n"
    "1. Study 📚\n2. Business 💼\n3. Gaming 🎮\nJust let me know!"
)

def is_greeting_or_smalltalk(user_input):
    user_input = user_input.lower().strip()
    close = get_close_matches(user_input, greeting_keywords, cutoff=0.6)
    return bool(close)

def get_random_greeting():
    return random.choice(greeting_responses)

# ========== PDF Handling ==========
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=3000, overlap=500):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

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

# ========== LLM Logic ==========
def ask_llm_with_history(question, context, history, openrouter_api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": 
        "You are a friendly AI assistant who gives casual and helpful laptop advice. "
        "ONLY use the internal knowledge you gain from the info below — but NEVER mention, refer to, or hint at it in your answers. "
        "Avoid formal tones or sign-offs. Be friendly, clear, and conversational.\n\n"
        f"[INFO SOURCE]\n{context}"}]

    for entry in history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})

    messages.append({"role": "user", "content": question})

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": messages,
        "temperature": 0.3,
        "top_p": 0.9
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return format_response(response.json()["choices"][0]["message"]["content"])
    else:
        return f"❌ Error {response.status_code}: {response.text}"

# ========== Emoji Formatting ==========
def format_response(text):
    text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n\n", text)
    text = re.sub(r"●", "\n\n●", text)
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

    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ========== Streamlit UI ==========
def truncate_text(text, limit=1500):
    if len(text) <= limit:
        return text
    return text[:limit] + "..."

st.set_page_config(page_title="💻 Laptop Chatbot", page_icon="💬", layout="wide")
st.title("💻 Laptop Recommendation Chatbot")

hf_token = st.text_input("🔑 Enter your OpenRouter API Key", type="password")
uploaded_file = st.file_uploader("📄 Upload a Laptop Specification PDF", type=["pdf"])

# Initialize session state for conversation
if "history" not in st.session_state:
    st.session_state.history = []

if hf_token and uploaded_file:
    with st.spinner("🔍 Extracting and processing your document..."):
        document_text = extract_text_from_pdf(uploaded_file)
        pdf_chunks = chunk_text(document_text)

    with st.container():
        st.subheader("🧠 Chat with your PDF")

        for entry in st.session_state.history:
            with st.chat_message("user"):
                st.markdown(entry["user"])
            with st.chat_message("assistant"):
                short_reply = truncate_text(entry["assistant"])
                st.write(short_reply)

                if len(entry["assistant"]) > 1500:
                    with st.expander("🔎 View full response"):
                        st.write(entry["assistant"])

        question = st.chat_input("💬 Your message")

        if question:
            if is_greeting_or_smalltalk(question):
                ai_reply = f"{get_random_greeting()}\n\n{category_suggestion}"
            else:
                with st.spinner("🤔 Thinking..."):
                    context = find_relevant_chunk(question, pdf_chunks)
                    ai_reply = ask_llm_with_history(question, context, st.session_state.history, hf_token)

            st.session_state.history.append({"user": question, "assistant": ai_reply})
            st.rerun()

elif not hf_token:
    st.info("Please enter your HuggingFace API token to start chatting.")
elif not uploaded_file:
    st.info("Please upload a PDF with laptop specifications.")
