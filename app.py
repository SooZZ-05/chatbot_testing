import os
import nltk
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
import re
import requests
import random
import pytz
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from io import BytesIO
from datetime import datetime

# ===== Load API Key =====
load_dotenv()
hf_token = os.getenv("OPENROUTER_API_KEY", st.secrets.get("OPENROUTER_API_KEY"))

# ===== NLTK Resource Setup =====
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ===== Greeting Logic =====
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
    "Would you like suggestions for laptops used in:\n\n"
    "1. Study 📚\n"
    "2. Business 💼\n"
    "3. Gaming 🎮\n\n"
    "Just let me know! 😃"
)

def is_greeting_or_smalltalk(user_input):
    user_input = user_input.lower().strip()
    close = get_close_matches(user_input, greeting_keywords, cutoff=0.6)
    return bool(close)

def get_random_greeting():
    return random.choice(greeting_responses)

# ===== PDF Handling =====
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
    documents = chunks + [question]
    vectorizer = TfidfVectorizer().fit(documents)
    chunk_vectors = vectorizer.transform(chunks)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
    best_index = similarities.argmax()
    return chunks[best_index]

# ===== LLM Logic =====
def ask_llm_with_history(question, context, history, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
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

# ===== Emoji Formatting =====
def format_response(text):
    text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n\n", text)
    text = re.sub(r"●", "\n\n●", text)
    used_emojis = set()
    replacements = {
        r"\bCPU\b": "🧠 CPU", r"\bprocessor\b": "🧠 Processor",
        r"\bRAM\b": "💾 RAM", r"\bSSD\b": "💽 SSD",
        r"\bstorage\b": "💽 Storage", r"\bdisplay\b": "🖥️ Display",
        r"\bscreen\b": "🖥️ Screen", r"\bbattery\b": "🔋 Battery",
        r"\bgraphics\b": "🎮 Graphics", r"\bprice\b": "💰 Price",
        r"\bweight\b": "⚖️ Weight",
    }
    for word, emoji in replacements.items():
        if emoji not in used_emojis:
            text = re.sub(word, emoji, text, count=1, flags=re.IGNORECASE)
            used_emojis.add(emoji)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def truncate_text(text, limit=1500):
    if len(text) <= limit:
        return text
    return text[:limit] + "..."

# ===== Chat Saving Button =====
def save_chat_to_pdf(chat_history):
    def strip_emojis(text):
        return re.sub(r'[^\x00-\x7F]+', '', text)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ===== Header =====
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 10, "Chat History", ln=True, align="C")

    pdf.set_font("Arial", '', 10)
    malaysia_time = datetime.now(pytz.timezone("Asia/Kuala_Lumpur")).strftime("%B %d, %Y %H:%M")
    pdf.cell(0, 10, f"Exported on {malaysia_time} (MYT)", ln=True, align="C")
    pdf.ln(5)

    for idx, entry in enumerate(chat_history):
        user_msg = strip_emojis(entry['user']).strip()
        bot_msg = strip_emojis(entry['assistant']).strip()

        # Alternate backgrounds for each pair
        r, g, b = (245, 245, 245) if idx % 2 == 0 else (255, 255, 255)

        # User message block
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, "You:", ln=True, fill=True)

        pdf.set_font("Arial", '', 11)
        pdf.set_fill_color(r, g, b)
        pdf.multi_cell(0, 8, user_msg, fill=True)
        pdf.ln(1)

        # Assistant message block
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 8, "Assistant:", ln=True, fill=True)

        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(r, g, b)
        pdf.multi_cell(0, 8, bot_msg, fill=True)
        pdf.ln(3)

        # Divider line
        pdf.set_draw_color(210, 210, 210)
        pdf.set_line_width(0.3)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)
    
# ===== Streamlit UI =====
st.set_page_config(page_title="💻 Laptop Chatbot", page_icon="💬", layout="wide")
st.title("💻 Laptop Recommendation Chatbot")

uploaded_file = st.file_uploader("📄 Upload a Laptop Specification PDF", type=["pdf"])

if "history" not in st.session_state:
    st.session_state.history = []

if hf_token and uploaded_file:
    with st.spinner("🔍 Extracting and processing your document..."):
        document_text = extract_text_from_pdf(uploaded_file)
        pdf_chunks = chunk_text(document_text)

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
            greeting = get_random_greeting()
            if "recommendation" not in greeting.lower() and "suggestion" not in greeting.lower():
                greeting += "\n\n" + category_suggestion
            ai_reply = greeting
        else:
            with st.spinner("🤔 Thinking..."):
                context = find_relevant_chunk(question, pdf_chunks)
                ai_reply = ask_llm_with_history(question, context, st.session_state.history, hf_token)

        st.session_state.history.append({"user": question, "assistant": ai_reply})
        st.rerun()

    #save chat to pdf
    with st.sidebar:
        st.markdown("### 💬 Options")
        if st.session_state.history:
            pdf_file = save_chat_to_pdf(st.session_state.history)
            st.download_button(
                label="📥 Download PDF",
                data=pdf_file,
                file_name="chat_history.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("No conversation to download yet!")

elif not hf_token:
    st.error("🔐 API key not found.")
elif not uploaded_file:
    st.info("Please upload a PDF with laptop specifications.")
