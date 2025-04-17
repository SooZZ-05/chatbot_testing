import os
import nltk
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
import re
import requests
import random
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from io import BytesIO
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet

# ===== Load API Key =====
load_dotenv()
hf_token = os.getenv("OPENROUTER_API_KEY", st.secrets.get("OPENROUTER_API_KEY"))

# ===== NLTK Resource Setup =====
NLTK_DATA_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)

for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_DIR)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# ===== Greeting Logic =====
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
    
# === Helper for POS mapping ===
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# === Domain-Specific Synonym Mapping ===
synonym_map = {
    "notebook": "laptop",
    "macbook": "laptop",
    "apple": "macbook",
    "macos": "macbook",
    "windows": "os",
    "linux": "os",
    "intel": "cpu",
    "amd": "cpu",
    "ryzen": "cpu",
    "core": "cpu",
    "processor": "cpu",
    "cpu": "cpu",
    "gpu": "graphics",
    "graphics": "graphics",
    "nvidia": "graphics",
    "geforce": "graphics",
    "radeon": "graphics",
    "memory": "ram",
    "ram": "ram",
    "hdd": "storage",
    "ssd": "storage",
    "storage": "storage",
    "battery": "battery",
    "screen": "display",
    "display": "display",
    "resolution": "display",
    "inch": "display",
    "weight": "weight",
    "portable": "weight",
    "light": "weight",
    "heavy": "weight",
    "price": "price",
    "cost": "price",
    "budget": "price"
}

# === NLP Preprocessing ===
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stop_words]
    tagged = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]

    normalized = [synonym_map.get(word, word) for word in lemmatized]

    return ' '.join(normalized)

def is_comparison_query(query):
    comparison_keywords = ["compare", "difference", "versus", "vs", "better than", "which is better"]
    return any(kw in query.lower() for kw in comparison_keywords)

def generate_comparison_table(keywords, chunks):
    headers = ["Model", "Category", "CPU", "RAM", "Storage", "Display", "GPU", "Battery", "Weight", "Price"]
    rows = []

    for chunk in chunks:
        match = any(kw.lower() in chunk.lower() for kw in keywords)
        if match:
            row = ["-"] * len(headers)
            row[0] = next((kw for kw in keywords if kw.lower() in chunk.lower()), "Unknown")
            for i, field in enumerate(headers[1:], 1):
                pattern = re.search(f"{field}[^\\n]*", chunk, re.IGNORECASE)
                if pattern:
                    row[i] = pattern.group().split(":")[-1].strip()
            rows.append(row)

    if not rows:
        return "⚠️ Could not extract detailed comparison."
    
    table_md = "| " + " | ".join(headers) + " |\n"
    table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        table_md += "| " + " | ".join(row) + " |\n"
    return table_md

    
# ===== PDF Handling =====
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_multiple_pdfs(uploaded_files):
    combined_text = ""
    for file in uploaded_files:
        raw = extract_text_from_pdf(file)
        cleaned = preprocess_text(raw)
        combined_text += cleaned + "\n\n"
    return chunk_text(combined_text)

def chunk_text(text, chunk_size=3000, overlap=500):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def find_relevant_chunk(question, chunks):
    preprocessed_question = preprocess_text(question)  # 🔧 preprocess question
    documents = chunks + [preprocessed_question]
    vectorizer = TfidfVectorizer().fit(documents)
    chunk_vectors = vectorizer.transform(chunks)
    question_vector = vectorizer.transform([preprocessed_question])
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
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Chat History", ln=True, align="C")
    pdf.ln(5)

    for idx, entry in enumerate(chat_history, 1):
        user_msg = strip_emojis(entry['user']).strip()
        bot_msg = strip_emojis(entry['assistant']).strip()

        # User Message
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(33, 33, 33)
        pdf.cell(0, 8, "You:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, user_msg)
        pdf.ln(2)

        # Bot Message
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 8, "Assistant:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, bot_msg)
        pdf.ln(5)

        # Divider
        pdf.set_draw_color(220, 220, 220)
        pdf.set_line_width(0.3)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

    # Export safely
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)
    
# ===== Streamlit UI =====
st.set_page_config(page_title="💻 Laptop Chatbot", page_icon="💬", layout="wide")
st.title("💻 Laptop Recommendation Chatbot")

uploaded_files = st.file_uploader("📄 Upload One or More PDF Documents", type=["pdf"], accept_multiple_files=True)

if "history" not in st.session_state:
    st.session_state.history = []

if hf_token and uploaded_files:
    with st.spinner("🔍 Extracting and processing documents..."):
        pdf_chunks = process_multiple_pdfs(uploaded_files)

    st.subheader("🧠 Chat with your PDF")
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(entry["user"])
        with st.chat_message("assistant"):
            short_reply = truncate_text(entry["assistant"])
            st.markdown(short_reply)
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
                if is_comparison_query(question):
                    # Extract keywords from the question for comparison
                    keywords = [synonym_map.get(word.lower(), word.lower()) 
                                for word in word_tokenize(question) if word.isalnum()]
                    table = generate_comparison_table(keywords, pdf_chunks)
                    ai_reply = "Here’s a comparison based on the available data:\n\n"
                    st.markdown(ai_reply)
                    st.markdown(table)
                else:
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
elif not uploaded_files:
    st.info("Please upload a PDF with laptop specifications.")



