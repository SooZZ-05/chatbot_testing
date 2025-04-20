import os
import nltk
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
import re
import requests
import random
import pytz
from fpdf import FPDF
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from datetime import datetime
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer


# ===== Load API Key =====
load_dotenv()
hf_token = os.getenv("OPENROUTER_API_KEY", st.secrets.get("OPENROUTER_API_KEY"))

# ===== NLTK Resource Setup =====
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Setup nltk data path
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

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

farewells = [
        "bye", "goodbye", "see ya", "see you", "later",
        "i'm done", "thank you", "thanks, that's all", "talk to you later",
        "exit", "quit", "close", "end", "good night", "goodbye for now"
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

# Define farewell checking function
def is_farewell(user_input):
    user_input = user_input.lower().strip()
    close = get_close_matches(user_input, farewells, cutoff=0.6)
    return bool(close)

# # Ensure required resources are available
# def download_nltk_data():
#     for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
#         try:
#             nltk.data.find(pkg)
#         except LookupError:
#             nltk.download(pkg, download_dir=nltk_data_path)

# download_nltk_data()

# # NLP Word Count Function
# def count_nlp_words(text):
#     tokens = word_tokenize(text)
#     tokens = [w.lower() for w in tokens if w.isalpha()]  # remove punctuation/numbers
#     tokens = [w for w in tokens if w not in stopwords.words("english")]  # remove stopwords
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(w) for w in tokens]  # lemmatize
#     return len(tokens)

# PDF Text Extractor
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# def count_words_from_pdf(uploaded_file):
#     text = extract_text_from_pdf(uploaded_file)
#     return count_nlp_words(text)

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
# def format_response(text):
#     text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n\n", text)
#     text = re.sub(r"●", "\n\n●", text)
#     used_emojis = set()
#     replacements = {
#         r"\bCPU\b": "🧠 CPU", r"\bprocessor\b": "🧠 Processor",
#         r"\bRAM\b": "💾 RAM", r"\bSSD\b": "💽 SSD",
#         r"\bstorage\b": "💽 Storage", r"\bdisplay\b": "🖥️ Display",
#         r"\bscreen\b": "🖥️ Screen", r"\bbattery\b": "🔋 Battery",
#         r"\bgraphics\b": "🎮 Graphics", r"\bprice\b": "💰 Price",
#         r"\bweight\b": "⚖️ Weight",
#     }
#     for word, emoji in replacements.items():
#         if emoji not in used_emojis:
#             text = re.sub(word, emoji, text, count=1, flags=re.IGNORECASE)
#             used_emojis.add(emoji)
#     text = re.sub(r'\n{3,}', '\n\n', text)
#     return text.strip()

# def truncate_text(text, limit=1500):
#     if len(text) <= limit:
#         return text
#     return text[:limit] + "..."

def format_response(text):
    # Add double newline after sentence-ending punctuation followed by a capital letter
    text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n\n", text)
    
    # Format bullet points (e.g., "●" should be followed by a newline for better formatting)
    text = re.sub(r"●", "\n\n●", text)
    
    # Define a set to track used emojis (to avoid multiple replacements for the same keyword)
    used_emojis = set()
    
    # Mapping of words to emojis
    replacements = {
        r"\bCPU\b": "🧠 CPU", 
        r"\bprocessor\b": "🧠 Processor",
        r"\bRAM\b": "💾 RAM", 
        r"\bSSD\b": "💽 SSD",
        r"\bstorage\b": "💽 Storage", 
        r"\bdisplay\b": "🖥️ Display",
        r"\bscreen\b": "🖥️ Screen", 
        r"\bbattery\b": "🔋 Battery",
        r"\bgraphics\b": "🎮 Graphics", 
        r"\bprice\b": "💰 Price",
        r"\bweight\b": "⚖️ Weight",
    }
    
    # Perform replacements while ensuring no emoji is replaced more than once
    for word, emoji in replacements.items():
        if emoji not in used_emojis:
            text = re.sub(word, emoji, text, count=1, flags=re.IGNORECASE)
            used_emojis.add(emoji)
    
    # Add RM symbol to prices (assuming the price is a numeric value)
    text = re.sub(r"(\bprice\b.*?)(\d+)", r"\1RM \2", text)
    
    # Remove excessive newlines (more than two consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Return the formatted text with any leading/trailing whitespace stripped
    return text.strip()

def truncate_text(text, limit=1500):
    # If the text is within the limit, return it as is
    if len(text) <= limit:
        return text
    
    # Otherwise, truncate the text and append "..." to indicate more content
    return text[:limit] + "..."

# ===== Chat Saving Button =====
def estimate_multicell_height(pdf, text, width, line_height):
    lines = pdf.multi_cell(width, line_height, text, split_only=True)
    return len(lines) * line_height + 4  # +4 for padding

def save_chat_to_pdf(chat_history):
    from fpdf import FPDF
    from datetime import datetime
    import pytz
    from io import BytesIO
    import re

    def strip_emojis(text):
        return re.sub(r'[^\x00-\x7F]+', '', text)

    def remove_newlines(text):
        return re.sub(r'\s*\n\s*', ' ', text.strip())

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)
    page_height = 297  # A4 height in mm
    margin_top = 10
    margin_bottom = 10
    usable_height = page_height - margin_top - margin_bottom
    line_height = 8
    box_spacing = 6
    box_width = 190

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Chat History", ln=True, align="C")
    pdf.set_font("Arial", '', 10)
    malaysia_time = datetime.now(pytz.timezone("Asia/Kuala_Lumpur")).strftime("%B %d, %Y %H:%M")
    pdf.cell(0, 10, f"Exported on {malaysia_time} (MYT)", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)

    for entry in chat_history:
        user_msg = strip_emojis(entry["user"]).strip()
        assistant_msg = remove_newlines(strip_emojis(entry["assistant"]).strip())

        label_user = f"You:\n{user_msg}"
        label_assistant = f"Assistant:\n{assistant_msg}"

        # Estimate heights
        user_box_height = estimate_multicell_height(pdf, label_user, box_width, line_height)
        assistant_box_height = estimate_multicell_height(pdf, label_assistant, box_width, line_height)
        total_pair_height = user_box_height + assistant_box_height + box_spacing

        # If not enough space, start new page
        if pdf.get_y() + total_pair_height > usable_height:
            pdf.add_page()

        # Render You box
        y_start = pdf.get_y()
        pdf.rect(10, y_start, box_width, user_box_height)
        pdf.set_xy(12, y_start + 2)
        pdf.multi_cell(0, line_height, label_user)
        pdf.ln(2)

        # Render Assistant box
        y_start = pdf.get_y()
        pdf.rect(10, y_start, box_width, assistant_box_height)
        pdf.set_xy(12, y_start + 2)
        pdf.set_text_color(0, 102, 204)
        pdf.multi_cell(0, line_height, label_assistant)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # Output PDF
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
             # Farewell check
        elif is_farewell(question):
            ai_reply = "👋 Alright, take care! Let me know if you need help again later. Bye!"
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
