import os
import nltk
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
import re
import requests
import random
import pytz
import string
from fpdf import FPDF
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from datetime import datetime
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.corpus import stopwords
import pdfplumber
import openai

# Load API Key from environment
load_dotenv()
hf_token = os.getenv("OPENROUTER_API_KEY", st.secrets.get("OPENROUTER_API_KEY"))

# NLTK Setup
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]

    return ' '.join(filtered_words)

# Chunking PDF content into smaller sections for better relevance matching
def chunk_text(text, chunk_size=3000, overlap=500):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Function to extract keywords using TF-IDF
def extract_keywords_tfidf(text, top_n=30):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray().flatten()
    keywords = np.array(vectorizer.get_feature_names_out())
    
    top_indices = scores.argsort()[-top_n:][::-1]
    top_keywords = keywords[top_indices]
    
    return top_keywords.tolist()

# Function to query the OpenRouter API and get response
def ask_llm_with_history(question, context, history, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Constructing the message body for OpenRouter API
    messages = [{"role": "system", "content": 
        "You are a friendly assistant, helping users find relevant information about laptops from uploaded PDFs."}]
    
    for entry in history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})

    messages.append({"role": "user", "content": question})

    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 200
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error {response.status_code}: {response.text}"

# Function to find the relevant chunk based on the user's query using semantic search
def find_relevant_chunk(question, chunks, api_key):
    # Query the OpenRouter API to analyze the question and the document chunks
    context = "\n".join(chunks)
    response = ask_llm_with_history(question, context, [], api_key)
    
    return response

# Function to format the response with emoji and other formatting
def format_response(text):
    text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n\n", text)
    text = re.sub(r"●", "\n\n●", text)
    
    used_emojis = set()
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
    
    for word, emoji in replacements.items():
        if emoji not in used_emojis:
            text = re.sub(word, emoji, text, count=1, flags=re.IGNORECASE)
            used_emojis.add(emoji)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

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

# Streamlit UI
st.set_page_config(page_title="💻 Laptop Chatbot", page_icon="💬", layout="wide")
st.title("💻 Laptop Specification Chatbot")

# PDF file uploader
uploaded_files = st.file_uploader("📄 Upload Laptop Specification PDFs", type=["pdf"], accept_multiple_files=True)

if "history" not in st.session_state:
    st.session_state.history = []

if hf_token and uploaded_files:
    with st.spinner("🔍 Extracting and processing your documents..."):
        all_text = ""
        for uploaded_file in uploaded_files:
            document_text = extract_text_from_pdf(uploaded_file)
            all_text += document_text + "\n\n"  # Combine the text from all PDFs
        pdf_chunks = chunk_text(all_text)
        keywords = extract_keywords_tfidf(all_text, top_n=30)

    # Display chat interface
    st.subheader("🧠 Chat with your PDF")
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(entry["user"])
        with st.chat_message("assistant"):
            short_reply = format_response(entry["assistant"])
            st.write(short_reply)
            if len(entry["assistant"]) > 1500:
                with st.expander("🔎 View full response"):
                    st.write(entry["assistant"])

    # User input for question
    question = st.chat_input("💬 Your message")

    if question:
        if any(greeting in question.lower() for greeting in ["hi", "hello", "hey"]):
            ai_reply = "Hello! How can I help you with laptop specifications today?"
        elif any(farewell in question.lower() for farewell in ["bye", "goodbye"]):
            ai_reply = "👋 Goodbye! Come back anytime if you need more help!"
        else:
            with st.spinner("🤔 Analyzing your question..."):
                # Find the most relevant chunk and ask the OpenRouter API
                context = find_relevant_chunk(question, pdf_chunks, hf_token)
                ai_reply = format_response(context)

        st.session_state.history.append({"user": question, "assistant": ai_reply})
        st.rerun()

    # Save the chat history as a PDF
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
