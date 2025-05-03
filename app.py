import os
import nltk
import streamlit as st
from dotenv import load_dotenv
import fitz
import re
import requests
import random
import pytz
import string
import numpy as np
import pdfplumber
import faiss

import streamlit.components.v1 as components
import base64
from gtts import gTTS

from fpdf import FPDF
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from datetime import datetime
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ===== Load API Key =====
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))

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

def is_follow_up_question(question, history, embedding_model, faiss_index, pdf_chunks):
    # Define possible follow-up indicators
    follow_up_phrases = [
        "any more",
        "what else",
        "another example",
        "can you show more",
        "give me more",
        "anything else"
    ]
    
    # Convert the user's question and follow-up phrases to embeddings
    question_embedding = embedding_model.encode([question.lower()])
    phrases_embeddings = embedding_model.encode(follow_up_phrases)
    
    # Compute cosine similarity between the question and each follow-up phrase
    similarities = cosine_similarity(question_embedding, phrases_embeddings)
    print(f"Similarities for '{question}': {similarities}")
    
    # Check if the maximum similarity is above a threshold
    if np.max(similarities) > 0.75:
        return True
    
    return False

def handle_follow_up_question(question, history, pdf_chunks, embedding_model, faiss_index):
    # Handle follow-up questions by providing more examples or details
    if is_follow_up_question(question, history, embedding_model, faiss_index, pdf_chunks):
        # Retrieve additional relevant chunks from the PDF
        query_embedding = embedding_model.encode([question])[0]
        relevant_chunk_indices = search_faiss(query_embedding, faiss_index, k=5)
        relevant_chunks = [pdf_chunks[i] for i in relevant_chunk_indices]
        additional_info = "\n".join(relevant_chunks)
        return additional_info

    # If no follow-up detected, ask for clarification
    return "❓ Could you clarify your request? I need more context to provide additional information."

def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    return text

def chunk_text_by_paragraph(text, chunk_size=3000, overlap=500):
    paragraphs = text.split("\n\n")  # Split the text into paragraphs
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_size:
            chunks.append(current_chunk)  # Add the current chunk to the list
            current_chunk = paragraph  # Start a new chunk
        else:
            current_chunk += "\n\n" + paragraph  # Add the paragraph to the current chunk
    
    # Add the last chunk if it's non-empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # Handle overlap by combining the last `overlap` characters from the previous chunk with the next
    if overlap > 0:
        for i in range(1, len(chunks)):
            chunks[i] = chunks[i-1][-overlap:] + chunks[i]
    
    return chunks

# Replace this function to use OpenAI API
def ask_llm_with_history(question, context, history, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    context = truncate_text(context, limit=1000)

    messages = [{"role": "system", "content": 
        "You are a friendly AI assistant that answers **briefly and directly**. "
        "Answer user questions based ONLY on the info below. "
        "Give **concise** answers, not more than **3 short sentences**."
        "Use maximum 3 short sentences or a simple table if needed."
        "Avoid unnecessary explanations, greetings, or sign-offs.\n\n"
        f"[INFO SOURCE]\n{context}"}]

    for entry in history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})

    messages.append({"role": "user", "content": question})

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return format_response(response.json()["choices"][0]["message"]["content"])
    else:
        return f"❌ Error {response.status_code}: {response.text}"

def is_relevant_question(question, pdf_chunks, keywords, faiss_index):
    # Combine keywords from the uploaded PDF and additional keywords
    additional_keywords = ["study", "business", "gaming", "laptop", "processor", "ram", "ssd", "battery", "weight", "price", "graphics", "display", "screen", "documents", "pdf", "similarities", "differences", "compare", "summary", "count"]
    relevant_keywords = keywords + additional_keywords  # Combine PDF and additional keywords
    
    # Check if any relevant keyword is in the question
    question = question.lower()
    if any(keyword in question for keyword in relevant_keywords):
        return True

    # If no relevant keyword, use FAISS to find semantically relevant content in chunks
    question_embedding = embedding_model.encode([question])[0]
    relevant_chunk_indices = search_faiss(question_embedding, faiss_index, k=3)

    # Check if relevant chunks are found
    if relevant_chunk_indices.size > 0:
        return True
    
    return False


# ===== Emoji Formatting =====
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
    
    # Ensure that product numbers and names stay on the same line (no break)
    text = re.sub(r"(\d+)\.\s*(\S.*?)(?=\s*\d+\.|\n|$)", r"\1. \2", text)  # Ensures number and name together
    
    # Add RM symbol to prices (assuming the price is a numeric value followed by "RM")
    text = re.sub(r"(\d{1,3}(?:,\d{3})*)(RM)", r"RM \1", text)  # Ensure 'RM' comes before the number

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
 
# uploaded_file = st.file_uploader("📄 Upload a Laptop Specification PDF", type=["pdf"])
uploaded_files = st.file_uploader("📄 Upload Laptop Specification PDFs", type=["pdf"], accept_multiple_files=True)


if "history" not in st.session_state:
    st.session_state.history = []

if openai_api_key and uploaded_files:
    with st.spinner("🔍 Extracting and processing your documents..."):
        all_text = ""
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            document_text = extract_text_from_pdf(file_bytes)
            all_text += document_text + "\n\n"  # Combine the text from all PDFs
        pdf_text = document_text
        pdf_chunks = chunk_text_by_paragraph(all_text)
        keywords = extract_keywords_tfidf(all_text, top_n=30)
        chunk_embeddings = get_embeddings(pdf_chunks)
        
        # Create FAISS index
        faiss_index = create_faiss_index(chunk_embeddings)
 
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
            ai_reply = get_random_greeting()
            if "recommendation" not in ai_reply.lower() and "suggestion" not in ai_reply.lower():
                ai_reply += "\n\n" + category_suggestion
        elif is_farewell(question):
            ai_reply = "👋 Alright, take care! Let me know if you need help again later. Bye!"
        else:
            if not is_relevant_question(question, pdf_chunks, keywords, faiss_index):
                ai_reply = "❓ Sorry, I can only help with questions related to the laptop specifications you uploaded."
            elif is_follow_up_question(question, st.session_state.history, embedding_model, faiss_index, pdf_chunks):
                ai_reply = handle_follow_up_question(question, st.session_state.history, pdf_chunks, embedding_model, faiss_index)
            else:
                with st.spinner("🤔 Thinking..."):
                    query_embedding = get_embeddings([question])[0]
                    relevant_chunk_indices = search_faiss(query_embedding, faiss_index, k=3)
                    relevant_chunks = [pdf_chunks[i] for i in relevant_chunk_indices]
                    context = "\n".join(relevant_chunks)
                    ai_reply = ask_llm_with_history(question, context, st.session_state.history, openai_api_key)

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

elif not openai_api_key:
    st.error("🔐 API key not found.")
elif not uploaded_files:
    st.info("Please upload a PDF with laptop specifications.")
    
st.title("🔊 Audio Playback Test")

text = st.text_input("Enter text:", "Hello, Streamlit!")

if st.button("Generate Audio"):
    html_code = generate_audio_controls(text)
    components.html(html_code, height=250)  # 🔥 KEY: height must be >200
