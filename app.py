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
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

nltk.download('stopwords')

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

stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]

    return ' '.join(filtered_words)

def chunk_text(text, chunk_size=3000, overlap=500):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def extract_keywords_tfidf(text, top_n=100):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray().flatten()
    keywords = np.array(vectorizer.get_feature_names_out())
    
    top_indices = scores.argsort()[-top_n:][::-1]
    top_keywords = keywords[top_indices]
    
    return top_keywords.tolist()

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
        "Respond in a clear, structured format using numbered bullet points for lists. Each item should start on a new line. "
        "Avoid formal tones or sign-offs. Be friendly, clear, and conversational.\n\n"
        f"[INFO SOURCE]\n{context}"}]

    for entry in history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})

    messages.append({"role": "user", "content": question})

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": messages,
        "temperature": 1.0,
        "top_p": 0.2,
        "max_tokens": 200
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return format_response(response.json()["choices"][0]["message"]["content"])
    else:
        return f"❌ Error {response.status_code}: {response.text}"

def is_relevant_question(question, pdf_chunks, keywords):
    # Here we check for the presence of any relevant keywords from the uploaded PDFs
    question = question.lower()
    additional_keywords = [
        "study", "business", "gaming", "laptop", "processor", "ram", "ssd", "battery", "weight", "price", 
        "graphics", "display", "screen", "documents", "pdf", "similarities", "differences", "compare", 
        "summary", "word count", "count", "overview", "total words", "total pages", "summary of document"
    ]
    relevant_keywords = keywords + additional_keywords
    for words in additional_keywords:
        relevant_keywords.append(words)
    if any(keyword in question for keyword in relevant_keywords):
        return True
    if "summary" in question or "word" in question or "total word" in question:
        return True
    return False

def handle_special_queries(question, pdf_text):
    if "summary" in question:
        # Return a truncated summary of the document text (or the first few paragraphs)
        summary = pdf_text[:1000]  # Adjust this as needed for your summary length
        return f"Here's a brief summary of the document:\n\n{summary}..."
    
    # Check if the question is asking for the word count
    elif "word count" in question.lower() or "total words" in question.lower():
        word_count = len(pdf_text.split())
        return f"The document contains approximately {word_count} words."
    
    return None

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

# ===== Streamlit UI =====
st.set_page_config(page_title="💻 Laptop Chatbot", page_icon="💬", layout="wide")
st.title("💻 Laptop Recommendation Chatbot")

uploaded_files = st.file_uploader("📄 Upload Laptop Specification PDFs", type=["pdf"], accept_multiple_files=True)

if "history" not in st.session_state:
    st.session_state.history = []

if hf_token and uploaded_files:
    with st.spinner("🔍 Extracting and processing your documents..."):
        all_text = ""
        document_texts = []
        for uploaded_file in uploaded_files:
            document_text = extract_text_from_pdf(uploaded_file)
            document_texts.append(document_text)
            all_text += document_text + "\n\n"  # Combine the text from all PDFs
        pdf_chunks = chunk_text(all_text)
        keywords = extract_keywords_tfidf(all_text, top_n=30)

    for idx, document_text in enumerate(document_texts):
            # Example of handling a summary or word count for each document
            document_word_count = len(document_text.split())
            st.write(f"Document {idx + 1} Word Count: {document_word_count}")

            # If you want to summarize each document separately
            document_chunks = chunk_text(document_text)
            document_keywords = extract_keywords_tfidf(document_text, top_n=10)  # Extract top 10 keywords for each document
            st.write(f"Keywords for Document {idx + 1}: {', '.join(document_keywords)}")

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
            if not is_relevant_question(question, pdf_chunks, keywords):
                ai_reply = "❓ Sorry, I can only help with questions related to the laptop specifications you uploaded."
            else:
                special_reply = handle_special_queries(question, all_text)  # `all_text` is the text from all PDFs combined
                if special_reply:
                    ai_reply = special_reply
                else:
                    with st.spinner("🤔 Thinking..."):
                        context = find_relevant_chunk(question, pdf_chunks)
                        ai_reply = ask_llm_with_history(question, context, st.session_state.history, hf_token)

        st.session_state.history.append({"user": question, "assistant": ai_reply})
        st.rerun()

    # Save chat to pdf
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
