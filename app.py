import nltk
nltk.download('wordnet')

import streamlit as st
import fitz  # PyMuPDF
import re
import requests
import random
from nltk.stem import WordNetLemmatizer

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

category_suggestion = (
    "Would you like suggestions for laptops used in:\n"
    "1. Study 📚\n2. Business 💼\n3. Gaming 🎮\nJust let me know!"
)

def is_greeting_or_smalltalk(user_input):
    user_input = user_input.lower().strip()
    patterns = [
        r"hi", r"hello", r"hey", r"good (morning|afternoon|evening)",
        r"how are you", r"what's up", r"how's it going", r"yo",
        r"sup", r"greetings", r"nice to meet you", r"howdy",
        r"everything okay", r"how are things", r"hello there", r"hiya",
        r"anyone there", r"can you help me", r"is this working", r"test",
        r"heyy+", r"helo+", r"bello", r"hallo", r"yo"
    ]
    for pattern in patterns:
        if re.search(rf"\b{pattern}\b", user_input):
            return True
    return False

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
def ask_llm(question, context, hf_token):
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

# ========== Emoji Formatting ==========
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

# ========== Streamlit App ==========
st.set_page_config(page_title="Laptop Chatbot", page_icon="💻")
st.title("💻 Laptop Recommendation Chatbot")

hf_token = st.text_input("🔑 Enter your HuggingFace API Token", type="password")
uploaded_file = st.file_uploader("📄 Upload a Laptop Specification PDF", type=["pdf"])

if hf_token and uploaded_file:
    with st.spinner("Extracting and processing your document..."):
        document_text = extract_text_from_pdf(uploaded_file)
        pdf_chunks = chunk_text(document_text)

    question = st.text_input("💬 Ask me anything about the laptops in the document!")

    if question:
        if is_greeting_or_smalltalk(question):
            st.markdown(f"**AI Assistant:**\n\n{get_random_greeting()}\n\n{category_suggestion}")
        else:
            with st.spinner("Thinking..."):
                context = find_relevant_chunk(question, pdf_chunks)
                answer = ask_llm(question, context, hf_token)
            st.markdown(f"**AI Assistant:**\n\n{answer}")
elif not hf_token:
    st.info("Please enter your HuggingFace API token to start chatting.")
elif not uploaded_file:
    st.info("Please upload a PDF with laptop specifications.")
