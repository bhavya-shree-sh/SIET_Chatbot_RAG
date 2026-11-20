import streamlit as st
import faiss
import numpy as np
import os

from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# PAGE CONFIGURATION
st.set_page_config(
    page_title="SIET ChatBot",
    page_icon="🎓",
    layout="centered"
)


# 1. LOAD PDF
def load_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))

    mypages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    full_text = "\n".join(mypages).strip()

    if not full_text:
        raise ValueError("No readable text found")

    return full_text


# 2. CREATE CHUNKS
def create_chunks(full_text, chunk_size=500, overlap=100):

    chunks = []
    start = 0

    while start < len(full_text):

        end = start + chunk_size
        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# 3. BUILD VECTORS
def build_vectors(chunks):

    embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(
        embed_vectors.shape[1]
    )

    index.add(embed_vectors)

    return index, embed_model


# 4. LOAD BASE
@st.cache_resource
def load_base():

    full_text = load_pdf(
        "01_College_Information.pdf"
    )

    chunks = create_chunks(full_text)

    index, embed_model = build_vectors(chunks)

    return embed_model, chunks, index


# 5. RETRIEVE CONTENT
def retrieve_content(
    question,
    embed_model,
    chunks,
    index,
    num_chunks=3
):

    if not question.strip():
        raise ValueError("Question cannot be blank")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embed)

    scores, indices = index.search(
        query_embed,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in indices[0]:

        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# 6. ASK RAG
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=3,
    model_name="openai/gpt-oss-120b"
):

    if not api_key:
        raise ValueError("API Key Missing")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks
    )

    prompt = f"""
    You are a professional AI Assistant.

    Answer using ONLY the document background below.
    Do not invent or assume facts.

    If information is not available,
    simply say Information Missing.

    Document Background:
    {background}

    User Question:
    {question}

    Answer:
    """

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# STREAMLIT APPLICATION

st.title("Shridevi Institute of Engineering")
st.title("and Technology - ChatBot")

st.caption("Ask anything about college information....")

st.caption("Created by Gnanesh N")


# LOAD DATABASE
try:

    embed_model, chunks, index = load_base()

except Exception as e:

    st.error(f"PDF Loading Error: {e}")
    st.stop()


# API KEY
api_key = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY", "")
)


# CHAT HISTORY
if "messages" not in st.session_state:

    st.session_state.messages = []


# DISPLAY OLD MESSAGES
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# CHAT INPUT
question = st.chat_input(
    "Ask anything about college...."
)


if question:

    # USER MESSAGE
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    # ASSISTANT RESPONSE
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key
                )

                st.write(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(f"Error: {e}")


# CLEAR CHAT BUTTON
if st.button("Clear Chat"):

    st.session_state.messages = []

    st.rerun()
