import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path


# 1. Load PDF files
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))

    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError(f"No readable text in {pdf_path}")

    return full_text


# 2. Create chunks
def create_chunks(full_text, chunksize=500, overlap=100):

    if chunksize <= 0 or overlap < 0 or overlap >= chunksize:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(full_text):
        end = min(start + chunksize, len(full_text))

        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# 3. Build vectors
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):

    if not chunks:
        raise ValueError("No chunks available")

    embed_model = SentenceTransformer(model_name)

    vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])

    index.add(vectors)

    return index, embed_model


# 4. Load database
def load_base(pdf_path="college_docs"):

    folder = Path(pdf_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {pdf_path}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Invalid folder: {pdf_path}")

    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF files found in college_docs")

    chunks = []

    for pdf in pdf_files:
        text = load_pdf(pdf)
        chunks.extend(create_chunks(text))

    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


# 5. Retrieve content
def retrieve_content(
    question,
    embed_model,
    chunks,
    index,
    num_chunks=3
):

    if not question or not question.strip():
        raise ValueError("Question cannot be blank")

    if index.ntotal == 0:
        raise ValueError("FAISS index is empty")

    query_vector = embed_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector,
        min(num_chunks, index.ntotal)
    )

    background = ""

    for i in indices[0]:
        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# 6. Ask RAG
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
        raise ValueError("Groq API key is missing")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks
    )

    if not background.strip():
        return "Information Missing."

    prompt = f"""
You are a professional SIET College AI assistant.

Answer using ONLY the document background.

Do not invent information.

If the answer is unavailable, say:
Information Missing.

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
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
