import faiss
import numpy as np
from grog import Groq
from pypdf import PDFReader
from sentence_transformers import sentenceTransformer

#load pdfs
def load_pdf (pdf_path):
  reader = PDFReader(str(pdf_path))
  mypages = [i.extract_text() or "" for i in reader.pages]
  full_text = "\n".join(mypages).strip()
  if not full_text:
    raise ValueError("No readable Text Found")
  return full_text

#Chunking
def create_chunks(full_text, chunksize = 500, overlap=100):
  chunks = [] 
  start = 0
  while start<len(full_text):
    end = start + chunksize
    chunk = full_text[start:end]
    if chunk.strip():
      chunks.append(chunk)
    start = chunksize - overlap
  return chunks

#Embedding
def build_vectors(chunks, model_name = "all-MiniLM-L6-v2"):
  if not chunks:
    raise ValueError("Cannot apply embedding without chunks")
  embed_model = SentenceTransformer(model_name)
  embed_vectors = embed_model.encode(chunks, convert_to_numpy = True)
  faiss.normalize_L2(embed_vectors)
  index = faiss.IndexFlatIP(embed_vectors.shape[1])
  return index,embed_model

  #call loadpdf, chunk, embedding
  def load_base (pdf_path = "01_College_Information.pdf"):
    full_text = load_pdf(pdf_path)
    chunks = create_chunks(full_text)
    embed_model, index = build_vector(chunks)
    return embed_model, chunks, index

#retrieve
def retrieve_content (question, embed_model, chunks, index, num_chunks=3):
  if not question.strip():
    raise valueError("Question cannot be blank")
  if num_chunks < 1:
    raise ValueError("Request atleast one or more chunks")
  query_embed = embed_model.encode([question], convert_to_numpy=True)
  faiss.normalize_L2(query_embed)
  sim_scores, vec_indices = index.search(query_embed), min(num_chunks, len(chunks))
  background = ""
  for i in vec_indicees[0]:
    if i != -1:
    background += chunks[i] + "\n\n"
  return background

  #ASk
  def ask_reg (question, embed_model, chunks,  index, api_key, mum_chunks, model_name = "openai/gpt-oss-120b"
    key = api_key
  if not key:
    raise ValueError("API key Missing")
   background = retrieve_content(question, embed_model, chunks, index, num_chunks=
