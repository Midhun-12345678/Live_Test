import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader

load_dotenv()

API_KEY = os.getenv("API_KEY")

pdf_path = "resume-org.pdf (1) (1).pdf"

reader = PdfReader(pdf_path)
resume_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        resume_text += text + "\n"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection(name="resume_collection")

chunks = [resume_text[i:i+500] for i in range(0, len(resume_text), 500)]

embeddings = model.encode(chunks).tolist()

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(chunks))]
)

query = input("Ask questions about the resume: ")

query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

context = " ".join(results["documents"][0])

groq_client = Groq(api_key=API_KEY)

response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Answer the question using only the provided resume context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

print(response.choices[0].message.content)