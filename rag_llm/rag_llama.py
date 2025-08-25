import os
import ssl
import pdfplumber
from nltk.tokenize import sent_tokenize
import subprocess
from flask import Flask, request, render_template
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------
# SSL fix & NLTK download
# -----------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
nltk.download('punkt')

# -----------------------
# Disable HuggingFace parallelism
# -----------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------
# Flask setup
# -----------------------
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------
# Global variables
# -----------------------
vector_store = None
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
MAX_CONTEXT_CHARS = 4000

# -----------------------
# PDF Processing
# -----------------------
def process_pdfs(pdf_paths):
    all_texts = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            pdf_text = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        table_text += " | ".join([str(cell) for cell in row]) + "\n"
                pdf_text.append(text + "\n" + table_text)
            all_texts.append("\n".join(pdf_text))
    data = "\n".join(all_texts)
    return data

# -----------------------
# Chunk text
# -----------------------
def chunk_text(text, max_chunk_size=2000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        current_chunk.append(sent)
        current_len += len(sent)
        if current_len >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# -----------------------
# Build vector store
# -----------------------
def build_vector_store(text_chunks):
    return FAISS.from_texts(texts=text_chunks, embedding=embedding_model)

# -----------------------
# Retrieve context
# -----------------------
def retrieve_context(query, vector_store, k=30):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs

# -----------------------
# Merge context
# -----------------------
def merge_context(docs):
    merged = ""
    for doc in docs:
        if len(merged) + len(doc.page_content) > MAX_CONTEXT_CHARS:
            break
        merged += doc.page_content + "\n\n"
    return merged

# -----------------------
# Generate answer via Ollama
# -----------------------
def generate_answer_llama(question, context):
    prompt = f"Answer the question based on the context. Be detailed and descriptive.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    print("\n---PROMPT SENT TO OLLAMA (first 1000 chars)---\n")
    print(prompt[:1000])
    print("\n---END OF PROMPT---\n")
    
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

# -----------------------
# Flask routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global vector_store
    answer = None
    context = None

    if request.method == "POST":
        # Upload PDFs
        uploaded_files = request.files.getlist("pdfs")
        pdf_paths = []
        for file in uploaded_files:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)
            pdf_paths.append(path)

        # Process PDFs
        data = process_pdfs(pdf_paths)
        text_chunks = chunk_text(data)
        vector_store = build_vector_store(text_chunks)

        answer = "✅ PDFs processed successfully. You can now ask questions."
    
    return render_template("index.html", answer=answer, context=context)

@app.route("/ask", methods=["POST"])
def ask():
    global vector_store
    if vector_store is None:
        return "⚠️ Upload PDFs first!"
    
    question = request.form.get("question")
    docs = retrieve_context(question, vector_store)
    merged_context = merge_context(docs)
    
    # Print context for debugging
    print("\n---Merged Context (first 2000 chars)---\n")
    print(merged_context[:2000])
    print("\n---END OF Merged Context---\n")
    
    answer = generate_answer_llama(question, merged_context)
    
    return render_template("index.html", answer=answer, context=merged_context)

# -----------------------
# Run Flask app
# -----------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
