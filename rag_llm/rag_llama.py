

import os
import ssl
import pdfplumber
from nltk.tokenize import sent_tokenize


# Fix SSL for NLTK punkt

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
nltk.download('punkt')

# 2Disable Hugging Face tokenizers parallelism

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# LangChain imports (updated for v0.2+)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import subprocess


#  Load PDFs from 'pdfs/' folder

pdf_folder = "/Users/shubhamchapagain/Desktop/Ragpipeline/pdfs:"
all_texts = []
total_tables = 0
sample_tables = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        with pdfplumber.open(os.path.join(pdf_folder, filename)) as pdf:
            pdf_text = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                
                tables = page.extract_tables()
                total_tables += len(tables)
                table_text = ""
                for t_idx, table in enumerate(tables):
                    for row in table:
                        table_text += " | ".join([str(cell) for cell in row]) + "\n"
                    if len(sample_tables) < 5:
                        sample_tables.append(table_text)
                
                pdf_text.append(text + "\n" + table_text)
            all_texts.append("\n".join(pdf_text))

data = "\n".join(all_texts)
print(f"Total characters extracted: {len(data)}")
print(f"Total tables extracted: {total_tables}")


#  Split text into chunks

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

text_chunks = chunk_text(data)
print(f"Number of text chunks: {len(text_chunks)}")


#  Create embeddings & vector store

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
print("Vector store created.")


#  Retrieve relevant context

def retrieve_context(query, k=30):  # increase k for more chunks
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs


#  Merge context safely for LLaMA

MAX_CONTEXT_CHARS = 4000  # Adjust based on model token limit

def merge_context(docs):
    merged = ""
    for doc in docs:
        if len(merged) + len(doc.page_content) > MAX_CONTEXT_CHARS:
            break
        merged += doc.page_content + "\n\n"
    return merged


# Generate descriptive answer using Ollama LLaMA 3

def generate_answer_llama(question, context):
    prompt = f"Answer the question based on the context. Be detailed and descriptive.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Error running Ollama:", e)
        print("stderr:", e.stderr)
        return ""

# -------------------------
# 10️⃣ Main execution
# -------------------------
if __name__ == "__main__":
    query = "How has Apple's total net sales changed over time?"
    docs = retrieve_context(query)
    merged_context = merge_context(docs)
    
    print("\n--- Retrieved Context ---")
    for i, doc in enumerate(docs[:5]):  # show first 5 docs for preview
        print(f"\nContext {i+1}:\n{doc.page_content[:500]}...")  # first 500 chars
    
    answer = generate_answer_llama(query, merged_context)
    print("\n--- Generated Answer ---")
    print(answer)
