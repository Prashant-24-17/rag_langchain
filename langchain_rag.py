from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util

# Load the PDF
loader = PDFPlumberLoader('yolo.pdf')
docs = loader.load()

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
chunked_docs = splitter.split_documents(docs)

# Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS vector store
vector = FAISS.from_documents(chunked_docs, embedding_model)

# Create retriever
retriever = vector.as_retriever(search_type='similarity', search_kwargs={'k': 10})

# Initialize the Ollama LLM
llm = Ollama(model="mistral")

# Define the QA chain prompt
prompt = """
1. Use the following information and answer the question at the end.
2. If you don't know the answer, just say "I don't know" but don't make up an answer on your own.
3. Keep the answer crisp and limited to 3-4 sentences.

Context: {context}

Question: {question}

Answer:
"""
QA_Chain_prompt = PromptTemplate.from_template(prompt)

# Define the LLM chain
llm_chain = LLMChain(
    llm=llm,
    prompt=QA_Chain_prompt,
    callbacks=None,
    verbose=True
)

# Define document prompt
document_prompt = PromptTemplate(
    input_variables=['page_content', 'source'],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

# Combine documents into a single context
combine_document_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None,
)

# Define the RetrievalQA chain
qa = RetrievalQA(
    combine_documents_chain=combine_document_chain,
    verbose=True,
    retriever=retriever,
    return_source_documents=False,  # Disable returning source documents
)

# Reranking function
def rerank_document(query, retrieved_docs):
    """
    Rerank the retrieved documents based on relevance scores using SentenceTransformer.
    """
    # Load the reranker model
    reranker_model = SentenceTransformer("cross-encoder/ms-marco-MiniLM-L-2-v2")
    
    # Prepare document texts
    doc_texts = [doc.page_content for doc in retrieved_docs]
    
    # Compute similarity scores between the query and document texts
    scores = reranker_model.encode([query], convert_to_tensor=True)
    doc_embeddings = reranker_model.encode(doc_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(scores, doc_embeddings).squeeze(0).tolist()
    print(cosine_scores[:3])
    
    # Sort documents by scores in descending order
    ranked_indices = sorted(range(len(cosine_scores)), key=lambda i: cosine_scores[i], reverse=True)
    return [retrieved_docs[i] for i in ranked_indices[:3]]  # Select top 3

# Query the system
query = "What is YOLO?"
retrieved_docs = retriever.get_relevant_documents(query)
reranked_docs = rerank_document(query, retrieved_docs)

# Prepare context using reranked documents
context = " ".join([doc.page_content for doc in reranked_docs])

# Query the QA chain
response = qa.run(query)

print("Answer:", response)
