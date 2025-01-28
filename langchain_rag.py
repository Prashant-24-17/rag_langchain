from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate


# Load the PDF
loader = PDFPlumberLoader('yolo.pdf')
docs = loader.load()

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=250 , chunk_overlap=25)
chunked_docs = splitter.split_documents(docs)

# Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS vector store
vector = FAISS.from_documents(chunked_docs, embedding_model)

# Create retriever
retriever = vector.as_retriever(search_type='similarity', search_kwargs={'k': 3})

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

# Query the system
response = qa.run("What is YOLO?")
print("Answer:", response)