from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()
from typing import List, Dict, Any, Optional, TypedDict, Literal
from pydantic import BaseModel, Field
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from fastapi.concurrency import run_in_threadpool
import os
openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
app = FastAPI()




load_dotenv()  # load from .env file into os.environ

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
astra_token = os.getenv("ASTRA_DB_TOKEN")
astra_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
astra_namespace = os.getenv("ASTRA_DB_NAMESPACE")


from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(api_key=anthropic_api_key,model="claude-3-5-sonnet-20240620")




app = FastAPI()

origins = [
    "*",  
    "http://localhost",  # if you test locally
    "https://chrome-chat-assistant.onrender.com", 
      # your deployed API (optional for CORS if backend calls only)
    "https://your-frontend-domain.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # restrict to specified origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




model = ChatAnthropic(
    model="claude-3-5-haiku-latest", 
    temperature=0
)

class QueryTypeSchema(BaseModel):
    """Schema for classifying query types"""
    query_type: Literal["summary", "chronology", "general_question"] = Field(
        description='Type of the query'
    )

class GraphState(BaseModel):
    """State management for the graph workflow"""
    class Config:
        arbitrary_types_allowed = True
     
    file_path: str = ""
    question: str = ""
    query_type: str = ""
    retriever: Any = None
    retrieved_docs: List[Any] = []
    All_docs: List[Any] = []
    answer: str = ""
    citations: List[Dict] = []
    cited_pages_metadata: List[Dict] = []
    history: List[str] = [],
    current_case_id: Optional[str] = None 

# Structured output model
structured_model = model.with_structured_output(QueryTypeSchema)

import fitz  # PyMuPDF

def safe_get_metadata(doc, key: str, default=None):
    """Safely extract metadata from document objects"""
    if hasattr(doc, 'metadata'):
        metadata = doc.metadata
        if isinstance(metadata, dict):
            return metadata.get(key, default)
        elif hasattr(metadata, key):
            return getattr(metadata, key, default)
    return default

def safe_get_content(doc):
    """Safely extract content from document objects"""
    if hasattr(doc, 'page_content'):
        return doc.page_content
    elif hasattr(doc, 'content'):
        return doc.content
    else:
        return str(doc)

from typing import List
import os
import fitz  # PyMuPDF
from langchain.schema import Document  # or wherever your Document class is from

def load_pdf_chunks(pdf_path: str, case_id: str) -> List[Document]:
    """
    Load PDF and split into chunks with enhanced metadata including case_id
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    all_chunks = []
    doc_no = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", ". ", "? ", "! ", "; ", ", ", "\n"],
    )

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()

        if not text:
            continue

        page_chunks = splitter.split_text(text)
        char_offset = 0

        for chunk_idx, chunk in enumerate(page_chunks):
            chunk = chunk.strip()
            if not chunk:
                continue

            metadata = {
                "page": page_num + 1,
                "offset": char_offset,
                "doc_no": doc_no,
                "source_id": len(all_chunks),
                "chunk_index": chunk_idx,
                "file_name": os.path.basename(pdf_path),
                "case_id": case_id,  # add case_id here
            }

            all_chunks.append(Document(page_content=chunk, metadata=metadata))
            char_offset += len(chunk)

    doc.close()

    return all_chunks


def to_anthropic_document_format(docs) -> Dict:
    """Convert documents to Anthropic document format with automatic citations"""
    # Safely extract content from all documents
    all_content = []
    for doc in docs:
        content = safe_get_content(doc)
        all_content.append(str(content))
    
    # Combine all content into a single document
    combined_text = "\n\n".join(all_content)
    
    return {
        "type": "document",
        "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": combined_text
        },
        "title": "Legal Document",
        "context": "Retrieved legal document content",
        "citations": {"enabled": True}
    }

def extract_citations_from_response(response) -> List[Dict]:
    """Extract citations from Anthropic response format - automatic citations"""
    citations = []
    
    if hasattr(response, 'content') and isinstance(response.content, list):
        for item in response.content:
            if item.get('type') == 'text' and 'citations' in item:
                for citation in item['citations']:
                    citations.append({
                        'cited_text': citation.get('cited_text', ''),
                        'document_title': citation.get('document_title', ''),
                        'document_index': citation.get('document_index', ''),
                        'start_char_index': citation.get('start_char_index', ''),
                        'end_char_index': citation.get('end_char_index', ''),
                        'type': citation.get('type', 'char_location')
                    })
    
    return citations

import uuid
import os

def generate_case_id() -> str:
    return str(uuid.uuid4())


def pdf_processing_node(state: GraphState) -> GraphState:
    """Process PDF and create vector store"""
    file_path = state.file_path

    if not file_path:
        raise ValueError("No file path provided")

    try:
        case_id = generate_case_id()
        state.current_case_id = case_id  # store in state for retrieval filtering

        # Pass case_id to load_pdf_chunks so chunks have it in metadata
        docs = load_pdf_chunks(file_path, case_id)
        state['All_docs']=docs


        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=openai_api_key
        )

        # Create vector store
        vectorstore = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="CaseStudy",
            api_endpoint=astra_endpoint,
            token=astra_token,
            namespace=astra_namespace,
        )

        vectorstore.add_documents(docs)
        state.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        state.history.append("pdf_processing_node")

    except Exception as e:
        print(f"Error in pdf_processing_node: {str(e)}")
        raise Exception(f"Error processing PDF: {str(e)}")

    return state

def query_node(state: GraphState) -> GraphState:
    """Classify the query type"""
    question = state.question

    prompt = f"""You are an intelligent legal assistant. Your task is to classify a user's question into **one** of the following categories:

    1. "summary" — if the question asks for a summary, overview, or gist of the legal document.
    2. "chronology" — if the question asks about the sequence of events, timeline, or what happened before/after certain events.
    3. "general_question" — if the question asks a factual, legal, or specific question that does not fall into summary or chronology.

    Examples:
    - "What is this case about?" → summary
    - "What happened first in this case?" → chronology
    - "Who was the plaintiff?" → general_question

    Question to classify: {question}"""

    try:
        result = structured_model.invoke(prompt).query_type
        state.query_type = result
        state.history.append("query_node")
    except Exception as e:
        print(f"Error in query classification: {str(e)}")
        state.query_type = "general_question"
        state.history.append("query_node_error")

    return state

def check_query_type(state: GraphState) -> Literal["summary_node", "chronology_node", "general_question_node"]:
    """Route to appropriate processing node based on query type"""
    query_type = state.query_type
    if query_type == "summary":
        return "summary_node"
    elif query_type == "chronology":
        return "chronology_node"
    else:
        return "general_question_node"

def summary_node(state: GraphState) -> GraphState:
    """Handle summary queries with automatic citations"""
    question = state.question
   

    case_id = getattr(state, "current_case_id", None)
    if not case_id:
        state.answer = "Error: Case ID not set."
        return state

    try:
        docs = state['All_docs']
        

        # Convert to Anthropic document format with automatic citations
        document_content = to_anthropic_document_format(docs)
  
        messages = [
            {
                "role": "user",
                "content": [
                    document_content,
                    {"type": "text", "text": f"""You are a legal assistant helping summarize legal documents.

Instructions:
- Provide a comprehensive but concise summary
- Focus on key legal points, parties involved, and main issues  
- Use clear, professional language
- Structure your response with clear headings if appropriate
- Answer ONLY using the provided context
- If the answer is not in the context, say "I don't have enough information to provide a complete summary"
-Do not genrate answer by imagine  use given context only to genrate answer

Please provide a summary for: {question}"""}
                ]
            }
        ]

        response = model.invoke(messages)
        
        # Extract text content and citations - citations are automatic
        if hasattr(response, 'content') and isinstance(response.content, list):
            state.answer = "".join(item.get('text', '') for item in response.content if item.get('type') == 'text')
            state.citations = extract_citations_from_response(response)
        else:
            state.answer = str(response.content)

    except Exception as e:
        print(f"Error in summary_node: {str(e)}")
        state.answer = f"Error generating summary: {str(e)}"

    state.history.append("summary_node")
    return state

def chronology_node(state: GraphState) -> GraphState:
    """Handle chronology queries with automatic citations"""
    question = state.question
   

    case_id = getattr(state, "current_case_id", None)
    if not case_id:
        state.answer = "Error: Case ID not set."
        return state

    try:
        docs = state['All_docs']
        state.retrieved_docs = docs
        
        # Convert to Anthropic document format with automatic citations
        document_content = to_anthropic_document_format(docs)

        messages = [
            {
                "role": "user",
                "content": [
                    document_content,
                    {"type": "text", "text": f"""You are a legal assistant specializing in chronological analysis of legal documents.

Instructions:
- Extract all events with specific dates mentioned in the documents
- Sort events chronologically from earliest to latest
- Format as a clear timeline
- Include the specific date and brief description of each event
- Only include events that have explicit dates
- If no dated events are found, explain what information is available
- Use the format: "Date: Event description"

Create a chronological timeline for: {question}"""}
                ]
            }
        ]

        response = model.invoke(messages)
        
        # Extract text content and citations - citations are automatic
        if hasattr(response, 'content') and isinstance(response.content, list):
            state.answer = "".join(item.get('text', '') for item in response.content if item.get('type') == 'text')
            state.citations = extract_citations_from_response(response)
        else:
            state.answer = str(response.content)

    except Exception as e:
        print(f"Error in chronology_node: {str(e)}")
        state.answer = f"Error generating chronology: {str(e)}"

    state.history.append("chronology_node")
    return state

def general_question_node(state: GraphState) -> GraphState:
    """Handle general legal questions with automatic citations"""
    question = state.question
    retriever = state.retriever

    if not retriever:
        state.answer = "Error: Document not processed. Please upload a document first."
        return state

    case_id = getattr(state, "current_case_id", None)
    if not case_id:
        state.answer = "Error: Case ID not set."
        return state

    try:
        docs = retriever.get_relevant_documents(question, filters={"case_id": case_id})
        state.retrieved_docs = docs

        # Convert to Anthropic document format with automatic citations
        document_content = to_anthropic_document_format(docs)

        messages = [
            {
                "role": "user",
                "content": [
                    document_content,
                    {"type": "text", "text": f"""You are a legal assistant providing factual answers to specific legal questions.

Instructions:
- Answer the question directly and concisely
- Use only information from the provided documents
- Be precise and factual
- If the answer requires information not in the documents, say "The document does not provide this information"
- Quote relevant sections when appropriate
- Maintain professional legal terminology
- Provide context when necessary for understanding

Question: {question}"""}
                ]
            }
        ]

        response = model.invoke(messages)
        
        # Extract text content and citations - citations are automatic
        if hasattr(response, 'content') and isinstance(response.content, list):
            state.answer = "".join(item.get('text', '') for item in response.content if item.get('type') == 'text')
            state.citations = extract_citations_from_response(response)
        else:
            state.answer = str(response.content)

    except Exception as e:
        print(f"Error in general_question_node: {str(e)}")
        state.answer = f"Error answering question: {str(e)}"

    state.history.append("general_question_node")
    return state






def extract_citation_metadata_node(state: GraphState) -> GraphState:
    """
    Graph node to extract comprehensive metadata for pages cited by the LLM.
    Returns formatted list ready for frontend consumption.

    Args:
        state: GraphState containing 'citations' and 'retrieved_docs'

    Returns:
        GraphState with added 'cited_pages_metadata' list for frontend
    """
    # Access GraphState attributes directly (not like a dict)
    citations_state = getattr(state, 'citations', [])
    retrieved_docs = getattr(state, 'retrieved_docs', []) # Ensure retrieved_docs is treated as a list

    cited_pages_metadata = _extract_citation_metadata(citations_state, retrieved_docs)

    # Add to state for frontend
    state.cited_pages_metadata = cited_pages_metadata

    return state

def _extract_citation_metadata(citations_state, retrieved_docs):
    """
    Internal function to extract comprehensive metadata for pages cited by Anthropic's automatic citations.

    Args:
        citations_state: List of Anthropic citation dictionaries with document_index, start_char_index, etc.
        retrieved_docs: List of document objects with metadata

    Returns:
        List of metadata dictionaries for cited pages
    """
    cited_pages_metadata = []
    seen_combinations = set()  # To avoid duplicates

    for citation in citations_state:
        # Anthropic citations have different structure
        document_index = citation.get('document_index', 0)
        start_char = citation.get('start_char_index', 0)
        end_char = citation.get('end_char_index', 0)
        cited_text = citation.get('cited_text', '')
        
        # Since we combined docs, we need to map back to original docs by char position
        current_char = 0
        source_doc_index = 0
        
        # Find which retrieved doc this citation maps to
        for idx, doc in enumerate(retrieved_docs):
            content = safe_get_content(doc)
            content_length = len(str(content)) + 2  # +2 for "\n\n" separator
            
            if start_char < current_char + content_length:
                source_doc_index = idx
                break
            current_char += content_length
        
        # Get the source document
        if source_doc_index >= len(retrieved_docs):
            print(f"Warning: Citation char index {start_char} maps beyond available documents")
            continue
            
        doc = retrieved_docs[source_doc_index]
        
        # Extract metadata from the source document
        page_num = safe_get_metadata(doc, 'page', 1)
        file_name = safe_get_metadata(doc, 'file_name', 'Unknown')
        
        # Get or derive source_id - this is essential for your application
        source_id = safe_get_metadata(doc, 'source_id', source_doc_index)
        
        # If source_id is still None or not found, create one based on document index
        if source_id is None or source_id == -1:
            source_id = source_doc_index
        
        # Create unique identifier to avoid duplicate entries
        unique_key = (source_id, page_num, start_char)  # Include start_char for uniqueness
        if unique_key in seen_combinations:
            continue
        seen_combinations.add(unique_key)

        page_metadata = {
            'source_id': source_id,  # This is required for your application
            'page': page_num,
            'file_name': file_name,
            'document_title': safe_get_metadata(doc, 'title', file_name),
            'content_preview': cited_text[:100] + '...' if len(cited_text) > 100 else cited_text,
            'quote': cited_text,
            'start_char_index': start_char,
            'end_char_index': end_char,
            'document_index': document_index,  # Anthropic's document index (usually 0 for combined doc)
            'source_doc_index': source_doc_index,  # Our mapped source doc index
            # Additional metadata fields
            'author': safe_get_metadata(doc, 'author'),
            'creation_date': safe_get_metadata(doc, 'creation_date'),
            'file_path': safe_get_metadata(doc, 'file_path'),
            'total_pages': safe_get_metadata(doc, 'total_pages'),
            'file_size': safe_get_metadata(doc, 'file_size'),
            'document_type': safe_get_metadata(doc, 'document_type'),
        }

        # Remove None values to keep it clean
        page_metadata = {k: v for k, v in page_metadata.items() if v is not None}

        cited_pages_metadata.append(page_metadata)

    return cited_pages_metadata


   
import logging
import aiofiles

@app.post("/ask")
async def ask_endpoint(question: str = Form(...), file: Optional[UploadFile] = File(None)):
    try:
        file_path = None
        if file is not None:
            file_location = f"./uploaded_{file.filename}"
            async with aiofiles.open(file_location, "wb") as out_file:
                content = await file.read()  # reads entire file into memory
                await out_file.write(content)
            file_path = file_location

        graph = StateGraph(GraphState)

        graph.add_node("pdf_processing_node", pdf_processing_node)
        graph.add_node("query_node", query_node)
        graph.add_node("summary_node", summary_node)
        graph.add_node("chronology_node", chronology_node)
        graph.add_node("general_question_node", general_question_node)
        graph.add_node("extract_citation_metadata_node",extract_citation_metadata_node)


        graph.add_edge(START, "pdf_processing_node")
        graph.add_edge("pdf_processing_node", "query_node")
        graph.add_conditional_edges("query_node", check_query_type)
        graph.add_edge("summary_node","extract_citation_metadata_node" )
        graph.add_edge("chronology_node","extract_citation_metadata_node")
        graph.add_edge("general_question_node","extract_citation_metadata_node")
        graph.add_edge("extract_citation_metadata_node",END)


        workflow = graph.compile()
        initial_state = {
            "file_path": file_path,
            "question": question,
            'history': []
        }

        final_state =workflow.invoke(initial_state)
        print(final_state['cited_pages_metadata'])

        answer =final_state['answer']
        cited_pages_metadata =final_state['cited_pages_metadata']
        return {
            "answer": answer,
            "cited_pages_metadata": cited_pages_metadata,
        }
    except Exception as e:
        logging.error("Error in /ask endpoint", exc_info=True)
        return {"error": "Internal server error"}

