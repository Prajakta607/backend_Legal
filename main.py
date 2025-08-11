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




class GraphState(TypedDict):
    file_path: str
    question: str
    query_type: Literal["summary", "chronology", "general_question"]
    answer: Optional[str]
    retriever: Any
    citations: List[Dict[str, Any]]  # [{source_id, quote, page, rects, color}]
    history: List[str]


class Citation(BaseModel):
    source_id: int = Field(..., description="ID of the specific source chunk")
    quote: str = Field(..., description="Exact text snippet from source")

class CitedAnswer(BaseModel):
    answer: str = Field(..., description="Final answer to user")
    citations: List[Citation] = Field(..., description="List of citations with quotes")




class querytypeSchema(BaseModel):
   query_type:Literal["summary", "chronology", "general_question"]= Field(description='type  of the query')


structured_model= model.with_structured_output(querytypeSchema)



import fitz  # PyMuPDF

def extract_text_with_coordinates(pdf_path: str):
    pdf = fitz.open(pdf_path)
    results = []
    for page_number, page in enumerate(pdf, start=1):
        blocks = page.get_text("blocks")  # returns (x0, y0, x1, y1, text, block_no, ...)
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if text.strip():  # ignore empty blocks
                results.append({
                    "page": page_number,
                    "text": text.strip(),
                    "rects": [(x0, y0, x1, y1)]
                })
    pdf.close()
    return results


import asyncio
async def pdf_processing_node(state: GraphState) -> GraphState:
    logging.info("PDF processing node started")
    file_path = state["file_path"]
    
    if not file_path:
        state["retriever"] = None
        state["history"].append("pdf_processing_node skipped")
        return state

    # Extract with coordinates
    raw_chunks = await asyncio.to_thread(extract_text_with_coordinates, file_path)

    # Convert to LangChain Documents
    from langchain.schema import Document
    documents = []
    for i, chunk in enumerate(raw_chunks):
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "source_id": i,
                "page": chunk["page"],
                "rects": chunk["rects"]
            }
        )
        documents.append(doc)

    # Split into smaller chunks if needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key
    )

    vectorstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="CaseStudy",
        api_endpoint=astra_endpoint,
        token=astra_token,
        namespace=astra_namespace,
    )
    vectorstore.add_documents(documents)

    state["retriever"] = vectorstore.as_retriever()
    state["history"].append("pdf_processing_node")
    return state


def check_query_type(state:GraphState)->Literal["summary_node", "chronology_node", "general_question_node"]:
   query_type = state["query_type"]
   if query_type == "summary":
        return "summary_node"
   elif query_type == "chronology":
        return "chronology_node"
   else:
        return "general_question_node"
   


async def query_node(state:GraphState)->GraphState:
   logging.info(" query_node started")
   question=state['question']
   prompt=f"""You are an intelligent legal assistant. Your task is to classify a user's question into **one** of the following categories:

            1. "summary" — if the question asks for a summary, overview, or gist of the legal document.
            2. "chronology" — if the question asks about the sequence of events, timeline, or what happened before/after certain events.
            3. "general_question" — if the question asks a factual, legal, or specific question that does not fall into summary or chronology.

            Respond **only** with one of the following strings exactly:
            - summary
            - chronology
            - general_question

            Now classify the following question: {question}"""
   result = (await structured_model.ainvoke(prompt)).query_type
   state["query_type"] = result
   state["history"].append("query_node")
   logging.info(" query_node started")
   return state

   

def _attach_metadata_to_citations(docs, citations_raw):
    logging.info("  _attach_metadata_to_citations started")
    colors = ["#FFEB3B", "#42A5F5", "#66BB6A", "#AB47BC", "#FF7043"]
    final = []
    for c in citations_raw:
        doc = next(d for d in docs if d.metadata["source_id"] == c.source_id)
        final.append({
            "source_id": c.source_id,
            "quote": c.quote,
            "page": doc.metadata.get("page"),
            "rects": doc.metadata.get("rects", []),
            "color": colors[c.source_id % len(colors)]
        })
    logging.info("  _attach_metadata_to_citations started")
    return final

async def summary_node(state: GraphState) -> GraphState:
    logging.info("  summary_node_citations started")
    question = state["question"]
    retriever = state["retriever"]
    if not retriever:
        state["answer"] = "No document uploaded."
        state["citations"] = []
        state["history"].append("no_retriever")
        return state
    docs = await retriever.ainvoke(question)

    print(docs[1].page_content)
    sources_str = "\n\n".join([
        f"Source ID: {d.metadata['source_id']}\nPage: {d.metadata.get('page')}\nText: {d.page_content}"
        for d in docs
    ])

    system_prompt = """
    You are a legal assistant helping summarize legal documents based on user queries.
    Summarize the relevant content concisely.
    Answer ONLY using the provided context.
    If the answer is not in the context, say "I don't know".
    Do NOT make up information.
    For each citation, use the exact text from the chunk and only use provided sources.
    Return ONLY JSON:
    {
      "answer": "...",
      "citations": [
        { "source_id": <int>, "quote": "<verbatim snippet>" }
      ]
    }
    """

    structured_llm = model.with_structured_output(CitedAnswer)
    result = await structured_llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"QUESTION: {question}\n\nSOURCES:\n{sources_str}"}
    ])

    state["answer"] = result.answer
    state["citations"] = _attach_metadata_to_citations(docs, result.citations)
    state["history"].append("summary_node")
    logging.info("  summary_node_citations started")
    return state


async def chronology_node(state: GraphState) -> GraphState:
    question = state["question"]
    retriever = state["retriever"]
    if not retriever:
        state["answer"] = "No document uploaded."
        state["citations"] = []
        state["history"].append("no_retriever")
        return state
    docs = await retriever.ainvoke(question)
    
    sources_str = "\n\n".join([
        f"Source ID: {d.metadata['source_id']}\nPage: {d.metadata.get('page')}\nText: {d.page_content}"
        for d in docs
    ])

    system_prompt = """
    You are a legal assistant. Extract a chronological list of dated legal events relevant to the question.
    Sort from earliest to latest, only include events with dates.
    For each citation, use the exact text from the chunk and only use provided sources.
    Return ONLY JSON:
    {
      "answer": "...",
      "citations": [
        { "source_id": <int>, "quote": "<verbatim snippet>" }
      ]
    }
    """

    structured_llm = model.with_structured_output(CitedAnswer)
    result = await structured_llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"QUESTION: {question}\n\nSOURCES:\n{sources_str}"}
    ])

    state["answer"] = result.answer
    state["citations"] = _attach_metadata_to_citations(docs, result.citations)
    state["history"].append("chronology_node")
    return state


async def general_question_node(state: GraphState) -> GraphState:
    question = state["question"]
    retriever = state["retriever"]
    if not retriever:
        state["answer"] = "No document uploaded."
        state["citations"] = []
        state["history"].append("no_retriever")
        return state
    docs = await retriever.ainvoke(question)
    

    sources_str = "\n\n".join([
        f"Source ID: {d.metadata['source_id']}\nPage: {d.metadata.get('page')}\nText: {d.page_content}"
        for d in docs
    ])

    system_prompt = """
    You are a legal assistant. Answer the question factually and concisely using only the provided sources.
    If the answer is not found, say "The document does not provide this information."
    For each citation, use the exact text from the chunk and only use provided sources.
    Return ONLY JSON:
    {
      "answer": "...",
      "citations": [
        { "source_id": <int>, "quote": "<verbatim snippet>" }
      ]
    }
    """

    structured_llm = model.with_structured_output(CitedAnswer)
    result = await  structured_llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"QUESTION: {question}\n\nSOURCES:\n{sources_str}"}
    ])

    state["answer"] = result.answer
    state["citations"] = _attach_metadata_to_citations(docs, result.citations)
    state["history"].append("general_question_node")
    return state


   
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

        graph.add_edge(START, "pdf_processing_node")
        graph.add_edge("pdf_processing_node", "query_node")
        graph.add_conditional_edges("query_node", check_query_type)
        graph.add_edge("summary_node", END)
        graph.add_edge("chronology_node", END)
        graph.add_edge("general_question_node", END)

        workflow = graph.compile()

        initial_state = {
            "file_path": file_path,
            "question": question,
            'history': []
        }

        final_state = await workflow.ainvoke(initial_state)

        answer = final_state.get("answer", "")
        citations = final_state.get("citations", [])

        return {
            "answer": answer,
            "citations": citations,
        }
    except Exception as e:
        logging.error("Error in /ask endpoint", exc_info=True)
        return {"error": "Internal server error"}

