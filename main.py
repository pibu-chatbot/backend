from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from contextlib import asynccontextmanager

from schemas.request_response import get_chain
from models.rag_model import search_documents, set_embedding_model, load_chromadbs, create_ensemble_retriever


class QueryRequest(BaseModel):
    query: str
    session_id: str


def load_api_key():
    load_dotenv()


def set_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def build_model():
    return ChatOpenAI(
        model_name='gpt-4o-mini',
        temperature=0.4
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_api_key()
    model = build_model()
    chat_memory = get_chain(model)
    
    embedding_model = set_embedding_model()
    chroma1, chroma2 = load_chromadbs(embedding_model)
    ensemble_retriever = create_ensemble_retriever(chroma1, chroma2)

    app.state.chat_memory = chat_memory
    app.state.ensemble_retriever = ensemble_retriever

    yield


app = FastAPI(lifespan=lifespan)
set_cors(app)


@app.post("/ask")
def ask_question(request: QueryRequest):
    search_results = search_documents(app.state.ensemble_retriever, request.query)
    response = app.state.chat_memory.invoke(
        {"query": request.query, "search_results": search_results},
        config={"configurable": {"session_id": request.session_id}}
    )
    print(response)
    return {"answer": response.content}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
