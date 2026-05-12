from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.backend.pipeline_service import verify_claim
from app.backend.random_claims import get_random_claim

app = FastAPI(title="Claim Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VerifyRequest(BaseModel):
    claim: str
    top_k: int = 5
    retriever: str = "minilm"
    retrieval_mode: str = "fever"
    use_reranker: bool = True


@app.get("/")
def root():
    return {"message": "Claim Verification API is running"}


@app.post("/verify")
def verify(request: VerifyRequest):
    return verify_claim(
        claim=request.claim,
        top_k=request.top_k,
        retriever=request.retriever,
        retrieval_mode=request.retrieval_mode,
        use_reranker=request.use_reranker,
    )

@app.get("/random-claim")
def random_claim():

    return get_random_claim()