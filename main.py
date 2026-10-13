"""FastAPI entry point."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import run_agent
from eval import run_eval

app = FastAPI(title="Research Agent", version="1.0.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    result = run_agent(body.question)
    return AskResponse(**result)


@app.post("/eval")
def run_evaluation() -> dict:
    return run_eval()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
