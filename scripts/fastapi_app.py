import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="AgentTune Distilled Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = None


class QueryRequest(BaseModel):
    prompt: str
    model_path: str = "./distilled_student"


class QueryResponse(BaseModel):
    prompt: str
    response: str


@app.on_event("startup")
async def startup_event():
    global generator
    # We load it lazily on first request to support changing models, but could load here


@app.post("/v1/agent/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    global generator
    try:
        if generator is None:
            print(f"Loading transformer pipeline with model {request.model_path}...")
            generator = pipeline("text-generation", model=request.model_path, device_map="auto")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to search tools.",
            },
            {"role": "user", "content": request.prompt},
        ]

        prompt = generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print("Generating response...")
        results = generator(prompt, max_new_tokens=512, return_full_text=False)

        return QueryResponse(prompt=request.prompt, response=results[0]["generated_text"])
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("scripts.fastapi_app:app", host="0.0.0.0", port=8001, reload=False)
