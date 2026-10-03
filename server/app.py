from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(...)

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    seed: Optional[int] = 42

@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    task = request.task_id if request else "easy"
    return {"status": "ok", "task_id": task, "observation": {}}

@app.post("/step")
async def step():
    return {"reward": 0.5, "done": False, "observation": {}}

@app.get("/health")
async def health():
    return {"status": "healthy"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
