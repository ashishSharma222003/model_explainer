from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import os
from models import (
    GlobalExplanation, TransactionExplanation
)
from chat import chat_manager

app = FastAPI(title="Model Explainer API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Model Explainer API is running"}

@app.post("/chat")
async def chat_endpoint(
    session_id: str = Body(..., embed=True),
    message: str = Body(..., embed=True),
    context: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """
    Chat with the explainer assistant.
    """
    try:
        response = await chat_manager.chat(session_id, message, context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/guide-code")
async def guide_code_endpoint(
    session_id: str = Body(..., embed=True),
    code: str = Body(..., embed=True)
):
    """
    Analyze user code and provide implementation guide for explainer functions.
    """
    try:
        response = await chat_manager.guide_code_to_json(session_id, code)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
