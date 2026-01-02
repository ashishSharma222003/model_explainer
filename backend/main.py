from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
import os
import json
import traceback
from models import (
    GlobalExplanation, TransactionExplanation
)
from chat import chat_manager
from sessions import session_manager

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
    Returns structured response with general answer and optional suggestion.
    """
    try:
        response = await chat_manager.chat(session_id, message, context)
        return {
            "response": response.general_answer,
            "global_json_suggestion": response.global_json_suggestion
        }
    except Exception as e:
        print(f"Error in chat: {e}")
        traceback.print_exc()
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
        print(f"Error in guide-code: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============ Session Management Endpoints ============

@app.get("/sessions")
async def get_all_sessions():
    """Get all saved sessions with full data."""
    try:
        sessions = session_manager.get_all_sessions()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        print(f"Error getting sessions: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/summaries")
async def get_session_summaries():
    """Get summaries of all sessions (for list display)."""
    try:
        summaries = session_manager.get_all_summaries()
        return {"summaries": summaries, "count": len(summaries)}
    except Exception as e:
        print(f"Error getting session summaries: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session by ID with all its data."""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session": session}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting session: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions")
async def save_session(session: Dict[str, Any] = Body(...)):
    """Save or update a session with all its data."""
    try:
        saved_session = session_manager.save_session(session)
        return {
            "success": True, 
            "session_id": saved_session['id'],
            "message": f"Session saved with {len(saved_session.get('mlCode', ''))} chars of code"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error saving session: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    try:
        success = session_manager.delete_session(session_id)
        return {"success": success}
    except Exception as e:
        print(f"Error deleting session: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get a summary of a specific session."""
    try:
        summary = session_manager.get_session_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"summary": summary}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting session summary: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
