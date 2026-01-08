from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from typing import Dict, Any, Optional, List
import os
import json
import traceback
import io
from models import (
    GlobalExplanation, TransactionExplanation
)
from chat import chat_manager
from sessions import session_manager
from report_generator import report_generator, ReportRequest, ReportType, ReportFormat
from kernel_manager import kernel_manager, CodeExecutionRequest

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
    Analyze user code and provide implementation guide for explain_global function.
    Uses focused prompt with only global JSON schema.
    """
    try:
        response = await chat_manager.guide_code_to_global_json(session_id, code)
        return {"response": response}
    except Exception as e:
        print(f"Error in guide-code: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/guide-txn-code")
async def guide_txn_code_endpoint(
    session_id: str = Body(..., embed=True),
    code: str = Body(..., embed=True),
    global_json_context: Optional[str] = Body(None, embed=True)
):
    """
    Analyze user code and provide implementation guide for explain_txn function.
    Uses focused prompt with only transaction JSON schema.
    Optionally accepts global_json_context for consistency.
    """
    try:
        response = await chat_manager.guide_code_to_txn_json(session_id, code, global_json_context)
        return {"response": response}
    except Exception as e:
        print(f"Error in guide-txn-code: {e}")
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


# ============ Report Generation Endpoints ============

@app.post("/reports/generate")
async def generate_report(
    session_id: str = Body(..., embed=True),
    report_type: str = Body("executive", embed=True),
    format: str = Body("markdown", embed=True),
    include_code: bool = Body(False, embed=True),
    include_chat_history: bool = Body(True, embed=True),
    include_json_data: bool = Body(False, embed=True)
):
    """Generate a downloadable report from a session."""
    try:
        # Get session data
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Create request
        request = ReportRequest(
            session_id=session_id,
            report_type=ReportType(report_type),
            format=ReportFormat(format),
            include_code=include_code,
            include_chat_history=include_chat_history,
            include_json_data=include_json_data
        )
        
        # Generate report
        report = await report_generator.generate_report(session_data, request)
        
        # Save report record to session
        import uuid
        full_content = report.get("content", "")
        report_record = {
            "id": str(uuid.uuid4())[:8],
            "reportType": report_type,
            "format": format,
            "title": report.get("title", f"{report_type.capitalize()} Report"),
            "generatedAt": datetime.now().isoformat(),
            "includeCode": include_code,
            "includeChatHistory": include_chat_history,
            "includeJsonData": include_json_data,
            "filename": report.get("filename", f"report.{format}"),
            "summary": full_content[:200] + "..." if len(full_content) > 200 else full_content,
            "content": full_content  # Store full content for viewing later
        }
        
        # Get current reports and add new one
        current_reports = session_data.get("reports", [])
        current_reports.append(report_record)
        
        # Update session with new report
        session_manager.save_session(session_id, {"reports": current_reports})
        
        return {
            "success": True,
            "report": report,
            "reportRecord": report_record
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error generating report: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ Developer Mode - Kernel Endpoints ============

@app.post("/kernel/execute")
async def execute_code(
    session_id: str = Body(..., embed=True),
    code: str = Body(..., embed=True),
    timeout_seconds: int = Body(30, embed=True)
):
    """Execute Python code in an isolated kernel session."""
    try:
        request = CodeExecutionRequest(
            session_id=session_id,
            code=code,
            timeout_seconds=min(timeout_seconds, 60)  # Max 60 seconds
        )
        
        result = await kernel_manager.execute_code(request)
        
        return {
            "success": result.success,
            "outputs": result.outputs,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
            "variables": result.variables_captured
        }
    except Exception as e:
        print(f"Error executing code: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kernel/reset")
async def reset_kernel(session_id: str = Body(..., embed=True)):
    """Reset a kernel session (clear all variables)."""
    try:
        success = kernel_manager.reset_session(session_id)
        return {"success": success, "message": "Kernel reset" if success else "No kernel found"}
    except Exception as e:
        print(f"Error resetting kernel: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kernel/{session_id}/info")
async def get_kernel_info(session_id: str):
    """Get information about a kernel session."""
    try:
        info = kernel_manager.get_session_info(session_id)
        if not info:
            return {"exists": False}
        return {"exists": True, "info": info}
    except Exception as e:
        print(f"Error getting kernel info: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kernel/inject-context")
async def inject_context(
    session_id: str = Body(..., embed=True),
    ml_code: Optional[str] = Body(None, embed=True),
    global_json: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """Inject context data into a kernel session."""
    try:
        kernel_manager.inject_context(session_id, ml_code, global_json)
        return {"success": True, "message": "Context injected into kernel"}
    except Exception as e:
        print(f"Error injecting context: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kernel/upload-file")
async def upload_file_to_kernel(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    variable_name: str = Form("uploaded_data")
):
    """Upload a CSV/JSON file and load it into the kernel as a DataFrame or dict."""
    try:
        content = await file.read()
        filename = file.filename or "uploaded_file"
        
        # Get or create kernel session
        session = kernel_manager.get_or_create_session(session_id)
        
        # Determine file type and load accordingly
        if filename.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(io.BytesIO(content))
            session.set_variable(variable_name, df)
            preview = df.head(5).to_string()
            return {
                "success": True,
                "message": f"CSV loaded as '{variable_name}' ({len(df)} rows, {len(df.columns)} columns)",
                "preview": preview,
                "shape": [len(df), len(df.columns)],
                "columns": list(df.columns)
            }
        elif filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            session.set_variable(variable_name, data)
            preview = json.dumps(data, indent=2)[:500]
            return {
                "success": True,
                "message": f"JSON loaded as '{variable_name}'",
                "preview": preview
            }
        else:
            # Try to load as CSV by default
            import pandas as pd
            try:
                df = pd.read_csv(io.BytesIO(content))
                session.set_variable(variable_name, df)
                preview = df.head(5).to_string()
                return {
                    "success": True,
                    "message": f"File loaded as '{variable_name}' ({len(df)} rows)",
                    "preview": preview,
                    "shape": [len(df), len(df.columns)],
                    "columns": list(df.columns)
                }
            except:
                # Load as text
                text_content = content.decode('utf-8')
                session.set_variable(variable_name, text_content)
                return {
                    "success": True,
                    "message": f"File loaded as text '{variable_name}' ({len(text_content)} chars)",
                    "preview": text_content[:500]
                }
    except Exception as e:
        print(f"Error uploading file: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
