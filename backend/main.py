from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from typing import Dict, Any, Optional, List
import os
import json
import traceback
import io
from datetime import datetime
from models import (
    GlobalExplanation, TransactionExplanation,
    SimilarShadowRule, DeduplicationResult, VectorStoreStats,
    AddRuleToIndexRequest, AddRulesBulkRequest, CheckDuplicateRequest,
    ExtractShadowRulesRequest
)
from chat import chat_manager
from sessions import session_manager
from report_generator import report_generator, ReportRequest, ReportType, ReportFormat
from kernel_manager import kernel_manager, CodeExecutionRequest
from xgboost_analyzer import xgboost_analyzer
from shadow_rules_vectorstore import get_vector_store, clear_vector_store_cache

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
    Chat with the explainer assistant (global mode).
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


@app.post("/chat-txn")
async def chat_txn_endpoint(
    session_id: str = Body(..., embed=True),
    message: str = Body(..., embed=True),
    context: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """
    Chat about transaction-level predictions and analyst behavior.
    Specialized prompting for:
    - Individual prediction analysis
    - What-if scenarios  
    - Risk assessment
    - Shadow rule detection (undocumented analyst patterns)
    - Guideline compliance checking
    
    Returns structured response with insights for banking fraud analysis workflows.
    """
    try:
        response = await chat_manager.chat_txn(session_id, message, context)
        return {
            "response": response.general_answer,
            "txn_json_suggestion": response.txn_json_suggestion,
            "what_if_insight": response.what_if_insight,
            "risk_flag": response.risk_flag,
            "shadow_rule_detected": response.shadow_rule_detected,
            "guideline_reference": response.guideline_reference,
            "compliance_note": response.compliance_note,
        }
    except Exception as e:
        print(f"Error in chat_txn: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-shadow-rules")
async def extract_shadow_rules_endpoint(request: ExtractShadowRulesRequest):
    """
    Dedicated endpoint for extracting shadow rules from selected transactions.
    
    This analyzes wrong predictions (false positives/negatives) to discover
    hidden patterns (shadow rules) that explain why analyst decisions
    differed from model predictions.
    
    Args:
        request: Contains selected_transactions, data_schema, decision_tree_rules, chat_history
        
    Returns:
        Structured response with discovered shadow rules, summary, and count
    """
    try:
        response = await chat_manager.extract_shadow_rules_from_transactions(
            session_id=request.session_id,
            selected_transactions=request.selected_transactions,
            data_schema=request.data_schema,
            decision_tree_rules=request.decision_tree_rules,
            chat_history=request.chat_history
        )
        
        return {
            "success": True,
            "shadow_rules": [rule.model_dump() for rule in response.shadow_rules],
            "summary": response.summary,
            "transactions_analyzed": response.transactions_analyzed,
            "rules_count": len(response.shadow_rules)
        }
    except Exception as e:
        print(f"Error extracting shadow rules: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/guide-code")
async def guide_code_endpoint(
    session_id: str = Body(..., embed=True),
    code: str = Body(..., embed=True),
    data_schema: Optional[str] = Body(None, embed=True)
):
    """
    Generate mathematical model code to understand analyst decisions.
    Uses data schema context to build appropriate ML/statistical models.
    """
    try:
        response = await chat_manager.guide_code_to_global_json(session_id, code, data_schema)
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

@app.post("/guide-schema-code")
async def guide_schema_code_endpoint(
    session_id: str = Body(..., embed=True),
    code: str = Body(..., embed=True)
):
    """
    Analyze user code and provide implementation guide for data schema analysis.
    Generates analyze_data_schema() function.
    """
    try:
        response = await chat_manager.guide_schema_analysis(session_id, code)
        return {"response": response}
    except Exception as e:
        print(f"Error in guide-schema-code: {e}")
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

# ============ CSV Data Endpoints ============

@app.post("/sessions/{session_id}/csv")
async def upload_csv_data(
    session_id: str,
    csv_data: List[Dict[str, Any]] = Body(..., embed=True),
    file_name: str = Body("", embed=True)
):
    """Upload CSV data for a session (stored on backend to avoid localStorage limits)."""
    try:
        success = session_manager.save_csv_data(session_id, csv_data, file_name)
        if success:
            return {
                "success": True,
                "message": f"Saved {len(csv_data)} rows for session {session_id}",
                "rowCount": len(csv_data)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save CSV data")
    except Exception as e:
        print(f"Error uploading CSV data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/csv")
async def get_csv_data(session_id: str):
    """Get CSV data for a session."""
    try:
        csv_data = session_manager.get_csv_data(session_id)
        if csv_data:
            return csv_data
        else:
            return {"fileName": "", "rowCount": 0, "data": []}
    except Exception as e:
        print(f"Error getting CSV data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ XGBoost Analysis Endpoints ============

@app.post("/sessions/{session_id}/analyze")
async def analyze_session_data(session_id: str):
    """
    Run XGBoost analysis on session CSV data.
    Automatically:
    - Detects L1/L2 decision columns
    - Trains XGBoost models
    - Generates SHAP analysis
    - Identifies wrong predictions
    """
    try:
        # Get CSV data
        csv_data = session_manager.get_csv_data(session_id)
        if not csv_data or not csv_data.get('data'):
            raise HTTPException(status_code=404, detail="No CSV data found for this session")
        
        # Run analysis
        result = await xgboost_analyzer.analyze(csv_data['data'])
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        # Save analysis result to session
        session = session_manager.get_session(session_id)
        if session:
            session['analysisResult'] = result.model_dump()
            session_manager.save_session(session)
        
        return result.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error analyzing session data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/analysis")
async def get_analysis_result(session_id: str):
    """Get the stored analysis result for a session."""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        analysis_result = session.get('analysisResult')
        if not analysis_result:
            raise HTTPException(status_code=404, detail="No analysis result found. Run /analyze first.")
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting analysis result: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/hyperparameters")
async def get_hyperparameters():
    """Get current XGBoost hyperparameters."""
    return xgboost_analyzer.get_hyperparameters()


@app.post("/analysis/hyperparameters")
async def set_hyperparameters(
    n_estimators: int = Body(100, embed=True),
    max_depth: int = Body(5, embed=True),
    learning_rate: float = Body(0.1, embed=True),
    min_child_weight: int = Body(1, embed=True),
    subsample: float = Body(0.8, embed=True),
    colsample_bytree: float = Body(0.8, embed=True),
    gamma: float = Body(0, embed=True),
    reg_alpha: float = Body(0, embed=True),
    reg_lambda: float = Body(1, embed=True),
    random_state: int = Body(42, embed=True)
):
    """
    Set XGBoost hyperparameters for analysis.
    These will be used in subsequent analysis runs.
    
    Parameters:
    - n_estimators: Number of boosting rounds (trees)
    - max_depth: Maximum depth of each tree (controls complexity)
    - learning_rate: Step size shrinkage (lower = more conservative)
    - min_child_weight: Minimum sum of instance weight in a child
    - subsample: Subsample ratio of training instances
    - colsample_bytree: Subsample ratio of columns when constructing each tree
    - gamma: Minimum loss reduction required to make a split
    - reg_alpha: L1 regularization term on weights
    - reg_lambda: L2 regularization term on weights
    - random_state: Random seed for reproducibility
    """
    try:
        xgboost_analyzer.set_hyperparameters({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state
        })
        return {
            "success": True,
            "message": "Hyperparameters updated",
            "hyperparameters": xgboost_analyzer.get_hyperparameters()
        }
    except Exception as e:
        print(f"Error setting hyperparameters: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sessions/{session_id}/wrong-predictions")
async def get_wrong_predictions(
    session_id: str,
    case_type: Optional[str] = None  # 'false_positive' or 'false_negative'
):
    """Get wrong predictions from the analysis result."""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        analysis_result = session.get('analysisResult')
        if not analysis_result:
            raise HTTPException(status_code=404, detail="No analysis result found. Run /analyze first.")
        
        wrong_predictions = analysis_result.get('wrong_predictions', [])
        
        # Filter by case type if specified
        if case_type:
            wrong_predictions = [p for p in wrong_predictions if p.get('case_type') == case_type]
        
        return {
            "total": len(wrong_predictions),
            "predictions": wrong_predictions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting wrong predictions: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/csv/row/{row_index}")
async def get_csv_row(session_id: str, row_index: int):
    """Get a single row from the CSV data."""
    try:
        row = session_manager.get_csv_row(session_id, row_index)
        if row:
            return {"row": row, "index": row_index}
        else:
            raise HTTPException(status_code=404, detail="Row not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting CSV row: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/csv/search")
async def search_csv_data(
    session_id: str,
    column: str,
    value: str
):
    """Search CSV data by column value."""
    try:
        rows = session_manager.get_csv_rows_by_column(session_id, column, value)
        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        print(f"Error searching CSV data: {e}")
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
    include_schema: bool = Body(True, embed=True),
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
            include_schema=include_schema,
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
            "includeSchema": include_schema,
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
        session_data["reports"] = current_reports
        session_manager.save_session(session_data)
        
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


# ============ Shadow Rules Conversion ============

@app.post("/convert-shadow-rules")
async def convert_shadow_rules(
    l1_rules: List[str] = Body(default=[]),
    l2_rules: List[str] = Body(default=[]),
    data_schema: Optional[Dict[str, Any]] = Body(default=None),
    existing_rules: List[str] = Body(default=[])
):
    """
    Convert decision tree rules to human-readable shadow rules using LLM.
    Avoids duplicates by checking against existing_rules.
    """
    try:
        response = await chat_manager.convert_rules_to_human_readable(
            l1_rules=l1_rules,
            l2_rules=l2_rules,
            data_schema=data_schema,
            existing_rules=existing_rules
        )
        
        return {
            "success": True,
            "rules": [rule.model_dump() for rule in response.rules],
            "count": len(response.rules)
        }
    except Exception as e:
        print(f"Error converting shadow rules: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ Shadow Rules Vector Store Endpoints ============

@app.post("/sessions/{session_id}/shadow-rules/check-duplicate")
async def check_shadow_rule_duplicate(
    session_id: str,
    request: CheckDuplicateRequest
):
    """
    Check if a shadow rule is semantically similar to existing rules.
    Returns deduplication result with similar rules found.
    """
    try:
        vector_store = get_vector_store(session_id)
        similar_rules = vector_store.find_similar(
            query_text=request.rule_text,
            threshold=request.threshold,
            top_k=5
        )
        
        # Convert to response models
        similar_rule_models = [
            SimilarShadowRule(
                rule_id=r.rule_id,
                rule_text=r.rule_text,
                simple_rule=r.simple_rule,
                similarity_score=r.similarity_score,
                source_analysis=r.source_analysis,
                target_decision=r.target_decision,
                predicted_outcome=r.predicted_outcome,
                is_duplicate=r.is_duplicate
            )
            for r in similar_rules
        ]
        
        # Determine if it's a duplicate
        is_duplicate = any(r.is_duplicate for r in similar_rules)
        
        # Suggest action
        if is_duplicate:
            suggested_action = "use_existing"
        elif similar_rules and similar_rules[0].similarity_score > 0.8:
            suggested_action = "review"
        else:
            suggested_action = "save_new"
        
        return DeduplicationResult(
            is_duplicate=is_duplicate,
            similar_rules=similar_rule_models,
            suggested_action=suggested_action
        )
        
    except Exception as e:
        print(f"Error checking duplicate: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/shadow-rules/add-to-index")
async def add_shadow_rule_to_index(
    session_id: str,
    request: AddRuleToIndexRequest
):
    """
    Add a shadow rule to the session's vector index.
    """
    try:
        vector_store = get_vector_store(session_id)
        success = vector_store.add_rule(
            rule_id=request.rule_id,
            rule_text=request.rule_text,
            source_analysis=request.source_analysis,
            simple_rule=request.simple_rule,
            target_decision=request.target_decision,
            predicted_outcome=request.predicted_outcome,
            confidence_level=request.confidence_level,
            samples_affected=request.samples_affected
        )
        
        return {
            "success": success,
            "message": "Rule added to index" if success else "Rule already exists"
        }
        
    except Exception as e:
        print(f"Error adding rule to index: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/shadow-rules/add-bulk")
async def add_shadow_rules_bulk(
    session_id: str,
    request: AddRulesBulkRequest
):
    """
    Add multiple shadow rules at once (efficient for decision tree rules).
    """
    try:
        vector_store = get_vector_store(session_id)
        
        # Convert request models to dicts
        rules_data = [
            {
                "id": r.rule_id,
                "text": r.rule_text,
                "source_analysis": r.source_analysis,
                "simple_rule": r.simple_rule,
                "target_decision": r.target_decision,
                "predicted_outcome": r.predicted_outcome,
                "confidence_level": r.confidence_level,
                "samples_affected": r.samples_affected
            }
            for r in request.rules
        ]
        
        count = vector_store.add_rules_bulk(rules_data)
        
        return {
            "success": True,
            "added_count": count,
            "message": f"Added {count} rules to index"
        }
        
    except Exception as e:
        print(f"Error adding rules in bulk: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}/shadow-rules/remove-from-index/{rule_id}")
async def remove_shadow_rule_from_index(
    session_id: str,
    rule_id: str
):
    """
    Remove a shadow rule from the session's vector index.
    """
    try:
        vector_store = get_vector_store(session_id)
        success = vector_store.delete_rule(rule_id)
        
        return {
            "success": success,
            "message": "Rule removed from index" if success else "Rule not found"
        }
        
    except Exception as e:
        print(f"Error removing rule from index: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}/shadow-rules/clear-by-source/{source}")
async def clear_shadow_rules_by_source(
    session_id: str,
    source: str
):
    """
    Delete all shadow rules with a given source.
    Used before re-running analysis to clear old decision tree rules.
    
    Args:
        source: 'decision-tree', 'chat-discovered', or 'manual'
    """
    try:
        vector_store = get_vector_store(session_id)
        deleted_count = vector_store.delete_rules_by_source(source)
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} '{source}' rules from index"
        }
        
    except Exception as e:
        print(f"Error clearing rules by source: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/shadow-rules/search")
async def search_similar_shadow_rules(
    session_id: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.0
):
    """
    Search for shadow rules similar to the query text.
    """
    try:
        vector_store = get_vector_store(session_id)
        similar_rules = vector_store.find_similar(
            query_text=query,
            threshold=threshold,
            top_k=top_k
        )
        
        return {
            "results": [r.to_dict() for r in similar_rules],
            "count": len(similar_rules)
        }
        
    except Exception as e:
        print(f"Error searching similar rules: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/shadow-rules/stats")
async def get_shadow_rules_stats(session_id: str):
    """
    Get statistics about the session's shadow rules vector store.
    """
    try:
        vector_store = get_vector_store(session_id)
        stats = vector_store.get_stats()
        
        return VectorStoreStats(
            session_id=stats["session_id"],
            total_rules=stats["total_rules"],
            decision_tree_rules=stats["decision_tree_rules"],
            chat_discovered_rules=stats["chat_discovered_rules"],
            manual_rules=stats["manual_rules"],
            index_size=stats["index_size"]
        )
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/shadow-rules/all")
async def get_all_shadow_rules_from_index(session_id: str):
    """
    Get all shadow rules from the vector index with their metadata.
    """
    try:
        vector_store = get_vector_store(session_id)
        rules = vector_store.get_all_rules()
        
        return {
            "rules": rules,
            "count": len(rules)
        }
        
    except Exception as e:
        print(f"Error getting all rules: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/shadow-rules/rebuild-index")
async def rebuild_shadow_rules_index(session_id: str):
    """
    Rebuild the FAISS index from metadata.
    Useful if the index gets corrupted or out of sync.
    """
    try:
        vector_store = get_vector_store(session_id)
        vector_store._rebuild_index()
        
        stats = vector_store.get_stats()
        return {
            "success": True,
            "message": f"Index rebuilt with {stats['total_rules']} rules"
        }
        
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

