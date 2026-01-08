"""
Jupyter Kernel Manager
Manages Python kernels for executing user code in developer mode.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging
import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr

logger = logging.getLogger(__name__)


class ExecutionResult(BaseModel):
    success: bool
    outputs: List[Dict[str, Any]]
    error: Optional[str] = None
    execution_time_ms: int
    variables_captured: List[str] = []


class CodeExecutionRequest(BaseModel):
    session_id: str
    code: str
    timeout_seconds: int = 30
    capture_variables: bool = True


class KernelSession:
    """Represents an isolated Python execution environment."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.namespace: Dict[str, Any] = {
            '__builtins__': __builtins__,
            '__name__': '__main__',
        }
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.execution_count = 0
        
        # Pre-import common libraries
        self._setup_environment()
    
    def _setup_environment(self):
        """Pre-import common ML libraries."""
        setup_code = """
import warnings
warnings.filterwarnings('ignore')

# Try to import common libraries
try:
    import numpy as np
except ImportError:
    pass

try:
    import pandas as pd
except ImportError:
    pass

try:
    import json
except ImportError:
    pass
"""
        try:
            exec(setup_code, self.namespace)
        except Exception as e:
            logger.warning(f"Error setting up kernel environment: {e}")
    
    def execute(self, code: str, timeout: int = 30) -> ExecutionResult:
        """Execute code in this kernel's namespace."""
        
        start_time = datetime.now()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        outputs: List[Dict[str, Any]] = []
        error = None
        
        self.last_used = datetime.now()
        self.execution_count += 1
        
        try:
            # Redirect stdout/stderr
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to execute as expression first (to capture return value)
                try:
                    result = eval(code, self.namespace)
                    if result is not None:
                        outputs.append({
                            'type': 'result',
                            'data': self._serialize_output(result)
                        })
                except SyntaxError:
                    # Not an expression, execute as statement
                    exec(code, self.namespace)
            
            # Capture stdout
            stdout_val = stdout_capture.getvalue()
            if stdout_val:
                outputs.append({
                    'type': 'stream',
                    'name': 'stdout',
                    'text': stdout_val
                })
            
            # Capture stderr
            stderr_val = stderr_capture.getvalue()
            if stderr_val:
                outputs.append({
                    'type': 'stream',
                    'name': 'stderr',
                    'text': stderr_val
                })
                
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            outputs.append({
                'type': 'error',
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': traceback.format_exc()
            })
        
        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Get user-defined variables
        user_vars = [
            k for k in self.namespace.keys() 
            if not k.startswith('_') and k not in ['np', 'pd', 'json', 'warnings']
        ]
        
        return ExecutionResult(
            success=error is None,
            outputs=outputs,
            error=error,
            execution_time_ms=execution_time,
            variables_captured=user_vars[:20]  # Limit to 20 vars
        )
    
    def _serialize_output(self, obj: Any) -> str:
        """Serialize output for JSON transport."""
        try:
            # Handle numpy arrays
            if hasattr(obj, 'tolist'):
                return str(obj.tolist())
            # Handle pandas DataFrames
            if hasattr(obj, 'to_string'):
                return obj.to_string(max_rows=20)
            # Handle dicts
            if isinstance(obj, dict):
                import json
                return json.dumps(obj, indent=2, default=str)[:2000]
            # Default
            return str(obj)[:2000]
        except Exception:
            return str(obj)[:2000]
    
    def get_variable(self, name: str) -> Optional[Any]:
        """Get a variable from the namespace."""
        return self.namespace.get(name)
    
    def set_variable(self, name: str, value: Any):
        """Set a variable in the namespace."""
        self.namespace[name] = value
    
    def list_variables(self) -> Dict[str, str]:
        """List all user-defined variables with their types."""
        return {
            k: type(v).__name__
            for k, v in self.namespace.items()
            if not k.startswith('_') and k not in ['np', 'pd', 'json', 'warnings']
        }


class KernelManager:
    """Manages multiple kernel sessions."""
    
    def __init__(self, max_sessions: int = 50, session_timeout_minutes: int = 60):
        self.sessions: Dict[str, KernelSession] = {}
        self.max_sessions = max_sessions
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def get_or_create_session(self, session_id: str) -> KernelSession:
        """Get existing session or create a new one."""
        
        # Cleanup old sessions first
        self._cleanup_old_sessions()
        
        if session_id not in self.sessions:
            if len(self.sessions) >= self.max_sessions:
                # Remove oldest session
                oldest_id = min(
                    self.sessions.keys(),
                    key=lambda k: self.sessions[k].last_used
                )
                del self.sessions[oldest_id]
                logger.info(f"Removed oldest kernel session: {oldest_id}")
            
            self.sessions[session_id] = KernelSession(session_id)
            logger.info(f"Created new kernel session: {session_id}")
        
        return self.sessions[session_id]
    
    def _cleanup_old_sessions(self):
        """Remove sessions that haven't been used recently."""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_used > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"Cleaned up expired kernel session: {sid}")
    
    async def execute_code(self, request: CodeExecutionRequest) -> ExecutionResult:
        """Execute code in a session's kernel."""
        
        session = self.get_or_create_session(request.session_id)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: session.execute(request.code, request.timeout_seconds)
        )
        
        return result
    
    def reset_session(self, session_id: str) -> bool:
        """Reset a session's kernel (clear all variables)."""
        if session_id in self.sessions:
            self.sessions[session_id] = KernelSession(session_id)
            return True
        return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return {
            'session_id': session_id,
            'created_at': session.created_at.isoformat(),
            'last_used': session.last_used.isoformat(),
            'execution_count': session.execution_count,
            'variables': session.list_variables()
        }
    
    def inject_context(self, session_id: str, ml_code: str = None, global_json: dict = None):
        """Inject context data into a session's namespace."""
        session = self.get_or_create_session(session_id)
        
        if ml_code:
            session.set_variable('_ml_code', ml_code)
        
        if global_json:
            session.set_variable('global_explanation', global_json)
            # Also set feature importances as a convenience
            if 'global_importance' in global_json:
                session.set_variable('feature_importances', global_json['global_importance'])


# Global instance
kernel_manager = KernelManager()

