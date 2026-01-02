"""
Session Management Module
Handles saving and loading user sessions with all their data:
- ML Code
- Global JSON
- Transaction JSON  
- Chat histories (code analyzer, global chat, txn chat)
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

SESSIONS_FILE = "user_sessions.json"


class ContextSnapshot(BaseModel):
    """Snapshot of context at the time of a chat message."""
    hasCode: bool = False
    codeLength: Optional[int] = None
    globalJsonVersion: Optional[str] = None
    globalJsonFeatureCount: Optional[int] = None
    txnId: Optional[str] = None
    timestamp: str


class ChatMessage(BaseModel):
    type: str  # 'user' or 'ai'
    content: str
    context: Optional[ContextSnapshot] = None  # Context snapshot for this message


class ChatHistory(BaseModel):
    codeAnalyzer: List[ChatMessage] = []
    globalChat: List[ChatMessage] = []
    txnChat: List[ChatMessage] = []


class CodeSuggestion(BaseModel):
    """Suggestion from chat to update code."""
    id: str
    type: str  # 'update_explain_global', 'update_explain_txn', 'add_feature', 'fix_issue'
    title: str
    description: str
    code: Optional[str] = None
    timestamp: str
    fromStep: str  # 'global-chat' or 'txn-chat'
    dismissed: bool = False


class Session(BaseModel):
    id: str
    name: str
    createdAt: str
    updatedAt: str
    mlCode: str = ""
    globalJson: Optional[Dict[str, Any]] = None
    txnJson: Optional[Dict[str, Any]] = None
    chatHistory: ChatHistory = ChatHistory()
    suggestions: List[CodeSuggestion] = []
    
    # Metadata for quick display
    step: str = "code"  # Current step: code, global-json, global-chat, txn-json, txn-chat


class SessionManager:
    def __init__(self, file_path: str = SESSIONS_FILE):
        self.file_path = file_path
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        if not os.path.exists(self.file_path):
            self._save_to_file({})
    
    def _load_from_file(self) -> Dict[str, Dict]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return {}
    
    def _save_to_file(self, data: Dict[str, Dict]):
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions, sorted by updatedAt (newest first)."""
        sessions = self._load_from_file()
        session_list = list(sessions.values())
        session_list.sort(key=lambda x: x.get('updatedAt', ''), reverse=True)
        return session_list
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific session by ID."""
        sessions = self._load_from_file()
        return sessions.get(session_id)
    
    def save_session(self, session_data: Dict) -> Dict:
        """Save or update a session."""
        sessions = self._load_from_file()
        session_id = session_data.get('id')
        
        if not session_id:
            raise ValueError("Session ID is required")
        
        # Update timestamp
        session_data['updatedAt'] = datetime.now().isoformat()
        
        # Ensure all required fields exist
        if 'createdAt' not in session_data:
            session_data['createdAt'] = session_data['updatedAt']
        if 'chatHistory' not in session_data:
            session_data['chatHistory'] = {
                'codeAnalyzer': [],
                'globalChat': [],
                'txnChat': []
            }
        if 'suggestions' not in session_data:
            session_data['suggestions'] = []
        
        sessions[session_id] = session_data
        self._save_to_file(sessions)
        
        logger.info(f"Session saved: {session_id}, code length: {len(session_data.get('mlCode', ''))}")
        return session_data
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        sessions = self._load_from_file()
        if session_id in sessions:
            del sessions[session_id]
            self._save_to_file(sessions)
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get a summary of a session (for list display)."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            'id': session['id'],
            'name': session['name'],
            'createdAt': session['createdAt'],
            'updatedAt': session['updatedAt'],
            'hasCode': bool(session.get('mlCode')),
            'codeLength': len(session.get('mlCode', '')),
            'hasGlobalJson': session.get('globalJson') is not None,
            'hasTxnJson': session.get('txnJson') is not None,
            'messageCount': (
                len(session.get('chatHistory', {}).get('codeAnalyzer', [])) +
                len(session.get('chatHistory', {}).get('globalChat', [])) +
                len(session.get('chatHistory', {}).get('txnChat', []))
            ),
            'suggestionCount': len([s for s in session.get('suggestions', []) if not s.get('dismissed', False)]),
            'step': session.get('step', 'code')
        }
    
    def get_all_summaries(self) -> List[Dict]:
        """Get summaries of all sessions."""
        sessions = self._load_from_file()
        summaries = []
        for session_id in sessions:
            summary = self.get_session_summary(session_id)
            if summary:
                summaries.append(summary)
        summaries.sort(key=lambda x: x.get('updatedAt', ''), reverse=True)
        return summaries


# Global instance
session_manager = SessionManager()

