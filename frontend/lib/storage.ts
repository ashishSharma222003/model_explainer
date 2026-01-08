// Session storage utilities

// Suggestion from chat to update code
export interface CodeSuggestion {
  id: string;
  type: 'update_explain_global' | 'update_explain_txn' | 'add_feature' | 'fix_issue';
  title: string;
  description: string;
  code?: string;
  timestamp: string;
  fromStep: 'global-chat' | 'txn-chat';
  dismissed: boolean;
}

// Context snapshot for tracking what data was active during a chat message
export interface ContextSnapshot {
  hasCode: boolean;
  codeLength?: number;
  globalJsonVersion?: string;
  globalJsonFeatureCount?: number;
  txnId?: string;
  timestamp: string;
}

// Chat message with optional context tracking
export interface ChatMessageWithContext {
  type: 'user' | 'ai';
  content: string;
  context?: ContextSnapshot;
}

// Record of a generated report (stored in session history)
export interface GeneratedReport {
  id: string;
  reportType: string;
  format: string;
  title: string;
  generatedAt: string;
  includeCode: boolean;
  includeChatHistory: boolean;
  includeJsonData: boolean;
  filename: string;
  summary?: string;  // Preview of the report content
  content?: string;  // Full content for viewing later
}

export interface Session {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  mlCode: string;
  globalJson: any;
  txnJson: any;
  chatHistory: {
    codeAnalyzer: ChatMessageWithContext[];
    globalChat: ChatMessageWithContext[];
    txnChat: ChatMessageWithContext[];
  };
  suggestions?: CodeSuggestion[];
  reports?: GeneratedReport[];  // History of generated reports
}

const SESSIONS_KEY = 'model_explainer_sessions';
const CURRENT_SESSION_KEY = 'model_explainer_current_session';

export function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

export function createNewSession(name?: string): Session {
  const now = new Date().toISOString();
  return {
    id: generateSessionId(),
    name: name || `Session ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}`,
    createdAt: now,
    updatedAt: now,
    mlCode: '',
    globalJson: null,
    txnJson: null,
    chatHistory: {
      codeAnalyzer: [],
      globalChat: [],
      txnChat: [],
    },
    suggestions: [],
    reports: [],
  };
}

export function getAllSessions(): Session[] {
  if (typeof window === 'undefined') return [];
  try {
    const data = localStorage.getItem(SESSIONS_KEY);
    return data ? JSON.parse(data) : [];
  } catch (e) {
    console.error('Failed to load sessions:', e);
    return [];
  }
}

export function getSession(sessionId: string): Session | null {
  const sessions = getAllSessions();
  return sessions.find(s => s.id === sessionId) || null;
}

export function saveSession(session: Session): void {
  if (typeof window === 'undefined') return;
  try {
    const sessions = getAllSessions();
    const existingIndex = sessions.findIndex(s => s.id === session.id);
    
    // Create a copy to avoid mutating React state
    const sessionToSave: Session = {
      ...session,
      updatedAt: new Date().toISOString(),
      // Deep copy chat history to avoid reference issues
      chatHistory: {
        codeAnalyzer: [...(session.chatHistory?.codeAnalyzer || [])],
        globalChat: [...(session.chatHistory?.globalChat || [])],
        txnChat: [...(session.chatHistory?.txnChat || [])],
      },
      // Deep copy suggestions
      suggestions: [...(session.suggestions || [])],
    };
    
    if (existingIndex >= 0) {
      sessions[existingIndex] = sessionToSave;
    } else {
      sessions.unshift(sessionToSave); // Add to beginning
    }
    
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
    localStorage.setItem(CURRENT_SESSION_KEY, session.id);
    console.log('Session saved to localStorage:', session.id, 'Code length:', session.mlCode?.length || 0);
  } catch (e) {
    console.error('Failed to save session:', e);
  }
}

export function deleteSession(sessionId: string): void {
  if (typeof window === 'undefined') return;
  try {
    const sessions = getAllSessions().filter(s => s.id !== sessionId);
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  } catch (e) {
    console.error('Failed to delete session:', e);
  }
}

export function getCurrentSessionId(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(CURRENT_SESSION_KEY);
}

export function setCurrentSessionId(sessionId: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(CURRENT_SESSION_KEY, sessionId);
}

// Export session to JSON file
export function exportSession(session: Session): void {
  const blob = new Blob([JSON.stringify(session, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${session.name.replace(/[^a-z0-9]/gi, '_')}_${session.id}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Import session from JSON file
export function importSession(file: File): Promise<Session> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const session = JSON.parse(e.target?.result as string);
        // Generate new ID to avoid conflicts
        session.id = generateSessionId();
        session.name = `Imported: ${session.name}`;
        session.updatedAt = new Date().toISOString();
        resolve(session);
      } catch (err) {
        reject(new Error('Invalid session file'));
      }
    };
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

// Sync session to backend (optional, for backup)
export async function syncSessionToBackend(session: Session): Promise<void> {
  try {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    await fetch(`${API_URL}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(session),
    });
  } catch (e) {
    console.warn('Failed to sync session to backend:', e);
    // Silently fail - localStorage is the primary storage
  }
}

// Load sessions from backend (for cross-device sync)
export async function loadSessionsFromBackend(): Promise<Session[]> {
  try {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const res = await fetch(`${API_URL}/sessions`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.sessions || [];
  } catch (e) {
    console.warn('Failed to load sessions from backend:', e);
    return [];
  }
}

