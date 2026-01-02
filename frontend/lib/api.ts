const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function explainGlobal(request: any) {
  const res = await fetch(`${API_URL}/explain/global`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to fetch global explanation');
  return res.json();
}

export async function explainTransaction(request: any) {
  const res = await fetch(`${API_URL}/explain/transaction`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to fetch transaction explanation');
  return res.json();
}

export interface ChatApiResponse {
  response: string;
  global_json_suggestion: string | null;
}

export async function chat(request: { session_id: string; message: string; context?: any }): Promise<ChatApiResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error('Failed to send chat message');
  return res.json();
}

export async function guideCode(session_id: string, code: string) {
  const res = await fetch(`${API_URL}/guide-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, code }),
  });
  if (!res.ok) throw new Error('Failed to get code guidance');
  return res.json();
}

// ============ Session API ============

export interface SessionSummary {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  hasCode: boolean;
  codeLength: number;
  hasGlobalJson: boolean;
  hasTxnJson: boolean;
  messageCount: number;
  step: string;
}

export interface CodeSuggestionApi {
  id: string;
  type: string;
  title: string;
  description: string;
  code?: string;
  timestamp: string;
  fromStep: string;
  dismissed: boolean;
}

export interface ContextSnapshotApi {
  hasCode: boolean;
  codeLength?: number;
  globalJsonVersion?: string;
  globalJsonFeatureCount?: number;
  txnId?: string;
  timestamp: string;
}

export interface ChatMessageApi {
  type: 'user' | 'ai';
  content: string;
  context?: ContextSnapshotApi;
}

export interface FullSession {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  mlCode: string;
  globalJson: any;
  txnJson: any;
  chatHistory: {
    codeAnalyzer: ChatMessageApi[];
    globalChat: ChatMessageApi[];
    txnChat: ChatMessageApi[];
  };
  suggestions?: CodeSuggestionApi[];
  step?: string;
}

export async function getAllSessionsFromBackend(): Promise<{ sessions: FullSession[]; count: number }> {
  try {
    const res = await fetch(`${API_URL}/sessions`);
    if (!res.ok) throw new Error('Failed to fetch sessions');
    return res.json();
  } catch (e) {
    console.warn('Failed to fetch sessions from backend:', e);
    return { sessions: [], count: 0 };
  }
}

export async function getSessionSummaries(): Promise<{ summaries: SessionSummary[]; count: number }> {
  try {
    const res = await fetch(`${API_URL}/sessions/summaries`);
    if (!res.ok) throw new Error('Failed to fetch session summaries');
    return res.json();
  } catch (e) {
    console.warn('Failed to fetch session summaries:', e);
    return { summaries: [], count: 0 };
  }
}

export async function getSessionFromBackend(sessionId: string): Promise<FullSession | null> {
  try {
    const res = await fetch(`${API_URL}/sessions/${sessionId}`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.session;
  } catch (e) {
    console.warn('Failed to fetch session from backend:', e);
    return null;
  }
}

export async function saveSessionToBackend(session: FullSession): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(session),
    });
    if (!res.ok) throw new Error('Failed to save session');
    const data = await res.json();
    console.log('Session saved to backend:', data.message);
    return true;
  } catch (e) {
    console.warn('Failed to save session to backend:', e);
    return false;
  }
}

export async function deleteSessionFromBackend(sessionId: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/sessions/${sessionId}`, {
      method: 'DELETE',
    });
    return res.ok;
  } catch (e) {
    console.warn('Failed to delete session from backend:', e);
    return false;
  }
}
