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

// Selected case filter info
export interface CaseFilterInfo {
  id: string;
  column: string;
  operator: string;
  value: string;
  label: string;
}

// Saved smart filter (user-created or cached)
export interface SavedSmartFilter {
  id: string;
  label: string;
  description: string;
  icon?: string;  // Icon name for display
  color?: string;  // Color class for display
  filters: CaseFilterInfo[];
  isBuiltIn: boolean;  // true for default filters, false for user-created
  createdAt: string;
  usageCount: number;  // Track popularity
}

// Selected cases context for transaction analysis
export interface SelectedCasesContext {
  selectedCases: Record<string, any>[];
  filters: CaseFilterInfo[];
  totalCases: number;
  selectedAt: string;
}

// Context snapshot for tracking what data was active during a chat message
export interface ContextSnapshot {
  hasCode: boolean;
  codeLength?: number;
  globalJsonVersion?: string;
  globalJsonFeatureCount?: number;
  txnId?: string;  // Deprecated: use selectedCasesInfo instead
  // New: selected cases info for transaction analysis
  selectedCasesCount?: number;
  activeFilters?: string[];  // Filter labels for display
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
  includeSchema: boolean;
  includeChatHistory: boolean;
  includeJsonData: boolean;
  filename: string;
  summary?: string;  // Preview of the report content
  content?: string;  // Full content for viewing later
}

// Bank guidelines for analyst decision making
export interface BankGuideline {
  id: string;
  title: string;
  category: 'regulatory' | 'internal' | 'risk' | 'compliance' | 'operational' | 'custom';
  description: string;
  rules?: string[];  // Specific rules or criteria
  source?: string;   // e.g., "RBI Guidelines 2024", "Internal Policy v2.1"
  priority?: 'critical' | 'high' | 'medium' | 'low';
  addedAt: string;
}

// Analyst decision record for a transaction
export interface AnalystDecision {
  txnId: string;
  analystId?: string;
  modelPrediction: 'fraud' | 'legit';
  modelScore: number;
  analystDecision: 'confirmed_fraud' | 'false_positive' | 'escalate' | 'needs_review';
  decisionReason?: string;
  reviewNotes?: string;
  timeSpentMinutes?: number;
  guidelinesReferenced?: string[];  // IDs of guidelines used
  decidedAt: string;
}

// Shadow rule detected from analyst behavior
export interface DetectedShadowRule {
  id: string;
  description: string;
  pattern: string;  // What pattern was detected
  frequency: number;  // How often this pattern appears
  examples: string[];  // Transaction IDs showing this pattern
  potentialBias?: string;  // If this represents a bias
  alignsWithGuidelines: boolean;
  guidelineViolations?: string[];  // Which guidelines this might violate
  detectedAt: string;
}

export interface Session {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  mlCode: string;
  dataSchema: any;  // Data schema analysis (features, target, feature engineering)
  // CSV data is stored on BACKEND to avoid localStorage quota limits
  // Use getCsvData() API to fetch it
  csvFileName: string;  // Original CSV file name
  hasCsvData: boolean;  // Flag indicating if CSV data is stored on backend
  csvRowCount: number;  // Number of rows in CSV (for display)
  globalJson: any;
  txnJson: any;
  chatHistory: {
    codeAnalyzer: ChatMessageWithContext[];
    globalChat: ChatMessageWithContext[];
    txnChat: ChatMessageWithContext[];
  };
  suggestions?: CodeSuggestion[];
  reports?: GeneratedReport[];  // History of generated reports
  // Bank-specific context
  guidelines?: BankGuideline[];  // Bank's policies and regulatory guidelines
  analystDecisions?: AnalystDecision[];  // Historical analyst decisions for analysis
  detectedShadowRules?: DetectedShadowRule[];  // Detected patterns from analyst behavior
  // Saved smart filters (user-created filters that persist)
  savedSmartFilters?: SavedSmartFilter[];
  // Random Forest analysis result
  analysisResult?: any;
  // Selected wrong predictions for chat
  selectedWrongPredictions?: any[];
}

const SESSIONS_KEY = 'model_explainer_sessions';
const CURRENT_SESSION_KEY = 'model_explainer_current_session';
const SMART_FILTERS_KEY = 'model_explainer_smart_filters';

export function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// ============ Smart Filters Storage ============

export function getGlobalSmartFilters(): SavedSmartFilter[] {
  if (typeof window === 'undefined') return [];
  try {
    const data = localStorage.getItem(SMART_FILTERS_KEY);
    return data ? JSON.parse(data) : [];
  } catch (e) {
    console.error('Failed to load smart filters:', e);
    return [];
  }
}

export function saveGlobalSmartFilter(filter: Omit<SavedSmartFilter, 'id' | 'createdAt' | 'usageCount'>): SavedSmartFilter {
  const filters = getGlobalSmartFilters();
  
  const newFilter: SavedSmartFilter = {
    ...filter,
    id: `filter_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    createdAt: new Date().toISOString(),
    usageCount: 0,
  };
  
  filters.push(newFilter);
  
  if (typeof window !== 'undefined') {
    localStorage.setItem(SMART_FILTERS_KEY, JSON.stringify(filters));
  }
  
  return newFilter;
}

export function updateSmartFilterUsage(filterId: string): void {
  const filters = getGlobalSmartFilters();
  const filter = filters.find(f => f.id === filterId);
  
  if (filter) {
    filter.usageCount++;
    if (typeof window !== 'undefined') {
      localStorage.setItem(SMART_FILTERS_KEY, JSON.stringify(filters));
    }
  }
}

export function deleteGlobalSmartFilter(filterId: string): void {
  const filters = getGlobalSmartFilters();
  const updatedFilters = filters.filter(f => f.id !== filterId);
  
  if (typeof window !== 'undefined') {
    localStorage.setItem(SMART_FILTERS_KEY, JSON.stringify(updatedFilters));
  }
}

export function createNewSession(name?: string): Session {
  const now = new Date().toISOString();
  return {
    id: generateSessionId(),
    name: name || `Session ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}`,
    createdAt: now,
    updatedAt: now,
    mlCode: '',
    dataSchema: null,
    csvFileName: '',
    hasCsvData: false,
    csvRowCount: 0,
    globalJson: null,
    txnJson: null,
    chatHistory: {
      codeAnalyzer: [],
      globalChat: [],
      txnChat: [],
    },
    suggestions: [],
    reports: [],
    guidelines: [],
    analystDecisions: [],
    detectedShadowRules: [],
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

