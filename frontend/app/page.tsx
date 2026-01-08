"use client";
import { useState, createContext, useContext, useEffect, useCallback, useRef } from 'react';
import { Layers, ChevronRight, Check, FileText, Terminal, User, Code2 } from 'lucide-react';
import CodeAnalyzer from '@/components/CodeAnalyzer';
import GlobalJsonInput from '@/components/GlobalJsonInput';
import ChatPanel from '@/components/ChatPanel';
import TxnJsonInput from '@/components/TxnJsonInput';
import SessionPicker from '@/components/SessionPicker';
import ReportGenerator from '@/components/ReportGenerator';
import DeveloperPanel from '@/components/DeveloperPanel';
import { 
  Session, 
  CodeSuggestion,
  ChatMessageWithContext,
  ContextSnapshot,
  createNewSession, 
  getAllSessions, 
  saveSession, 
  getCurrentSessionId,
  setCurrentSessionId,
  deleteSession as deleteSessionFromStorage,
} from '@/lib/storage';
import { saveSessionToBackend, getAllSessionsFromBackend } from '@/lib/api';

// Re-export types from storage for convenience
export type { CodeSuggestion, ChatMessageWithContext, ContextSnapshot } from '@/lib/storage';

// Alias for backwards compatibility
export type ChatMessage = ChatMessageWithContext;

// Context for sharing data across steps
interface AppContextType {
  session: Session;
  updateSession: (updates: Partial<Session>) => void;
  // Convenience accessors
  mlCode: string;
  setMlCode: (code: string) => void;
  globalJson: any;
  setGlobalJson: (json: any) => void;
  txnJson: any;
  setTxnJson: (json: any) => void;
  // Chat history
  codeAnalyzerMessages: ChatMessage[];
  setCodeAnalyzerMessages: (messages: ChatMessage[]) => void;
  globalChatMessages: ChatMessage[];
  setGlobalChatMessages: (messages: ChatMessage[]) => void;
  txnChatMessages: ChatMessage[];
  setTxnChatMessages: (messages: ChatMessage[]) => void;
  // Context snapshot helper
  createContextSnapshot: () => ContextSnapshot;
  // Suggestions system
  suggestions: CodeSuggestion[];
  addSuggestion: (suggestion: Omit<CodeSuggestion, 'id' | 'timestamp' | 'dismissed'>) => void;
  dismissSuggestion: (id: string) => void;
  clearSuggestions: () => void;
  // Navigation
  goToStep: (step: Step) => void;
  currentStep: Step;
}

export const AppContext = createContext<AppContextType | null>(null);

type Step = 'code' | 'global-json' | 'global-chat' | 'txn-json' | 'txn-chat';

const STEPS: { id: Step; label: string; description: string }[] = [
  { id: 'code', label: 'Share Code', description: 'Paste your ML code' },
  { id: 'global-json', label: 'Global JSON', description: 'Paste generated JSON' },
  { id: 'global-chat', label: 'Explore Global', description: 'Chat about model behavior' },
  { id: 'txn-json', label: 'Txn JSON', description: 'Add transaction context' },
  { id: 'txn-chat', label: 'Explore Txn', description: 'Chat about predictions' },
];

function getStepFromSession(session: Session): Step {
  if (session.txnJson) return 'txn-chat';
  if (session.globalJson) return 'global-chat';
  if (session.mlCode && session.chatHistory?.codeAnalyzer?.length > 0) return 'code';
  return 'code';
}

export default function Home() {
  const [currentStep, setCurrentStep] = useState<Step>('code');
  const [session, setSession] = useState<Session>(() => createNewSession());
  const [allSessions, setAllSessions] = useState<Session[]>([]);
  const [isHydrated, setIsHydrated] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle');
  const [showReportGenerator, setShowReportGenerator] = useState(false);
  const [developerMode, setDeveloperMode] = useState(false);
  const [kernelOutputToChat, setKernelOutputToChat] = useState<string | null>(null);
  
  const shouldSave = useRef(false);
  
  // Suggestions are derived from session but managed separately for reactivity
  const suggestions = session.suggestions || [];

  // Load sessions on mount
  useEffect(() => {
    const loadSessions = async () => {
      let sessions = getAllSessions();
      
      try {
        const backendData = await getAllSessionsFromBackend();
        if (backendData.sessions.length > 0) {
          const sessionMap = new Map<string, Session>();
          sessions.forEach(s => sessionMap.set(s.id, s));
          backendData.sessions.forEach(s => {
            const existing = sessionMap.get(s.id);
            if (!existing || new Date(s.updatedAt) > new Date(existing.updatedAt)) {
              sessionMap.set(s.id, s as Session);
            }
          });
          sessions = Array.from(sessionMap.values());
          sessions.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
          sessions.forEach(s => saveSession(s));
        }
      } catch (e) {
        console.warn('Could not sync with backend:', e);
      }
      
      setAllSessions(sessions);
      
      const lastSessionId = getCurrentSessionId();
      if (lastSessionId) {
        const lastSession = sessions.find(s => s.id === lastSessionId);
        if (lastSession) {
          setSession(lastSession);
          setCurrentStep(getStepFromSession(lastSession));
        }
      }
      
      setIsHydrated(true);
      setTimeout(() => {
        shouldSave.current = true;
      }, 500);
    };
    
    loadSessions();
  }, []);

  // Auto-save session
  useEffect(() => {
    if (!isHydrated || !shouldSave.current) return;
    
    setSaveStatus('saving');
    saveSession(session);
    setCurrentSessionId(session.id);
    saveSessionToBackend(session as any);
    setAllSessions(getAllSessions());
    
    setSaveStatus('saved');
    const timeout = setTimeout(() => setSaveStatus('idle'), 1500);
    return () => clearTimeout(timeout);
  }, [session, isHydrated]);

  const updateSession = useCallback((updates: Partial<Session>) => {
    setSession(prev => ({ ...prev, ...updates }));
  }, []);

  const handleSelectSession = (selectedSession: Session) => {
    shouldSave.current = false;
    setSession(selectedSession);
    setCurrentSessionId(selectedSession.id);
    setCurrentStep(getStepFromSession(selectedSession));
    // Suggestions are now loaded from the session automatically
    setTimeout(() => {
      shouldSave.current = true;
    }, 100);
  };

  const handleNewSession = () => {
    const newSession = createNewSession();
    setSession(newSession);
    setCurrentSessionId(newSession.id);
    setCurrentStep('code');
    // New sessions start with empty suggestions (already in createNewSession)
    saveSession(newSession);
    setAllSessions(getAllSessions());
  };

  const handleDeleteSession = (sessionId: string) => {
    deleteSessionFromStorage(sessionId);
    const remainingSessions = getAllSessions();
    setAllSessions(remainingSessions);
    
    if (sessionId === session.id) {
      if (remainingSessions.length > 0) {
        handleSelectSession(remainingSessions[0]);
      } else {
        handleNewSession();
      }
    }
  };

  // Suggestion system - persisted in session
  const addSuggestion = useCallback((suggestion: Omit<CodeSuggestion, 'id' | 'timestamp' | 'dismissed'>) => {
    const newSuggestion: CodeSuggestion = {
      ...suggestion,
      id: `suggestion_${Date.now()}`,
      timestamp: new Date().toISOString(),
      dismissed: false,
    };
    updateSession({ suggestions: [newSuggestion, ...(session.suggestions || [])] });
  }, [session.suggestions, updateSession]);

  const dismissSuggestion = useCallback((id: string) => {
    updateSession({ 
      suggestions: (session.suggestions || []).map(s => s.id === id ? { ...s, dismissed: true } : s) 
    });
  }, [session.suggestions, updateSession]);

  const clearSuggestions = useCallback(() => {
    updateSession({ suggestions: [] });
  }, [updateSession]);

  const goToStep = useCallback((step: Step) => {
    setCurrentStep(step);
  }, []);

  // Create a snapshot of current context for chat messages
  const createContextSnapshot = useCallback((): ContextSnapshot => {
    return {
      hasCode: Boolean(session.mlCode && session.mlCode.length > 0),
      codeLength: session.mlCode?.length || 0,
      globalJsonVersion: session.globalJson?.model_version,
      globalJsonFeatureCount: session.globalJson?.global_importance?.length,
      txnId: session.txnJson?.txn_id,
      timestamp: new Date().toISOString(),
    };
  }, [session.mlCode, session.globalJson, session.txnJson]);

  // Context value
  const contextValue: AppContextType = {
    session,
    updateSession,
    mlCode: session.mlCode,
    setMlCode: (code) => updateSession({ mlCode: code }),
    globalJson: session.globalJson,
    setGlobalJson: (json) => updateSession({ globalJson: json }),
    txnJson: session.txnJson,
    setTxnJson: (json) => updateSession({ txnJson: json }),
    codeAnalyzerMessages: session.chatHistory?.codeAnalyzer || [],
    setCodeAnalyzerMessages: (messages) => updateSession({ 
      chatHistory: { ...session.chatHistory, codeAnalyzer: messages }
    }),
    globalChatMessages: session.chatHistory?.globalChat || [],
    setGlobalChatMessages: (messages) => updateSession({ 
      chatHistory: { ...session.chatHistory, globalChat: messages }
    }),
    txnChatMessages: session.chatHistory?.txnChat || [],
    setTxnChatMessages: (messages) => updateSession({ 
      chatHistory: { ...session.chatHistory, txnChat: messages }
    }),
    createContextSnapshot,
    suggestions,
    addSuggestion,
    dismissSuggestion,
    clearSuggestions,
    goToStep,
    currentStep,
  };

  const currentStepIndex = STEPS.findIndex(s => s.id === currentStep);

  const canNavigateTo = (step: Step) => {
    const stepIndex = STEPS.findIndex(s => s.id === step);
    if (stepIndex < currentStepIndex) return true;
    if (step === 'code') return true;
    if (step === 'global-json') return session.mlCode.length > 0;
    if (step === 'global-chat') return session.globalJson !== null;
    if (step === 'txn-json') return session.globalJson !== null;
    if (step === 'txn-chat') return session.txnJson !== null;
    return false;
  };

  const handleNextStep = () => {
    const nextIndex = currentStepIndex + 1;
    if (nextIndex < STEPS.length) {
      setCurrentStep(STEPS[nextIndex].id);
    }
  };

  if (!isHydrated) {
    return (
      <main className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
          <div className="text-slate-400">Loading sessions...</div>
        </div>
      </main>
    );
  }

  return (
    <AppContext.Provider value={contextValue}>
      <main className="min-h-screen bg-[#0a0a0f] text-slate-200 font-sans selection:bg-cyan-500/30">
        {/* Background */}
        <div className="fixed inset-0 bg-gradient-to-br from-[#0a0a0f] via-[#0d1117] to-[#0a0a0f] -z-10" />
        <div className="fixed inset-0 opacity-30 -z-10" style={{
          backgroundImage: `radial-gradient(circle at 20% 50%, rgba(56, 189, 248, 0.08) 0%, transparent 50%),
                            radial-gradient(circle at 80% 80%, rgba(168, 85, 247, 0.06) 0%, transparent 50%)`
        }} />

        {/* Header */}
        <header className="border-b border-slate-800/60 bg-[#0a0a0f]/80 backdrop-blur-xl sticky top-0 z-30">
          <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-tr from-cyan-500 to-blue-600 p-2.5 rounded-xl shadow-lg shadow-cyan-500/20">
                <Layers className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white tracking-tight">
                  Model Explainer
                </h1>
                <p className="text-xs text-slate-500 -mt-0.5">Chat-first ML analysis</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              {saveStatus !== 'idle' && (
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  saveStatus === 'saving' 
                    ? 'bg-amber-500/20 text-amber-300' 
                    : 'bg-emerald-500/20 text-emerald-300'
                }`}>
                  {saveStatus === 'saving' ? (
                    <>
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-pulse" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Check className="w-3 h-3" />
                      Saved
                    </>
                  )}
                </div>
              )}

              {/* Mode Toggle */}
              <div className="flex items-center bg-slate-800/50 rounded-lg border border-slate-700/50 p-0.5">
                <button
                  onClick={() => setDeveloperMode(false)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    !developerMode 
                      ? 'bg-slate-700 text-white' 
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <User className="w-3.5 h-3.5" />
                  Standard
                </button>
                <button
                  onClick={() => setDeveloperMode(true)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    developerMode 
                      ? 'bg-emerald-600 text-white' 
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Terminal className="w-3.5 h-3.5" />
                  Developer
                </button>
              </div>

              {/* Report Generator Button */}
              <button
                onClick={() => setShowReportGenerator(true)}
                className="flex items-center gap-2 px-3 py-1.5 bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/40 rounded-lg text-xs font-medium text-violet-300 transition-all"
              >
                <FileText className="w-3.5 h-3.5" />
                Report
              </button>
              
              <SessionPicker
                sessions={allSessions}
                currentSession={session}
                onSelectSession={handleSelectSession}
                onNewSession={handleNewSession}
                onDeleteSession={handleDeleteSession}
              />
              
              <div className="px-3 py-1.5 bg-slate-800/50 rounded-full border border-slate-700/50">
                <span className="text-xs font-mono text-slate-400">v0.4</span>
              </div>
            </div>
          </div>
        </header>

        {/* Step Navigator */}
        <div className="border-b border-slate-800/40 bg-[#0d1117]/50 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center gap-2 overflow-x-auto pb-2">
              {STEPS.map((step, idx) => {
                const isActive = step.id === currentStep;
                const isCompleted = idx < currentStepIndex;
                const isClickable = canNavigateTo(step.id);
                const hasSuggestions = suggestions.filter(s => !s.dismissed && s.fromStep === step.id).length > 0;

                return (
                  <button
                    key={step.id}
                    onClick={() => isClickable && setCurrentStep(step.id)}
                    disabled={!isClickable}
                    className={`relative flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all whitespace-nowrap ${
                      isActive
                        ? 'bg-gradient-to-r from-cyan-500/20 to-blue-600/20 border border-cyan-500/40 shadow-lg shadow-cyan-500/10'
                        : isCompleted
                        ? 'bg-emerald-500/10 border border-emerald-500/30 hover:bg-emerald-500/20'
                        : isClickable
                        ? 'bg-slate-800/30 border border-slate-700/40 hover:bg-slate-800/50'
                        : 'bg-slate-900/30 border border-slate-800/30 opacity-40 cursor-not-allowed'
                    }`}
                  >
                    {/* Suggestion indicator */}
                    {hasSuggestions && step.id === 'code' && (
                      <div className="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full animate-pulse" />
                    )}
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                      isActive
                        ? 'bg-cyan-500 text-white'
                        : isCompleted
                        ? 'bg-emerald-500 text-white'
                        : 'bg-slate-700 text-slate-400'
                    }`}>
                      {isCompleted ? <Check className="w-3.5 h-3.5" /> : idx + 1}
                    </div>
                    <div className="text-left">
                      <div className={`text-sm font-medium ${isActive ? 'text-cyan-300' : isCompleted ? 'text-emerald-300' : 'text-slate-300'}`}>
                        {step.label}
                      </div>
                      <div className="text-xs text-slate-500">{step.description}</div>
                    </div>
                    {idx < STEPS.length - 1 && (
                      <ChevronRight className="w-4 h-4 text-slate-600 ml-2" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Main Content - with side panel space when developer mode is on */}
        <div className={`transition-all duration-300 ${
          developerMode && (currentStep === 'global-chat' || currentStep === 'txn-chat')
            ? 'mr-[420px]' // Make room for the developer panel
            : ''
        }`}>
        <div className="max-w-7xl mx-auto px-6 py-8">
          {currentStep === 'code' && (
            <CodeAnalyzer onComplete={() => handleNextStep()} />
          )}
          {currentStep === 'global-json' && (
            <GlobalJsonInput onComplete={() => handleNextStep()} />
          )}
          {currentStep === 'global-chat' && (
            <ChatPanel 
              mode="global"
              onAddTxn={() => setCurrentStep('txn-json')}
                kernelOutput={kernelOutputToChat}
                onKernelOutputUsed={() => setKernelOutputToChat(null)}
            />
          )}
          {currentStep === 'txn-json' && (
            <TxnJsonInput onComplete={() => handleNextStep()} />
          )}
          {currentStep === 'txn-chat' && (
              <ChatPanel 
                mode="txn" 
                kernelOutput={kernelOutputToChat}
                onKernelOutputUsed={() => setKernelOutputToChat(null)}
              />
          )}
          </div>
        </div>

        {/* Developer Panel - Fixed Side Panel */}
        {developerMode && (currentStep === 'global-chat' || currentStep === 'txn-chat') && (
          <DeveloperPanel 
            isVisible={true}
            onSendToChat={(output) => {
              setKernelOutputToChat(output);
            }}
          />
        )}

        {/* Report Generator Modal */}
        <ReportGenerator 
          isOpen={showReportGenerator} 
          onClose={() => setShowReportGenerator(false)} 
        />
      </main>
    </AppContext.Provider>
  );
}
