"use client";
import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronDown, Plus, History, Code2, FileJson, MessageSquare, 
  Trash2, Check, Clock, FolderOpen
} from 'lucide-react';
import { Session } from '@/lib/storage';

interface SessionPickerProps {
  sessions: Session[];
  currentSession: Session;
  onSelectSession: (session: Session) => void;
  onNewSession: () => void;
  onDeleteSession: (sessionId: string) => void;
}

export default function SessionPicker({
  sessions,
  currentSession,
  onSelectSession,
  onNewSession,
  onDeleteSession,
}: SessionPickerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setDeleteConfirm(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
    return date.toLocaleDateString();
  };

  const getSessionStatus = (session: Session) => {
    if (session.txnJson) return { step: 'Txn Analysis', color: 'text-amber-400' };
    if (session.globalJson) return { step: 'Global Chat', color: 'text-cyan-400' };
    if (session.mlCode) return { step: 'Code Shared', color: 'text-violet-400' };
    return { step: 'New Session', color: 'text-slate-400' };
  };

  const handleDelete = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    if (deleteConfirm === sessionId) {
      onDeleteSession(sessionId);
      setDeleteConfirm(null);
    } else {
      setDeleteConfirm(sessionId);
      setTimeout(() => setDeleteConfirm(null), 3000);
    }
  };

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Main Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-3 px-4 py-2 bg-slate-800/70 hover:bg-slate-700/70 border border-slate-700 rounded-xl transition-all"
      >
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-sm font-medium text-white max-w-[150px] truncate">
            {currentSession.name}
          </span>
        </div>
        <div className="flex items-center gap-2 border-l border-slate-600 pl-3">
          <span className="text-xs text-slate-400">{sessions.length} sessions</span>
          <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </div>
      </button>

      {/* Dropdown */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute top-full left-0 mt-2 w-80 bg-[#0d1117] border border-slate-700 rounded-xl shadow-2xl overflow-hidden z-50"
          >
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-amber-400" />
                <span className="text-sm font-semibold text-white">Your Sessions</span>
              </div>
              <button
                onClick={() => {
                  onNewSession();
                  setIsOpen(false);
                }}
                className="flex items-center gap-1 px-2 py-1 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 rounded-lg text-cyan-300 text-xs font-medium transition-colors"
              >
                <Plus className="w-3 h-3" />
                New
              </button>
            </div>

            {/* Sessions List */}
            <div className="max-h-[400px] overflow-y-auto">
              {sessions.length === 0 ? (
                <div className="p-8 text-center">
                  <FolderOpen className="w-10 h-10 text-slate-600 mx-auto mb-2" />
                  <p className="text-sm text-slate-500">No sessions yet</p>
                </div>
              ) : (
                sessions.map((session) => {
                  const isActive = session.id === currentSession.id;
                  const status = getSessionStatus(session);
                  
                  return (
                    <div
                      key={session.id}
                      onClick={() => {
                        if (!isActive) {
                          onSelectSession(session);
                        }
                        setIsOpen(false);
                      }}
                      className={`px-4 py-3 border-b border-slate-800/50 cursor-pointer transition-all ${
                        isActive 
                          ? 'bg-cyan-500/10 border-l-2 border-l-cyan-500' 
                          : 'hover:bg-slate-800/50'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            {isActive && <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
                            <span className={`text-sm font-medium truncate ${isActive ? 'text-cyan-300' : 'text-white'}`}>
                              {session.name}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 mt-1">
                            <Clock className="w-3 h-3 text-slate-500" />
                            <span className="text-xs text-slate-500">{formatDate(session.updatedAt)}</span>
                            <span className={`text-xs ${status.color}`}>• {status.step}</span>
                          </div>
                        </div>
                        
                        {!isActive && (
                          <button
                            onClick={(e) => handleDelete(e, session.id)}
                            className={`p-1 rounded transition-colors ${
                              deleteConfirm === session.id
                                ? 'bg-red-500/20 text-red-400'
                                : 'hover:bg-slate-700 text-slate-500 hover:text-slate-300'
                            }`}
                            title={deleteConfirm === session.id ? 'Click again to confirm' : 'Delete'}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>

                      {/* Session content indicators */}
                      <div className="flex flex-wrap gap-2">
                        {session.mlCode && (
                          <div className="flex items-center gap-1 px-2 py-0.5 bg-violet-500/10 rounded text-xs text-violet-400">
                            <Code2 className="w-3 h-3" />
                            <span>{session.mlCode.length} chars</span>
                          </div>
                        )}
                        {session.globalJson && (
                          <div className="flex items-center gap-1 px-2 py-0.5 bg-cyan-500/10 rounded text-xs text-cyan-400">
                            <FileJson className="w-3 h-3" />
                            <span>Global</span>
                          </div>
                        )}
                        {session.txnJson && (
                          <div className="flex items-center gap-1 px-2 py-0.5 bg-amber-500/10 rounded text-xs text-amber-400">
                            <FileJson className="w-3 h-3" />
                            <span>Txn</span>
                          </div>
                        )}
                        {(session.chatHistory?.codeAnalyzer?.length > 0 ||
                          session.chatHistory?.globalChat?.length > 0 ||
                          session.chatHistory?.txnChat?.length > 0) && (
                          <div className="flex items-center gap-1 px-2 py-0.5 bg-emerald-500/10 rounded text-xs text-emerald-400">
                            <MessageSquare className="w-3 h-3" />
                            <span>
                              {(session.chatHistory?.codeAnalyzer?.length || 0) +
                               (session.chatHistory?.globalChat?.length || 0) +
                               (session.chatHistory?.txnChat?.length || 0)} msgs
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {/* Footer */}
            {sessions.length > 0 && (
              <div className="px-4 py-2 border-t border-slate-800 bg-slate-900/50">
                <p className="text-xs text-slate-500 text-center">
                  Click a session to resume where you left off
                </p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

