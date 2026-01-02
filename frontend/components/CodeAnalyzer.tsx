"use client";
import { useState, useRef, useEffect, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Code2, Send, Sparkles, Copy, Check, ArrowRight, Bot, RotateCcw, History, ChevronDown, ChevronUp, FileCode, Lightbulb, X, MessageSquare, Wand2 } from 'lucide-react';
import { guideCode } from '../lib/api';
import { AppContext, ChatMessage } from '@/app/page';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ReactNode } from 'react';
import { getAllSessions, Session } from '@/lib/storage';

export default function CodeAnalyzer({ onComplete }: { onComplete: () => void }) {
  const context = useContext(AppContext);
  const [userCode, setUserCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(`sess_code_${context?.session.id || Math.random().toString(36).substr(2, 9)}`);
  const [copied, setCopied] = useState(false);
  const [copiedSuggestionId, setCopiedSuggestionId] = useState<string | null>(null);
  const [initialized, setInitialized] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [previousSessions, setPreviousSessions] = useState<Session[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Get suggestions from context
  const suggestions = context?.suggestions?.filter(s => !s.dismissed) || [];
  const hasActiveSuggestions = suggestions.length > 0;

  // Use messages from context
  const messages = context?.codeAnalyzerMessages || [];
  const setMessages = (newMessages: ChatMessage[]) => {
    context?.setCodeAnalyzerMessages(newMessages);
  };

  // Load previous sessions with code
  useEffect(() => {
    const sessions = getAllSessions();
    const sessionsWithCode = sessions.filter(s => 
      s.mlCode && s.mlCode.length > 0 && s.id !== context?.session.id
    );
    setPreviousSessions(sessionsWithCode);
  }, [context?.session.id]);

  // Initialize code from context on mount
  useEffect(() => {
    if (!initialized && context?.mlCode) {
      setUserCode(context.mlCode);
      setInitialized(true);
    } else if (!initialized && context) {
      setInitialized(true);
    }
  }, [context?.mlCode, initialized]);

  // Sync if session changes
  useEffect(() => {
    if (context?.session.id) {
      setUserCode(context.mlCode || '');
      const sessions = getAllSessions();
      const sessionsWithCode = sessions.filter(s => 
        s.mlCode && s.mlCode.length > 0 && s.id !== context?.session.id
      );
      setPreviousSessions(sessionsWithCode);
    }
  }, [context?.session.id]);

  // Save code to context (debounced)
  const contextRef = useRef(context);
  contextRef.current = context;

  useEffect(() => {
    if (!initialized) return;
    
    const timeout = setTimeout(() => {
      if (contextRef.current && userCode !== contextRef.current.mlCode) {
        contextRef.current.setMlCode(userCode);
      }
    }, 300);
    
    return () => clearTimeout(timeout);
  }, [userCode, initialized]);

  const handleAnalyzeCode = async () => {
    if (!userCode.trim()) return;
    
    context?.setMlCode(userCode);
    
    setLoading(true);
    const newMessages: ChatMessage[] = [...messages, { 
      type: 'user', 
      content: `Analyze my code and generate the \`explain_global()\` function:\n\n\`\`\`python\n${userCode.substring(0, 200)}${userCode.length > 200 ? '...' : ''}\n\`\`\``
    }];
    setMessages(newMessages);

    try {
      const res = await guideCode(sessionId, 
        `Here is my model code. Please analyze it and write a complete Python function called 'explain_global()' that generates a Global JSON.

MY CODE:

${userCode}`
      );
      setMessages([...newMessages, { type: 'ai', content: res.response }]);
    } catch (e) {
      setMessages([...newMessages, { type: 'ai', content: "Failed to analyze code. Please check your connection." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleFollowUp = async (question: string) => {
    if (!question.trim()) return;
    setLoading(true);
    const newMessages: ChatMessage[] = [...messages, { type: 'user', content: question }];
    setMessages(newMessages);

    try {
      const res = await guideCode(sessionId, question);
      setMessages([...newMessages, { type: 'ai', content: res.response }]);
    } catch (e) {
      setMessages([...newMessages, { type: 'ai', content: "Failed to respond." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  const handleLoadPreviousCode = (code: string) => {
    setUserCode(code);
    context?.setMlCode(code);
    setShowHistory(false);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const copySuggestion = (suggestionId: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSuggestionId(suggestionId);
    setTimeout(() => setCopiedSuggestionId(null), 2000);
  };

  const useSuggestionInChat = async (suggestion: typeof suggestions[0]) => {
    if (!userCode.trim()) {
      // If no code, just show a message
      setMessages([...messages, { 
        type: 'ai', 
        content: "Please paste your ML code first, then I can apply the suggestion." 
      }]);
      return;
    }
    
    setLoading(true);
    const suggestionPrompt = `Based on this suggestion from our previous chat, please update the explain_global() function:

SUGGESTION: ${suggestion.description}

Please regenerate the explain_global() function incorporating this improvement.

MY CODE:
${userCode}`;

    const newMessages: ChatMessage[] = [...messages, { 
      type: 'user', 
      content: `Apply suggestion: "${suggestion.title}"\n\n${suggestion.description}`
    }];
    setMessages(newMessages);

    try {
      const res = await guideCode(sessionId, suggestionPrompt);
      setMessages([...newMessages, { type: 'ai', content: res.response }]);
      // Dismiss the suggestion after using it
      context?.dismissSuggestion(suggestion.id);
    } catch (e) {
      setMessages([...newMessages, { type: 'ai', content: "Failed to apply suggestion. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
    return date.toLocaleDateString();
  };

  const getCodePreview = (code: string, maxLength: number = 100) => {
    const preview = code.substring(0, maxLength);
    return preview.length < code.length ? preview + '...' : preview;
  };

  return (
    <div className="space-y-4 h-[calc(100vh-220px)] flex flex-col">
      {/* Suggestions Banner from Global Chat */}
      <AnimatePresence>
        {hasActiveSuggestions && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex-shrink-0"
          >
            <div className="bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/40 rounded-xl p-4">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-amber-500/20 rounded-lg flex-shrink-0">
                  <Lightbulb className="w-5 h-5 text-amber-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <h4 className="text-sm font-semibold text-amber-300">
                      Suggestions from Chat ({suggestions.length})
                    </h4>
                    <span className="text-xs text-amber-400/60 flex items-center gap-1">
                      <MessageSquare className="w-3 h-3" />
                      From Explore Global
                    </span>
                  </div>
                  <div className="space-y-2 max-h-[180px] overflow-y-auto">
                    {suggestions.map((suggestion) => (
                      <div key={suggestion.id} className="bg-slate-900/50 rounded-lg p-3">
                        <div className="flex items-start gap-2">
                          <div className="flex-1 min-w-0">
                            <p className="text-xs text-white font-medium mb-1">{suggestion.title}</p>
                            <p className="text-xs text-amber-200/70 mb-2">{suggestion.description}</p>
                          </div>
                          <div className="flex items-center gap-1 flex-shrink-0">
                            <button
                              onClick={() => copySuggestion(suggestion.id, suggestion.description)}
                              className="p-1.5 hover:bg-slate-800 rounded text-slate-400 hover:text-emerald-400 transition-colors"
                              title="Copy suggestion"
                            >
                              {copiedSuggestionId === suggestion.id ? (
                                <Check className="w-3.5 h-3.5 text-emerald-400" />
                              ) : (
                                <Copy className="w-3.5 h-3.5" />
                              )}
                            </button>
                            <button
                              onClick={() => context?.dismissSuggestion(suggestion.id)}
                              className="p-1.5 hover:bg-slate-800 rounded text-slate-400 hover:text-red-400 transition-colors"
                              title="Dismiss"
                            >
                              <X className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 mt-2 pt-2 border-t border-slate-800/50">
                          <button
                            onClick={() => useSuggestionInChat(suggestion)}
                            disabled={loading}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-amber-500/20 to-orange-500/20 hover:from-amber-500/30 hover:to-orange-500/30 border border-amber-500/40 rounded-md text-xs text-amber-200 hover:text-white transition-all disabled:opacity-50"
                          >
                            <Wand2 className="w-3 h-3" />
                            <span>Apply & Regenerate</span>
                          </button>
                          <button
                            onClick={() => copySuggestion(suggestion.id, suggestion.description)}
                            className="flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-800/50 hover:bg-slate-800 rounded-md text-xs text-slate-300 hover:text-white transition-colors"
                          >
                            {copiedSuggestionId === suggestion.id ? (
                              <>
                                <Check className="w-3 h-3 text-emerald-400" />
                                <span className="text-emerald-400">Copied!</span>
                              </>
                            ) : (
                              <>
                                <Copy className="w-3 h-3" />
                                <span>Copy</span>
                              </>
                            )}
                          </button>
                          <span className="text-xs text-slate-500 ml-auto">
                            {new Date(suggestion.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="flex items-center gap-2 mt-3">
                    <button
                      onClick={() => context?.clearSuggestions()}
                      className="text-xs text-slate-400 hover:text-white transition-colors"
                    >
                      Dismiss all
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Load from Previous Sessions */}
      {previousSessions.length > 0 && (
        <div className="bg-[#0d1117] border border-slate-800 rounded-xl overflow-hidden flex-shrink-0">
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-800/30 transition-colors"
          >
            <div className="flex items-center gap-2">
              <History className="w-4 h-4 text-amber-400" />
              <span className="text-sm font-medium text-white">Load Code from Previous Sessions</span>
              <span className="text-xs text-slate-500">({previousSessions.length} available)</span>
            </div>
            {showHistory ? (
              <ChevronUp className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            )}
          </button>
          
          <AnimatePresence>
            {showHistory && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="p-3 border-t border-slate-800 space-y-2 max-h-[180px] overflow-y-auto">
                  {previousSessions.map((session) => (
                    <div
                      key={session.id}
                      className="group p-3 bg-slate-900/50 hover:bg-slate-800/50 border border-slate-800 rounded-lg cursor-pointer transition-all"
                      onClick={() => handleLoadPreviousCode(session.mlCode)}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <FileCode className="w-4 h-4 text-violet-400" />
                          <span className="text-sm font-medium text-white">{session.name}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-slate-500">{formatDate(session.updatedAt)}</span>
                          <span className="text-xs text-emerald-400 opacity-0 group-hover:opacity-100 transition-opacity">
                            Click to load →
                          </span>
                        </div>
                      </div>
                      <pre className="text-xs text-slate-400 font-mono truncate bg-slate-950/50 p-2 rounded">
                        {getCodePreview(session.mlCode, 120)}
                      </pre>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Main Editor and Chat */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
        {/* Left: Code Input */}
        <div className="flex flex-col min-h-0">
          <div className="mb-4 flex-shrink-0">
            <h2 className="text-2xl font-bold text-white flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg">
                <Code2 className="w-5 h-5 text-white" />
              </div>
              Share Your ML Code
            </h2>
            <p className="text-slate-400 mt-2 text-sm">
              Paste your model training/inference code. I'll generate a function to create the Global JSON.
            </p>
          </div>

          <div className="flex-1 flex flex-col bg-[#0d1117] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl min-h-0">
            <div className="flex items-center gap-2 px-4 py-3 bg-slate-900/50 border-b border-slate-800 flex-shrink-0">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
              </div>
              <span className="text-xs text-slate-500 font-mono ml-2">your_model.py</span>
              {userCode && (
                <span className="text-xs text-emerald-400 ml-auto flex items-center gap-1">
                  <Check className="w-3 h-3" />
                  {userCode.length} chars
                </span>
              )}
            </div>
            
            <textarea
              value={userCode}
              onChange={e => setUserCode(e.target.value)}
              placeholder={`# Paste your ML code here...\n\nimport sklearn\nfrom sklearn.ensemble import RandomForestClassifier\n\n# model = RandomForestClassifier(...)\n# model.fit(X_train, y_train)`}
              className="flex-1 bg-transparent p-4 text-sm font-mono text-emerald-300 resize-none outline-none placeholder-slate-600 min-h-0"
              spellCheck={false}
            />
            
            <div className="p-4 border-t border-slate-800 bg-slate-900/30 flex-shrink-0">
              <button
                onClick={handleAnalyzeCode}
                disabled={loading || !userCode.trim()}
                className="w-full bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all shadow-lg shadow-violet-500/20"
              >
                <Sparkles className="w-4 h-4" />
                {loading ? 'Analyzing...' : 'Generate explain_global() Function'}
              </button>
            </div>
          </div>
        </div>

        {/* Right: Chat */}
        <div className="flex flex-col bg-[#0d1117] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl min-h-0">
          <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-violet-900/30 to-purple-900/30 border-b border-slate-800 flex-shrink-0">
            <div className="flex items-center gap-2">
              <Bot className="w-4 h-4 text-violet-400" />
              <span className="text-sm font-medium text-violet-300">Code Assistant</span>
              {messages.length > 0 && (
                <span className="text-xs text-slate-500">({messages.length} messages)</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {messages.length > 0 && (
                <button
                  onClick={handleClearChat}
                  className="flex items-center gap-1 px-2 py-1 text-xs text-slate-400 hover:text-white hover:bg-slate-800 rounded transition-colors"
                  title="Clear chat"
                >
                  <RotateCcw className="w-3 h-3" />
                </button>
              )}
              {messages.length > 0 && userCode && (
                <button
                  onClick={onComplete}
                  className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 rounded-lg text-emerald-300 text-xs font-medium transition-all"
                >
                  Continue <ArrowRight className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center py-12">
                <div className="w-16 h-16 bg-violet-500/10 rounded-full flex items-center justify-center mb-4">
                  <Sparkles className="w-8 h-8 text-violet-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {context?.mlCode ? 'Code Loaded' : 'Ready to Analyze'}
                </h3>
                <p className="text-slate-400 text-sm max-w-xs">
                  {context?.mlCode 
                    ? 'Click "Generate" to create the explain_global() function.'
                    : 'Paste your code and I\'ll generate a custom function for your model.'
                  }
                </p>
              </div>
            )}
            
            {messages.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${m.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[90%] rounded-xl p-4 text-sm ${
                  m.type === 'user'
                    ? 'bg-violet-600/30 border border-violet-500/30 text-violet-100'
                    : 'bg-slate-800/50 border border-slate-700/50 text-slate-200'
                }`}>
                  {m.type === 'user' ? (
                    <div className="whitespace-pre-wrap text-sm">{m.content}</div>
                  ) : (
                    <div className="markdown-content text-sm">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          h1: ({ children }: { children?: ReactNode }) => <h1 className="text-xl font-bold text-white mt-4 mb-2">{children}</h1>,
                          h2: ({ children }: { children?: ReactNode }) => <h2 className="text-lg font-bold text-white mt-4 mb-2">{children}</h2>,
                          h3: ({ children }: { children?: ReactNode }) => <h3 className="text-base font-bold text-white mt-3 mb-1">{children}</h3>,
                          h4: ({ children }: { children?: ReactNode }) => <h4 className="text-sm font-bold text-white mt-2 mb-1">{children}</h4>,
                          p: ({ children }: { children?: ReactNode }) => <p className="mb-3 leading-relaxed">{children}</p>,
                          ul: ({ children }: { children?: ReactNode }) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
                          ol: ({ children }: { children?: ReactNode }) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
                          li: ({ children }: { children?: ReactNode }) => <li className="text-slate-300">{children}</li>,
                          code: ({ className, children, ...props }: { className?: string; children?: ReactNode }) => {
                            const match = /language-(\w+)/.exec(className || '');
                            const isInline = !match;
                            const codeString = String(children).replace(/\n$/, '');
                            
                            if (isInline) {
                              return (
                                <code className="bg-slate-900 text-violet-300 px-1.5 py-0.5 rounded text-xs font-mono" {...props}>
                                  {children}
                                </code>
                              );
                            }
                            
                            return (
                              <div className="relative group my-3">
                                <div className="flex items-center justify-between bg-slate-900 px-3 py-1.5 rounded-t-lg border-b border-slate-700">
                                  <span className="text-xs text-slate-500 font-mono">{match?.[1] || 'code'}</span>
                                  <button
                                    onClick={() => copyToClipboard(codeString)}
                                    className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors"
                                  >
                                    {copied ? (
                                      <>
                                        <Check className="w-3 h-3 text-emerald-400" />
                                        <span className="text-emerald-400">Copied</span>
                                      </>
                                    ) : (
                                      <>
                                        <Copy className="w-3 h-3" />
                                        <span>Copy</span>
                                      </>
                                    )}
                                  </button>
                                </div>
                                <pre className="bg-slate-950 p-3 rounded-b-lg overflow-x-auto max-h-[300px] overflow-y-auto">
                                  <code className="text-xs font-mono text-emerald-300" {...props}>
                                    {children}
                                  </code>
                                </pre>
                              </div>
                            );
                          },
                          blockquote: ({ children }: { children?: ReactNode }) => (
                            <blockquote className="border-l-4 border-violet-500 pl-4 my-3 text-slate-400 italic">
                              {children}
                            </blockquote>
                          ),
                          table: ({ children }: { children?: ReactNode }) => (
                            <div className="overflow-x-auto my-3">
                              <table className="min-w-full border border-slate-700 rounded-lg">{children}</table>
                            </div>
                          ),
                          thead: ({ children }: { children?: ReactNode }) => <thead className="bg-slate-800">{children}</thead>,
                          th: ({ children }: { children?: ReactNode }) => <th className="px-3 py-2 text-left text-xs font-semibold text-white border-b border-slate-700">{children}</th>,
                          td: ({ children }: { children?: ReactNode }) => <td className="px-3 py-2 text-xs text-slate-300 border-b border-slate-800">{children}</td>,
                          a: ({ href, children }: { href?: string; children?: ReactNode }) => (
                            <a href={href} target="_blank" rel="noopener noreferrer" className="text-violet-400 hover:text-violet-300 underline">
                              {children}
                            </a>
                          ),
                          strong: ({ children }: { children?: ReactNode }) => <strong className="font-bold text-white">{children}</strong>,
                          em: ({ children }: { children?: ReactNode }) => <em className="italic text-slate-300">{children}</em>,
                          hr: () => <hr className="border-slate-700 my-4" />,
                        }}
                      >
                        {m.content}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
            
            {loading && (
              <div className="flex items-center gap-2 text-slate-400 text-sm">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span>Generating function...</span>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {messages.length > 0 && (
            <div className="p-4 border-t border-slate-800 bg-slate-900/30 flex-shrink-0">
              <FollowUpInput onSend={handleFollowUp} loading={loading} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function FollowUpInput({ onSend, loading }: { onSend: (msg: string) => void; loading: boolean }) {
  const [input, setInput] = useState('');

  const handleSubmit = () => {
    if (input.trim() && !loading) {
      onSend(input);
      setInput('');
    }
  };

  return (
    <div className="flex gap-2">
      <input
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && handleSubmit()}
        placeholder="Ask a follow-up question..."
        className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-white text-sm focus:ring-2 focus:ring-violet-500 outline-none"
      />
      <button
        onClick={handleSubmit}
        disabled={loading || !input.trim()}
        className="bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white p-2.5 rounded-lg transition-colors"
      >
        <Send className="w-4 h-4" />
      </button>
    </div>
  );
}
