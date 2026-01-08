"use client";
import { useState, useRef, useEffect, useContext } from 'react';
import { motion } from 'framer-motion';
import { Send, MessageSquare, Sparkles, Plus, Code2, FileJson, Layers, Copy, Check, ArrowRight, Lightbulb, X, Info } from 'lucide-react';
import { chat, ChatApiResponse } from '../lib/api';
import { AppContext, ChatMessage } from '@/app/page';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatPanelProps {
  mode: 'global' | 'txn';
  onAddTxn?: () => void;
  kernelOutput?: string | null;
  onKernelOutputUsed?: () => void;
}


export default function ChatPanel({ mode, onAddTxn, kernelOutput, onKernelOutputUsed }: ChatPanelProps) {
  const context = useContext(AppContext);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(`sess_${mode}_${context?.session.id || Math.random().toString(36).substr(2, 9)}`);
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  const [showSuggestionBanner, setShowSuggestionBanner] = useState(false);
  const [currentSuggestion, setCurrentSuggestion] = useState<string | null>(null);
  const [showKernelOutputBanner, setShowKernelOutputBanner] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  
  // Handle incoming kernel output
  useEffect(() => {
    if (kernelOutput) {
      setShowKernelOutputBanner(true);
    }
  }, [kernelOutput]);

  // Use messages from context based on mode
  const messages = mode === 'global' 
    ? (context?.globalChatMessages || [])
    : (context?.txnChatMessages || []);
  
  const setMessages = (newMessages: ChatMessage[]) => {
    if (mode === 'global') {
      context?.setGlobalChatMessages(newMessages);
    } else {
      context?.setTxnChatMessages(newMessages);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    
    const userMessage = input;
    setInput('');
    
    // Create context snapshot for this message
    const contextSnapshot = context?.createContextSnapshot();
    
    const newMessages: ChatMessage[] = [...messages, { 
      type: 'user', 
      content: userMessage,
      context: contextSnapshot 
    }];
    setMessages(newMessages);
    setLoading(true);

    try {
      const res: ChatApiResponse = await chat({
        session_id: sessionId,
        message: userMessage,
        context: {
          ml_code: context?.mlCode,
          global: context?.globalJson,
          txn: mode === 'txn' ? context?.txnJson : null,
        }
      });
      
      const aiResponse = res.response;
      setMessages([...newMessages, { 
        type: 'ai', 
        content: aiResponse,
        context: contextSnapshot // AI response uses same context as the question
      }]);
      
      // Check if AI provided a suggestion for improving global JSON function
      if (res.global_json_suggestion) {
        setCurrentSuggestion(res.global_json_suggestion);
        setShowSuggestionBanner(true);
        
        // Also add to context suggestions
        context?.addSuggestion({
          type: 'update_explain_global',
          title: 'Update explain_global() function',
          description: res.global_json_suggestion,
          fromStep: mode === 'global' ? 'global-chat' : 'txn-chat',
        });
      }
    } catch (e) {
      setMessages([...newMessages, { 
        type: 'ai', 
        content: "Error: Could not reach assistant.",
        context: contextSnapshot
      }]);
    } finally {
      setLoading(false);
    }
  };

  const copyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const handleGoToCode = () => {
    setShowSuggestionBanner(false);
    context?.goToStep('code');
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const suggestedQuestions = mode === 'global' ? [
    "What are the most important features in this model?",
    "Are there any potential issues with this model?",
    "Explain the feature importance rankings",
    "What are the model's limitations?",
  ] : [
    "Why did the model make this prediction?",
    "Which features contributed most to this decision?",
    "Is this prediction reliable?",
    "What would change the prediction?",
  ];

  return (
    <div className="flex gap-6 h-[calc(100vh-220px)]">
      {/* Context Sidebar */}
      <div className="w-72 flex-shrink-0 space-y-4 overflow-y-auto">
        <div className="bg-[#0d1117] border border-slate-800 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            Active Context
          </h3>
          
          <div className="space-y-3">
            <ContextItem
              icon={<Code2 className="w-4 h-4" />}
              label="ML Code"
              status={context?.mlCode ? 'active' : 'missing'}
              detail={context?.mlCode ? `${context.mlCode.length} chars` : 'Not provided'}
              color="violet"
            />
            
            <ContextItem
              icon={<FileJson className="w-4 h-4" />}
              label="Global JSON"
              status={context?.globalJson ? 'active' : 'missing'}
              detail={context?.globalJson?.model_version || 'Not loaded'}
              color="cyan"
            />
            
            <ContextItem
              icon={<FileJson className="w-4 h-4" />}
              label="Transaction JSON"
              status={context?.txnJson ? 'active' : (mode === 'txn' ? 'required' : 'optional')}
              detail={context?.txnJson?.txn_id || (mode === 'txn' ? 'Required' : 'Optional')}
              color="amber"
            />
          </div>
        </div>

        {mode === 'global' && onAddTxn && (
          <button
            onClick={onAddTxn}
            className="w-full p-4 bg-gradient-to-r from-amber-500/10 to-orange-500/10 hover:from-amber-500/20 hover:to-orange-500/20 border border-amber-500/30 rounded-xl text-amber-300 text-sm font-medium flex items-center justify-center gap-2 transition-all"
          >
            <Plus className="w-4 h-4" />
            Add Transaction JSON
          </button>
        )}

        <div className="bg-[#0d1117] border border-slate-800 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-violet-400" />
            Suggested Questions
          </h3>
          <div className="space-y-2">
            {suggestedQuestions.map((q, i) => (
              <button
                key={i}
                onClick={() => setInput(q)}
                className="w-full text-left text-xs text-slate-400 hover:text-white hover:bg-slate-800/50 p-2 rounded-lg transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        {messages.length > 0 && (
          <div className="bg-[#0d1117] border border-slate-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-white mb-2">Chat Stats</h3>
            <div className="text-xs text-slate-400 space-y-1">
              <div>Messages: {messages.length}</div>
              <div>Your questions: {messages.filter(m => m.type === 'user').length}</div>
            </div>
          </div>
        )}
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-[#0d1117] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl min-h-0">
        {/* Header */}
        <div className="flex-shrink-0 flex items-center justify-between px-6 py-4 bg-gradient-to-r from-cyan-900/20 to-blue-900/20 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/20 rounded-lg">
              <MessageSquare className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">
                {mode === 'global' ? 'Explore Global Model Behavior' : 'Analyze Transaction Prediction'}
              </h2>
              <p className="text-xs text-slate-500">
                {mode === 'global' 
                  ? 'Ask about feature importance, model behavior, and potential issues'
                  : 'Understand why the model made this specific prediction'}
              </p>
            </div>
          </div>
          {messages.length > 0 && (
            <span className="text-xs text-slate-500">{messages.length} messages</span>
          )}
        </div>

        {/* Suggestion Banner */}
        {showSuggestionBanner && currentSuggestion && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex-shrink-0 mx-4 mt-4 p-4 bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/40 rounded-xl"
          >
            <div className="flex items-start gap-3">
              <div className="p-2 bg-amber-500/20 rounded-lg">
                <Lightbulb className="w-5 h-5 text-amber-400" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-semibold text-amber-300 mb-1">
                  Suggestion: Update explain_global() function
                </h4>
                <p className="text-xs text-amber-200/80 mb-3 line-clamp-2">
                  {currentSuggestion}
                </p>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleGoToCode}
                    className="flex items-center gap-2 px-3 py-1.5 bg-amber-500/30 hover:bg-amber-500/40 border border-amber-500/50 rounded-lg text-xs font-medium text-amber-200 transition-all"
                  >
                    <ArrowRight className="w-3.5 h-3.5" />
                    Go to Share Code
                  </button>
                  <button
                    onClick={() => setShowSuggestionBanner(false)}
                    className="p-1.5 hover:bg-slate-800/50 rounded-lg text-slate-400 hover:text-white transition-colors"
                    title="Dismiss"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Kernel Output Banner */}
        {showKernelOutputBanner && kernelOutput && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex-shrink-0 mx-4 mt-4 p-4 bg-gradient-to-r from-emerald-500/20 to-green-500/20 border border-emerald-500/40 rounded-xl"
          >
            <div className="flex items-start gap-3">
              <div className="p-2 bg-emerald-500/20 rounded-lg">
                <Code2 className="w-5 h-5 text-emerald-400" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-semibold text-emerald-300 mb-1">
                  Kernel Output Ready
                </h4>
                <div className="bg-slate-950 rounded-lg p-2 mb-3 max-h-24 overflow-y-auto">
                  <pre className="text-xs text-emerald-200/80 font-mono whitespace-pre-wrap">
                    {kernelOutput.length > 300 ? kernelOutput.slice(0, 300) + '...' : kernelOutput}
                  </pre>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      setInput(prev => {
                        const prefix = prev ? prev + '\n\n' : '';
                        return `${prefix}Here's the output from my Python kernel:\n\`\`\`\n${kernelOutput}\n\`\`\`\n\nCan you help me understand this?`;
                      });
                      setShowKernelOutputBanner(false);
                      onKernelOutputUsed?.();
                    }}
                    className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/30 hover:bg-emerald-500/40 border border-emerald-500/50 rounded-lg text-xs font-medium text-emerald-200 transition-all"
                  >
                    <Send className="w-3.5 h-3.5" />
                    Add to Message
                  </button>
                  <button
                    onClick={() => {
                      setShowKernelOutputBanner(false);
                      onKernelOutputUsed?.();
                    }}
                    className="p-1.5 hover:bg-slate-800/50 rounded-lg text-slate-400 hover:text-white transition-colors"
                    title="Dismiss"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Messages - this is the scrollable area */}
        <div 
          ref={messagesContainerRef}
          className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0"
        >
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <div className="w-20 h-20 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-full flex items-center justify-center mb-4">
                <MessageSquare className="w-10 h-10 text-cyan-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                {mode === 'global' ? 'Ready to Explore' : 'Analyze This Prediction'}
              </h3>
              <p className="text-slate-400 text-sm max-w-md mb-6">
                {mode === 'global'
                  ? "I have your ML code and Global JSON loaded. Ask me anything about how your model works, feature importance, or potential issues."
                  : "I can explain why this prediction was made, which features contributed most, and whether it's reliable."}
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {suggestedQuestions.slice(0, 2).map((q, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(q)}
                    className="px-4 py-2 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-300 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => {
            // Check if context changed from previous user message
            const prevUserMsg = messages.slice(0, i).reverse().find(msg => msg.type === 'user');
            const contextChanged = m.type === 'user' && prevUserMsg?.context && m.context && (
              prevUserMsg.context.globalJsonVersion !== m.context.globalJsonVersion ||
              prevUserMsg.context.txnId !== m.context.txnId
            );
            
            return (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex flex-col ${m.type === 'user' ? 'items-end' : 'items-start'}`}
            >
              {/* Context change indicator */}
              {contextChanged && (
                <div className="flex items-center gap-2 mb-2 px-3 py-1.5 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                  <Info className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-xs text-amber-300">
                    Context updated: {m.context?.globalJsonVersion !== prevUserMsg?.context?.globalJsonVersion && (
                      <span>Global JSON → v{m.context?.globalJsonVersion || 'new'}</span>
                    )}
                    {m.context?.txnId !== prevUserMsg?.context?.txnId && m.context?.txnId && (
                      <span> Txn: {m.context.txnId}</span>
                    )}
                  </span>
                </div>
              )}
              
              <div className={`max-w-[80%] rounded-2xl p-4 ${
                m.type === 'user'
                  ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white'
                  : 'bg-slate-800/50 border border-slate-700 text-slate-200'
              }`}>
                {/* Context badge for user messages */}
                {m.type === 'user' && m.context && (
                  <div className="flex items-center gap-2 mb-2 pb-2 border-b border-white/20">
                    <div className="flex items-center gap-1.5 text-xs text-cyan-100/80">
                      {m.context.globalJsonVersion && (
                        <span className="flex items-center gap-1 bg-white/10 px-1.5 py-0.5 rounded">
                          <FileJson className="w-3 h-3" />
                          v{m.context.globalJsonVersion}
                        </span>
                      )}
                      {m.context.txnId && (
                        <span className="flex items-center gap-1 bg-white/10 px-1.5 py-0.5 rounded">
                          <Layers className="w-3 h-3" />
                          {m.context.txnId}
                        </span>
                      )}
                      {m.context.hasCode && (
                        <span className="flex items-center gap-1 bg-white/10 px-1.5 py-0.5 rounded">
                          <Code2 className="w-3 h-3" />
                          {m.context.codeLength} chars
                        </span>
                      )}
                    </div>
                  </div>
                )}
                
                {m.type === 'user' ? (
                  <div className="text-sm whitespace-pre-wrap">{m.content}</div>
                ) : (
                  <div className="markdown-content text-sm">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        // Headings
                        h1: ({ children }) => <h1 className="text-xl font-bold text-white mt-4 mb-2">{children}</h1>,
                        h2: ({ children }) => <h2 className="text-lg font-bold text-white mt-4 mb-2">{children}</h2>,
                        h3: ({ children }) => <h3 className="text-base font-bold text-white mt-3 mb-1">{children}</h3>,
                        h4: ({ children }) => <h4 className="text-sm font-bold text-white mt-2 mb-1">{children}</h4>,
                        
                        // Paragraphs
                        p: ({ children }) => <p className="mb-3 leading-relaxed">{children}</p>,
                        
                        // Lists
                        ul: ({ children }) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
                        li: ({ children }) => <li className="text-slate-300">{children}</li>,
                        
                        // Code blocks
                        code: ({ className, children, ...props }) => {
                          const match = /language-(\w+)/.exec(className || '');
                          const isInline = !match;
                          const codeString = String(children).replace(/\n$/, '');
                          
                          if (isInline) {
                            return (
                              <code className="bg-slate-900 text-cyan-300 px-1.5 py-0.5 rounded text-xs font-mono" {...props}>
                                {children}
                              </code>
                            );
                          }
                          
                          return (
                            <div className="relative group my-3">
                              <div className="flex items-center justify-between bg-slate-900 px-3 py-1.5 rounded-t-lg border-b border-slate-700">
                                <span className="text-xs text-slate-500 font-mono">{match?.[1] || 'code'}</span>
                                <button
                                  onClick={() => copyCode(codeString)}
                                  className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors"
                                >
                                  {copiedCode === codeString ? (
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
                              <pre className="bg-slate-950 p-3 rounded-b-lg overflow-x-auto">
                                <code className="text-xs font-mono text-emerald-300" {...props}>
                                  {children}
                                </code>
                              </pre>
                            </div>
                          );
                        },
                        
                        // Blockquotes
                        blockquote: ({ children }) => (
                          <blockquote className="border-l-4 border-cyan-500 pl-4 my-3 text-slate-400 italic">
                            {children}
                          </blockquote>
                        ),
                        
                        // Tables
                        table: ({ children }) => (
                          <div className="overflow-x-auto my-3">
                            <table className="min-w-full border border-slate-700 rounded-lg">{children}</table>
                          </div>
                        ),
                        thead: ({ children }) => <thead className="bg-slate-800">{children}</thead>,
                        th: ({ children }) => <th className="px-3 py-2 text-left text-xs font-semibold text-white border-b border-slate-700">{children}</th>,
                        td: ({ children }) => <td className="px-3 py-2 text-xs text-slate-300 border-b border-slate-800">{children}</td>,
                        
                        // Links
                        a: ({ href, children }) => (
                          <a href={href} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 underline">
                            {children}
                          </a>
                        ),
                        
                        // Strong/Bold
                        strong: ({ children }) => <strong className="font-bold text-white">{children}</strong>,
                        
                        // Emphasis/Italic
                        em: ({ children }) => <em className="italic text-slate-300">{children}</em>,
                        
                        // Horizontal rule
                        hr: () => <hr className="border-slate-700 my-4" />,
                      }}
                    >
                      {m.content}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            </motion.div>
          );
          })}

          {loading && (
            <div className="flex items-center gap-3 text-slate-400">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
              <span className="text-sm">Thinking...</span>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <div className="flex-shrink-0 p-4 border-t border-slate-800 bg-slate-900/30">
          <div className="flex gap-3">
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder={mode === 'global' ? "Ask about your model..." : "Ask about this prediction..."}
              className="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-5 py-3 text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none text-sm"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white px-6 rounded-xl transition-all flex items-center gap-2 font-medium shadow-lg shadow-cyan-500/20"
            >
              <Send className="w-4 h-4" />
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ContextItem({ 
  icon, 
  label, 
  status, 
  detail, 
  color 
}: { 
  icon: React.ReactNode;
  label: string;
  status: 'active' | 'missing' | 'required' | 'optional';
  detail: string;
  color: string;
}) {
  const statusColors = {
    active: 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400',
    missing: 'bg-slate-800/50 border-slate-700 text-slate-500',
    required: 'bg-red-500/10 border-red-500/30 text-red-400',
    optional: 'bg-slate-800/30 border-slate-700/50 text-slate-600',
  };

  const iconColors: Record<string, string> = {
    violet: 'text-violet-400',
    cyan: 'text-cyan-400',
    amber: 'text-amber-400',
  };

  return (
    <div className={`p-3 rounded-lg border ${statusColors[status]}`}>
      <div className="flex items-center gap-2 mb-1">
        <span className={iconColors[color]}>{icon}</span>
        <span className="text-xs font-medium">{label}</span>
      </div>
      <div className="text-xs truncate opacity-80">{detail}</div>
    </div>
  );
}
