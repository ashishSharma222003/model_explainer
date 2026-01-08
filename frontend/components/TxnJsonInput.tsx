"use client";
import { useState, useRef, useEffect, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileJson, Check, AlertCircle, ArrowRight, Eye, Send, Sparkles, Bot, Copy, Code2, MessageSquare } from 'lucide-react';
import { guideTxnCode } from '../lib/api';
import { AppContext } from '@/app/page';

type TabType = 'generate' | 'paste';

export default function TxnJsonInput({ onComplete }: { onComplete: () => void }) {
  const context = useContext(AppContext);
  const [activeTab, setActiveTab] = useState<TabType>('generate');
  const [jsonInput, setJsonInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<any>(null);

  // Code generation state
  const [messages, setMessages] = useState<{ type: 'user' | 'ai'; content: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(`sess_txn_${Math.random().toString(36).substr(2, 9)}`);
  const [copied, setCopied] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleGenerateFunction = async () => {
    if (!context?.mlCode) {
      setError('No ML code found. Please go back to Step 1 and share your code.');
      return;
    }

    setLoading(true);
    setMessages(prev => [...prev, { 
      type: 'user', 
      content: 'Generate the explain_txn() function for my model'
    }]);

    try {
      // Simple user message - the backend now has the focused prompt
      const userMessage = `Please generate an explain_txn() function for my model.

MY CODE:
${context.mlCode}`;

      // Pass global JSON context separately to the backend
      const globalContext = context?.globalJson 
        ? JSON.stringify(context.globalJson, null, 2)
        : undefined;

      const res = await guideTxnCode(sessionId, userMessage, globalContext);
      setMessages(prev => [...prev, { type: 'ai', content: res.response }]);
    } catch (e) {
      setMessages(prev => [...prev, { type: 'ai', content: "Failed to generate function." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleChatSend = async () => {
    if (!chatInput.trim() || loading) return;

    const userMessage = chatInput;
    setChatInput('');
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setLoading(true);

    try {
      // Simple message - backend has the focused txn prompt
      let contextInfo = `User request about the explain_txn() function:\n\n${userMessage}`;
      
      if (context?.mlCode) {
        contextInfo += `\n\nML CODE:\n${context.mlCode}`;
      }

      // Pass global JSON context separately
      const globalContext = context?.globalJson 
        ? JSON.stringify(context.globalJson, null, 2)
        : undefined;

      const res = await guideTxnCode(sessionId, contextInfo, globalContext);
      setMessages(prev => [...prev, { type: 'ai', content: res.response }]);
    } catch (e) {
      setMessages(prev => [...prev, { type: 'ai', content: "Failed to process request. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  const validateAndParse = (input: string) => {
    try {
      const parsed = JSON.parse(input);
      
      if (typeof parsed !== 'object' || parsed === null) {
        return { valid: false, error: 'JSON must be an object', data: null };
      }
      
      const warnings: string[] = [];
      if (!parsed.prediction) warnings.push('Missing prediction object');
      if (!parsed.local_contributions) warnings.push('Missing local_contributions array');
      
      return { valid: true, error: null, data: parsed, warnings };
    } catch (e: any) {
      return { valid: false, error: `Invalid JSON: ${e.message}`, data: null };
    }
  };

  const handlePreview = () => {
    const result = validateAndParse(jsonInput);
    if (result.valid) {
      setPreview(result.data);
      setError(null);
    } else {
      setError(result.error);
      setPreview(null);
    }
  };

  const handleSubmit = () => {
    const result = validateAndParse(jsonInput);
    if (result.valid) {
      context?.setTxnJson(result.data);
      onComplete();
    } else {
      setError(result.error);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const extractCodeBlock = (content: string) => {
    const match = content.match(/```python\n([\s\S]*?)```/);
    return match ? match[1] : null;
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="max-w-5xl mx-auto">
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-amber-500 to-orange-600 rounded-2xl mb-4 shadow-lg shadow-amber-500/20">
          <FileJson className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-3xl font-bold text-white mb-2">Add Transaction JSON</h2>
        <p className="text-slate-400 max-w-md mx-auto">
          Generate a function to explain individual predictions, then paste the JSON output.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 justify-center">
        <button
          onClick={() => setActiveTab('generate')}
          className={`px-6 py-3 rounded-xl font-medium flex items-center gap-2 transition-all ${
            activeTab === 'generate'
              ? 'bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/40 text-amber-300'
              : 'bg-slate-800/30 border border-slate-700/50 text-slate-400 hover:text-white'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          Generate Function
        </button>
        <button
          onClick={() => setActiveTab('paste')}
          className={`px-6 py-3 rounded-xl font-medium flex items-center gap-2 transition-all ${
            activeTab === 'paste'
              ? 'bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/40 text-amber-300'
              : 'bg-slate-800/30 border border-slate-700/50 text-slate-400 hover:text-white'
          }`}
        >
          <FileJson className="w-4 h-4" />
          Paste JSON
        </button>
      </div>

      <AnimatePresence mode="wait">
        {activeTab === 'generate' ? (
          <motion.div
            key="generate"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="bg-[#0d1117] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl"
          >
            {/* Chat Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-amber-900/20 to-orange-900/20 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Bot className="w-4 h-4 text-amber-400" />
                <span className="text-sm font-medium text-amber-300">Transaction Function Generator</span>
              </div>
              <div className="flex items-center gap-2">
                {context?.globalJson && (
                  <span className="text-xs bg-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded-full">
                    ✓ Global JSON
                  </span>
                )}
                {context?.mlCode ? (
                  <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
                    ✓ ML Code
                  </span>
                ) : (
                  <span className="text-xs text-red-400">⚠️ No ML code</span>
                )}
              </div>
            </div>

            {/* Chat Messages */}
            <div className="h-[400px] overflow-y-auto p-4 space-y-4">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center py-12">
                  <div className="w-16 h-16 bg-amber-500/10 rounded-full flex items-center justify-center mb-4">
                    <Code2 className="w-8 h-8 text-amber-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">Generate explain_txn()</h3>
                  <p className="text-slate-400 text-sm max-w-xs mb-6">
                    I'll analyze your ML code and generate a function to explain individual predictions.
                  </p>
                  <button
                    onClick={handleGenerateFunction}
                    disabled={loading || !context?.mlCode}
                    className="bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 disabled:opacity-50 text-white px-6 py-3 rounded-xl font-medium flex items-center gap-2 transition-all shadow-lg shadow-amber-500/20"
                  >
                    <Sparkles className="w-4 h-4" />
                    Generate Function
                  </button>
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
                      ? 'bg-amber-600/30 border border-amber-500/30 text-amber-100'
                      : 'bg-slate-800/50 border border-slate-700/50 text-slate-200'
                  }`}>
                    {m.type === 'ai' && extractCodeBlock(m.content) && (
                      <div className="mb-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-emerald-400 font-mono">Generated Function</span>
                          <button
                            onClick={() => copyToClipboard(extractCodeBlock(m.content) || '')}
                            className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors"
                          >
                            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                            {copied ? 'Copied!' : 'Copy'}
                          </button>
                        </div>
                        <pre className="bg-slate-950 p-3 rounded-lg overflow-x-auto text-xs text-emerald-300 font-mono">
                          {extractCodeBlock(m.content)}
                        </pre>
                      </div>
                    )}
                    <div className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
                      {m.type === 'ai' ? m.content.replace(/```python\n[\s\S]*?```/g, '[Code shown above]') : m.content}
                    </div>
                  </div>
                </motion.div>
              ))}

              {loading && (
                <div className="flex items-center gap-2 text-slate-400 text-sm">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <span>Generating...</span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input & Actions Footer */}
            {messages.length > 0 && (
              <div className="border-t border-slate-800 bg-slate-900/30">
                {/* Chat Input */}
                <div className="p-3 border-b border-slate-800/50">
                  <div className="flex gap-2">
                    <input
                      ref={inputRef}
                      type="text"
                      value={chatInput}
                      onChange={e => setChatInput(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleChatSend()}
                      placeholder="Ask to refine the function... (e.g., 'add more SHAP features', 'use different explainer')"
                      className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-white placeholder-slate-500 outline-none focus:border-amber-500/50 transition-colors"
                      disabled={loading}
                    />
                    <button
                      onClick={handleChatSend}
                      disabled={loading || !chatInput.trim()}
                      className="px-4 py-2.5 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 disabled:opacity-50 rounded-lg text-white transition-all"
                    >
                      <Send className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="flex gap-2 mt-2">
                    <button
                      onClick={() => setChatInput('Add more local contribution features')}
                      className="text-xs px-2 py-1 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded text-slate-400 hover:text-white transition-colors"
                    >
                      More features
                    </button>
                    <button
                      onClick={() => setChatInput('Add confidence intervals to the prediction')}
                      className="text-xs px-2 py-1 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded text-slate-400 hover:text-white transition-colors"
                    >
                      Add confidence
                    </button>
                    <button
                      onClick={() => setChatInput('Make the narrative more detailed for compliance')}
                      className="text-xs px-2 py-1 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded text-slate-400 hover:text-white transition-colors"
                    >
                      Better narrative
                    </button>
                  </div>
                </div>
                
                {/* Action Buttons */}
                <div className="p-3 flex justify-between items-center">
                  <button
                    onClick={handleGenerateFunction}
                    disabled={loading}
                    className="text-sm text-slate-400 hover:text-white transition-colors"
                  >
                    Start Over
                  </button>
                  <button
                    onClick={() => setActiveTab('paste')}
                    className="flex items-center gap-2 px-4 py-2 bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 rounded-lg text-amber-300 text-sm font-medium transition-all"
                  >
                    Continue to Paste JSON <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="paste"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="bg-[#0d1117] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl"
          >
            {/* Editor Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-slate-900/50 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-500/80" />
                  <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                  <div className="w-3 h-3 rounded-full bg-green-500/80" />
                </div>
                <span className="text-xs text-slate-500 font-mono ml-2">transaction_explanation.json</span>
              </div>
            </div>

            {/* JSON Input */}
            <textarea
              value={jsonInput}
              onChange={e => {
                setJsonInput(e.target.value);
                setError(null);
              }}
              placeholder={`{
  "txn_id": "txn_12345",
  "model_version": "fraud_detector_v2",
  "generated_at": "2025-01-02T10:30:00Z",
  "prediction": {
    "score": 0.87,
    "threshold": 0.5,
    "label": "FRAUD"
  },
  "local_contributions": [
    { "feature": "amount", "value": 5000, "contribution": 0.35, "direction": "positive" },
    { "feature": "merchant_category", "value": "electronics", "contribution": 0.22, "direction": "positive" }
  ],
  "narrative_plain": ["High transaction amount from unusual merchant triggered fraud flag"]
}`}
              className="w-full h-[350px] bg-transparent p-4 text-sm font-mono text-amber-300 resize-none outline-none placeholder-slate-600"
              spellCheck={false}
            />

            {/* Error */}
            {error && (
              <div className="mx-4 mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-300 text-sm">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                {error}
              </div>
            )}

            {/* Preview */}
            {preview && (
              <div className="border-t border-slate-800 bg-slate-900/30 p-4">
                <h4 className="text-sm font-medium text-slate-300 mb-3">Parsed Preview</h4>
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-xs text-slate-500">Transaction ID</div>
                    <div className="text-sm font-medium text-white">{preview.txn_id || 'N/A'}</div>
                  </div>
                  <div className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-xs text-slate-500">Prediction</div>
                    <div className={`text-sm font-medium ${preview.prediction?.label === 'FRAUD' ? 'text-red-400' : 'text-emerald-400'}`}>
                      {preview.prediction?.label || 'N/A'} ({preview.prediction?.score?.toFixed(2) || 'N/A'})
                    </div>
                  </div>
                  <div className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-xs text-slate-500">Contributing Features</div>
                    <div className="text-sm font-medium text-white">{preview.local_contributions?.length || 0}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="p-4 border-t border-slate-800 bg-slate-900/30 flex gap-3">
              <button
                onClick={handlePreview}
                disabled={!jsonInput.trim()}
                className="flex-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all border border-slate-700"
              >
                <Eye className="w-4 h-4" />
                Preview
              </button>
              <button
                onClick={handleSubmit}
                disabled={!jsonInput.trim()}
                className="flex-1 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 disabled:opacity-50 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all shadow-lg shadow-amber-500/20"
              >
                <Check className="w-4 h-4" />
                Save & Analyze
                <ArrowRight className="w-4 h-4 ml-1" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

