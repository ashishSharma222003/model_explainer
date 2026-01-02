"use client";
import { useState, useEffect, useRef } from 'react';
import { Send, Play, RefreshCw, MessageSquare, AlertCircle, FileJson } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { chat, guideCode } from '../lib/api';

export default function TransactionExplainer({
    modelSpecs,
    globalContext,
    complianceMode = false
}: {
    modelSpecs: any,
    globalContext: any,
    complianceMode?: boolean
}) {
    const [txnData, setTxnData] = useState<any>(null);
    const [txnJsonInput, setTxnJsonInput] = useState("");
    const [chatOpen, setChatOpen] = useState(false);

    // Chat state
    const [messages, setMessages] = useState<{ type: 'user' | 'ai', content: string }[]>([]);
    const [input, setInput] = useState("");
    const [sessionId] = useState(`sess_${Math.random().toString(36).substr(2, 9)}`);
    const [chatLoading, setChatLoading] = useState(false);
    const chatEndRef = useRef<HTMLDivElement>(null);

    // Tools state
    const [codeMode, setCodeMode] = useState(false);
    const [userCode, setUserCode] = useState("");

    const handleVisualizeTxn = () => {
        try {
            const parsed = JSON.parse(txnJsonInput);
            if (!parsed.prediction || !parsed.local_contributions) {
                alert("Invalid Transaction JSON. Must contain 'prediction' and 'local_contributions'.");
                return;
            }
            setTxnData(parsed);
        } catch (e) {
            alert("Invalid JSON format");
        }
    };

    const handleSendChat = async () => {
        if (!input.trim()) return;
        const msg = input;
        setInput("");
        setMessages(prev => [...prev, { type: 'user', content: msg }]);
        setChatLoading(true);

        try {
            const res = await chat({
                session_id: sessionId,
                message: msg,
                context: { txn: txnData, global: globalContext }
            });
            setMessages(prev => [...prev, { type: 'ai', content: res.response }]);
        } catch (e) {
            setMessages(prev => [...prev, { type: 'ai', content: "Error: Could not reach assistant." }]);
        } finally {
            setChatLoading(false);
        }
    };

    const handleCodeGuide = async () => {
        if (!userCode.trim()) return;
        const code = userCode;
        setMessages(prev => [...prev, { type: 'user', content: "Analyze my code: " + code.substring(0, 30) + "..." }]);
        setUserCode("");
        setChatLoading(true);
        setChatOpen(true);
        try {
            const res = await guideCode(sessionId, code);
            setMessages(prev => [...prev, { type: 'ai', content: res.response }]);
        } catch (e) {
            setMessages(prev => [...prev, { type: 'ai', content: "Error analyzing code." }]);
        } finally {
            setChatLoading(false);
        }
    }

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[600px]">
            {/* Inputs / Config Pane */}
            <div className="lg:col-span-1 bg-slate-900/50 backdrop-blur-md border border-slate-700 rounded-xl p-6 shadow-xl flex flex-col">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="font-semibold text-slate-100">Transaction Input</h3>
                    <button onClick={() => setCodeMode(!codeMode)} className="text-xs text-blue-400 hover:text-blue-300">
                        {codeMode ? "Paste Transaction JSON" : "Get Code Guide"}
                    </button>
                </div>

                {codeMode ? (
                    <div className="flex-1 flex flex-col gap-2">
                        <textarea
                            value={userCode}
                            onChange={e => setUserCode(e.target.value)}
                            placeholder="Paste your model code here..."
                            className="flex-1 bg-slate-950/50 border border-slate-700 rounded-lg p-3 text-xs font-mono text-slate-300 resize-none outline-none focus:border-blue-500"
                        />
                        <button
                            onClick={handleCodeGuide}
                            disabled={chatLoading}
                            className="bg-purple-600 hover:bg-purple-500 text-white py-2 rounded-lg text-sm font-medium transition-colors"
                        >
                            Get Implementation Guide
                        </button>
                    </div>
                ) : (
                    <div className="flex-1 flex flex-col gap-2">
                        <label className="text-xs text-slate-400 uppercase font-bold flex items-center gap-2">
                            <FileJson className="w-3 h-3" /> Transaction JSON
                        </label>
                        <textarea
                            value={txnJsonInput}
                            onChange={e => setTxnJsonInput(e.target.value)}
                            placeholder='{ "prediction": {...}, "local_contributions": [...] }'
                            className="flex-1 bg-slate-950/50 border border-slate-700 rounded-lg p-3 text-xs font-mono text-emerald-300 resize-none outline-none focus:border-emerald-500"
                        />
                        <button
                            onClick={handleVisualizeTxn}
                            className="bg-emerald-600 hover:bg-emerald-500 text-white py-2 rounded-lg text-sm font-medium transition-colors flex justify-center items-center gap-2"
                        >
                            <Play className="w-4 h-4 fill-current" /> VISUALIZE
                        </button>
                    </div>
                )}
            </div>

            {/* Main Visualization Pane */}
            <div className="lg:col-span-2 flex flex-col gap-6">
                {txnData ? (
                    <div className="flex-1 bg-slate-900/50 backdrop-blur-md border border-slate-700 rounded-xl p-6 shadow-xl relative overflow-hidden">

                        {/* Header Result */}
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <h2 className="text-3xl font-bold text-white mb-1">
                                    {txnData.prediction.label.toUpperCase()}
                                </h2>
                                <div className="flex items-center gap-2">
                                    <span className="text-slate-400 text-sm">Score:</span>
                                    <div className="h-2 w-24 bg-slate-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-white" style={{ width: `${txnData.prediction.score * 100}%` }}></div>
                                    </div>
                                    <span className="text-white font-mono">{txnData.prediction.score}</span>
                                </div>
                            </div>
                            <button
                                onClick={() => setChatOpen(!chatOpen)}
                                className="bg-blue-600/20 hover:bg-blue-600/40 text-blue-300 p-2 rounded-lg transition-colors"
                            >
                                <MessageSquare className="w-6 h-6" />
                            </button>
                        </div>

                        {/* Contributions */}
                        <div className="space-y-2 mb-6">
                            <h4 className="text-sm text-slate-400 font-semibold uppercase tracking-wider mb-2">Key Drivers</h4>
                            {txnData.local_contributions.slice(0, 5).map((c: any, i: number) => (
                                <div key={i} className="flex items-center gap-3 group">
                                    <div className="w-24 text-right text-sm text-slate-300 truncate">{c.feature}</div>
                                    <div className="flex-1 h-8 bg-slate-800/50 rounded flex items-center px-2 relative">
                                        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-600"></div>
                                        <motion.div
                                            initial={{ scaleX: 0 }}
                                            animate={{ scaleX: 1 }}
                                            className={`h-4 rounded ${c.direction === 'positive' ? 'bg-emerald-500' : 'bg-rose-500'}`}
                                            style={{
                                                width: `${Math.min(Math.abs(c.contribution) * 100 * 3, 50)}%`, // Scale for visual
                                                marginLeft: c.direction === 'positive' ? '50%' : 'auto',
                                                marginRight: c.direction === 'positive' ? 'auto' : '50%',
                                                transformOrigin: c.direction === 'positive' ? 'left' : 'right'
                                            }}
                                        />
                                    </div>
                                    <div className={`w-12 text-sm font-mono ${c.direction === 'positive' ? 'text-emerald-400' : 'text-rose-400'}`}>
                                        {c.contribution > 0 ? '+' : ''}{c.contribution}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Narrative */}
                        <div className={`p-4 rounded-lg border ${complianceMode ? 'bg-indigo-900/20 border-indigo-500/30' : 'bg-slate-800/40 border-slate-700/50'}`}>
                            {complianceMode ? (
                                <div>
                                    <div className="text-xs uppercase text-indigo-400 font-bold mb-1 tracking-wider">Compliance Report</div>
                                    <ul className="text-indigo-200 text-sm space-y-1 list-disc pl-4">
                                        {txnData.narrative_compliance && txnData.narrative_compliance.length > 0
                                            ? txnData.narrative_compliance.map((line: string, i: number) => <li key={i}>{line}</li>)
                                            : <li>No compliance notes available.</li>
                                        }
                                    </ul>
                                </div>
                            ) : (
                                <p className="text-slate-200 text-sm italic">"{txnData.narrative_plain?.[0] || 'No narrative available'}"</p>
                            )}
                        </div>

                        {/* Chat Overlay */}
                        <AnimatePresence>
                            {chatOpen && (
                                <motion.div
                                    initial={{ x: '100%', opacity: 0 }}
                                    animate={{ x: 0, opacity: 1 }}
                                    exit={{ x: '100%', opacity: 0 }}
                                    className="absolute inset-0 bg-slate-900 z-10 flex flex-col"
                                >
                                    <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-900">
                                        <h3 className="font-semibold text-white">Analysis Assistant</h3>
                                        <button onClick={() => setChatOpen(false)} className="text-slate-400 hover:text-white">✕</button>
                                    </div>
                                    <div className="flex-1 overflow-y-auto p-4 space-y-4">
                                        {messages.map((m, i) => (
                                            <div key={i} className={`flex ${m.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                                                <div className={`max-w-[80%] rounded-lg p-3 text-sm ${m.type === 'user' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-200'
                                                    }`}>
                                                    {m.type === 'ai' ? (
                                                        <div className="prose prose-invert prose-sm">
                                                            {/* Very Basic Markdown-ish rendering for POC */}
                                                            {m.content.split('\n').map((line, idx) => <p key={idx} className="mb-1">{line}</p>)}
                                                        </div>
                                                    ) : m.content}
                                                </div>
                                            </div>
                                        ))}
                                        {chatLoading && <div className="text-slate-400 text-xs animate-pulse">Assistant is typing...</div>}
                                        <div ref={chatEndRef} />
                                    </div>
                                    <div className="p-4 border-t border-slate-700">
                                        <div className="flex gap-2">
                                            <input
                                                value={input}
                                                onChange={e => setInput(e.target.value)}
                                                onKeyDown={e => e.key === 'Enter' && handleSendChat()}
                                                placeholder="Ask why, what if..."
                                                className="flex-1 bg-slate-800 border-none rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500 outline-none"
                                            />
                                            <button onClick={handleSendChat} className="bg-blue-600 p-2 rounded-lg text-white hover:bg-blue-500">
                                                <Send className="w-5 h-5" />
                                            </button>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                    </div>
                ) : (
                    <div className="flex-1 bg-slate-900/50 backdrop-blur-md border border-slate-700/50 dashed border-2 rounded-xl flex flex-col items-center justify-center text-slate-500 p-10">
                        <AlertCircle className="w-12 h-12 mb-4 opacity-50" />
                        <p>Paste Transaction JSON and click Visualize</p>
                    </div>
                )}
            </div>
        </div>
    );
}
