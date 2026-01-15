"use client";
import React, { useState, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Zap, ChevronDown, ChevronUp, RefreshCw, Target,
    Users, Lightbulb, Code, AlertTriangle, CheckCircle, XCircle
} from 'lucide-react';
import { AppContext } from '@/app/page';

interface GeneralPurposeRule {
    original_rule: string;
    description: string;
    target: string;
    accuracy: number;
    coverage: number;
    confidence: string;
}

export default function DiscoverySection() {
    const context = useContext(AppContext);
    const [expandedRow, setExpandedRow] = useState<number | null>(null);
    const [isRegenerating, setIsRegenerating] = useState(false);

    const analysisResult = context?.session.analysisResult;
    const generalRules = analysisResult?.general_purpose_rules || [];

    const handleRegenerate = async () => {
        if (!context?.session.id) return;

        setIsRegenerating(true);
        try {
            // Re-run analysis to regenerate rules
            const res = await fetch(`http://localhost:8000/sessions/${context.session.id}/analyze`, {
                method: 'POST',
            });

            if (res.ok) {
                const data = await res.json();
                context?.updateSession({ analysisResult: data });
            }
        } catch (error) {
            console.error('Error regenerating rules:', error);
        } finally {
            setIsRegenerating(false);
        }
    };

    if (!analysisResult) {
        return (
            <div className="max-w-7xl mx-auto text-center py-12">
                <AlertTriangle className="w-16 h-16 text-amber-400 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-white mb-2">No Analysis Available</h2>
                <p className="text-slate-400">
                    Please run the analysis in the Insights section first.
                </p>
            </div>
        );
    }

    if (generalRules.length === 0) {
        return (
            <div className="max-w-7xl mx-auto text-center py-12">
                <Lightbulb className="w-16 h-16 text-amber-400 mx-auto mb-4 opacity-50" />
                <h2 className="text-2xl font-bold text-white mb-2">No Rules Discovered</h2>
                <p className="text-slate-400 mb-6">
                    Run or re-run the analysis to generate shadow rules from your data.
                </p>
                <button
                    onClick={handleRegenerate}
                    disabled={isRegenerating}
                    className="px-6 py-3 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 rounded-xl text-white font-medium flex items-center gap-2 mx-auto disabled:opacity-50"
                >
                    {isRegenerating ? (
                        <>
                            <RefreshCw className="w-5 h-5 animate-spin" />
                            Generating Rules...
                        </>
                    ) : (
                        <>
                            <Zap className="w-5 h-5" />
                            Generate Rules
                        </>
                    )}
                </button>
            </div>
        );
    }

    const totalTransactions = analysisResult.total_samples || 0;

    return (
        <div className="max-w-7xl mx-auto">
            {/* Header */}
            <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3">
                    <div className="p-3 bg-gradient-to-br from-amber-500 to-orange-600 rounded-2xl shadow-lg shadow-amber-500/20">
                        <Lightbulb className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h2 className="text-3xl font-bold text-white">Shadow Rules Discovery</h2>
                        <p className="text-slate-400 mt-1">
                            {generalRules.length} business rules discovered from {totalTransactions.toLocaleString()} transactions
                        </p>
                    </div>
                </div>

                <button
                    onClick={handleRegenerate}
                    disabled={isRegenerating}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-300 font-medium flex items-center gap-2 disabled:opacity-50"
                >
                    <RefreshCw className={`w-4 h-4 ${isRegenerating ? 'animate-spin' : ''}`} />
                    Regenerate
                </button>
            </div>

            {/* Table */}
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
                {/* Table Header */}
                <div className="grid grid-cols-12 gap-4 px-6 py-4 bg-slate-800/50 border-b border-slate-700 text-sm font-semibold text-slate-300">
                    <div className="col-span-1 text-center">#</div>
                    <div className="col-span-4">Shadow Rule</div>
                    <div className="col-span-2 text-center">Coverage</div>
                    <div className="col-span-3 text-center">Classification</div>
                    <div className="col-span-2 text-center">Accuracy</div>
                </div>

                {/* Table Body */}
                <div className="divide-y divide-slate-800/50">
                    {generalRules.map((rule: GeneralPurposeRule, idx: number) => {
                        const isExpanded = expandedRow === idx;
                        const transactionsAffected = Math.round((rule.coverage / 100) * totalTransactions);

                        return (
                            <div key={idx} className="hover:bg-slate-800/30 transition-colors">
                                {/* Main Row */}
                                <div
                                    className="grid grid-cols-12 gap-4 px-6 py-4 cursor-pointer items-center"
                                    onClick={() => setExpandedRow(isExpanded ? null : idx)}
                                >
                                    {/* Serial No */}
                                    <div className="col-span-1 text-center text-slate-400 font-mono text-sm">
                                        {idx + 1}
                                    </div>

                                    {/* Shadow Rule Heading */}
                                    <div className="col-span-4">
                                        <div className="flex items-center gap-2">
                                            {isExpanded ? (
                                                <ChevronUp className="w-4 h-4 text-slate-500 flex-shrink-0" />
                                            ) : (
                                                <ChevronDown className="w-4 h-4 text-slate-500 flex-shrink-0" />
                                            )}
                                            <span className="text-white font-medium line-clamp-1">
                                                {rule.description}
                                            </span>
                                        </div>
                                    </div>

                                    {/* Coverage */}
                                    <div className="col-span-2">
                                        <div className="text-center">
                                            <div className="text-lg font-bold text-violet-400">
                                                {rule.coverage.toFixed(1)}%
                                            </div>
                                            <div className="text-xs text-slate-500">
                                                {transactionsAffected.toLocaleString()} / {totalTransactions.toLocaleString()}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Classification (Class) */}
                                    <div className="col-span-3">
                                        <div className="flex items-center justify-center gap-2">
                                            <div className={`px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider ${rule.target === 'Fraud'
                                                ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                                                : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                                                }`}>
                                                {rule.target === 'Fraud' ? (
                                                    <span className="flex items-center gap-1">
                                                        <XCircle className="w-3 h-3" />
                                                        Fraud
                                                    </span>
                                                ) : (
                                                    <span className="flex items-center gap-1">
                                                        <CheckCircle className="w-3 h-3" />
                                                        Legit
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Accuracy */}
                                    <div className="col-span-2">
                                        <div className="text-center">
                                            <div className="text-lg font-bold text-emerald-400">
                                                {(rule.accuracy * 100).toFixed(1)}%
                                            </div>
                                            <div className="text-xs text-slate-500 capitalize">
                                                {rule.confidence} confidence
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Expanded Details */}
                                <AnimatePresence>
                                    {isExpanded && (
                                        <motion.div
                                            initial={{ height: 0, opacity: 0 }}
                                            animate={{ height: 'auto', opacity: 1 }}
                                            exit={{ height: 0, opacity: 0 }}
                                            transition={{ duration: 0.2 }}
                                            className="overflow-hidden border-t border-slate-800"
                                        >
                                            <div className="px-6 py-6 bg-slate-950/50 grid grid-cols-1 gap-4">
                                                {/* General Purpose Rule */}
                                                <div>
                                                    <div className="flex items-center gap-2 mb-2">
                                                        <Lightbulb className="w-4 h-4 text-amber-400" />
                                                        <h4 className="text-sm font-semibold text-slate-300">General Purpose Rule</h4>
                                                    </div>
                                                    <div className="pl-6 text-slate-200 leading-relaxed">
                                                        {rule.description}
                                                    </div>
                                                </div>

                                                {/* Actual Technical Rule */}
                                                <div>
                                                    <div className="flex items-center gap-2 mb-2">
                                                        <Code className="w-4 h-4 text-blue-400" />
                                                        <h4 className="text-sm font-semibold text-slate-300">Technical Rule (Original)</h4>
                                                    </div>
                                                    <div className="pl-6 bg-black/40 rounded-lg p-3 font-mono text-xs text-emerald-300 overflow-x-auto">
                                                        {rule.original_rule}
                                                    </div>
                                                </div>

                                                {/* Metrics Summary */}
                                                <div className="grid grid-cols-3 gap-4 mt-2">
                                                    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3">
                                                        <div className="flex items-center gap-2 mb-1">
                                                            <Target className="w-4 h-4 text-slate-400" />
                                                            <span className="text-xs text-slate-500">Accuracy</span>
                                                        </div>
                                                        <div className="text-xl font-bold text-emerald-400">
                                                            {(rule.accuracy * 100).toFixed(1)}%
                                                        </div>
                                                    </div>
                                                    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3">
                                                        <div className="flex items-center gap-2 mb-1">
                                                            <Users className="w-4 h-4 text-slate-400" />
                                                            <span className="text-xs text-slate-500">Coverage</span>
                                                        </div>
                                                        <div className="text-xl font-bold text-violet-400">
                                                            {rule.coverage.toFixed(1)}%
                                                        </div>
                                                        <div className="text-xs text-slate-500 mt-1">
                                                            {transactionsAffected.toLocaleString()} transactions
                                                        </div>
                                                    </div>
                                                    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3">
                                                        <div className="flex items-center gap-2 mb-1">
                                                            <Zap className="w-4 h-4 text-slate-400" />
                                                            <span className="text-xs text-slate-500">Confidence</span>
                                                        </div>
                                                        <div className="text-xl font-bold text-amber-400 capitalize">
                                                            {rule.confidence}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Summary Stats */}
            <div className="mt-6 grid grid-cols-3 gap-4">
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">Total Rules Discovered</div>
                    <div className="text-2xl font-bold text-white">{generalRules.length}</div>
                </div>
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">Average Accuracy</div>
                    <div className="text-2xl font-bold text-emerald-400">
                        {(generalRules.reduce((sum: number, r: GeneralPurposeRule) => sum + r.accuracy, 0) / generalRules.length * 100).toFixed(1)}%
                    </div>
                </div>
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
                    <div className="text-sm text-slate-400 mb-1">Average Coverage</div>
                    <div className="text-2xl font-bold text-violet-400">
                        {(generalRules.reduce((sum: number, r: GeneralPurposeRule) => sum + r.coverage, 0) / generalRules.length).toFixed(1)}%
                    </div>
                </div>
            </div>
        </div>
    );
}
