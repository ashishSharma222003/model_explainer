"use client";
import React, { useState, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    FileText,
    Download,
    Loader2,
    TrendingUp,
    TrendingDown,
    AlertTriangle,
    Users,
    Shield,
    BarChart2,
    CheckCircle,
    DollarSign,
    Target,
    Lightbulb,
    ArrowRight,
    Copy,
    Check,
    Clock,
    History,
    Eye
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { AppContext } from '@/app/page';
import { generateReport } from '@/lib/api';

export default function ExecutiveReportSection() {
    const context = useContext(AppContext);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [generatedReport, setGeneratedReport] = useState<{ content: string; filename: string; title: string } | null>(null);
    const [selectedReportType, setSelectedReportType] = useState<'business_impact' | 'analyst_performance' | null>(null);
    const [copied, setCopied] = useState(false);

    // Check if there are shadow rules to analyze
    const shadowRules = context?.session.shadowRules || [];
    const hasShadowRules = shadowRules.length > 0;
    const txnChatHistory = context?.session.chatHistory?.txnChat || [];
    const hasTxnChat = txnChatHistory.length > 0;

    // Past reports from session
    const pastReports = context?.session.reports || [];

    // View a past report
    const handleViewPastReport = (report: any) => {
        setGeneratedReport({
            content: report.content || report.summary || 'Report content not available',
            filename: report.filename || `report_${report.id}.md`,
            title: report.title || 'Executive Report'
        });
    };

    const handleGenerateReport = async (reportType: 'business_impact' | 'analyst_performance') => {
        if (!context?.session.id) return;

        setLoading(true);
        setError(null);
        setSelectedReportType(reportType);

        try {
            // Use executive report type with specific focus based on selection
            // Include schema for context, but no raw JSON data or code
            const result = await generateReport({
                session_id: context.session.id,
                report_type: reportType === 'business_impact' ? 'executive' : 'shadow_report',
                format: 'markdown',
                include_code: false,
                include_schema: true,  // Include schema for business context
                include_chat_history: true,
                include_json_data: false,  // No raw JSON data - keep report clean
            });

            if (result.success && result.report) {
                setGeneratedReport({
                    content: typeof result.report.content === 'string'
                        ? result.report.content
                        : JSON.stringify(result.report.content, null, 2),
                    filename: result.report.filename || `executive_report_${Date.now()}.md`,
                    title: reportType === 'business_impact'
                        ? 'Business Impact Report'
                        : 'Decision Pattern Analysis Report',
                });

                // Add report record to session
                if (result.reportRecord) {
                    const reports = context?.session.reports || [];
                    const updatedReports = [...reports, result.reportRecord];
                    context?.updateSession({ reports: updatedReports });
                }
            }
        } catch (e: any) {
            setError(e.message || 'Failed to generate report');
        } finally {
            setLoading(false);
        }
    };

    const handleDownload = () => {
        if (!generatedReport) return;

        const blob = new Blob([generatedReport.content], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = generatedReport.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const handleCopyToClipboard = () => {
        if (!generatedReport) return;
        navigator.clipboard.writeText(generatedReport.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const reportTypes = [
        {
            id: 'business_impact' as const,
            title: 'Business Impact Report',
            subtitle: 'For Executive Leadership',
            description: 'Comprehensive analysis of how decision patterns affect your bottom line. Identifies areas of potential revenue loss and opportunities for improvement.',
            icon: DollarSign,
            color: 'from-emerald-600 to-green-600',
            hoverColor: 'from-emerald-500 to-green-500',
            borderColor: 'border-emerald-500/30',
            iconBg: 'bg-emerald-500/20',
            iconColor: 'text-emerald-400',
            highlights: [
                { icon: TrendingDown, text: 'Revenue Impact Analysis', color: 'text-rose-400' },
                { icon: TrendingUp, text: 'Improvement Opportunities', color: 'text-emerald-400' },
                { icon: Target, text: 'Strategic Recommendations', color: 'text-cyan-400' },
            ],
            disabled: !hasShadowRules && !hasTxnChat,
            disabledMessage: 'Complete analysis to unlock this report',
        },
        {
            id: 'analyst_performance' as const,
            title: 'Decision Pattern Analysis',
            subtitle: 'Analyst & Model Behavior',
            description: 'Detailed review of decision-making patterns, identifying potential inconsistencies or opportunities to improve decision quality and fairness.',
            icon: Users,
            color: 'from-violet-600 to-purple-600',
            hoverColor: 'from-violet-500 to-purple-500',
            borderColor: 'border-violet-500/30',
            iconBg: 'bg-violet-500/20',
            iconColor: 'text-violet-400',
            highlights: [
                { icon: AlertTriangle, text: 'Decision Consistency Review', color: 'text-amber-400' },
                { icon: Shield, text: 'Compliance Alignment', color: 'text-blue-400' },
                { icon: Lightbulb, text: 'Best Practice Insights', color: 'text-yellow-400' },
            ],
            disabled: !hasTxnChat,
            disabledMessage: 'Analyze transactions to unlock this report',
        },
    ];

    return (
        <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/50 border border-slate-700/50 rounded-2xl p-6 backdrop-blur-sm">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="p-2.5 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-xl">
                    <FileText className="w-5 h-5 text-cyan-400" />
                </div>
                <div>
                    <h2 className="text-lg font-semibold text-white">Executive Reports</h2>
                    <p className="text-xs text-slate-400">Generate comprehensive reports for leadership review</p>
                </div>
            </div>

            {/* Generated Report View */}
            <AnimatePresence mode="wait">
                {generatedReport ? (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="space-y-6"
                    >
                        {/* Success Header with Gradient */}
                        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-emerald-600 via-green-600 to-teal-600 p-6">
                            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zNiAxOGMzLjMxNCAwIDYgMi42ODYgNiA2cy0yLjY4NiA2LTYgNi02LTIuNjg2LTYtNiAyLjY4Ni02IDYtNiIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuMSkiLz48L2c+PC9zdmc+')] opacity-30"></div>
                            <div className="relative flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <div className="p-3 bg-white/20 backdrop-blur-sm rounded-xl">
                                        <CheckCircle className="w-8 h-8 text-white" />
                                    </div>
                                    <div>
                                        <h3 className="text-xl font-bold text-white">{generatedReport.title}</h3>
                                        <p className="text-emerald-100 text-sm">Report generated successfully • Ready for download</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => setGeneratedReport(null)}
                                    className="text-sm text-white/80 hover:text-white transition-colors bg-white/10 px-4 py-2 rounded-lg backdrop-blur-sm"
                                >
                                    ← New Report
                                </button>
                            </div>
                        </div>

                        {/* Action Buttons */}
                        <div className="grid grid-cols-2 gap-4">
                            <button
                                onClick={handleDownload}
                                className="group relative overflow-hidden bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white py-5 rounded-xl font-semibold flex items-center justify-center gap-3 transition-all shadow-xl shadow-violet-500/25 text-lg"
                            >
                                <div className="absolute inset-0 bg-white/10 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
                                <Download className="w-6 h-6 relative z-10" />
                                <span className="relative z-10">Download Report</span>
                            </button>
                            <button
                                onClick={handleCopyToClipboard}
                                className={`py-5 rounded-xl font-semibold flex items-center justify-center gap-3 transition-all text-lg ${copied
                                    ? 'bg-emerald-500/20 border-2 border-emerald-400 text-emerald-300'
                                    : 'bg-slate-800/80 hover:bg-slate-700/80 border-2 border-slate-600 text-white'
                                    }`}
                            >
                                {copied ? (
                                    <>
                                        <Check className="w-6 h-6" />
                                        <span>Copied to Clipboard!</span>
                                    </>
                                ) : (
                                    <>
                                        <Copy className="w-6 h-6" />
                                        <span>Copy Report</span>
                                    </>
                                )}
                            </button>
                        </div>

                        {/* Report Preview - Dark Theme Markdown Viewer */}
                        <div className="bg-slate-900 rounded-xl overflow-hidden border border-slate-700/50">
                            {/* Document Header */}
                            <div className="px-6 py-3 bg-slate-800/80 border-b border-slate-700/50 flex items-center justify-between">
                                <span className="text-sm font-medium text-slate-300">Executive Report</span>
                                <span className="text-xs text-slate-500">{generatedReport.filename}</span>
                            </div>

                            {/* Document Content - Dark Theme */}
                            <div className="p-8 max-h-[600px] overflow-auto bg-slate-900">
                                <div className="markdown-report">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                        {generatedReport.content}
                                    </ReactMarkdown>
                                </div>
                            </div>
                        </div>

                        {/* Board-Ready Badge */}
                        <div className="flex items-center gap-4 p-5 bg-gradient-to-r from-blue-500/10 to-indigo-500/10 border border-blue-500/20 rounded-xl">
                            <div className="p-3 bg-blue-500/20 rounded-xl">
                                <Shield className="w-6 h-6 text-blue-400" />
                            </div>
                            <div className="flex-1">
                                <p className="font-semibold text-blue-300">Board-Ready Executive Format</p>
                                <p className="text-sm text-slate-400 mt-1">
                                    This report uses business terminology only. No technical jargon included. Share confidently with C-suite executives and board members.
                                </p>
                            </div>
                        </div>
                    </motion.div>
                ) : (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="space-y-4"
                    >
                        {/* Report Type Cards */}
                        {reportTypes.map((report) => {
                            const Icon = report.icon;
                            const isDisabled = report.disabled;

                            return (
                                <motion.button
                                    key={report.id}
                                    onClick={() => !isDisabled && handleGenerateReport(report.id)}
                                    disabled={isDisabled || loading}
                                    className={`w-full text-left p-5 rounded-xl border transition-all ${isDisabled
                                        ? 'bg-slate-900/30 border-slate-800/50 opacity-50 cursor-not-allowed'
                                        : `bg-gradient-to-br from-slate-900/90 to-slate-800/70 ${report.borderColor} hover:border-opacity-60 hover:shadow-lg cursor-pointer`
                                        }`}
                                    whileHover={!isDisabled ? { scale: 1.01 } : undefined}
                                    whileTap={!isDisabled ? { scale: 0.99 } : undefined}
                                >
                                    <div className="flex items-start gap-4">
                                        {/* Icon */}
                                        <div className={`p-3 ${report.iconBg} rounded-xl flex-shrink-0`}>
                                            <Icon className={`w-6 h-6 ${report.iconColor}`} />
                                        </div>

                                        {/* Content */}
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center justify-between mb-1">
                                                <h3 className="font-semibold text-white">{report.title}</h3>
                                                {loading && selectedReportType === report.id ? (
                                                    <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
                                                ) : (
                                                    <ArrowRight className={`w-5 h-5 ${isDisabled ? 'text-slate-600' : 'text-slate-400'}`} />
                                                )}
                                            </div>
                                            <p className="text-xs text-cyan-400 font-medium mb-2">{report.subtitle}</p>
                                            <p className="text-sm text-slate-400 mb-3">{report.description}</p>

                                            {/* Highlights */}
                                            <div className="flex flex-wrap gap-2">
                                                {report.highlights.map((highlight, i) => {
                                                    const HighlightIcon = highlight.icon;
                                                    return (
                                                        <span
                                                            key={i}
                                                            className="flex items-center gap-1.5 text-xs bg-slate-800/60 px-2.5 py-1 rounded-lg"
                                                        >
                                                            <HighlightIcon className={`w-3.5 h-3.5 ${highlight.color}`} />
                                                            <span className="text-slate-300">{highlight.text}</span>
                                                        </span>
                                                    );
                                                })}
                                            </div>

                                            {/* Disabled Message */}
                                            {isDisabled && (
                                                <div className="flex items-center gap-2 mt-3 text-xs text-amber-400/80">
                                                    <AlertTriangle className="w-3.5 h-3.5" />
                                                    {report.disabledMessage}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </motion.button>
                            );
                        })}

                        {/* Error Message */}
                        {error && (
                            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-300 text-sm">
                                {error}
                            </div>
                        )}

                        {/* Past Reports Section */}
                        {pastReports.length > 0 && (
                            <div className="mt-6 pt-6 border-t border-slate-700/50">
                                <div className="flex items-center gap-2 mb-4">
                                    <History className="w-4 h-4 text-slate-400" />
                                    <h3 className="text-sm font-medium text-slate-300">Previously Generated Reports</h3>
                                    <span className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full">{pastReports.length}</span>
                                </div>
                                <div className="space-y-2">
                                    {pastReports.slice().reverse().map((report: any, index: number) => (
                                        <div
                                            key={report.id || index}
                                            className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700/30 hover:border-slate-600/50 transition-colors"
                                        >
                                            <div className="flex items-center gap-3">
                                                <div className="p-1.5 bg-slate-700/50 rounded">
                                                    <FileText className="w-4 h-4 text-slate-400" />
                                                </div>
                                                <div>
                                                    <p className="text-sm font-medium text-slate-300">{report.title}</p>
                                                    <div className="flex items-center gap-2 text-xs text-slate-500">
                                                        <Clock className="w-3 h-3" />
                                                        <span>{new Date(report.generatedAt).toLocaleDateString()} at {new Date(report.generatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                                    </div>
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => handleViewPastReport(report)}
                                                className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10 rounded-lg transition-colors"
                                            >
                                                <Eye className="w-4 h-4" />
                                                View
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Info Note */}
                        <div className="flex items-start gap-3 p-4 bg-slate-800/50 border border-slate-700/50 rounded-xl">
                            <BarChart2 className="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" />
                            <div>
                                <p className="text-sm font-medium text-slate-300">About Executive Reports</p>
                                <p className="text-xs text-slate-500 mt-1">
                                    These reports are specifically designed for senior leadership and board presentations.
                                    They focus on business impact, strategic recommendations, and actionable insights
                                    without technical jargon.
                                </p>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
