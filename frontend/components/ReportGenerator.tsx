"use client";
import { useState, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FileText, 
  X, 
  Download, 
  FileCode, 
  MessageSquare, 
  FileJson,
  Loader2,
  Check,
  Users,
  Code2,
  ClipboardList,
  History,
  Clock,
  Database,
  UserSearch,
  Shield
} from 'lucide-react';
import { AppContext } from '@/app/page';
import { generateReport, ReportType, ReportFormat, GenerateReportResponse } from '@/lib/api';
import { GeneratedReport } from '@/lib/storage';

interface ReportGeneratorProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ReportGenerator({ isOpen, onClose }: ReportGeneratorProps) {
  const context = useContext(AppContext);
  const [reportType, setReportType] = useState<ReportType>('executive');
  const [format, setFormat] = useState<ReportFormat>('markdown');
  const [includeCode, setIncludeCode] = useState(false);
  const [includeSchema, setIncludeSchema] = useState(true);  // Include data schema by default
  const [includeChatHistory, setIncludeChatHistory] = useState(true);
  const [includeJsonData, setIncludeJsonData] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generatedReport, setGeneratedReport] = useState<{ content: string; filename: string; title: string } | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [viewingReport, setViewingReport] = useState<GeneratedReport | null>(null);

  // Get reports from session
  const reports = context?.session.reports || [];

  const handleGenerate = async () => {
    if (!context?.session.id) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await generateReport({
        session_id: context.session.id,
        report_type: reportType,
        format,
        include_code: includeCode,
        include_schema: includeSchema,
        include_chat_history: includeChatHistory,
        include_json_data: includeJsonData,
      });
      
      if (result.success && result.report) {
        setGeneratedReport({
          content: typeof result.report.content === 'string' 
            ? result.report.content 
            : JSON.stringify(result.report.content, null, 2),
          filename: result.report.filename,
          title: result.report.title,
        });
        
        // Add report record to session
        if (result.reportRecord) {
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
    
    const blob = new Blob([generatedReport.content], { 
      type: format === 'markdown' ? 'text/markdown' : 'application/json' 
    });
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
  };

  // Check if txn chat has messages (required for shadow report)
  const hasTxnChat = (context?.session.chatHistory?.txnChat?.length || 0) > 0;

  const reportTypes: { id: ReportType; label: string; description: string; icon: React.ReactNode; requiresTxnChat?: boolean }[] = [
    { 
      id: 'executive', 
      label: 'Executive Summary', 
      description: 'Clear overview for management and stakeholders',
      icon: <Users className="w-5 h-5" />
    },
    { 
      id: 'full_export', 
      label: 'Operations Report', 
      description: 'Complete findings and recommendations',
      icon: <ClipboardList className="w-5 h-5" />
    },
    { 
      id: 'shadow_report', 
      label: 'Shadow Report', 
      description: 'Hidden patterns, biases, and compliance gaps',
      icon: <UserSearch className="w-5 h-5" />,
      requiresTxnChat: true
    },
    { 
      id: 'technical', 
      label: 'Technical Report', 
      description: 'Code and model details for developers',
      icon: <Code2 className="w-5 h-5" />
    },
  ];

  // Filter report types based on availability
  const availableReportTypes = reportTypes.filter(type => {
    if (type.requiresTxnChat && !hasTxnChat) return false;
    return true;
  });

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="bg-[#0d1117] border border-slate-800 rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden shadow-2xl"
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-gradient-to-r from-violet-900/20 to-purple-900/20">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-violet-500/20 rounded-lg">
                <FileText className="w-5 h-5 text-violet-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">
                  {showHistory ? 'Report History' : 'Generate Report'}
                </h2>
                <p className="text-xs text-slate-500">
                  {showHistory ? `${reports.length} reports generated` : 'Export your analysis as a document'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {reports.length > 0 && (
                <button
                  onClick={() => setShowHistory(!showHistory)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                    showHistory 
                      ? 'bg-violet-500/20 text-violet-300' 
                      : 'hover:bg-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  <History className="w-4 h-4" />
                  {showHistory ? 'New' : 'History'}
                  {!showHistory && reports.length > 0 && (
                    <span className="bg-violet-500/30 text-violet-300 text-xs px-1.5 py-0.5 rounded-full">
                      {reports.length}
                    </span>
                  )}
                </button>
              )}
              <button
                onClick={onClose}
                className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
            {viewingReport ? (
              /* Viewing a past report */
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white">{viewingReport.title}</h3>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        viewingReport.reportType === 'executive' ? 'bg-violet-500/20 text-violet-300' :
                        viewingReport.reportType === 'technical' ? 'bg-cyan-500/20 text-cyan-300' :
                        viewingReport.reportType === 'compliance' ? 'bg-amber-500/20 text-amber-300' :
                        'bg-slate-700 text-slate-300'
                      }`}>
                        {viewingReport.reportType}
                      </span>
                      <span className="text-xs text-slate-500">
                        {new Date(viewingReport.generatedAt).toLocaleDateString()}{' '}
                        {new Date(viewingReport.generatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => setViewingReport(null)}
                    className="text-sm text-slate-400 hover:text-white transition-colors"
                  >
                    ← Back to History
                  </button>
                </div>
                
                {/* Full content */}
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 max-h-[350px] overflow-auto mb-4">
                  <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap">
                    {viewingReport.content || 'Content not available'}
                  </pre>
                </div>
                
                {/* Actions */}
                <div className="flex gap-3">
                  <button
                    onClick={() => {
                      if (!viewingReport.content) return;
                      const blob = new Blob([viewingReport.content], { 
                        type: viewingReport.format === 'markdown' ? 'text/markdown' : 'application/json' 
                      });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = viewingReport.filename;
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                      URL.revokeObjectURL(url);
                    }}
                    disabled={!viewingReport.content}
                    className="flex-1 bg-gradient-to-r from-emerald-600 to-green-600 hover:from-emerald-500 hover:to-green-500 disabled:opacity-50 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all"
                  >
                    <Download className="w-4 h-4" />
                    Download
                  </button>
                  <button
                    onClick={() => {
                      if (viewingReport.content) {
                        navigator.clipboard.writeText(viewingReport.content);
                      }
                    }}
                    disabled={!viewingReport.content}
                    className="flex-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all border border-slate-700"
                  >
                    Copy to Clipboard
                  </button>
                </div>
              </div>
            ) : showHistory ? (
              /* Report History View */
              <div className="space-y-3">
                {reports.length === 0 ? (
                  <div className="text-center py-12 text-slate-500">
                    <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No reports generated yet</p>
                  </div>
                ) : (
                  [...reports].reverse().map((report, i) => (
                    <button
                      key={report.id || i}
                      onClick={() => setViewingReport(report)}
                      className="w-full p-4 bg-slate-900/50 border border-slate-800 rounded-xl hover:border-violet-500/50 hover:bg-slate-900 transition-colors text-left group"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h4 className="font-medium text-white group-hover:text-violet-300 transition-colors">
                            {report.title}
                          </h4>
                          <div className="flex items-center gap-2 mt-1">
                            <span className={`text-xs px-2 py-0.5 rounded-full ${
                              report.reportType === 'executive' ? 'bg-violet-500/20 text-violet-300' :
                              report.reportType === 'technical' ? 'bg-cyan-500/20 text-cyan-300' :
                              report.reportType === 'compliance' ? 'bg-amber-500/20 text-amber-300' :
                              'bg-slate-700 text-slate-300'
                            }`}>
                              {report.reportType}
                            </span>
                            <span className="text-xs text-slate-500">{report.format}</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="flex items-center gap-1 text-xs text-slate-500">
                            <Clock className="w-3 h-3" />
                            {new Date(report.generatedAt).toLocaleDateString()}{' '}
                            {new Date(report.generatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </div>
                          <span className="text-xs text-violet-400 opacity-0 group-hover:opacity-100 transition-opacity">
                            View →
                          </span>
                        </div>
                      </div>
                      
                      {/* Included content badges */}
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {report.includeChatHistory && (
                          <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded flex items-center gap-1">
                            <MessageSquare className="w-3 h-3" /> Chat
                          </span>
                        )}
                        {report.includeCode && (
                          <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded flex items-center gap-1">
                            <FileCode className="w-3 h-3" /> Code
                          </span>
                        )}
                        {report.includeSchema && (
                          <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded flex items-center gap-1">
                            <Database className="w-3 h-3" /> Schema
                          </span>
                        )}
                        {report.includeJsonData && (
                          <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded flex items-center gap-1">
                            <FileJson className="w-3 h-3" /> JSON
                          </span>
                        )}
                      </div>
                      
                      {/* Summary preview */}
                      {report.summary && (
                        <p className="mt-2 text-xs text-slate-500 line-clamp-2">
                          {report.summary}
                        </p>
                      )}
                    </button>
                  ))
                )}
              </div>
            ) : !generatedReport ? (
              <>
                {/* Report Type Selection */}
                <div className="mb-6">
                  <h3 className="text-sm font-medium text-white mb-3">Report Type</h3>
                  <div className="grid grid-cols-1 gap-3">
                    {availableReportTypes.map(type => (
                      <button
                        key={type.id}
                        onClick={() => setReportType(type.id)}
                        className={`flex items-center gap-4 p-4 rounded-xl border transition-all text-left ${
                          reportType === type.id
                            ? 'bg-violet-500/20 border-violet-500/50 text-violet-300'
                            : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        <div className={`p-2 rounded-lg ${
                          reportType === type.id ? 'bg-violet-500/20' : 'bg-slate-800'
                        }`}>
                          {type.icon}
                        </div>
                        <div className="flex-1">
                          <div className="font-medium text-white">{type.label}</div>
                          <div className="text-xs opacity-70">{type.description}</div>
                        </div>
                        {reportType === type.id && (
                          <Check className="w-5 h-5 text-violet-400" />
                        )}
                      </button>
                    ))}
                  </div>
                  
                  {/* Shadow Report note when not available */}
                  {!hasTxnChat && (
                    <div className="mt-3 p-3 bg-slate-900/30 border border-slate-800 rounded-lg">
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <UserSearch className="w-4 h-4" />
                        <span>
                          <strong>Shadow Report</strong> will be available after you start analyzing transactions in Transaction Chat.
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Include Options */}
                <div className="mb-6">
                  <h3 className="text-sm font-medium text-white mb-3">Include in Report</h3>
                  <div className="space-y-2">
                    <label className="flex items-center gap-3 p-3 bg-slate-900/50 border border-slate-800 rounded-lg cursor-pointer hover:border-slate-700 transition-colors">
                      <input
                        type="checkbox"
                        checked={includeChatHistory}
                        onChange={e => setIncludeChatHistory(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-violet-500 focus:ring-violet-500"
                      />
                      <MessageSquare className="w-4 h-4 text-cyan-400" />
                      <span className="text-sm text-slate-300">Analysis Conversations</span>
                    </label>
                    
                    {/* Only show code option for technical reports */}
                    {reportType === 'technical' && (
                      <label className="flex items-center gap-3 p-3 bg-slate-900/50 border border-slate-800 rounded-lg cursor-pointer hover:border-slate-700 transition-colors">
                        <input
                          type="checkbox"
                          checked={includeCode}
                          onChange={e => setIncludeCode(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-violet-500 focus:ring-violet-500"
                        />
                        <FileCode className="w-4 h-4 text-violet-400" />
                        <span className="text-sm text-slate-300">ML Code</span>
                        {!context?.mlCode && <span className="text-xs text-slate-500">(not provided)</span>}
                      </label>
                    )}
                    
                    <label className="flex items-center gap-3 p-3 bg-slate-900/50 border border-slate-800 rounded-lg cursor-pointer hover:border-slate-700 transition-colors">
                      <input
                        type="checkbox"
                        checked={includeSchema}
                        onChange={e => setIncludeSchema(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-violet-500 focus:ring-violet-500"
                      />
                      <Database className="w-4 h-4 text-emerald-400" />
                      <span className="text-sm text-slate-300">Data Overview</span>
                      {!context?.dataSchema && <span className="text-xs text-slate-500">(not provided)</span>}
                    </label>
                    
                    <label className="flex items-center gap-3 p-3 bg-slate-900/50 border border-slate-800 rounded-lg cursor-pointer hover:border-slate-700 transition-colors">
                      <input
                        type="checkbox"
                        checked={includeJsonData}
                        onChange={e => setIncludeJsonData(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-violet-500 focus:ring-violet-500"
                      />
                      <FileJson className="w-4 h-4 text-amber-400" />
                      <span className="text-sm text-slate-300">Model Findings Details</span>
                    </label>
                  </div>
                </div>

                {/* Format Selection */}
                <div className="mb-6">
                  <h3 className="text-sm font-medium text-white mb-3">Format</h3>
                  <div className="flex gap-3">
                    <button
                      onClick={() => setFormat('markdown')}
                      className={`flex-1 p-3 rounded-lg border transition-all ${
                        format === 'markdown'
                          ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
                          : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="font-medium">Markdown</div>
                      <div className="text-xs opacity-70">.md file</div>
                    </button>
                    <button
                      onClick={() => setFormat('json')}
                      className={`flex-1 p-3 rounded-lg border transition-all ${
                        format === 'json'
                          ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
                          : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="font-medium">JSON</div>
                      <div className="text-xs opacity-70">.json file</div>
                    </button>
                  </div>
                </div>

                {/* Error Message */}
                {error && (
                  <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-300 text-sm">
                    {error}
                  </div>
                )}

                {/* Generate Button */}
                <button
                  onClick={handleGenerate}
                  disabled={loading}
                  className="w-full bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 disabled:opacity-50 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all shadow-lg shadow-violet-500/20"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <FileText className="w-4 h-4" />
                      Generate Report
                    </>
                  )}
                </button>
              </>
            ) : (
              /* Report Preview */
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">{generatedReport.title}</h3>
                  <button
                    onClick={() => setGeneratedReport(null)}
                    className="text-sm text-slate-400 hover:text-white transition-colors"
                  >
                    ← Back
                  </button>
                </div>
                
                {/* Preview */}
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 max-h-[300px] overflow-auto mb-4">
                  <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap">
                    {generatedReport.content.substring(0, 2000)}
                    {generatedReport.content.length > 2000 && '\n\n... (truncated preview)'}
                  </pre>
                </div>
                
                {/* Actions */}
                <div className="flex gap-3">
                  <button
                    onClick={handleDownload}
                    className="flex-1 bg-gradient-to-r from-emerald-600 to-green-600 hover:from-emerald-500 hover:to-green-500 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all"
                  >
                    <Download className="w-4 h-4" />
                    Download
                  </button>
                  <button
                    onClick={handleCopyToClipboard}
                    className="flex-1 bg-slate-800 hover:bg-slate-700 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all border border-slate-700"
                  >
                    Copy to Clipboard
                  </button>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

