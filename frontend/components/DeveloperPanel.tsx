"use client";
import { useState, useContext, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Terminal, 
  Play, 
  RotateCcw, 
  Copy, 
  Check, 
  AlertCircle,
  Loader2,
  Zap,
  MessageSquare,
  Upload,
  FileSpreadsheet,
  Table
} from 'lucide-react';
import { AppContext } from '@/app/page';
import { executeCode, resetKernel, injectKernelContext, uploadFileToKernel, KernelOutput } from '@/lib/api';

interface DeveloperPanelProps {
  isVisible: boolean;
  onSendToChat?: (output: string) => void;
}

// Generate smart code snippet based on uploaded files
const generateDataLoadSnippet = (files: {name: string; varName: string; shape?: [number, number]}[]) => {
  if (files.length === 0) {
    return `# No files uploaded yet
# Click the upload button to add CSV, JSON, or text files
# They will be automatically loaded into pandas DataFrames`;
  }

  const lines = ['import pandas as pd', ''];
  
  // Show how to load each file (already loaded, but show the code for reference)
  lines.push('# Files are already loaded! Here\'s how they were loaded:');
  files.forEach(file => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext === 'csv') {
      lines.push(`# ${file.varName} = pd.read_csv("${file.name}")`);
    } else if (ext === 'json') {
      lines.push(`# ${file.varName} = pd.read_json("${file.name}")`);
    } else {
      lines.push(`# ${file.varName} loaded from ${file.name}`);
    }
  });
  
  lines.push('');
  lines.push('# Explore your data:');
  
  files.forEach(file => {
    lines.push(`print(f"=== ${file.varName} ${file.shape ? `(${file.shape[0]} rows × ${file.shape[1]} cols)` : ''} ===")`);
    lines.push(`print(${file.varName}.dtypes)`);
    lines.push(`print(${file.varName}.head())`);
    lines.push('');
  });

  return lines.join('\n');
};

export default function DeveloperPanel({ isVisible, onSendToChat }: DeveloperPanelProps) {
  const context = useContext(AppContext);
  const [code, setCode] = useState('# Write Python code here\nprint("Hello from kernel!")');
  const [outputs, setOutputs] = useState<KernelOutput[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [executionTime, setExecutionTime] = useState<number | null>(null);
  const [variables, setVariables] = useState<string[]>([]);
  const [copied, setCopied] = useState(false);
  const [contextLoaded, setContextLoaded] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<{name: string; varName: string; shape?: [number, number]}[]>([]);
  const [uploading, setUploading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const sessionId = `kernel_${context?.session.id || 'default'}`;

  const handleExecute = async () => {
    if (!code.trim()) return;
    
    setLoading(true);
    setError(null);
    setOutputs([]);
    
    try {
      const result = await executeCode(sessionId, code);
      setOutputs(result.outputs);
      setExecutionTime(result.execution_time_ms);
      setVariables(result.variables);
      
      if (!result.success && result.error) {
        setError(result.error);
      }
    } catch (e: any) {
      setError(e.message || 'Failed to execute code');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      await resetKernel(sessionId);
      setOutputs([]);
      setVariables([]);
      setContextLoaded(false);
      setUploadedFiles([]);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleLoadContext = async () => {
    try {
      setLoading(true);
      await injectKernelContext(
        sessionId,
        context?.mlCode,
        context?.globalJson
      );
      setContextLoaded(true);
      setOutputs([{ type: 'stream', name: 'stdout', text: '✓ Context loaded! Variables available: global_explanation, feature_importances' }]);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Generate variable name from filename
    const varName = file.name
      .replace(/\.[^/.]+$/, '') // Remove extension
      .replace(/[^a-zA-Z0-9_]/g, '_') // Replace special chars with underscore
      .replace(/^[0-9]/, '_$&') // Prefix with underscore if starts with number
      .toLowerCase();

    setUploading(true);
    setError(null);

    try {
      const result = await uploadFileToKernel(sessionId, file, varName);
      
      setUploadedFiles(prev => [...prev, { 
        name: file.name, 
        varName,
        shape: result.shape 
      }]);
      
      setOutputs([{ 
        type: 'stream', 
        name: 'stdout', 
        text: `✓ ${result.message}\n\nPreview:\n${result.preview}` 
      }]);
      
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (e: any) {
      setError(e.message || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  const handleLoadDataSchema = () => {
    setCode(generateDataLoadSnippet(uploadedFiles));
  };

  const handleCopyOutput = () => {
    const outputText = outputs.map(o => o.text || o.data || '').join('\n');
    navigator.clipboard.writeText(outputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSendToChat = () => {
    if (onSendToChat && outputs.length > 0) {
      const outputText = outputs.map(o => o.text || o.data || '').join('\n');
      onSendToChat(outputText);
    }
  };

  const formatOutput = (output: KernelOutput): string => {
    if (output.type === 'error') {
      return `${output.ename}: ${output.evalue}`;
    }
    return output.text || output.data || '';
  };

  if (!isVisible) return null;

  return (
    <motion.div
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 100 }}
      className="fixed right-0 top-16 bottom-0 w-[400px] bg-[#0d1117] border-l border-slate-800 z-20 flex flex-col shadow-2xl"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-emerald-900/20 to-green-900/20 border-b border-slate-800 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span className="text-sm font-medium text-emerald-300">Python Kernel</span>
          {contextLoaded && (
            <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
              ✓
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {/* File Upload */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.json,.txt"
            onChange={handleFileUpload}
            className="hidden"
            id="kernel-file-upload"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="flex items-center gap-1 p-1.5 text-xs hover:bg-amber-500/20 rounded text-amber-400 transition-colors disabled:opacity-50"
            title="Upload CSV, JSON, or text file"
          >
            {uploading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Upload className="w-4 h-4" />
            )}
          </button>
          <button
            onClick={handleLoadContext}
            disabled={loading || !context?.globalJson}
            className="flex items-center gap-1 p-1.5 text-xs hover:bg-slate-800 disabled:opacity-50 rounded text-slate-400 hover:text-white transition-colors"
            title="Load ML code and JSON into kernel"
          >
            <Zap className="w-4 h-4" />
          </button>
          <button
            onClick={handleReset}
            className="flex items-center gap-1 p-1.5 text-xs text-slate-400 hover:text-white hover:bg-slate-800 rounded transition-colors"
            title="Reset kernel"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* Uploaded Files */}
        {uploadedFiles.length > 0 && (
          <div className="flex flex-wrap gap-1.5 p-2 bg-slate-900/50 rounded-lg">
            {uploadedFiles.map((file, i) => (
              <div 
                key={i}
                className="flex items-center gap-1 px-2 py-0.5 bg-amber-500/10 border border-amber-500/30 rounded text-xs text-amber-300"
                title={`Variable: ${file.varName}${file.shape ? ` (${file.shape[0]}×${file.shape[1]})` : ''}`}
              >
                <FileSpreadsheet className="w-3 h-3" />
                <span className="font-mono">{file.varName}</span>
              </div>
            ))}
          </div>
        )}

        {/* Code Editor Section */}
        <div className="bg-slate-900/50 rounded-lg overflow-hidden border border-slate-800">
          <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800">
            <span className="text-xs text-slate-400 font-medium">Code</span>
            {uploadedFiles.length > 0 && (
              <button
                onClick={handleLoadDataSchema}
                className="flex items-center gap-1 text-xs text-amber-400 hover:text-amber-300 hover:bg-amber-500/10 px-2 py-1 rounded transition-colors"
                title="Generate code to explore uploaded data"
              >
                <Table className="w-3 h-3" />
                Explore Data
              </button>
            )}
          </div>

          <textarea
            ref={textareaRef}
            value={code}
            onChange={e => setCode(e.target.value)}
            className="w-full h-[140px] bg-slate-950 p-3 text-xs font-mono text-emerald-300 resize-none outline-none"
            placeholder="# Write Python code here..."
            spellCheck={false}
          />
          
          <div className="p-2 border-t border-slate-800">
            <button
              onClick={handleExecute}
              disabled={loading || !code.trim()}
              className="w-full bg-gradient-to-r from-emerald-600 to-green-600 hover:from-emerald-500 hover:to-green-500 disabled:opacity-50 text-white py-2 rounded-lg font-medium flex items-center justify-center gap-2 transition-all text-sm"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run
                </>
              )}
            </button>
          </div>
        </div>

        {/* Output Section */}
        <div className="bg-slate-900/50 rounded-lg overflow-hidden border border-slate-800 flex-1 flex flex-col min-h-0">
          <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800 flex-shrink-0">
            <span className="text-xs text-slate-400 font-medium">
              Output
              {executionTime !== null && (
                <span className="ml-1.5 text-emerald-400">({executionTime}ms)</span>
              )}
            </span>
            <div className="flex items-center gap-1">
              {outputs.length > 0 && (
                <>
                  <button
                    onClick={handleCopyOutput}
                    className="flex items-center gap-1 px-2 py-1 text-xs text-slate-400 hover:text-white hover:bg-slate-800 rounded transition-colors"
                    title="Copy output"
                  >
                    {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  </button>
                  {onSendToChat && (
                    <button
                      onClick={handleSendToChat}
                      className="flex items-center gap-1 px-2 py-1 text-xs bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded transition-colors"
                      title="Send output to chat"
                    >
                      <MessageSquare className="w-3 h-3" />
                      <span>Chat</span>
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
          
          <div className="flex-1 overflow-auto p-3 min-h-[120px]">
            {outputs.length === 0 && !error ? (
              <div className="text-xs text-slate-600 italic">
                Output will appear here...
              </div>
            ) : (
              <div className="space-y-2">
                {outputs.map((output, i) => (
                  <div 
                    key={i} 
                    className={`text-xs font-mono whitespace-pre-wrap ${
                      output.type === 'error' 
                        ? 'text-red-400' 
                        : output.type === 'result'
                        ? 'text-cyan-300'
                        : 'text-slate-300'
                    }`}
                  >
                    {formatOutput(output)}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="flex items-start gap-2 p-2 m-2 bg-red-500/10 border border-red-500/30 rounded-lg text-xs text-red-300 flex-shrink-0">
              <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <span className="break-all">{error}</span>
            </div>
          )}

          {/* Variables Footer */}
          {variables.length > 0 && (
            <div className="px-3 py-2 border-t border-slate-800 text-xs text-slate-500 flex-shrink-0">
              <span className="font-medium">Vars:</span>{' '}
              {variables.slice(0, 5).join(', ')}
              {variables.length > 5 && ` +${variables.length - 5}`}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

