"use client";
import { useState, useContext, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileJson, Check, AlertCircle, ArrowRight, Eye, EyeOff, History, ChevronDown, ChevronUp } from 'lucide-react';
import { AppContext } from '@/app/page';

export default function GlobalJsonInput({ onComplete }: { onComplete: () => void }) {
  const context = useContext(AppContext);
  const [jsonInput, setJsonInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<any>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showExisting, setShowExisting] = useState(false);
  const [initialized, setInitialized] = useState(false);

  // Load existing JSON from session on mount
  useEffect(() => {
    if (!initialized && context?.globalJson) {
      setJsonInput(JSON.stringify(context.globalJson, null, 2));
      setPreview(context.globalJson);
      setInitialized(true);
    } else if (!initialized) {
      setInitialized(true);
    }
  }, [context?.globalJson, initialized]);

  // Re-sync when session changes
  useEffect(() => {
    if (context?.session.id && context?.globalJson) {
      setJsonInput(JSON.stringify(context.globalJson, null, 2));
      setPreview(context.globalJson);
    } else if (context?.session.id) {
      setJsonInput('');
      setPreview(null);
    }
  }, [context?.session.id]);

  const validateAndParse = (input: string) => {
    try {
      const parsed = JSON.parse(input);
      
      // Basic validation
      if (typeof parsed !== 'object' || parsed === null) {
        return { valid: false, error: 'JSON must be an object', data: null };
      }
      
      // Check for recommended fields
      const warnings: string[] = [];
      if (!parsed.global_importance) warnings.push('Missing global_importance array');
      if (!parsed.model_version) warnings.push('Missing model_version');
      
      return { 
        valid: true, 
        error: null, 
        data: parsed, 
        warnings: warnings.length > 0 ? warnings : null 
      };
    } catch (e: any) {
      return { valid: false, error: `Invalid JSON: ${e.message}`, data: null };
    }
  };

  const handlePreview = () => {
    const result = validateAndParse(jsonInput);
    if (result.valid) {
      setPreview(result.data);
      setError(null);
      setShowPreview(true);
    } else {
      setError(result.error);
      setPreview(null);
    }
  };

  const handleSubmit = () => {
    const result = validateAndParse(jsonInput);
    if (result.valid) {
      context?.setGlobalJson(result.data);
      onComplete();
    } else {
      setError(result.error);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl mb-4 shadow-lg shadow-cyan-500/20">
          <FileJson className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-3xl font-bold text-white mb-2">Paste Global JSON</h2>
        <p className="text-slate-400 max-w-md mx-auto">
          Run the <code className="text-cyan-300 bg-cyan-900/30 px-1.5 py-0.5 rounded">explain_global()</code> function 
          on your machine and paste the output JSON here.
        </p>
      </div>

      <div className="bg-[#0d1117] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
        {/* Editor Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-slate-900/50 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500/80" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
              <div className="w-3 h-3 rounded-full bg-green-500/80" />
            </div>
            <span className="text-xs text-slate-500 font-mono ml-2">global_explanation.json</span>
          </div>
          {jsonInput && (
            <button
              onClick={() => setShowPreview(!showPreview)}
              className="flex items-center gap-2 text-xs text-slate-400 hover:text-white transition-colors"
            >
              {showPreview ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              {showPreview ? 'Hide Preview' : 'Show Preview'}
            </button>
          )}
        </div>

        {/* JSON Input */}
        <div className="relative">
          <textarea
            value={jsonInput}
            onChange={e => {
              setJsonInput(e.target.value);
              setError(null);
            }}
            placeholder={`{
  "model_version": "fraud_detector_v2",
  "generated_at": "2025-01-02T10:00:00Z",
  "global_importance": [
    { "feature_or_group": "transaction_amount", "importance": 0.32, "direction": "positive" },
    { "feature_or_group": "merchant_category", "importance": 0.18, "direction": "negative" }
  ],
  "global_trends": [...],
  "reliability": { "sample_size": 10000, "stability_score": 0.94 },
  "limits": ["Model may underperform on new merchant categories"]
}`}
            className="w-full h-[400px] bg-transparent p-4 text-sm font-mono text-cyan-300 resize-none outline-none placeholder-slate-600"
            spellCheck={false}
          />
        </div>

        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mx-4 mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-300 text-sm"
          >
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {error}
          </motion.div>
        )}

        {/* Preview Panel */}
        {showPreview && preview && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="border-t border-slate-800 bg-slate-900/30 p-4"
          >
            <h4 className="text-sm font-medium text-slate-300 mb-3">Parsed Preview</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <PreviewCard 
                label="Model Version" 
                value={preview.model_version || 'N/A'} 
                color="cyan"
              />
              <PreviewCard 
                label="Features" 
                value={preview.global_importance?.length || 0} 
                color="emerald"
              />
              <PreviewCard 
                label="Sample Size" 
                value={preview.reliability?.sample_size?.toLocaleString() || 'N/A'} 
                color="violet"
              />
              <PreviewCard 
                label="Stability" 
                value={preview.reliability?.stability_score || 'N/A'} 
                color="amber"
              />
            </div>
            
            {preview.global_importance && preview.global_importance.length > 0 && (
              <div className="mt-4">
                <h5 className="text-xs text-slate-500 uppercase tracking-wider mb-2">Top Features</h5>
                <div className="space-y-2">
                  {preview.global_importance.slice(0, 3).map((item: any, idx: number) => (
                    <div key={idx} className="flex items-center gap-3">
                      <span className="text-sm text-slate-300 w-40 truncate">{item.feature_or_group}</span>
                      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${(item.importance || 0) * 100}%` }}
                          transition={{ delay: idx * 0.1 }}
                          className={`h-full ${item.direction === 'positive' ? 'bg-emerald-500' : 'bg-rose-500'}`}
                        />
                      </div>
                      <span className="text-xs text-slate-400 w-12 text-right">
                        {((item.importance || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
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
            className="flex-1 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all shadow-lg shadow-cyan-500/20"
          >
            <Check className="w-4 h-4" />
            Save & Continue
            <ArrowRight className="w-4 h-4 ml-1" />
          </button>
        </div>
      </div>

      {/* Existing JSON Preview */}
      {context?.globalJson && (
        <div className="mt-6 bg-[#0d1117] border border-slate-800 rounded-xl overflow-hidden">
          <button
            onClick={() => setShowExisting(!showExisting)}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-800/30 transition-colors"
          >
            <div className="flex items-center gap-2">
              <History className="w-4 h-4 text-cyan-400" />
              <span className="text-sm font-medium text-white">Current Saved Global JSON</span>
              <span className="text-xs text-emerald-400 bg-emerald-500/20 px-2 py-0.5 rounded-full">
                v{context.globalJson.model_version || 'unknown'}
              </span>
            </div>
            {showExisting ? (
              <ChevronUp className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            )}
          </button>
          
          <AnimatePresence>
            {showExisting && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="p-4 border-t border-slate-800">
                  {/* Quick Stats */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                    <PreviewCard 
                      label="Model Version" 
                      value={context.globalJson.model_version || 'N/A'} 
                      color="cyan"
                    />
                    <PreviewCard 
                      label="Features" 
                      value={context.globalJson.global_importance?.length || 0} 
                      color="emerald"
                    />
                    <PreviewCard 
                      label="Sample Size" 
                      value={context.globalJson.reliability?.sample_size?.toLocaleString() || 'N/A'} 
                      color="violet"
                    />
                    <PreviewCard 
                      label="Generated" 
                      value={context.globalJson.generated_at ? new Date(context.globalJson.generated_at).toLocaleDateString() : 'N/A'} 
                      color="amber"
                    />
                  </div>
                  
                  {/* JSON Preview */}
                  <div className="bg-slate-950 rounded-lg p-3 max-h-[200px] overflow-auto">
                    <pre className="text-xs font-mono text-cyan-300">
                      {JSON.stringify(context.globalJson, null, 2)}
                    </pre>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Context Info */}
      {context?.mlCode && (
        <div className="mt-6 p-4 bg-violet-500/10 border border-violet-500/30 rounded-xl">
          <div className="flex items-center gap-2 text-violet-300 text-sm">
            <Check className="w-4 h-4" />
            <span>Your ML code is saved in context ({context.mlCode.length} characters)</span>
          </div>
        </div>
      )}
    </div>
  );
}

function PreviewCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  const colorClasses: Record<string, string> = {
    cyan: 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300',
    emerald: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300',
    violet: 'bg-violet-500/10 border-violet-500/30 text-violet-300',
    amber: 'bg-amber-500/10 border-amber-500/30 text-amber-300',
  };

  return (
    <div className={`p-3 rounded-lg border ${colorClasses[color]}`}>
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className="font-semibold">{value}</div>
    </div>
  );
}

