"use client";
import { useState, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BookOpen, 
  Plus, 
  Trash2, 
  ChevronDown, 
  ChevronUp,
  Shield,
  Building2,
  AlertTriangle,
  FileCheck,
  Settings,
  Tag,
  Check,
  Upload,
  FileText,
  HelpCircle,
  Copy,
  Download
} from 'lucide-react';
import { AppContext } from '@/app/page';
import { BankGuideline } from '@/lib/storage';

const CATEGORY_CONFIG = {
  regulatory: { label: 'Regulatory', icon: Shield, color: 'text-red-400', bg: 'bg-red-500/20', border: 'border-red-500/30' },
  internal: { label: 'Internal Policy', icon: Building2, color: 'text-blue-400', bg: 'bg-blue-500/20', border: 'border-blue-500/30' },
  risk: { label: 'Risk Management', icon: AlertTriangle, color: 'text-amber-400', bg: 'bg-amber-500/20', border: 'border-amber-500/30' },
  compliance: { label: 'Compliance', icon: FileCheck, color: 'text-emerald-400', bg: 'bg-emerald-500/20', border: 'border-emerald-500/30' },
  operational: { label: 'Operational', icon: Settings, color: 'text-purple-400', bg: 'bg-purple-500/20', border: 'border-purple-500/30' },
  custom: { label: 'Custom', icon: Tag, color: 'text-slate-400', bg: 'bg-slate-500/20', border: 'border-slate-500/30' },
};

const PRIORITY_CONFIG = {
  critical: { label: 'Critical', color: 'text-red-400', bg: 'bg-red-500/20' },
  high: { label: 'High', color: 'text-amber-400', bg: 'bg-amber-500/20' },
  medium: { label: 'Medium', color: 'text-blue-400', bg: 'bg-blue-500/20' },
  low: { label: 'Low', color: 'text-slate-400', bg: 'bg-slate-500/20' },
};

interface GuidelinesInputProps {
  isOpen: boolean;
  onClose: () => void;
}

// Example JSON structure for import
const EXAMPLE_JSON = `[
  {
    "title": "High-Value Transaction Review",
    "category": "regulatory",
    "description": "All transactions above $10,000 must undergo enhanced review per AML regulations.",
    "rules": [
      "Verify source of funds for amounts over $10,000",
      "Cross-check against sanctions list",
      "Document customer relationship history"
    ],
    "source": "AML Compliance Guidelines 2024",
    "priority": "critical"
  },
  {
    "title": "Cross-Border Payment Protocol",
    "category": "risk",
    "description": "Additional verification required for international transfers.",
    "rules": [
      "Verify recipient country is not on restricted list",
      "Confirm purpose of transfer",
      "Check for unusual patterns in transfer history"
    ],
    "source": "Internal Risk Policy v3.2",
    "priority": "high"
  }
]`;

export default function GuidelinesInput({ isOpen, onClose }: GuidelinesInputProps) {
  const context = useContext(AppContext);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showJsonHelp, setShowJsonHelp] = useState(false);
  const [copiedExample, setCopiedExample] = useState(false);
  const [expandedGuideline, setExpandedGuideline] = useState<string | null>(null);
  const [editingRules, setEditingRules] = useState<string>('');
  
  // Form state
  const [title, setTitle] = useState('');
  const [category, setCategory] = useState<BankGuideline['category']>('internal');
  const [description, setDescription] = useState('');
  const [rules, setRules] = useState('');
  const [source, setSource] = useState('');
  const [priority, setPriority] = useState<BankGuideline['priority']>('medium');

  const guidelines = context?.session.guidelines || [];

  const handleAddGuideline = () => {
    if (!title.trim() || !description.trim()) return;

    const newGuideline: BankGuideline = {
      id: `guideline_${Date.now()}`,
      title: title.trim(),
      category,
      description: description.trim(),
      rules: rules.split('\n').filter(r => r.trim()),
      source: source.trim() || undefined,
      priority,
      addedAt: new Date().toISOString(),
    };

    context?.updateSession({
      guidelines: [...guidelines, newGuideline]
    });

    // Reset form
    setTitle('');
    setDescription('');
    setRules('');
    setSource('');
    setCategory('internal');
    setPriority('medium');
    setShowAddForm(false);
  };

  const handleDeleteGuideline = (id: string) => {
    context?.updateSession({
      guidelines: guidelines.filter(g => g.id !== id)
    });
  };

  const handleImportGuidelines = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const imported = JSON.parse(text);
      
      if (Array.isArray(imported)) {
        const newGuidelines = imported.map((g: any) => ({
          id: `guideline_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
          title: g.title || 'Untitled',
          category: g.category || 'custom',
          description: g.description || '',
          rules: g.rules || [],
          source: g.source,
          priority: g.priority || 'medium',
          addedAt: new Date().toISOString(),
        }));
        context?.updateSession({
          guidelines: [...guidelines, ...newGuidelines]
        });
      }
    } catch (err) {
      console.error('Failed to import guidelines:', err);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-[#0d1117] border border-slate-800 rounded-2xl w-full max-w-3xl max-h-[85vh] overflow-hidden shadow-2xl"
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 bg-gradient-to-r from-blue-900/20 to-purple-900/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/20 rounded-lg">
                <BookOpen className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Bank Guidelines</h2>
                <p className="text-sm text-slate-400">Policies, regulations, and decision criteria</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-white transition-colors text-xl"
            >
              ×
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(85vh-180px)]">
          {/* Actions Bar */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowAddForm(!showAddForm)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-white text-sm font-medium transition-colors"
              >
                <Plus className="w-4 h-4" />
                Add Guideline
              </button>
              <label className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 text-sm font-medium transition-colors cursor-pointer">
                <Upload className="w-4 h-4" />
                Import JSON
                <input
                  type="file"
                  accept=".json"
                  onChange={handleImportGuidelines}
                  className="hidden"
                />
              </label>
              <button
                onClick={() => setShowJsonHelp(!showJsonHelp)}
                className="flex items-center gap-1 px-2 py-2 text-slate-500 hover:text-blue-400 transition-colors"
                title="View JSON format"
              >
                <HelpCircle className="w-4 h-4" />
              </button>
            </div>
            <span className="text-sm text-slate-500">
              {guidelines.length} guideline{guidelines.length !== 1 ? 's' : ''} configured
            </span>
          </div>

          {/* JSON Format Help */}
          <AnimatePresence>
            {showJsonHelp && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6 p-4 bg-slate-900/50 border border-blue-500/30 rounded-xl"
              >
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-blue-300 flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    JSON Import Format
                  </h3>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(EXAMPLE_JSON);
                        setCopiedExample(true);
                        setTimeout(() => setCopiedExample(false), 2000);
                      }}
                      className="flex items-center gap-1 px-2 py-1 text-xs bg-slate-800 hover:bg-slate-700 rounded text-slate-300 transition-colors"
                    >
                      {copiedExample ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
                      {copiedExample ? 'Copied!' : 'Copy'}
                    </button>
                    <button
                      onClick={() => {
                        const blob = new Blob([EXAMPLE_JSON], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'guidelines_template.json';
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                      className="flex items-center gap-1 px-2 py-1 text-xs bg-slate-800 hover:bg-slate-700 rounded text-slate-300 transition-colors"
                    >
                      <Download className="w-3 h-3" />
                      Download Template
                    </button>
                    <button
                      onClick={() => setShowJsonHelp(false)}
                      className="text-slate-500 hover:text-white transition-colors"
                    >
                      ×
                    </button>
                  </div>
                </div>

                <div className="text-xs text-slate-400 mb-3">
                  Import a JSON array of guidelines with the following structure:
                </div>

                <pre className="bg-slate-950 p-3 rounded-lg text-xs font-mono text-blue-300 overflow-x-auto max-h-[300px] overflow-y-auto">
                  {EXAMPLE_JSON}
                </pre>

                <div className="mt-3 grid grid-cols-2 gap-4 text-xs">
                  <div>
                    <div className="text-slate-400 mb-1 font-medium">Required Fields:</div>
                    <ul className="text-slate-500 space-y-0.5">
                      <li>• <code className="text-blue-300">title</code> - Guideline name</li>
                      <li>• <code className="text-blue-300">description</code> - What it covers</li>
                    </ul>
                  </div>
                  <div>
                    <div className="text-slate-400 mb-1 font-medium">Optional Fields:</div>
                    <ul className="text-slate-500 space-y-0.5">
                      <li>• <code className="text-blue-300">category</code> - regulatory, internal, risk, compliance, operational, custom</li>
                      <li>• <code className="text-blue-300">rules</code> - Array of specific rule strings</li>
                      <li>• <code className="text-blue-300">source</code> - Document/policy reference</li>
                      <li>• <code className="text-blue-300">priority</code> - critical, high, medium, low</li>
                    </ul>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Add Form */}
          <AnimatePresence>
            {showAddForm && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6 p-4 bg-slate-900/50 border border-slate-800 rounded-xl"
              >
                <h3 className="text-sm font-medium text-white mb-4">Add New Guideline</h3>
                
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Title *</label>
                      <input
                        value={title}
                        onChange={e => setTitle(e.target.value)}
                        placeholder="e.g., High-Value Transaction Review"
                        className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Source</label>
                      <input
                        value={source}
                        onChange={e => setSource(e.target.value)}
                        placeholder="e.g., RBI Circular 2024, Internal Policy v2.1"
                        className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Category</label>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(CATEGORY_CONFIG).map(([key, config]) => {
                          const Icon = config.icon;
                          return (
                            <button
                              key={key}
                              onClick={() => setCategory(key as BankGuideline['category'])}
                              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                                category === key
                                  ? `${config.bg} ${config.color} ${config.border} border`
                                  : 'bg-slate-800 text-slate-400 hover:text-white'
                              }`}
                            >
                              <Icon className="w-3.5 h-3.5" />
                              {config.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Priority</label>
                      <div className="flex gap-2">
                        {Object.entries(PRIORITY_CONFIG).map(([key, config]) => (
                          <button
                            key={key}
                            onClick={() => setPriority(key as BankGuideline['priority'])}
                            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                              priority === key
                                ? `${config.bg} ${config.color}`
                                : 'bg-slate-800 text-slate-400 hover:text-white'
                            }`}
                          >
                            {config.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Description *</label>
                    <textarea
                      value={description}
                      onChange={e => setDescription(e.target.value)}
                      placeholder="Describe when and how this guideline applies..."
                      rows={2}
                      className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 resize-none"
                    />
                  </div>

                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Specific Rules (one per line)</label>
                    <textarea
                      value={rules}
                      onChange={e => setRules(e.target.value)}
                      placeholder={`e.g.,
Transactions > $10,000 require manager approval
Cross-border transactions need enhanced due diligence
Multiple failed attempts trigger account freeze`}
                      rows={3}
                      className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 resize-none font-mono"
                    />
                  </div>

                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => setShowAddForm(false)}
                      className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleAddGuideline}
                      disabled={!title.trim() || !description.trim()}
                      className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:hover:bg-blue-600 rounded-lg text-white text-sm font-medium transition-colors"
                    >
                      <Check className="w-4 h-4" />
                      Add Guideline
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Guidelines List */}
          {guidelines.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="w-12 h-12 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">No Guidelines Added</h3>
              <p className="text-sm text-slate-400 max-w-md mx-auto">
                Add your bank's policies, regulatory requirements, and decision criteria. 
                These will be used to analyze analyst behavior and detect shadow rules.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {guidelines.map((guideline) => {
                const catConfig = CATEGORY_CONFIG[guideline.category];
                const priorityConfig = PRIORITY_CONFIG[guideline.priority || 'medium'];
                const Icon = catConfig.icon;
                const isExpanded = expandedGuideline === guideline.id;

                return (
                  <div
                    key={guideline.id}
                    className={`bg-slate-900/50 border rounded-xl overflow-hidden transition-all ${
                      isExpanded ? 'border-blue-500/50' : 'border-slate-800'
                    }`}
                  >
                    <div
                      className="p-4 cursor-pointer hover:bg-slate-800/30 transition-colors"
                      onClick={() => setExpandedGuideline(isExpanded ? null : guideline.id)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div className={`p-2 rounded-lg ${catConfig.bg}`}>
                            <Icon className={`w-4 h-4 ${catConfig.color}`} />
                          </div>
                          <div>
                            <h4 className="text-sm font-medium text-white">{guideline.title}</h4>
                            <p className="text-xs text-slate-400 mt-0.5 line-clamp-1">{guideline.description}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`text-[10px] px-2 py-0.5 rounded ${priorityConfig.bg} ${priorityConfig.color}`}>
                            {priorityConfig.label}
                          </span>
                          <span className={`text-[10px] px-2 py-0.5 rounded ${catConfig.bg} ${catConfig.color}`}>
                            {catConfig.label}
                          </span>
                          {isExpanded ? (
                            <ChevronUp className="w-4 h-4 text-slate-500" />
                          ) : (
                            <ChevronDown className="w-4 h-4 text-slate-500" />
                          )}
                        </div>
                      </div>
                    </div>

                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          className="border-t border-slate-800"
                        >
                          <div className="p-4 space-y-3">
                            <p className="text-sm text-slate-300">{guideline.description}</p>
                            
                            {guideline.source && (
                              <div className="text-xs text-slate-500">
                                <span className="text-slate-400">Source:</span> {guideline.source}
                              </div>
                            )}

                            {guideline.rules && guideline.rules.length > 0 && (
                              <div>
                                <div className="text-xs text-slate-400 mb-2">Specific Rules:</div>
                                <ul className="space-y-1">
                                  {guideline.rules.map((rule, i) => (
                                    <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                                      <div className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-1.5" />
                                      {rule}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            <div className="flex justify-end pt-2">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteGuideline(guideline.id);
                                }}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                                Remove
                              </button>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-900/30">
          <div className="flex items-center justify-between">
            <div className="text-xs text-slate-500">
              Guidelines are used to analyze analyst decisions and detect shadow rules
            </div>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-white text-sm font-medium transition-colors"
            >
              Done
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

