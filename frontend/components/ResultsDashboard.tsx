"use client";
import React, { useState, useEffect, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart3, CheckCircle, XCircle, AlertTriangle, TrendingUp,
  RefreshCw, ArrowRight, MessageSquare, Filter, ChevronDown,
  ChevronUp, Target, Users, Zap, Settings, TreeDeciduous, Image as ImageIcon
} from 'lucide-react';
import { AppContext } from '@/app/page';

interface Hyperparameters {
  n_estimators: number;        // Number of trees in the forest
  max_depth: number;           // Maximum depth of each tree
  min_samples_split: number;   // Minimum samples to split a node
  min_samples_leaf: number;    // Minimum samples at a leaf
  max_features: string;        // Features to consider for best split
  bootstrap: boolean;          // Use bootstrap samples
  random_state: number;
}

interface AnalysisResult {
  success: boolean;
  error?: string;
  total_samples: number;
  features_used: string[];
  hyperparameters?: Hyperparameters;
  l1_analysis?: {
    column: string;
    metrics: {
      true_positives: number;
      false_positives: number;
      true_negatives: number;
      false_negatives: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      n_classes?: number;
    };
    feature_importance?: Record<string, number>;
    shap_importance?: Record<string, number>;  // Backward compatibility
    importance_summary?: string;
    shap_summary?: string;  // Backward compatibility
  };
  l2_analysis?: {
    column: string;
    metrics: any;
    feature_importance?: Record<string, number>;
    shap_importance?: Record<string, number>;  // Backward compatibility
    importance_summary?: string;
    shap_summary?: string;  // Backward compatibility
  };
  prediction_breakdown: {
    total_cases: number;
    false_positives: number;
    false_negatives: number;
    correct_predictions: number;
    accuracy_rate: number;
  };
  importance_summary: string;
  wrong_predictions: Array<{
    index: number;
    case_id: string;
    case_type: 'true_positive' | 'false_positive' | 'true_negative' | 'false_negative' | 'escalated_fraud' | 'escalated_legit';
    l1_decision: string;
    true_fraud: string;
    is_correct?: boolean;
    is_wrong?: boolean;
    is_escalated?: boolean;
    [key: string]: any;
  }>;
  // Tree visualization
  l1_tree_image?: string;
  l2_tree_image?: string;
  l1_decision_rules?: string[];
  l2_decision_rules?: string[];
  l1_tree_text?: string;
  l2_tree_text?: string;
  l1_tree_structure?: any[];
  l2_tree_structure?: any[];
  analyzed_at: string;
}

interface ResultsDashboardProps {
  onChatWithCase: (cases: any[]) => void;
}

const DEFAULT_HYPERPARAMS: Hyperparameters = {
  n_estimators: 100,
  max_depth: 5,
  min_samples_split: 10,
  min_samples_leaf: 5,
  max_features: "sqrt",
  bootstrap: true,
  random_state: 42,
};

// Hyperparameter Modal Component
function HyperparamModal({
  show,
  hyperparams,
  setHyperparams,
  onClose,
  onRunAnalysis
}: {
  show: boolean;
  hyperparams: Hyperparameters;
  setHyperparams: (h: Hyperparameters) => void;
  onClose: () => void;
  onRunAnalysis: () => void;
}) {
  if (!show) return null;
  
  return (
    <div 
      className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        onClick={e => e.stopPropagation()}
        className="bg-[#0d1117] border border-slate-800 rounded-2xl p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto"
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-xl font-bold text-white flex items-center gap-2">
              <Settings className="w-5 h-5 text-violet-400" />
              Random Forest Configuration
            </h3>
            <p className="text-xs text-slate-500 mt-1">Configure Random Forest hyperparameters for analysis</p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white p-2 hover:bg-slate-800 rounded-lg">
            <XCircle className="w-5 h-5" />
          </button>
        </div>

        {/* Model Info */}
        <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-4 mb-6">
          <h4 className="text-sm font-medium text-white mb-2">Model Used</h4>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded">
              <div className="font-medium text-emerald-300">Random Forest Classifier</div>
              <div className="text-slate-400">Ensemble of decision trees for predictions</div>
            </div>
            <div className="p-2 bg-blue-500/10 border border-blue-500/20 rounded">
              <div className="font-medium text-blue-300">Decision Tree Visualization</div>
              <div className="text-slate-400">Single tree for interpretable rules</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Number of Trees (n_estimators)</label>
            <input
              type="number"
              value={hyperparams.n_estimators}
              onChange={e => setHyperparams({ ...hyperparams, n_estimators: parseInt(e.target.value) || 100 })}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
              min={10}
              max={500}
            />
            <p className="text-xs text-slate-500 mt-1">More trees = more accurate but slower (10-500)</p>
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Max Depth</label>
            <input
              type="number"
              value={hyperparams.max_depth}
              onChange={e => setHyperparams({ ...hyperparams, max_depth: parseInt(e.target.value) || 5 })}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
              min={1}
              max={20}
            />
            <p className="text-xs text-slate-500 mt-1">Controls tree complexity. Lower = simpler trees (3-10)</p>
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Min Samples to Split</label>
            <input
              type="number"
              value={hyperparams.min_samples_split}
              onChange={e => setHyperparams({ ...hyperparams, min_samples_split: parseInt(e.target.value) || 10 })}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
              min={2}
              max={50}
            />
            <p className="text-xs text-slate-500 mt-1">Minimum samples required to split a node (2-50)</p>
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Min Samples at Leaf</label>
            <input
              type="number"
              value={hyperparams.min_samples_leaf}
              onChange={e => setHyperparams({ ...hyperparams, min_samples_leaf: parseInt(e.target.value) || 5 })}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
              min={1}
              max={20}
            />
            <p className="text-xs text-slate-500 mt-1">Minimum samples at a leaf node (1-20)</p>
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Max Features</label>
            <select
              value={hyperparams.max_features}
              onChange={e => setHyperparams({ ...hyperparams, max_features: e.target.value })}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
            >
              <option value="sqrt">sqrt (recommended)</option>
              <option value="log2">log2</option>
              <option value="auto">auto</option>
            </select>
            <p className="text-xs text-slate-500 mt-1">Features to consider for best split</p>
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Bootstrap</label>
            <select
              value={hyperparams.bootstrap ? "true" : "false"}
              onChange={e => setHyperparams({ ...hyperparams, bootstrap: e.target.value === "true" })}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
            >
              <option value="true">Yes (recommended)</option>
              <option value="false">No</option>
            </select>
            <p className="text-xs text-slate-500 mt-1">Use bootstrap samples when building trees</p>
          </div>

          <div className="col-span-2">
            <label className="block text-sm text-slate-400 mb-1">Random State (Seed)</label>
            <input
              type="number"
              value={hyperparams.random_state}
              onChange={e => setHyperparams({ ...hyperparams, random_state: parseInt(e.target.value) || 42 })}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
            />
            <p className="text-xs text-slate-500 mt-1">Seed for reproducibility</p>
          </div>
        </div>

        <div className="flex justify-between">
          <button
            onClick={() => setHyperparams(DEFAULT_HYPERPARAMS)}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300"
          >
            Reset to Defaults
          </button>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300"
            >
              Cancel
            </button>
            <button
              onClick={onRunAnalysis}
              className="px-4 py-2 bg-violet-600 hover:bg-violet-500 rounded-lg text-sm text-white font-medium flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Apply & Run Analysis
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default function ResultsDashboard({ onChatWithCase }: ResultsDashboardProps) {
  const context = useContext(AppContext);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [selectedTab, setSelectedTab] = useState<'all' | 'correct' | 'wrong' | 'escalated' | 'true_positives' | 'false_positives' | 'true_negatives' | 'false_negatives'>('all');
  const [expandedCase, setExpandedCase] = useState<number | null>(null);
  const [selectedCases, setSelectedCases] = useState<Set<number>>(new Set());
  
  // Hyperparameters state
  const [hyperparams, setHyperparams] = useState<Hyperparameters>(DEFAULT_HYPERPARAMS);
  const [showHyperparamModal, setShowHyperparamModal] = useState(false);
  const [showTreeModal, setShowTreeModal] = useState<'l1' | 'l2' | null>(null);
  const [progressMessage, setProgressMessage] = useState<string>('');
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(20);
  const itemsPerPageOptions = [10, 20, 50, 100];

  // Load or run analysis
  useEffect(() => {
    if (context?.session.id && context?.session.hasCsvData) {
      loadOrRunAnalysis();
    }
  }, [context?.session.id, context?.session.hasCsvData]);

  const loadOrRunAnalysis = async () => {
    if (!context?.session.id) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // First try to get existing analysis
      const existingRes = await fetch(`http://localhost:8000/sessions/${context.session.id}/analysis`);
      
      if (existingRes.ok) {
        const data = await existingRes.json();
        setAnalysisResult(data);
      } else {
        // Run new analysis
        await runAnalysis();
      }
    } catch (e) {
      // Run new analysis
      await runAnalysis();
    } finally {
      setLoading(false);
    }
  };

  const runAnalysis = async () => {
    if (!context?.session.id) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // First, set hyperparameters
      await fetch('http://localhost:8000/analysis/hyperparameters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(hyperparams),
      });
      
      // Then run analysis
      setProgressMessage('Training Random Forest model...');
      const res = await fetch(`http://localhost:8000/sessions/${context.session.id}/analyze`, {
        method: 'POST',
      });
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }
      
      setProgressMessage('Processing results...');
      const data = await res.json();
      setAnalysisResult(data);
      
      // Update hyperparams from result
      if (data.hyperparameters) {
        setHyperparams(data.hyperparameters);
      }
      
      // Save to context
      context?.updateSession({ analysisResult: data });
      
    } catch (e: any) {
      setError(e.message || 'Failed to run analysis');
    } finally {
      setLoading(false);
      setProgressMessage('');
    }
  };
  
  // Load hyperparameters on mount
  useEffect(() => {
    const loadHyperparams = async () => {
      try {
        const res = await fetch('http://localhost:8000/analysis/hyperparameters');
        if (res.ok) {
          const data = await res.json();
          setHyperparams(data);
        }
      } catch (e) {
        console.log('Using default hyperparameters');
      }
    };
    loadHyperparams();
  }, []);

  const toggleCaseSelection = (index: number) => {
    const newSelected = new Set(selectedCases);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedCases(newSelected);
  };

  const handleChatWithSelected = () => {
    if (selectedCases.size === 0 || !analysisResult) return;
    
    // Get all filtered predictions and filter by selected indices
    const allFiltered = getFilteredPredictions();
    const cases = Array.from(selectedCases).map(idx => allFiltered[idx]).filter(Boolean);
    onChatWithCase(cases);
  };
  
  // Reset page when tab changes
  useEffect(() => {
    setCurrentPage(1);
    setSelectedCases(new Set()); // Clear selection when changing tabs
  }, [selectedTab]);

  const getFilteredPredictions = () => {
    if (!analysisResult?.wrong_predictions) return [];
    
    switch (selectedTab) {
      case 'all':
        return analysisResult.wrong_predictions;
      case 'correct':
        return analysisResult.wrong_predictions.filter(p => p.is_correct);
      case 'wrong':
        return analysisResult.wrong_predictions.filter(p => p.is_wrong);
      case 'false_positives':
        return analysisResult.wrong_predictions.filter(p => p.case_type === 'false_positive');
      case 'false_negatives':
        return analysisResult.wrong_predictions.filter(p => p.case_type === 'false_negative');
      case 'escalated':
        return analysisResult.wrong_predictions.filter(p => p.is_escalated);
      case 'true_positives':
        return analysisResult.wrong_predictions.filter(p => p.case_type === 'true_positive');
      case 'true_negatives':
        return analysisResult.wrong_predictions.filter(p => p.case_type === 'true_negative');
      default:
        return analysisResult.wrong_predictions;
    }
  };
  
  // Get paginated predictions
  const getPaginatedPredictions = () => {
    const filtered = getFilteredPredictions();
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return filtered.slice(startIndex, endIndex);
  };
  
  // Calculate total pages
  const getTotalPages = () => {
    const filtered = getFilteredPredictions();
    return Math.ceil(filtered.length / itemsPerPage);
  };
  
  // Calculate counts for each category
  const getCategoryCounts = () => {
    if (!analysisResult?.wrong_predictions) return {};
    const predictions = analysisResult.wrong_predictions;
    return {
      all: predictions.length,
      correct: predictions.filter(p => p.is_correct).length,
      wrong: predictions.filter(p => p.is_wrong).length,
      true_positives: predictions.filter(p => p.case_type === 'true_positive').length,
      false_positives: predictions.filter(p => p.case_type === 'false_positive').length,
      true_negatives: predictions.filter(p => p.case_type === 'true_negative').length,
      false_negatives: predictions.filter(p => p.case_type === 'false_negative').length,
      escalated: predictions.filter(p => p.is_escalated).length,
      escalated_fraud: predictions.filter(p => p.case_type === 'escalated_fraud').length,
      escalated_legit: predictions.filter(p => p.case_type === 'escalated_legit').length,
    };
  };

  const hasData = context?.session.hasCsvData;

  if (!hasData) {
    return (
      <div className="max-w-4xl mx-auto text-center py-12">
        <AlertTriangle className="w-16 h-16 text-amber-400 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-white mb-2">No Data Available</h2>
        <p className="text-slate-400">
          Please upload your fraud alert data in the Data step first.
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto text-center py-12">
        <RefreshCw className="w-16 h-16 text-violet-400 mx-auto mb-4 animate-spin" />
        <h2 className="text-2xl font-bold text-white mb-2">Analyzing Data...</h2>
        <p className="text-slate-400 mb-4">
          {progressMessage || 'Training Random Forest models...'}
        </p>
        
        {/* Progress Steps */}
        <div className="max-w-md mx-auto mt-6">
          <div className="space-y-3">
            <div className="flex items-center gap-3 text-left">
              <div className="w-6 h-6 bg-violet-500 rounded-full flex items-center justify-center">
                <CheckCircle className="w-4 h-4 text-white" />
              </div>
              <span className="text-slate-300">Setting hyperparameters</span>
            </div>
            <div className="flex items-center gap-3 text-left">
              <div className="w-6 h-6 bg-violet-500/50 rounded-full flex items-center justify-center animate-pulse">
                <RefreshCw className="w-4 h-4 text-white animate-spin" />
              </div>
              <span className="text-slate-300">Training Random Forest on L1 decisions</span>
            </div>
            <div className="flex items-center gap-3 text-left opacity-50">
              <div className="w-6 h-6 bg-slate-700 rounded-full" />
              <span className="text-slate-500">Calculating feature importance</span>
            </div>
            <div className="flex items-center gap-3 text-left opacity-50">
              <div className="w-6 h-6 bg-slate-700 rounded-full" />
              <span className="text-slate-500">Building decision tree visualization</span>
            </div>
            <div className="flex items-center gap-3 text-left opacity-50">
              <div className="w-6 h-6 bg-slate-700 rounded-full" />
              <span className="text-slate-500">Identifying wrong predictions</span>
            </div>
          </div>
        </div>

        {/* Hyperparameters Used */}
        <div className="mt-8 p-4 bg-slate-900/50 border border-slate-800 rounded-xl max-w-md mx-auto">
          <h4 className="text-sm font-medium text-slate-400 mb-2">Current Hyperparameters</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-left"><span className="text-slate-500">Trees:</span> <span className="text-white">{hyperparams.n_estimators}</span></div>
            <div className="text-left"><span className="text-slate-500">Max Depth:</span> <span className="text-white">{hyperparams.max_depth}</span></div>
            <div className="text-left"><span className="text-slate-500">Min Split:</span> <span className="text-white">{hyperparams.min_samples_split}</span></div>
            <div className="text-left"><span className="text-slate-500">Min Leaf:</span> <span className="text-white">{hyperparams.min_samples_leaf}</span></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto text-center py-12">
        <XCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-white mb-2">Analysis Failed</h2>
        <p className="text-red-300 mb-4">{error}</p>
        <button
          onClick={runAnalysis}
          className="px-6 py-3 bg-violet-600 hover:bg-violet-500 rounded-xl text-white font-medium"
        >
          Retry Analysis
        </button>
      </div>
    );
  }

  if (!analysisResult) {
    return (
      <div className="max-w-4xl mx-auto text-center py-12">
        <BarChart3 className="w-16 h-16 text-slate-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-white mb-2">Ready to Analyze</h2>
        <p className="text-slate-400 mb-6">
          Run Random Forest analysis to understand L1 and L2 analyst decisions.
        </p>
        
        {/* Hyperparameter Preview */}
        <div className="max-w-md mx-auto mb-6 p-4 bg-slate-900/50 border border-slate-800 rounded-xl">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-slate-400">Random Forest Settings</h4>
            <button
              onClick={() => setShowHyperparamModal(true)}
              className="text-xs text-violet-400 hover:text-violet-300 flex items-center gap-1"
            >
              <Settings className="w-3 h-3" />
              Configure
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-left"><span className="text-slate-500">Trees:</span> <span className="text-white">{hyperparams.n_estimators}</span></div>
            <div className="text-left"><span className="text-slate-500">Max Depth:</span> <span className="text-white">{hyperparams.max_depth}</span></div>
            <div className="text-left"><span className="text-slate-500">Min Split:</span> <span className="text-white">{hyperparams.min_samples_split}</span></div>
            <div className="text-left"><span className="text-slate-500">Min Leaf:</span> <span className="text-white">{hyperparams.min_samples_leaf}</span></div>
          </div>
        </div>

        <div className="flex items-center justify-center gap-3">
          <button
            onClick={() => setShowHyperparamModal(true)}
            className="px-4 py-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl text-slate-300 font-medium flex items-center gap-2"
          >
            <Settings className="w-5 h-5" />
            Settings
          </button>
          <button
            onClick={runAnalysis}
            className="px-6 py-3 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 rounded-xl text-white font-medium flex items-center gap-2"
          >
            <Zap className="w-5 h-5" />
            Run Analysis
          </button>
        </div>

        {/* Hyperparameters Modal - Render here too for initial state */}
        <HyperparamModal 
          show={showHyperparamModal}
          hyperparams={hyperparams}
          setHyperparams={setHyperparams}
          onClose={() => setShowHyperparamModal(false)}
          onRunAnalysis={() => {
            setShowHyperparamModal(false);
            runAnalysis();
          }}
        />
      </div>
    );
  }

  const { prediction_breakdown, l1_analysis, l2_analysis } = analysisResult;
  const filteredPredictions = getFilteredPredictions();

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl mb-4 shadow-lg shadow-violet-500/20">
          <BarChart3 className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-3xl font-bold text-white mb-2">Analysis Results</h2>
        <p className="text-slate-400 max-w-lg mx-auto">
          ML analysis of {analysisResult.total_samples.toLocaleString()} fraud alert cases
        </p>
        <div className="flex items-center justify-center gap-4 mt-3">
          <span className="px-3 py-1 bg-emerald-500/20 border border-emerald-500/30 rounded-full text-xs text-emerald-300">
            Random Forest
          </span>
          <span className="px-3 py-1 bg-blue-500/20 border border-blue-500/30 rounded-full text-xs text-blue-300">
            Decision Tree Visualization
          </span>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5 text-emerald-400" />
            <span className="text-sm text-slate-400">Correct</span>
          </div>
          <div className="text-2xl font-bold text-emerald-400">
            {prediction_breakdown.correct_predictions.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500">
            {(prediction_breakdown.accuracy_rate * 100).toFixed(1)}% accuracy
          </div>
        </div>

        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <XCircle className="w-5 h-5 text-red-400" />
            <span className="text-sm text-slate-400">False Positives</span>
          </div>
          <div className="text-2xl font-bold text-red-400">
            {prediction_breakdown.false_positives.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500">
            Flagged fraud but was legit
          </div>
        </div>

        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            <span className="text-sm text-slate-400">False Negatives</span>
          </div>
          <div className="text-2xl font-bold text-amber-400">
            {prediction_breakdown.false_negatives.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500">
            Cleared but was fraud
          </div>
        </div>

        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-5 h-5 text-violet-400" />
            <span className="text-sm text-slate-400">Total Cases</span>
          </div>
          <div className="text-2xl font-bold text-violet-400">
            {prediction_breakdown.total_cases.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500">
            Analyzed
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap justify-center gap-4 mb-6">
        <button
          onClick={() => setShowHyperparamModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-sm text-slate-300"
        >
          <Settings className="w-4 h-4" />
          Hyperparameters
        </button>
        {/* Always show L1 tree button if we have L1 analysis */}
        {l1_analysis && (
          <button
            onClick={() => setShowTreeModal('l1')}
            className="flex items-center gap-2 px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/40 rounded-lg text-sm text-blue-300"
          >
            <TreeDeciduous className="w-4 h-4" />
            View L1 Decision Tree
          </button>
        )}
        {/* Show L2 tree button if we have L2 analysis */}
        {l2_analysis && (
          <button
            onClick={() => setShowTreeModal('l2')}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 rounded-lg text-sm text-emerald-300"
          >
            <TreeDeciduous className="w-4 h-4" />
            View L2 Decision Tree
          </button>
        )}
      </div>

      {/* L1 Analysis Summary */}
      {l1_analysis && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Users className="w-5 h-5 text-blue-400" />
                L1 Analyst Decision Analysis
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Predicting L1 decisions • Excludes all L2 data
              </p>
            </div>
            <div className="flex gap-2">
              <span className="px-2 py-1 bg-emerald-500/20 border border-emerald-500/30 rounded text-xs text-emerald-300">
                Random Forest
              </span>
            </div>
          </div>
          
          {/* Model Performance Metrics */}
          <div className="grid grid-cols-4 gap-4 mb-4">
            <div className="text-center p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xl font-bold text-white">
                {l1_analysis.metrics.accuracy != null ? `${(l1_analysis.metrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
              </div>
              <div className="text-xs text-slate-500">Accuracy</div>
            </div>
            <div className="text-center p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xl font-bold text-white">
                {l1_analysis.metrics.precision != null ? `${(l1_analysis.metrics.precision * 100).toFixed(1)}%` : 'N/A'}
              </div>
              <div className="text-xs text-slate-500">Precision</div>
            </div>
            <div className="text-center p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xl font-bold text-white">
                {l1_analysis.metrics.recall != null ? `${(l1_analysis.metrics.recall * 100).toFixed(1)}%` : 'N/A'}
              </div>
              <div className="text-xs text-slate-500">Recall</div>
            </div>
            <div className="text-center p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xl font-bold text-white">
                {l1_analysis.metrics.f1_score != null ? `${(l1_analysis.metrics.f1_score * 100).toFixed(1)}%` : 'N/A'}
              </div>
              <div className="text-xs text-slate-500">F1 Score</div>
            </div>
          </div>
          
          {/* Multi-class indicator */}
          {l1_analysis.metrics.n_classes && l1_analysis.metrics.n_classes > 2 && (
            <div className="text-xs text-amber-400 mb-2 text-center">
              ⚠️ Multi-class classification ({l1_analysis.metrics.n_classes} classes) - using weighted averages
            </div>
          )}

          {/* Confusion Matrix Summary */}
          <div className="grid grid-cols-4 gap-2 mb-4 text-center">
            <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded">
              <div className="text-sm font-bold text-emerald-400">{l1_analysis.metrics.true_positives}</div>
              <div className="text-[10px] text-slate-500">True Pos</div>
            </div>
            <div className="p-2 bg-red-500/10 border border-red-500/20 rounded">
              <div className="text-sm font-bold text-red-400">{l1_analysis.metrics.false_positives}</div>
              <div className="text-[10px] text-slate-500">False Pos</div>
            </div>
            <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded">
              <div className="text-sm font-bold text-emerald-400">{l1_analysis.metrics.true_negatives}</div>
              <div className="text-[10px] text-slate-500">True Neg</div>
            </div>
            <div className="p-2 bg-amber-500/10 border border-amber-500/20 rounded">
              <div className="text-sm font-bold text-amber-400">{l1_analysis.metrics.false_negatives}</div>
              <div className="text-[10px] text-slate-500">False Neg</div>
            </div>
          </div>

          {/* View Decision Tree Button */}
          {(analysisResult.l1_tree_image || analysisResult.l1_tree_text) && (
            <div className="mt-4 mb-4">
              <button
                onClick={() => setShowTreeModal('l1')}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 rounded-lg text-emerald-300 font-medium transition-colors"
              >
                <TreeDeciduous className="w-5 h-5" />
                View L1 Decision Tree
                {analysisResult.l1_decision_rules && (
                  <span className="text-xs text-emerald-400/70">({analysisResult.l1_decision_rules.length} rules)</span>
                )}
              </button>
            </div>
          )}

          {/* Top Features */}
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-slate-400">Top Influencing Factors</h4>
              <span className="text-xs text-slate-500">Random Forest feature importance</span>
            </div>
            <div className="space-y-2">
              {Object.entries(l1_analysis.feature_importance || l1_analysis.shap_importance || {}).slice(0, 5).map(([feature, importance], idx) => (
                <div key={feature} className="flex items-center gap-3">
                  <span className="text-xs text-slate-500 w-4">{idx + 1}</span>
                  <div className="flex-1 bg-slate-800 rounded-full h-2.5">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-cyan-400 h-2.5 rounded-full transition-all"
                      style={{ width: `${Math.min(100, (importance as number) * 100)}%` }}
                    />
                  </div>
                  <span className="text-sm text-white font-mono truncate w-44">{feature}</span>
                  <span className="text-xs text-slate-500 w-12 text-right">{((importance as number) * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* L2 Analysis Summary */}
      {l2_analysis && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Users className="w-5 h-5 text-emerald-400" />
                L2 Analyst Decision Analysis
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Predicting L2 decisions • <span className="text-emerald-400">Includes L1 decision as feature</span>
              </p>
            </div>
            <div className="flex gap-2">
              <span className="px-2 py-1 bg-emerald-500/20 border border-emerald-500/30 rounded text-xs text-emerald-300">
                Random Forest
              </span>
              <span className="px-2 py-1 bg-blue-500/20 border border-blue-500/30 rounded text-xs text-blue-300">
                Uses L1
              </span>
            </div>
          </div>
          
          {/* Model Performance Metrics */}
          <div className="grid grid-cols-4 gap-4 mb-4">
            <div className="text-center p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xl font-bold text-white">{(l2_analysis.metrics.accuracy * 100).toFixed(1)}%</div>
              <div className="text-xs text-slate-500">Accuracy</div>
            </div>
            <div className="text-center p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xl font-bold text-white">{(l2_analysis.metrics.precision * 100).toFixed(1)}%</div>
              <div className="text-xs text-slate-500">Precision</div>
            </div>
            <div className="text-center p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xl font-bold text-white">{(l2_analysis.metrics.recall * 100).toFixed(1)}%</div>
              <div className="text-xs text-slate-500">Recall</div>
            </div>
            <div className="text-center p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xl font-bold text-white">{(l2_analysis.metrics.f1_score * 100).toFixed(1)}%</div>
              <div className="text-xs text-slate-500">F1 Score</div>
            </div>
          </div>

          {/* View Decision Tree Button */}
          {(analysisResult.l2_tree_image || analysisResult.l2_tree_text) && (
            <div className="mt-4 mb-4">
              <button
                onClick={() => setShowTreeModal('l2')}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 rounded-lg text-blue-300 font-medium transition-colors"
              >
                <TreeDeciduous className="w-5 h-5" />
                View L2 Decision Tree
                {analysisResult.l2_decision_rules && (
                  <span className="text-xs text-blue-400/70">({analysisResult.l2_decision_rules.length} rules)</span>
                )}
              </button>
            </div>
          )}

          {/* Top Features */}
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-slate-400">Top Influencing Factors</h4>
              <span className="text-xs text-slate-500">Random Forest feature importance</span>
            </div>
            <div className="space-y-2">
              {Object.entries(l2_analysis.feature_importance || l2_analysis.shap_importance || {}).slice(0, 5).map(([feature, importance], idx) => (
                <div key={feature} className="flex items-center gap-3">
                  <span className="text-xs text-slate-500 w-4">{idx + 1}</span>
                  <div className="flex-1 bg-slate-800 rounded-full h-2.5">
                    <div 
                      className="bg-gradient-to-r from-emerald-500 to-teal-400 h-2.5 rounded-full transition-all"
                      style={{ width: `${Math.min(100, (importance as number) * 100)}%` }}
                    />
                  </div>
                  <span className={`text-sm font-mono truncate w-44 ${feature.toLowerCase().includes('l1') ? 'text-yellow-300' : 'text-white'}`}>
                    {feature}
                    {feature.toLowerCase().includes('l1') && <span className="text-[10px] ml-1">(L1)</span>}
                  </span>
                  <span className="text-xs text-slate-500 w-12 text-right">{((importance as number) * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Transaction Classifications Section */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
        {/* Header with explanation */}
        <div className="p-4 border-b border-slate-800 bg-slate-800/30">
          <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            Transaction Classifications
          </h3>
          <p className="text-sm text-slate-400 mb-3">
            All transactions classified by L1 analyst decision vs actual fraud status.
            Select cases to chat with AI about patterns and decisions.
          </p>
          <div className="flex flex-wrap gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-emerald-500/50"></div>
              <span className="text-slate-400">
                <strong className="text-emerald-300">TP</strong>: Block → Fraud ✓
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-red-500/50"></div>
              <span className="text-slate-400">
                <strong className="text-red-300">FP</strong>: Block → Legit ✗
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-emerald-500/50"></div>
              <span className="text-slate-400">
                <strong className="text-emerald-300">TN</strong>: Release → Legit ✓
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-amber-500/50"></div>
              <span className="text-slate-400">
                <strong className="text-amber-300">FN</strong>: Release → Fraud ✗
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-blue-500/50"></div>
              <span className="text-slate-400">
                <strong className="text-blue-300">ESC</strong>: Escalated
              </span>
            </div>
          </div>
        </div>
        
        {/* Tabs - All Classifications */}
        <div className="flex flex-wrap border-b border-slate-800">
          <button
            onClick={() => setSelectedTab('all')}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              selectedTab === 'all' 
                ? 'text-white bg-slate-800/50 border-b-2 border-violet-500' 
                : 'text-slate-400 hover:text-white'
            }`}
          >
            📋 All ({getCategoryCounts().all || 0})
          </button>
          <button
            onClick={() => setSelectedTab('correct')}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              selectedTab === 'correct' 
                ? 'text-white bg-slate-800/50 border-b-2 border-emerald-500' 
                : 'text-slate-400 hover:text-white'
            }`}
          >
            ✅ Correct ({getCategoryCounts().correct || 0})
          </button>
          <button
            onClick={() => setSelectedTab('wrong')}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              selectedTab === 'wrong' 
                ? 'text-white bg-slate-800/50 border-b-2 border-red-500' 
                : 'text-slate-400 hover:text-white'
            }`}
          >
            ❌ Wrong ({getCategoryCounts().wrong || 0})
          </button>
          <button
            onClick={() => setSelectedTab('escalated')}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              selectedTab === 'escalated' 
                ? 'text-white bg-slate-800/50 border-b-2 border-blue-500' 
                : 'text-slate-400 hover:text-white'
            }`}
          >
            🔄 Escalated ({getCategoryCounts().escalated || 0})
          </button>
        </div>
        
        {/* Sub-tabs for detailed breakdown */}
        <div className="flex flex-wrap bg-slate-900/50 border-b border-slate-800 text-xs">
          <button
            onClick={() => setSelectedTab('true_positives')}
            className={`px-3 py-2 transition-colors ${
              selectedTab === 'true_positives' 
                ? 'text-emerald-300 bg-emerald-500/20' 
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            TP: Block→Fraud ({getCategoryCounts().true_positives || 0})
          </button>
          <button
            onClick={() => setSelectedTab('false_positives')}
            className={`px-3 py-2 transition-colors ${
              selectedTab === 'false_positives' 
                ? 'text-red-300 bg-red-500/20' 
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            FP: Block→Legit ({getCategoryCounts().false_positives || 0})
          </button>
          <button
            onClick={() => setSelectedTab('true_negatives')}
            className={`px-3 py-2 transition-colors ${
              selectedTab === 'true_negatives' 
                ? 'text-emerald-300 bg-emerald-500/20' 
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            TN: Release→Legit ({getCategoryCounts().true_negatives || 0})
          </button>
          <button
            onClick={() => setSelectedTab('false_negatives')}
            className={`px-3 py-2 transition-colors ${
              selectedTab === 'false_negatives' 
                ? 'text-amber-300 bg-amber-500/20' 
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            FN: Release→Fraud ({getCategoryCounts().false_negatives || 0})
          </button>
        </div>

        {/* Action Bar with Selection + Pagination Controls */}
        <div className="px-4 py-3 bg-slate-800/30 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
          {/* Left: Selection info and quick select */}
          <div className="flex items-center gap-3">
            {/* Quick select buttons */}
            <div className="flex items-center gap-1 border-r border-slate-700 pr-3">
              <button
                onClick={() => {
                  const filtered = getFilteredPredictions();
                  const newSelected = new Set<number>();
                  for (let i = 0; i < Math.min(10, filtered.length); i++) {
                    newSelected.add(i);
                  }
                  setSelectedCases(newSelected);
                  setCurrentPage(1);
                }}
                className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs text-white"
                title="Select first 10 transactions"
              >
                Select 10
              </button>
              <button
                onClick={() => {
                  const filtered = getFilteredPredictions();
                  const newSelected = new Set<number>();
                  for (let i = 0; i < Math.min(20, filtered.length); i++) {
                    newSelected.add(i);
                  }
                  setSelectedCases(newSelected);
                  setCurrentPage(1);
                }}
                className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs text-white"
                title="Select first 20 transactions"
              >
                Select 20
              </button>
              <button
                onClick={() => {
                  const filtered = getFilteredPredictions();
                  const newSelected = new Set<number>();
                  for (let i = 0; i < filtered.length; i++) {
                    newSelected.add(i);
                  }
                  setSelectedCases(newSelected);
                }}
                className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs text-white"
                title="Select all transactions"
              >
                All
              </button>
            </div>
            
            {/* Selection status */}
            {selectedCases.size > 0 ? (
              <>
                <span className="text-sm text-violet-300 font-medium">
                  ✓ {selectedCases.size} selected
                </span>
                <button
                  onClick={handleChatWithSelected}
                  className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 rounded-lg text-sm text-white font-medium"
                >
                  <MessageSquare className="w-4 h-4" />
                  Chat About Selected
                </button>
                <button
                  onClick={() => setSelectedCases(new Set())}
                  className="text-sm text-slate-400 hover:text-white px-2 py-1"
                >
                  Clear
                </button>
              </>
            ) : (
              <span className="text-sm text-slate-500">
                Click rows or use buttons to select
              </span>
            )}
          </div>
          
          {/* Right: Pagination controls */}
          <div className="flex items-center gap-4">
            {/* Items per page */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500">Show:</span>
              <select
                value={itemsPerPage}
                onChange={(e) => {
                  setItemsPerPage(Number(e.target.value));
                  setCurrentPage(1);
                }}
                className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm text-white"
              >
                {itemsPerPageOptions.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>
            
            {/* Page navigation */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-2 py-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:hover:bg-slate-800 rounded text-sm text-white"
              >
                ←
              </button>
              <span className="text-sm text-slate-400">
                Page {currentPage} of {getTotalPages() || 1}
              </span>
              <button
                onClick={() => setCurrentPage(Math.min(getTotalPages(), currentPage + 1))}
                disabled={currentPage >= getTotalPages()}
                className="px-2 py-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:hover:bg-slate-800 rounded text-sm text-white"
              >
                →
              </button>
            </div>
            
            {/* Total count */}
            <span className="text-xs text-slate-500">
              ({getFilteredPredictions().length} total)
            </span>
          </div>
        </div>

        {/* Cases List - Paginated */}
        <div className="max-h-[600px] overflow-y-auto">
          {getPaginatedPredictions().length === 0 ? (
            <div className="text-center py-12 text-slate-500">
              <CheckCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No transactions in this category</p>
            </div>
          ) : (
            <div className="divide-y divide-slate-800/50">
              {getPaginatedPredictions().map((prediction, pageIdx) => {
                // Calculate the actual index in the filtered list for selection
                const actualIdx = (currentPage - 1) * itemsPerPage + pageIdx;
                return (
                <div 
                  key={prediction.index} 
                  className={`p-4 hover:bg-slate-800/30 transition-colors cursor-pointer ${
                    selectedCases.has(actualIdx) ? 'bg-violet-500/10 border-l-4 border-violet-500' : ''
                  }`}
                  onClick={() => toggleCaseSelection(actualIdx)}
                >
                  <div className="flex items-center gap-4">
                    {/* Checkbox */}
                    <div
                      className={`w-6 h-6 rounded border-2 flex items-center justify-center flex-shrink-0 ${
                        selectedCases.has(actualIdx) 
                          ? 'bg-violet-500 border-violet-500' 
                          : 'border-slate-600 hover:border-slate-500'
                      }`}
                    >
                      {selectedCases.has(actualIdx) && <CheckCircle className="w-4 h-4 text-white" />}
                    </div>

                    {/* Case Type Badge with better explanation */}
                    <div className={`px-3 py-1.5 rounded text-xs font-bold min-w-[80px] text-center ${
                      prediction.case_type === 'true_positive' ? 'bg-emerald-500/30 text-emerald-200 border border-emerald-500/50' :
                      prediction.case_type === 'false_positive' ? 'bg-red-500/30 text-red-200 border border-red-500/50' :
                      prediction.case_type === 'true_negative' ? 'bg-emerald-500/30 text-emerald-200 border border-emerald-500/50' :
                      prediction.case_type === 'false_negative' ? 'bg-amber-500/30 text-amber-200 border border-amber-500/50' :
                      prediction.case_type === 'escalated_fraud' ? 'bg-blue-500/30 text-blue-200 border border-blue-500/50' :
                      'bg-blue-500/30 text-blue-200 border border-blue-500/50'
                    }`}>
                      {prediction.case_type === 'true_positive' ? '✅ TP' :
                       prediction.case_type === 'false_positive' ? '🚨 FP' :
                       prediction.case_type === 'true_negative' ? '✅ TN' :
                       prediction.case_type === 'false_negative' ? '⚠️ FN' :
                       prediction.case_type === 'escalated_fraud' ? '🔄 ESC' :
                       '🔄 ESC'}
                    </div>

                    {/* Case ID and summary */}
                    <div className="flex-1 min-w-0">
                      <div className="text-white font-mono font-medium">Case #{prediction.case_id}</div>
                      <div className="text-xs text-slate-500 mt-0.5">
                        {prediction.case_type === 'true_positive' ? 'L1 blocked (fraud) → Actually fraud ✓' :
                         prediction.case_type === 'false_positive' ? 'L1 blocked (fraud) → Actually legit ✗' :
                         prediction.case_type === 'true_negative' ? 'L1 released (legit) → Actually legit ✓' :
                         prediction.case_type === 'false_negative' ? 'L1 released (legit) → Actually fraud ✗' :
                         prediction.case_type === 'escalated_fraud' ? 'L1 escalated → Actually fraud' :
                         'L1 escalated → Actually legit'}
                      </div>
                    </div>

                    {/* Decision flow visualization */}
                    <div className="flex items-center gap-2 text-sm bg-slate-800/50 rounded-lg px-3 py-2">
                      <div className="text-center">
                        <div className="text-[10px] text-slate-500 mb-0.5">L1 Said</div>
                        <div className={`font-bold ${
                          prediction.l1_decision.toLowerCase() === 'block' ? 'text-red-400' :
                          prediction.l1_decision.toLowerCase() === 'escalate' ? 'text-blue-400' :
                          prediction.l1_decision.toLowerCase() === 'release' ? 'text-green-400' :
                          'text-slate-300'
                        }`}>
                          {prediction.l1_decision}
                        </div>
                      </div>
                      <div className="text-slate-600">→</div>
                      <div className="text-center">
                        <div className="text-[10px] text-slate-500 mb-0.5">Actually</div>
                        <div className={`font-bold ${
                          prediction.true_fraud.toLowerCase().includes('fraud') || prediction.true_fraud === '1' 
                            ? 'text-red-400' 
                            : 'text-green-400'
                        }`}>
                          {prediction.true_fraud}
                        </div>
                      </div>
                      {/* Correctness indicator */}
                      <div className={`ml-2 text-lg ${prediction.is_correct ? 'text-emerald-400' : prediction.is_wrong ? 'text-red-400' : 'text-blue-400'}`}>
                        {prediction.is_correct ? '✓' : prediction.is_wrong ? '✗' : '?'}
                      </div>
                    </div>

                    {/* Expand button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setExpandedCase(expandedCase === actualIdx ? null : actualIdx);
                      }}
                      className="text-slate-400 hover:text-white p-2 hover:bg-slate-700 rounded"
                    >
                      {expandedCase === actualIdx ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                    </button>
                  </div>

                  {/* Expanded Details */}
                  <AnimatePresence>
                    {expandedCase === actualIdx && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 pt-4 border-t border-slate-800"
                      >
                        <div className="grid grid-cols-4 gap-4 text-sm">
                          {Object.entries(prediction)
                            .filter(([key]) => !['index', 'case_type'].includes(key))
                            .slice(0, 12)
                            .map(([key, value]) => (
                              <div key={key}>
                                <div className="text-slate-500 text-xs mb-0.5">{key}</div>
                                <div className="text-white font-mono text-xs truncate">{String(value)}</div>
                              </div>
                            ))}
                        </div>
                        <button
                          onClick={() => {
                            setSelectedCases(new Set([actualIdx]));
                            handleChatWithSelected();
                          }}
                          className="mt-4 flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs text-slate-300"
                        >
                          <MessageSquare className="w-3 h-3" />
                          Chat about this case
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Re-run Analysis Button */}
      <div className="mt-6 text-center">
        <button
          onClick={runAnalysis}
          disabled={loading}
          className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 flex items-center gap-2 mx-auto"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Re-run Analysis
        </button>
        <p className="text-xs text-slate-500 mt-2">
          Last analyzed: {new Date(analysisResult.analyzed_at).toLocaleString()}
        </p>
      </div>

      {/* Hyperparameters Modal */}
      <HyperparamModal
        show={showHyperparamModal}
        hyperparams={hyperparams}
        setHyperparams={setHyperparams}
        onClose={() => setShowHyperparamModal(false)}
        onRunAnalysis={() => {
          setShowHyperparamModal(false);
          runAnalysis();
        }}
      />

      {/* Tree Visualization Modal */}
      <AnimatePresence>
        {showTreeModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setShowTreeModal(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={e => e.stopPropagation()}
              className="bg-[#0d1117] border border-slate-800 rounded-2xl p-6 max-w-[95vw] w-full max-h-[90vh] overflow-auto"
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-xl font-bold text-white flex items-center gap-2">
                    <TreeDeciduous className="w-5 h-5 text-emerald-400" />
                    {showTreeModal === 'l1' ? 'L1' : 'L2'} Decision Tree Analysis
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">
                    sklearn DecisionTreeClassifier • Interpretable rules from your data
                  </p>
                </div>
                <button
                  onClick={() => setShowTreeModal(null)}
                  className="text-slate-400 hover:text-white p-2 hover:bg-slate-800 rounded-lg"
                >
                  <XCircle className="w-5 h-5" />
                </button>
              </div>

              {/* Info Banner */}
              <div className={`p-3 rounded-lg mb-4 ${showTreeModal === 'l1' ? 'bg-blue-500/10 border border-blue-500/30' : 'bg-emerald-500/10 border border-emerald-500/30'}`}>
                <p className="text-sm text-slate-300">
                  <strong className={showTreeModal === 'l1' ? 'text-blue-300' : 'text-emerald-300'}>
                    {showTreeModal === 'l1' ? 'L1 Analysis:' : 'L2 Analysis:'}
                  </strong>{' '}
                  {showTreeModal === 'l1' 
                    ? 'This tree shows how L1 analysts make decisions based on transaction features. All L2-related columns are excluded.'
                    : 'This tree shows how L2 analysts review L1 decisions. The L1 decision is included as a feature to show how L2 uses L1\'s judgment.'}
                </p>
              </div>

              {/* Decision Tree Text - Primary View */}
              {(showTreeModal === 'l1' ? analysisResult.l1_tree_text : analysisResult.l2_tree_text) && (
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                    <TreeDeciduous className="w-4 h-4 text-emerald-400" />
                    Full Decision Tree Structure
                  </h4>
                  <div className="bg-slate-950 rounded-lg p-4 max-h-[400px] overflow-auto font-mono text-xs">
                    <pre className="text-emerald-300 whitespace-pre">
                      {showTreeModal === 'l1' ? analysisResult.l1_tree_text : analysisResult.l2_tree_text}
                    </pre>
                  </div>
                </div>
              )}

              {/* Decision Rules - Summarized Paths */}
              {((showTreeModal === 'l1' ? analysisResult.l1_decision_rules : analysisResult.l2_decision_rules)?.length ?? 0) > 0 && (
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-white mb-3">Decision Paths (Human Readable)</h4>
                  <div className="bg-slate-950 rounded-lg p-4 max-h-64 overflow-y-auto">
                    {(showTreeModal === 'l1' ? analysisResult.l1_decision_rules : analysisResult.l2_decision_rules)?.map((rule, idx) => (
                      <div key={idx} className="text-xs font-mono text-cyan-300 mb-2 pb-2 border-b border-slate-800 last:border-0">
                        <span className="text-slate-500 mr-2">{idx + 1}.</span>
                        {rule}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tree Image - Optional Visual */}
              {(showTreeModal === 'l1' ? analysisResult.l1_tree_image : analysisResult.l2_tree_image) && (
                <div>
                  <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                    <ImageIcon className="w-4 h-4 text-blue-400" />
                    Visual Tree Diagram
                  </h4>
                  <div className="bg-white rounded-lg p-4 overflow-x-auto">
                    <img
                      src={`data:image/png;base64,${showTreeModal === 'l1' ? analysisResult.l1_tree_image : analysisResult.l2_tree_image}`}
                      alt={`${showTreeModal} Decision Tree`}
                      className="max-w-none"
                    />
                  </div>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

