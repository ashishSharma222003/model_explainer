"use client";
import React, { useState, useEffect, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3, CheckCircle, XCircle, AlertTriangle, TrendingUp,
  RefreshCw, ArrowRight, MessageSquare, Filter, ChevronDown,
  ChevronUp, Target, Users, Zap, Settings, TreeDeciduous, Image as ImageIcon,
  Lightbulb, Code
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

interface GeneralPurposeRule {
  original_rule: string;
  description: string;
  target: string;
  accuracy: number;
  coverage: number;
  confidence: string;
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
  // === NEW: General Purpose Rules ===
  general_purpose_rules?: GeneralPurposeRule[];

  // Tree visualization
  l1_tree_image?: string;
  l2_tree_image?: string;
  l1_decision_rules?: string[];
  l2_decision_rules?: string[];
  l1_tree_text?: string;
  l2_tree_text?: string;
  l1_tree_structure?: any[];
  l2_tree_structure?: any[];
  // NEW: Segment Analysis
  combined_target_distribution?: Record<string, number>;
  segment_analysis?: Array<{
    segment_id: string;
    tree_id: number;
    leaf_id: number;
    transaction_count: number;
    class_distribution: Record<string, number>;
    TP: number;
    FP: number;
    TN: number;
    FN: number;
    accuracy: number;
    precision: number;
    recall: number;
    fraud_rate: number;
    dominant_l1_decision: string;
    rule_text: string;
    fp_rate: number;
    fn_rate: number;
  }>;
  top_fp_segments?: any[];
  top_fn_segments?: any[];
  segment_summary?: string;
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


  // Hyperparameters state
  const [hyperparams, setHyperparams] = useState<Hyperparameters>(DEFAULT_HYPERPARAMS);
  const [showHyperparamModal, setShowHyperparamModal] = useState(false);
  const [showTreeModal, setShowTreeModal] = useState<'l1' | 'l2' | null>(null);
  const [progressMessage, setProgressMessage] = useState<string>('');

  // Segment analysis state - for interactive exploration
  const [expandedSegment, setExpandedSegment] = useState<string | null>(null);
  const [selectedSegment, setSelectedSegment] = useState<any | null>(null);
  const [showSegmentModal, setShowSegmentModal] = useState(false);
  const [segmentViewMode, setSegmentViewMode] = useState<'fp' | 'fn' | 'all'>('all');

  // Segment pagination state
  const [segmentPage, setSegmentPage] = useState(1);
  const [segmentsPerPage, setSegmentsPerPage] = useState(20);
  const segmentsPerPageOptions = [10, 20, 50, 100];



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

          {/* General Purpose Rules Section */}
          {analysisResult.general_purpose_rules && analysisResult.general_purpose_rules.length > 0 && (
            <div className="mb-6 mt-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-gradient-to-br from-amber-500 to-orange-600 rounded-lg shadow-lg shadow-amber-500/20">
                  <Lightbulb className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">General Purpose Rules</h3>
                  <p className="text-slate-400 text-sm">Simplified business logic discovered by AI</p>
                </div>
              </div>

              <div className="grid gap-3">
                {analysisResult.general_purpose_rules.map((rule: GeneralPurposeRule, idx: number) => (
                  <div key={idx} className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl hover:border-slate-700 transition-colors group/card">
                    <div className="flex justify-between items-start gap-4">
                      <div className="text-base text-slate-200 font-medium mb-1">
                        {rule.description}
                      </div>
                      <div className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider whitespace-nowrap ${rule.target === 'Fraud' ? 'bg-red-500/20 text-red-300 border border-red-500/30' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        }`}>
                        {rule.target}
                      </div>
                    </div>

                    <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
                      <div className="flex items-center gap-1" title="Accuracy">
                        <Target className="w-3 h-3" />
                        <span>{(rule.accuracy * 100).toFixed(0)}% Acc</span>
                      </div>
                      <div className="flex items-center gap-1" title="Coverage">
                        <Users className="w-3 h-3" />
                        <span>{(rule.coverage * 100).toFixed(1)}% Cov</span>
                      </div>

                      <div className="flex items-center gap-1 ml-auto">
                        <details className="relative group/details">
                          <summary className="list-none cursor-pointer hover:text-slate-300 transition-colors flex items-center gap-1 select-none">
                            <Code className="w-3 h-3" />
                            Logic
                          </summary>
                          <div className="absolute right-0 bottom-6 w-64 p-2 bg-black/90 backdrop-blur rounded text-slate-400 font-mono text-[10px] break-words border border-slate-800 z-10 shadow-xl">
                            {rule.original_rule}
                          </div>
                        </details>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

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

      {/* Segment Analysis Section - Interactive */}
      {analysisResult.segment_analysis && analysisResult.segment_analysis.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-cyan-400" />
                L1 Decision Segments (Pattern Discovery)
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Click on segments to explore decision patterns and metrics
              </p>
            </div>
            <span className="px-2 py-1 bg-cyan-500/20 border border-cyan-500/30 rounded text-xs text-cyan-300">
              {analysisResult.segment_analysis.length} Segments
            </span>
          </div>

          {/* Combined Target Distribution - Clickable badges */}
          {analysisResult.combined_target_distribution && (
            <div className="mb-4 p-3 bg-slate-800/50 rounded-lg">
              <h4 className="text-sm font-medium text-slate-300 mb-2">Combined Target Distribution (L1 Decision + Fraud)</h4>
              <div className="flex flex-wrap gap-2">
                {Object.entries(analysisResult.combined_target_distribution).map(([label, count]) => (
                  <button
                    key={label}
                    onClick={() => {
                      // Filter segments by this class
                      const filtered = analysisResult.segment_analysis?.filter(
                        s => s.class_distribution && s.class_distribution[label]
                      );
                      if (filtered && filtered.length > 0) {
                        setSelectedSegment(filtered[0]);
                        setShowSegmentModal(true);
                      }
                    }}
                    className={`px-3 py-1.5 rounded text-xs cursor-pointer transition-all hover:scale-105 ${label.includes('block_1') ? 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30' :
                      label.includes('block_0') ? 'bg-red-500/20 text-red-300 hover:bg-red-500/30' :
                        label.includes('release_0') ? 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30' :
                          label.includes('release_1') ? 'bg-amber-500/20 text-amber-300 hover:bg-amber-500/30' :
                            'bg-blue-500/20 text-blue-300 hover:bg-blue-500/30'
                      }`}
                  >
                    <span className="font-medium">{label}</span>: {count}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Segment View Tabs */}
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setSegmentViewMode('all')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${segmentViewMode === 'all'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
            >
              All Segments ({analysisResult.segment_analysis.length})
            </button>
            <button
              onClick={() => setSegmentViewMode('fp')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${segmentViewMode === 'fp'
                ? 'bg-red-500/20 text-red-300 border border-red-500/40'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
            >
              High FP ({analysisResult.top_fp_segments?.length || 0})
            </button>
            <button
              onClick={() => setSegmentViewMode('fn')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${segmentViewMode === 'fn'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
            >
              High FN ({analysisResult.top_fn_segments?.length || 0})
            </button>
          </div>

          {/* Pagination Controls */}
          {(() => {
            const currentSegments = segmentViewMode === 'fp' ? analysisResult.top_fp_segments :
              segmentViewMode === 'fn' ? analysisResult.top_fn_segments :
                analysisResult.segment_analysis;
            const totalSegments = currentSegments?.length || 0;
            const totalPages = Math.ceil(totalSegments / segmentsPerPage);
            const startIdx = (segmentPage - 1) * segmentsPerPage;
            const endIdx = startIdx + segmentsPerPage;
            const paginatedSegments = currentSegments?.slice(startIdx, endIdx) || [];

            return (
              <>
                {/* Pagination Header */}
                <div className="flex items-center justify-between mb-3 p-2 bg-slate-800/30 rounded-lg">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-400">Showing</span>
                    <select
                      value={segmentsPerPage}
                      onChange={(e) => { setSegmentsPerPage(Number(e.target.value)); setSegmentPage(1); }}
                      className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white"
                    >
                      {segmentsPerPageOptions.map(opt => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                    <span className="text-xs text-slate-400">per page</span>
                  </div>
                  <span className="text-xs text-slate-400">
                    {startIdx + 1}-{Math.min(endIdx, totalSegments)} of {totalSegments} segments
                  </span>
                </div>

                {/* Segments List */}
                <div className="space-y-2 max-h-[500px] overflow-y-auto">
                  {paginatedSegments.map((seg) => (
                    <div
                      key={seg.segment_id}
                      className={`border rounded-lg transition-all cursor-pointer ${expandedSegment === seg.segment_id
                        ? 'bg-slate-800/80 border-cyan-500/50'
                        : 'bg-slate-800/30 border-slate-700 hover:bg-slate-800/50 hover:border-slate-600'
                        }`}
                    >
                      {/* Segment Header - Clickable */}
                      <div
                        className="p-3 flex items-center justify-between"
                        onClick={() => setExpandedSegment(expandedSegment === seg.segment_id ? null : seg.segment_id)}
                      >
                        <div className="flex items-center gap-3">
                          <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${expandedSegment === seg.segment_id ? 'rotate-180' : ''}`} />
                          <span className="font-mono text-cyan-300 text-sm">{seg.segment_id}</span>
                          <span className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                            {seg.transaction_count} txns
                          </span>
                          <span className={`px-2 py-0.5 rounded text-xs ${seg.dominant_l1_decision === 'block' ? 'bg-red-500/20 text-red-300' :
                            seg.dominant_l1_decision === 'release' ? 'bg-emerald-500/20 text-emerald-300' :
                              'bg-blue-500/20 text-blue-300'
                            }`}>
                            {seg.dominant_l1_decision}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          {seg.FP > 0 && (
                            <span className="px-2 py-0.5 bg-red-500/20 rounded text-xs text-red-300">
                              FP: {seg.FP} ({(seg.fp_rate * 100).toFixed(0)}%)
                            </span>
                          )}
                          {seg.FN > 0 && (
                            <span className="px-2 py-0.5 bg-amber-500/20 rounded text-xs text-amber-300">
                              FN: {seg.FN} ({(seg.fn_rate * 100).toFixed(0)}%)
                            </span>
                          )}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedSegment(seg);
                              setShowSegmentModal(true);
                            }}
                            className="px-2 py-1 bg-violet-500/20 hover:bg-violet-500/30 rounded text-xs text-violet-300"
                          >
                            Details
                          </button>
                        </div>
                      </div>

                      {/* Expanded Content */}
                      <AnimatePresence>
                        {expandedSegment === seg.segment_id && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                          >
                            <div className="px-3 pb-3 border-t border-slate-700 pt-3">
                              {/* Metrics Grid */}
                              <div className="grid grid-cols-5 gap-2 mb-3">
                                <div className="text-center p-2 bg-emerald-500/10 rounded">
                                  <div className="text-lg font-bold text-emerald-400">{seg.TP}</div>
                                  <div className="text-[10px] text-slate-500">True Pos</div>
                                </div>
                                <div className="text-center p-2 bg-red-500/10 rounded">
                                  <div className="text-lg font-bold text-red-400">{seg.FP}</div>
                                  <div className="text-[10px] text-slate-500">False Pos</div>
                                </div>
                                <div className="text-center p-2 bg-emerald-500/10 rounded">
                                  <div className="text-lg font-bold text-emerald-400">{seg.TN}</div>
                                  <div className="text-[10px] text-slate-500">True Neg</div>
                                </div>
                                <div className="text-center p-2 bg-amber-500/10 rounded">
                                  <div className="text-lg font-bold text-amber-400">{seg.FN}</div>
                                  <div className="text-[10px] text-slate-500">False Neg</div>
                                </div>
                                <div className="text-center p-2 bg-violet-500/10 rounded">
                                  <div className="text-lg font-bold text-violet-400">{(seg.accuracy * 100).toFixed(0)}%</div>
                                  <div className="text-[10px] text-slate-500">Accuracy</div>
                                </div>
                              </div>

                              {/* Decision Rule */}
                              <div className="p-2 bg-slate-900/50 rounded border border-slate-700">
                                <div className="text-xs text-slate-400 mb-1">Decision Rule:</div>
                                <div className="text-sm text-cyan-300 font-mono">{seg.rule_text}</div>
                              </div>

                              {/* Class Distribution */}
                              {seg.class_distribution && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {Object.entries(seg.class_distribution).map(([cls, cnt]) => (
                                    <span key={cls} className="px-2 py-0.5 bg-slate-700 rounded text-[10px] text-slate-300">
                                      {cls}: {cnt as number}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  ))}
                </div>

                {/* Pagination Footer */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-center gap-2 mt-4">
                    <button
                      onClick={() => setSegmentPage(Math.max(1, segmentPage - 1))}
                      disabled={segmentPage === 1}
                      className={`px-3 py-1 rounded text-sm ${segmentPage === 1
                        ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                        : 'bg-slate-700 text-white hover:bg-slate-600'}`}
                    >
                      ← Prev
                    </button>
                    <div className="flex items-center gap-1">
                      {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                        let pageNum;
                        if (totalPages <= 5) {
                          pageNum = i + 1;
                        } else if (segmentPage <= 3) {
                          pageNum = i + 1;
                        } else if (segmentPage >= totalPages - 2) {
                          pageNum = totalPages - 4 + i;
                        } else {
                          pageNum = segmentPage - 2 + i;
                        }
                        return (
                          <button
                            key={pageNum}
                            onClick={() => setSegmentPage(pageNum)}
                            className={`w-8 h-8 rounded text-sm ${segmentPage === pageNum
                              ? 'bg-cyan-500 text-white'
                              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
                          >
                            {pageNum}
                          </button>
                        );
                      })}
                    </div>
                    <button
                      onClick={() => setSegmentPage(Math.min(totalPages, segmentPage + 1))}
                      disabled={segmentPage === totalPages}
                      className={`px-3 py-1 rounded text-sm ${segmentPage === totalPages
                        ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                        : 'bg-slate-700 text-white hover:bg-slate-600'}`}
                    >
                      Next →
                    </button>
                  </div>
                )}
              </>
            );
          })()}
        </div>
      )}

      {/* Segment Detail Modal */}
      {showSegmentModal && selectedSegment && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setShowSegmentModal(false)}
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
                  <Target className="w-5 h-5 text-cyan-400" />
                  Segment Details
                </h3>
                <p className="text-sm text-cyan-300 font-mono mt-1">{selectedSegment.segment_id}</p>
              </div>
              <button onClick={() => setShowSegmentModal(false)} className="text-slate-400 hover:text-white p-2 hover:bg-slate-800 rounded-lg">
                <XCircle className="w-5 h-5" />
              </button>
            </div>

            {/* Full Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
              <div className="text-center p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                <div className="text-2xl font-bold text-emerald-400">{selectedSegment.TP}</div>
                <div className="text-xs text-slate-400">True Positives</div>
              </div>
              <div className="text-center p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                <div className="text-2xl font-bold text-red-400">{selectedSegment.FP}</div>
                <div className="text-xs text-slate-400">False Positives</div>
              </div>
              <div className="text-center p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                <div className="text-2xl font-bold text-emerald-400">{selectedSegment.TN}</div>
                <div className="text-xs text-slate-400">True Negatives</div>
              </div>
              <div className="text-center p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                <div className="text-2xl font-bold text-amber-400">{selectedSegment.FN}</div>
                <div className="text-xs text-slate-400">False Negatives</div>
              </div>
            </div>

            {/* Derived Metrics */}
            <div className="grid grid-cols-4 gap-3 mb-4">
              <div className="text-center p-2 bg-slate-800 rounded-lg">
                <div className="text-lg font-bold text-white">{(selectedSegment.accuracy * 100).toFixed(1)}%</div>
                <div className="text-[10px] text-slate-500">Accuracy</div>
              </div>
              <div className="text-center p-2 bg-slate-800 rounded-lg">
                <div className="text-lg font-bold text-white">{(selectedSegment.precision * 100).toFixed(1)}%</div>
                <div className="text-[10px] text-slate-500">Precision</div>
              </div>
              <div className="text-center p-2 bg-slate-800 rounded-lg">
                <div className="text-lg font-bold text-white">{(selectedSegment.recall * 100).toFixed(1)}%</div>
                <div className="text-[10px] text-slate-500">Recall</div>
              </div>
              <div className="text-center p-2 bg-slate-800 rounded-lg">
                <div className="text-lg font-bold text-white">{(selectedSegment.fraud_rate * 100).toFixed(1)}%</div>
                <div className="text-[10px] text-slate-500">Fraud Rate</div>
              </div>
            </div>

            {/* Segment Info */}
            <div className="space-y-3">
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Transaction Count</div>
                <div className="text-lg font-bold text-white">{selectedSegment.transaction_count}</div>
              </div>

              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Dominant L1 Decision</div>
                <div className={`text-lg font-bold ${selectedSegment.dominant_l1_decision === 'block' ? 'text-red-400' :
                  selectedSegment.dominant_l1_decision === 'release' ? 'text-emerald-400' :
                    'text-blue-400'
                  }`}>{selectedSegment.dominant_l1_decision}</div>
              </div>

              <div className="p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Decision Rule (Pattern)</div>
                <div className="text-sm text-cyan-300 font-mono whitespace-pre-wrap">{selectedSegment.rule_text}</div>
              </div>

              {/* Class Distribution */}
              {selectedSegment.class_distribution && (
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <div className="text-xs text-slate-400 mb-2">Class Distribution</div>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(selectedSegment.class_distribution).map(([cls, cnt]) => (
                      <span
                        key={cls}
                        className={`px-2 py-1 rounded text-xs ${cls.includes('block_1') ? 'bg-emerald-500/20 text-emerald-300' :
                          cls.includes('block_0') ? 'bg-red-500/20 text-red-300' :
                            cls.includes('release_0') ? 'bg-emerald-500/20 text-emerald-300' :
                              cls.includes('release_1') ? 'bg-amber-500/20 text-amber-300' :
                                'bg-slate-700 text-slate-300'
                          }`}
                      >
                        {cls}: {cnt as number}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      )}

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

      {/* === NEW: Segment Analysis === */}
      {/* general_purpose_rules?: GeneralPurposeRule[]; */}
      {/* Tree visualization (base64 encoded PNG) */}
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

