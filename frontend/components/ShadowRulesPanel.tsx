"use client";
import React, { useState, useContext, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Zap, Plus, Trash2, ChevronDown, ChevronUp,
  RefreshCw, Copy, Eye, EyeOff, MessageSquare,
  AlertTriangle, Clock, Filter, Database, Sparkles,
  Send, Loader2, CheckCircle, Tag, Edit3, Save, X,
  TreeDeciduous, Search, Info
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { AppContext } from '@/app/page';
import { ShadowRule, ChatMessageWithContext } from '@/lib/storage';
import {
  chatTxn,
  convertShadowRules,
  checkShadowRuleDuplicate,
  addShadowRuleToIndex,
  addShadowRulesBulk,
  clearShadowRulesBySource,
  getShadowRulesStats,
  extractShadowRulesFromTransactions,
  VectorStoreStats,
  AddRuleToIndexRequest
} from '@/lib/api';

export default function ShadowRulesPanel() {
  const context = useContext(AppContext);
  const [expandedRule, setExpandedRule] = useState<string | null>(null);
  const [filterTarget, setFilterTarget] = useState<'all' | 'l1' | 'l2' | 'decision-tree' | 'discovered'>('all');
  const [showActiveOnly, setShowActiveOnly] = useState(false);
  const [editingRule, setEditingRule] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const [generatingFromTree, setGeneratingFromTree] = useState(false);

  // Chat state
  const [showChat, setShowChat] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessageWithContext[]>([]);
  const [extractingRules, setExtractingRules] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const shadowRules = context?.session.shadowRules || [];
  const selectedPredictions = context?.session.selectedWrongPredictions || [];
  const analysisResult = context?.session.analysisResult;

  // Vector store state for semantic deduplication
  const [vectorStoreStats, setVectorStoreStats] = useState<VectorStoreStats | null>(null);
  const [duplicateNotification, setDuplicateNotification] = useState<{
    show: boolean;
    message: string;
    type: 'duplicate' | 'review' | 'new';
  } | null>(null);
  const [syncingToVectorStore, setSyncingToVectorStore] = useState(false);

  // Load vector store stats on mount and when session changes
  useEffect(() => {
    const loadStats = async () => {
      if (!context?.session.id) return;
      try {
        const stats = await getShadowRulesStats(context.session.id);
        setVectorStoreStats(stats);
      } catch (error) {
        console.log('Vector store not initialized yet');
      }
    };
    loadStats();
  }, [context?.session.id, shadowRules.length]);

  // Check if decision tree rules have been imported
  const hasDecisionTreeRules = shadowRules.some(r => r.sourceAnalysis === 'decision-tree');
  const l1RulesCount = analysisResult?.l1_decision_rules?.length || 0;
  const l2RulesCount = analysisResult?.l2_decision_rules?.length || 0;

  // Generate shadow rules from decision tree rules using LLM for human-readable conversion
  const generateFromDecisionTree = async () => {
    if (!context || !analysisResult) return;

    setGeneratingFromTree(true);

    try {
      const l1Rules = analysisResult.l1_decision_rules || [];
      const l2Rules = analysisResult.l2_decision_rules || [];

      // STEP 1: Clear old decision tree rules from vector store (for re-analysis)
      try {
        await clearShadowRulesBySource(context.session.id, 'decision-tree');
        console.log('Cleared old decision tree rules from vector store');
      } catch (e) {
        console.log('Vector store not initialized, will create new');
      }

      // Also remove old decision tree rules from session
      const nonDecisionTreeRules = shadowRules.filter(r => r.sourceAnalysis !== 'decision-tree');

      // STEP 2: Get existing rule texts from chat-discovered rules to check for semantic duplicates
      const existingRuleTexts = nonDecisionTreeRules.map(r => r.ruleText);

      // STEP 3: Call API to convert rules to human-readable format
      const response = await convertShadowRules(
        l1Rules,
        l2Rules,
        context.session.dataSchema,
        existingRuleTexts
      );

      if (response.success && response.rules.length > 0) {
        const timestamp = new Date().toISOString();
        const newRules: ShadowRule[] = response.rules.map((rule, idx) => ({
          id: `rule_dt_${rule.target_decision}_${Date.now()}_${idx}`,
          ruleText: rule.simple_rule,  // Human-readable version
          decisionPath: rule.original_rule,  // Original technical rule
          sourceAnalysis: 'decision-tree',
          targetDecision: rule.target_decision as 'l1' | 'l2',
          predictedOutcome: rule.predicted_outcome,
          confidence: rule.confidence_level === 'high' ? 0.9 : rule.confidence_level === 'medium' ? 0.7 : 0.5,
          samplesAffected: rule.samples_affected,
          featureConditions: rule.key_factors.map(f => ({ feature: f, operator: '', value: '' })),
          isActive: true,
          createdAt: timestamp,
          notes: `Key factors: ${rule.key_factors.join(', ')}`,
        }));

        // STEP 4: Add new rules to session
        context.updateSession({
          shadowRules: [...nonDecisionTreeRules, ...newRules]
        });

        // STEP 5: Sync new rules to vector store for semantic search
        try {
          const rulesToIndex: AddRuleToIndexRequest[] = newRules.map(r => ({
            rule_id: r.id,
            rule_text: r.ruleText,
            source_analysis: 'decision-tree',
            simple_rule: r.ruleText,
            target_decision: r.targetDecision,
            predicted_outcome: r.predictedOutcome,
            confidence_level: r.confidence > 0.8 ? 'high' : r.confidence > 0.6 ? 'medium' : 'low',
            samples_affected: r.samplesAffected
          }));

          const bulkResult = await addShadowRulesBulk(context.session.id, rulesToIndex);
          console.log(`Synced ${bulkResult.added_count} rules to vector store`);

          // Refresh stats
          const stats = await getShadowRulesStats(context.session.id);
          setVectorStoreStats(stats);
        } catch (e) {
          console.error('Failed to sync to vector store:', e);
        }
      }

    } catch (error) {
      console.error('Failed to generate rules:', error);
      // Fallback to basic conversion without LLM
      fallbackGenerateFromDecisionTree();
    } finally {
      setGeneratingFromTree(false);
    }
  };

  // Fallback method if LLM conversion fails
  const fallbackGenerateFromDecisionTree = () => {
    if (!context || !analysisResult) return;

    const newRules: ShadowRule[] = [];
    const timestamp = new Date().toISOString();

    // Extract rules from L1 decision tree
    if (analysisResult.l1_decision_rules && analysisResult.l1_decision_rules.length > 0) {
      analysisResult.l1_decision_rules.forEach((ruleText: string, idx: number) => {
        const conditions = parseRuleConditions(ruleText);
        const outcome = extractOutcome(ruleText);
        const samples = extractSamplesCount(ruleText);

        const exists = shadowRules.some(r =>
          (r.decisionPath === ruleText || r.ruleText === ruleText) && r.sourceAnalysis === 'decision-tree'
        );

        if (!exists) {
          newRules.push({
            id: `rule_dt_l1_${Date.now()}_${idx}`,
            ruleText: ruleText,
            decisionPath: ruleText,
            sourceAnalysis: 'decision-tree',
            targetDecision: 'l1',
            predictedOutcome: outcome,
            confidence: 0.9,
            samplesAffected: samples,
            featureConditions: conditions,
            isActive: true,
            createdAt: timestamp,
            notes: 'Auto-generated from L1 Decision Tree (fallback)',
          });
        }
      });
    }

    // Extract rules from L2 decision tree
    if (analysisResult.l2_decision_rules && analysisResult.l2_decision_rules.length > 0) {
      analysisResult.l2_decision_rules.forEach((ruleText: string, idx: number) => {
        const conditions = parseRuleConditions(ruleText);
        const outcome = extractOutcome(ruleText);
        const samples = extractSamplesCount(ruleText);

        const exists = shadowRules.some(r =>
          (r.decisionPath === ruleText || r.ruleText === ruleText) && r.sourceAnalysis === 'decision-tree'
        );

        if (!exists) {
          newRules.push({
            id: `rule_dt_l2_${Date.now()}_${idx}`,
            ruleText: ruleText,
            decisionPath: ruleText,
            sourceAnalysis: 'decision-tree',
            targetDecision: 'l2',
            predictedOutcome: outcome,
            confidence: 0.9,
            samplesAffected: samples,
            featureConditions: conditions,
            isActive: true,
            createdAt: timestamp,
            notes: 'Auto-generated from L2 Decision Tree (fallback)',
          });
        }
      });
    }

    if (newRules.length > 0) {
      context.updateSession({
        shadowRules: [...shadowRules, ...newRules]
      });
    }
  };

  // Parse rule text to extract feature conditions
  const parseRuleConditions = (ruleText: string): ShadowRule['featureConditions'] => {
    const conditions: ShadowRule['featureConditions'] = [];
    const conditionPattern = /(\w+)\s*(<=|>=|<|>|==|!=)\s*([\d.]+|[\w]+)/g;
    let match;

    while ((match = conditionPattern.exec(ruleText)) !== null) {
      conditions.push({
        feature: match[1],
        operator: match[2],
        value: isNaN(Number(match[3])) ? match[3] : Number(match[3])
      });
    }

    return conditions;
  };

  // Extract predicted outcome from rule text
  const extractOutcome = (ruleText: string): string => {
    const lower = ruleText.toLowerCase();
    if (lower.includes('block')) return 'block';
    if (lower.includes('release')) return 'release';
    if (lower.includes('escalate')) return 'escalate';
    if (lower.includes('class: 0') || lower.includes('class 0')) return 'release';
    if (lower.includes('class: 1') || lower.includes('class 1')) return 'block';
    if (lower.includes('class: 2') || lower.includes('class 2')) return 'escalate';
    return 'unknown';
  };

  // Extract samples count from rule text
  const extractSamplesCount = (ruleText: string): number => {
    const samplesMatch = ruleText.match(/samples?\s*[=:]\s*(\d+)/i);
    return samplesMatch ? parseInt(samplesMatch[1]) : 0;
  };

  // Add a new shadow rule manually
  const addManualRule = () => {
    if (!context) return;

    const newRule: ShadowRule = {
      id: `rule_manual_${Date.now()}`,
      ruleText: 'New shadow rule - click to edit',
      decisionPath: '',
      sourceAnalysis: 'manual',
      targetDecision: 'l1',
      predictedOutcome: 'unknown',
      confidence: 0.5,
      samplesAffected: 0,
      featureConditions: [],
      isActive: true,
      createdAt: new Date().toISOString(),
      notes: '',
    };

    context.updateSession({
      shadowRules: [newRule, ...shadowRules]
    });
    setEditingRule(newRule.id);
    setEditText(newRule.ruleText);
  };

  // Extract shadow rules from selected transactions using dedicated endpoint
  const extractRulesFromChat = async () => {
    if (!context || selectedPredictions.length === 0) return;

    setExtractingRules(true);

    try {
      const analysis = context.session.analysisResult;

      // Build decision tree context
      const decisionTreeContext = {
        l1_decision_rules: analysis?.l1_decision_rules || [],
        l2_decision_rules: analysis?.l2_decision_rules || [],
        l1_accuracy: analysis?.l1_decision_analysis?.metrics?.accuracy,
        l2_accuracy: analysis?.l2_decision_analysis?.metrics?.accuracy,
        l1_top_features: analysis?.l1_decision_analysis?.feature_importance
          ? Object.keys(analysis.l1_decision_analysis.feature_importance).slice(0, 5)
          : [],
        l2_top_features: analysis?.l2_decision_analysis?.feature_importance
          ? Object.keys(analysis.l2_decision_analysis.feature_importance).slice(0, 5)
          : [],
      };

      // Build chat history context
      const chatHistoryContext = chatMessages.map(m => ({
        role: m.type === 'user' ? 'user' : 'assistant',
        content: m.content
      }));

      // Call the dedicated extraction endpoint
      const response = await extractShadowRulesFromTransactions({
        session_id: context.session.id,
        selected_transactions: selectedPredictions,
        data_schema: context.session.dataSchema,
        decision_tree_rules: decisionTreeContext,
        chat_history: chatHistoryContext.length > 0 ? chatHistoryContext : undefined
      });

      if (!response.success) {
        throw new Error('Extraction failed');
      }

      const newRules: ShadowRule[] = [];
      const duplicateRules: string[] = [];

      // Process each extracted rule with deduplication
      for (let i = 0; i < response.shadow_rules.length; i++) {
        const extractedRule = response.shadow_rules[i];

        // Check for duplicates
        try {
          const dedupeResult = await checkShadowRuleDuplicate(
            context.session.id,
            extractedRule.rule_description,
            0.95
          );

          if (dedupeResult.is_duplicate) {
            duplicateRules.push(extractedRule.rule_description);
            console.log(`Duplicate detected: ${extractedRule.rule_description}`);
            continue;
          }
        } catch (e) {
          // If vector store not available, check locally
          const exists = shadowRules.some(r =>
            r.ruleText.toLowerCase() === extractedRule.rule_description.toLowerCase()
          );
          if (exists) continue;
        }

        // Convert to ShadowRule format
        const newRule: ShadowRule = {
          id: `rule_extracted_${Date.now()}_${i}`,
          ruleText: extractedRule.rule_description,
          decisionPath: extractedRule.reasoning,
          sourceAnalysis: 'chat-discovered',
          targetDecision: extractedRule.target_decision.toLowerCase().includes('l2') ? 'l2' : 'l1',
          predictedOutcome: extractedRule.predicted_outcome.toLowerCase(),
          confidence: extractedRule.confidence === 'high' ? 0.9 : extractedRule.confidence === 'medium' ? 0.7 : 0.5,
          samplesAffected: extractedRule.affected_transactions || selectedPredictions.length,
          featureConditions: extractedRule.key_features.map(f => ({ feature: f, operator: '', value: '' })),
          isActive: true,
          createdAt: new Date().toISOString(),
          notes: extractedRule.reasoning,
        };

        newRules.push(newRule);
      }

      if (newRules.length > 0) {
        context.updateSession({
          shadowRules: [...newRules, ...shadowRules]
        });

        // Sync new rules to vector store
        try {
          const rulesToIndex: AddRuleToIndexRequest[] = newRules.map(r => ({
            rule_id: r.id,
            rule_text: r.ruleText,
            source_analysis: 'chat-discovered',
            simple_rule: r.ruleText,
            target_decision: r.targetDecision,
            predicted_outcome: r.predictedOutcome,
            confidence_level: r.confidence > 0.8 ? 'high' : r.confidence > 0.6 ? 'medium' : 'low',
            samples_affected: r.samplesAffected
          }));

          await addShadowRulesBulk(context.session.id, rulesToIndex);

          // Refresh stats
          const stats = await getShadowRulesStats(context.session.id);
          setVectorStoreStats(stats);
        } catch (e) {
          console.error('Failed to sync to vector store:', e);
        }

        // Build success message
        let message = `✅ Extracted ${newRules.length} shadow rule(s) from ${response.transactions_analyzed} transactions.\n\n**Summary:** ${response.summary}`;
        if (duplicateRules.length > 0) {
          message += `\n\n⚠️ Skipped ${duplicateRules.length} duplicate rule(s).`;
        }

        setChatMessages(prev => [...prev, {
          type: 'ai',
          content: message,
          context: context.createContextSnapshot()
        }]);
      } else {
        let message = response.shadow_rules.length === 0
          ? `No shadow rules could be identified from the ${response.transactions_analyzed} transactions. Try chatting more about specific patterns you notice.`
          : `Found ${response.shadow_rules.length} rule(s), but all were duplicates of existing rules.`;

        setChatMessages(prev => [...prev, {
          type: 'ai',
          content: message,
          context: context.createContextSnapshot()
        }]);
      }

    } catch (error) {
      console.error('Failed to extract rules:', error);
      setChatMessages(prev => [...prev, {
        type: 'ai',
        content: '❌ Failed to extract shadow rules. Please try again.',
        context: context.createContextSnapshot()
      }]);
    } finally {
      setExtractingRules(false);
    }
  };

  // Toggle rule active state
  const toggleRuleActive = (ruleId: string) => {
    if (!context) return;

    const updatedRules = shadowRules.map(rule =>
      rule.id === ruleId ? { ...rule, isActive: !rule.isActive } : rule
    );

    context.updateSession({ shadowRules: updatedRules });
  };

  // Delete a rule
  const deleteRule = (ruleId: string) => {
    if (!context) return;

    const updatedRules = shadowRules.filter(rule => rule.id !== ruleId);
    context.updateSession({ shadowRules: updatedRules });
  };

  // Save edited rule
  const saveEditedRule = (ruleId: string) => {
    if (!context || !editText.trim()) return;

    const updatedRules = shadowRules.map(rule =>
      rule.id === ruleId ? { ...rule, ruleText: editText.trim() } : rule
    );

    context.updateSession({ shadowRules: updatedRules });
    setEditingRule(null);
    setEditText('');
  };

  // Copy rule to clipboard
  const copyRule = (rule: ShadowRule) => {
    navigator.clipboard.writeText(rule.ruleText);
  };

  // Filter rules
  const filteredRules = shadowRules.filter(rule => {
    if (filterTarget === 'l1' && rule.targetDecision !== 'l1') return false;
    if (filterTarget === 'l2' && rule.targetDecision !== 'l2') return false;
    if (filterTarget === 'decision-tree' && rule.sourceAnalysis !== 'decision-tree') return false;
    if (filterTarget === 'discovered' && rule.sourceAnalysis === 'decision-tree') return false;
    if (showActiveOnly && !rule.isActive) return false;
    return true;
  });

  // Count by source
  const decisionTreeRulesCount = shadowRules.filter(r => r.sourceAnalysis === 'decision-tree').length;
  const discoveredRulesCount = shadowRules.filter(r => r.sourceAnalysis !== 'decision-tree').length;

  const getOutcomeColor = (outcome: string) => {
    switch (outcome.toLowerCase()) {
      case 'block': return 'text-red-400 bg-red-500/20 border-red-500/50';
      case 'release': return 'text-emerald-400 bg-emerald-500/20 border-emerald-500/50';
      case 'escalate': return 'text-blue-400 bg-blue-500/20 border-blue-500/50';
      default: return 'text-slate-400 bg-slate-500/20 border-slate-500/50';
    }
  };

  // Send chat message
  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading || !context) return;

    const userMessage: ChatMessageWithContext = {
      type: 'user',
      content: chatInput,
      context: context.createContextSnapshot()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);

    try {
      // Build comprehensive context for the chat
      const analysis = context.session.analysisResult;

      // Extract decision tree rules from analysis
      const decisionTreeContext = {
        l1_decision_rules: analysis?.l1_decision_rules || [],
        l2_decision_rules: analysis?.l2_decision_rules || [],
        l1_accuracy: analysis?.l1_decision_analysis?.metrics?.accuracy,
        l2_accuracy: analysis?.l2_decision_analysis?.metrics?.accuracy,
        l1_top_features: analysis?.l1_decision_analysis?.feature_importance
          ? Object.keys(analysis.l1_decision_analysis.feature_importance).slice(0, 5)
          : [],
        l2_top_features: analysis?.l2_decision_analysis?.feature_importance
          ? Object.keys(analysis.l2_decision_analysis.feature_importance).slice(0, 5)
          : [],
      };

      // Build the transaction context with selected predictions
      const txnJsonContext = {
        selectedCases: selectedPredictions,
        filters: [],
        totalCases: selectedPredictions.length,
        selectedAt: new Date().toISOString(),
        // Explain why we're chatting about these transactions
        context_explanation: `These ${selectedPredictions.length} transactions have predictions that DIFFER from the Random Forest models trained on this data. The L1 decision tree achieved ${decisionTreeContext.l1_accuracy ? (decisionTreeContext.l1_accuracy * 100).toFixed(1) + '% accuracy' : 'N/A'} and L2 achieved ${decisionTreeContext.l2_accuracy ? (decisionTreeContext.l2_accuracy * 100).toFixed(1) + '% accuracy' : 'N/A'}. We're trying to understand WHY these specific transactions were predicted differently (either false positives or false negatives).`
      };

      const response = await chatTxn({
        session_id: context.session.id,
        message: chatInput,
        context: {
          history: [...chatMessages, userMessage].map(m => ({
            role: m.type === 'user' ? 'user' : 'assistant',
            content: m.content
          })),
          // Full schema with all feature information
          dataSchema: context.session.dataSchema,
          // Transaction context with explanation
          txnJson: txnJsonContext,
          // Only include decision tree rules summary (not full analysis to save tokens)
          decision_tree_rules: decisionTreeContext,
          // ONLY the user-selected transactions (not all wrong predictions)
          wrong_predictions: selectedPredictions
        }
      });

      const aiMessage: ChatMessageWithContext = {
        type: 'ai',
        content: response.response,
        context: context.createContextSnapshot()
      };

      setChatMessages(prev => [...prev, aiMessage]);

      setTimeout(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);

    } catch (error) {
      console.error('Chat error:', error);
      setChatMessages(prev => [...prev, {
        type: 'ai',
        content: 'Sorry, there was an error processing your request.',
        context: context.createContextSnapshot()
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl">
              <Database className="w-6 h-6 text-white" />
            </div>
            Shadow Rules Database
          </h2>
          <p className="text-slate-400 mt-1">
            Pre-generated from decision trees + discovered from wrong predictions
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Generate from Decision Tree Button */}
          {analysisResult && (l1RulesCount > 0 || l2RulesCount > 0) && (
            <button
              onClick={generateFromDecisionTree}
              disabled={generatingFromTree}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium ${hasDecisionTreeRules
                ? 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                }`}
            >
              {generatingFromTree ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <TreeDeciduous className="w-4 h-4" />
              )}
              {hasDecisionTreeRules ? 'Refresh from Trees' : `Import ${l1RulesCount + l2RulesCount} Tree Rules`}
            </button>
          )}

          <button
            onClick={addManualRule}
            className="flex items-center gap-2 px-4 py-2 bg-amber-600 hover:bg-amber-500 rounded-lg text-sm text-white font-medium"
          >
            <Plus className="w-4 h-4" />
            Add Manual
          </button>
        </div>
      </div>

      {/* Decision Tree Rules Banner - if not imported yet */}
      {analysisResult && (l1RulesCount > 0 || l2RulesCount > 0) && !hasDecisionTreeRules && (
        <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <TreeDeciduous className="w-6 h-6 text-emerald-400" />
              <div>
                <div className="text-sm font-medium text-emerald-300">
                  Decision Tree Rules Available
                </div>
                <div className="text-xs text-emerald-400/70">
                  {l1RulesCount} L1 rules + {l2RulesCount} L2 rules from Random Forest analysis
                </div>
              </div>
            </div>
            <button
              onClick={generateFromDecisionTree}
              disabled={generatingFromTree}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-sm text-white font-medium"
            >
              {generatingFromTree ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Zap className="w-4 h-4" />
              )}
              Import to Database
            </button>
          </div>
        </div>
      )}

      {/* Two Column Layout */}
      <div className="grid grid-cols-2 gap-6">
        {/* Left: Wrong Predictions Analysis */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
          <div className="p-4 border-b border-slate-800 bg-slate-800/30">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-amber-400" />
              Discover New Rules
            </h3>
            <p className="text-sm text-slate-400 mt-1">
              Chat about wrong predictions to find additional patterns
            </p>
          </div>

          {selectedPredictions.length === 0 ? (
            <div className="p-8 text-center">
              <AlertTriangle className="w-12 h-12 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400 mb-2">No wrong predictions selected</p>
              <p className="text-sm text-slate-500">
                Go to Results → Select wrong predictions → Click "Chat About Selected"
              </p>
            </div>
          ) : (
            <div className="flex flex-col h-[500px]">
              {/* Selected Cases Summary */}
              <div className="p-3 bg-amber-500/10 border-b border-amber-500/30">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-amber-300 font-medium">
                      {selectedPredictions.length} cases selected
                    </span>
                    <div className="flex gap-1">
                      {selectedPredictions.slice(0, 5).map((pred: any, idx: number) => (
                        <span
                          key={idx}
                          className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${pred.case_type === 'false_positive'
                            ? 'bg-red-500/30 text-red-300'
                            : 'bg-amber-500/30 text-amber-300'
                            }`}
                        >
                          {pred.case_type === 'false_positive' ? 'FP' : 'FN'}
                        </span>
                      ))}
                      {selectedPredictions.length > 5 && (
                        <span className="text-xs text-slate-500">+{selectedPredictions.length - 5}</span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      context?.updateSession({ selectedWrongPredictions: [] });
                      setChatMessages([]);
                    }}
                    className="text-xs text-slate-400 hover:text-white"
                  >
                    Clear
                  </button>
                </div>
              </div>

              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-slate-950/30">
                {chatMessages.length === 0 ? (
                  <div className="text-center py-8">
                    <MessageSquare className="w-8 h-8 mx-auto mb-2 text-slate-600" />
                    <p className="text-sm text-slate-500 mb-3">
                      Ask questions to discover shadow rules
                    </p>
                    <div className="flex flex-col gap-2">
                      {[
                        "Why did the analyst make different decisions?",
                        "What patterns explain these wrong predictions?",
                        "Are there any hidden rules the model missed?"
                      ].map((q, i) => (
                        <button
                          key={i}
                          onClick={() => setChatInput(q)}
                          className="px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs text-slate-400 text-left"
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <>
                    {chatMessages.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div className={`max-w-[90%] rounded-lg p-3 ${msg.type === 'user'
                          ? 'bg-amber-600/30 text-amber-100'
                          : 'bg-slate-800 text-slate-200'
                          }`}>
                          {msg.type === 'ai' ? (
                            <div className="prose prose-sm prose-invert max-w-none text-sm">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {msg.content}
                              </ReactMarkdown>
                            </div>
                          ) : (
                            <p className="text-sm">{msg.content}</p>
                          )}
                        </div>
                      </div>
                    ))}
                    {chatLoading && (
                      <div className="flex justify-start">
                        <div className="bg-slate-800 rounded-lg p-3">
                          <Loader2 className="w-5 h-5 animate-spin text-amber-400" />
                        </div>
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </>
                )}
              </div>

              {/* Extract Rules Button */}
              {chatMessages.length > 0 && (
                <div className="p-2 border-t border-slate-800 bg-slate-900/50">
                  <button
                    onClick={extractRulesFromChat}
                    disabled={extractingRules}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded-lg text-sm text-white font-medium"
                  >
                    {extractingRules ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <Zap className="w-4 h-4" />
                    )}
                    Extract Shadow Rules from Chat
                  </button>
                </div>
              )}

              {/* Chat Input */}
              <form onSubmit={sendMessage} className="p-3 border-t border-slate-800 bg-slate-900/50">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask about the wrong predictions..."
                    className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-amber-500"
                    disabled={chatLoading}
                  />
                  <button
                    type="submit"
                    disabled={!chatInput.trim() || chatLoading}
                    className="px-3 py-2 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 rounded-lg text-white"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
              </form>
            </div>
          )}
        </div>

        {/* Right: Shadow Rules Database */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
          <div className="p-4 border-b border-slate-800 bg-slate-800/30">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Database className="w-5 h-5 text-blue-400" />
                  Rules Database
                </h3>
                <p className="text-sm text-slate-400 mt-1">
                  {shadowRules.length} rules • {shadowRules.filter(r => r.isActive).length} active
                </p>
              </div>

              {shadowRules.length > 0 && (
                <button
                  onClick={() => context?.updateSession({ shadowRules: [] })}
                  className="text-xs text-red-400 hover:text-red-300 px-2 py-1"
                >
                  Clear All
                </button>
              )}
            </div>

            {/* Filters */}
            <div className="flex flex-wrap items-center gap-3 mt-3">
              <select
                value={filterTarget}
                onChange={(e) => setFilterTarget(e.target.value as any)}
                className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white"
              >
                <option value="all">All Rules ({shadowRules.length})</option>
                <option value="decision-tree">🌳 From Decision Tree ({decisionTreeRulesCount})</option>
                <option value="discovered">💡 Discovered ({discoveredRulesCount})</option>
                <option value="l1">L1 Only</option>
                <option value="l2">L2 Only</option>
              </select>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showActiveOnly}
                  onChange={(e) => setShowActiveOnly(e.target.checked)}
                  className="w-3 h-3 rounded border-slate-600 bg-slate-800 text-amber-500"
                />
                <span className="text-xs text-slate-400">Active only</span>
              </label>
            </div>
          </div>

          {/* Rules List */}
          <div className="max-h-[500px] overflow-y-auto">
            {filteredRules.length === 0 ? (
              <div className="p-8 text-center">
                <Database className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                <p className="text-slate-400 mb-2">No shadow rules yet</p>
                <p className="text-sm text-slate-500">
                  Analyze wrong predictions to discover hidden patterns
                </p>
              </div>
            ) : (
              <div className="divide-y divide-slate-800/50">
                {filteredRules.map((rule) => (
                  <div
                    key={rule.id}
                    className={`transition-colors ${rule.isActive ? '' : 'opacity-50'}`}
                  >
                    <div className="p-3 hover:bg-slate-800/30">
                      <div className="flex items-start gap-3">
                        {/* Active Toggle */}
                        <button
                          onClick={() => toggleRuleActive(rule.id)}
                          className={`mt-0.5 p-1 rounded transition-colors ${rule.isActive
                            ? 'bg-emerald-500/20 text-emerald-400'
                            : 'bg-slate-700/50 text-slate-500'
                            }`}
                        >
                          {rule.isActive ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                        </button>

                        {/* Rule Content */}
                        <div className="flex-1 min-w-0">
                          {editingRule === rule.id ? (
                            <div className="flex gap-2">
                              <input
                                type="text"
                                value={editText}
                                onChange={(e) => setEditText(e.target.value)}
                                className="flex-1 bg-slate-800 border border-amber-500 rounded px-2 py-1 text-sm text-white"
                                autoFocus
                              />
                              <button
                                onClick={() => saveEditedRule(rule.id)}
                                className="p-1 text-emerald-400 hover:bg-emerald-500/20 rounded"
                              >
                                <Save className="w-4 h-4" />
                              </button>
                              <button
                                onClick={() => { setEditingRule(null); setEditText(''); }}
                                className="p-1 text-slate-400 hover:bg-slate-700 rounded"
                              >
                                <X className="w-4 h-4" />
                              </button>
                            </div>
                          ) : (
                            <>
                              <div className="flex items-center gap-2 mb-1 flex-wrap">
                                {/* Source Badge */}
                                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${rule.sourceAnalysis === 'decision-tree'
                                  ? 'bg-emerald-500/20 text-emerald-300'
                                  : rule.sourceAnalysis === 'manual'
                                    ? 'bg-amber-500/20 text-amber-300'
                                    : 'bg-violet-500/20 text-violet-300'
                                  }`}>
                                  {rule.sourceAnalysis === 'decision-tree' ? '🌳 Tree' :
                                    rule.sourceAnalysis === 'manual' ? '✏️ Manual' : '💡 Discovered'}
                                </span>
                                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${rule.targetDecision === 'l1'
                                  ? 'bg-blue-500/20 text-blue-300'
                                  : 'bg-purple-500/20 text-purple-300'
                                  }`}>
                                  {rule.targetDecision.toUpperCase()}
                                </span>
                                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${getOutcomeColor(rule.predictedOutcome)}`}>
                                  → {rule.predictedOutcome.toUpperCase()}
                                </span>
                                {rule.samplesAffected > 0 && (
                                  <span className="text-[10px] text-slate-500">
                                    {rule.samplesAffected} samples
                                  </span>
                                )}
                              </div>
                              {/* Simple human-readable rule */}
                              <p className="text-sm text-white">{rule.ruleText}</p>
                              {/* Show original technical rule if different */}
                              {rule.decisionPath && rule.decisionPath !== rule.ruleText && (
                                <details className="mt-2">
                                  <summary className="text-[10px] text-slate-500 cursor-pointer hover:text-slate-400">
                                    View original technical rule
                                  </summary>
                                  <pre className="mt-1 p-2 bg-slate-950 rounded text-[10px] text-slate-400 font-mono overflow-x-auto whitespace-pre-wrap">
                                    {rule.decisionPath}
                                  </pre>
                                </details>
                              )}
                              {rule.notes && (
                                <p className="text-xs text-slate-500 mt-1">{rule.notes}</p>
                              )}
                            </>
                          )}
                        </div>

                        {/* Actions */}
                        {editingRule !== rule.id && (
                          <div className="flex items-center gap-1">
                            <button
                              onClick={() => { setEditingRule(rule.id); setEditText(rule.ruleText); }}
                              className="p-1 text-slate-400 hover:text-white hover:bg-slate-700 rounded"
                              title="Edit"
                            >
                              <Edit3 className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => copyRule(rule)}
                              className="p-1 text-slate-400 hover:text-white hover:bg-slate-700 rounded"
                              title="Copy"
                            >
                              <Copy className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => deleteRule(rule.id)}
                              className="p-1 text-slate-400 hover:text-red-400 hover:bg-red-500/20 rounded"
                              title="Delete"
                            >
                              <Trash2 className="w-3 h-3" />
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats Footer */}
      <div className="grid grid-cols-5 gap-4">
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
          <div className="text-2xl font-bold text-white">{shadowRules.length}</div>
          <div className="text-xs text-slate-400">Total Rules</div>
        </div>
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
          <div className="text-2xl font-bold text-emerald-400">{decisionTreeRulesCount}</div>
          <div className="text-xs text-slate-400">🌳 From Trees</div>
        </div>
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
          <div className="text-2xl font-bold text-violet-400">{discoveredRulesCount}</div>
          <div className="text-xs text-slate-400">💡 Discovered</div>
        </div>
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
          <div className="text-2xl font-bold text-blue-400">
            {shadowRules.filter(r => r.targetDecision === 'l1').length}
          </div>
          <div className="text-xs text-slate-400">L1 Rules</div>
        </div>
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
          <div className="text-2xl font-bold text-purple-400">
            {shadowRules.filter(r => r.targetDecision === 'l2').length}
          </div>
          <div className="text-xs text-slate-400">L2 Rules</div>
        </div>
      </div>
    </div>
  );
}
