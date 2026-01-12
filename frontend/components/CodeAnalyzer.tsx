"use client";
import { useState, useRef, useEffect, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Code2, Send, Sparkles, Copy, Check, ArrowRight, Bot, RotateCcw, Wand2, Users, TrendingUp, AlertTriangle, Target, Layers, GitBranch, BarChart3, Brain } from 'lucide-react';
import { guideCode } from '../lib/api';
import { AppContext, ChatMessage } from '@/app/page';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Model types for understanding analyst decision-making
const MODEL_TYPES = [
  {
    id: 'decision_tree',
    label: 'Decision Tree',
    description: 'Extract interpretable IF-THEN rules from analyst decisions',
    icon: GitBranch,
    color: 'text-blue-400',
    bg: 'bg-blue-500/20',
    prompt: `Generate Python code to build a Decision Tree model that explains L1 analyst decisions.

The code should:
1. Train a DecisionTreeClassifier to predict l1_decision based on transaction features
2. Extract human-readable rules using sklearn's export_text()
3. Visualize the decision tree using plot_tree()
4. Calculate feature importance to show which factors drive decisions
5. Output rules in business language: "IF transaction_amount > 5000 AND geo_mismatch = 1 THEN decision = Fraud"
6. Compare tree predictions to true_fraud_flag to assess analyst accuracy

Output a function called analyze_l1_decisions() that returns a dictionary with:
- decision_rules: list of extracted rules
- feature_importance: dict of feature -> importance score
- accuracy_metrics: precision, recall, f1 vs true_fraud_flag`
  },
  {
    id: 'logistic_regression',
    label: 'Logistic Regression',
    description: 'Quantify how each feature influences analyst decisions',
    icon: TrendingUp,
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/20',
    prompt: `Generate Python code to build a Logistic Regression model that quantifies feature effects on analyst decisions.

The code should:
1. Train LogisticRegression to predict l1_decision and l2_decision
2. Calculate odds ratios (exp of coefficients) for each feature
3. Interpret coefficients: "1 unit increase in velocity_score = 1.5x more likely to flag as fraud"
4. Identify which features have strongest positive/negative effects on fraud decisions
5. Compare L1 vs L2 coefficient patterns to find differences
6. Include 95% confidence intervals for coefficients

Output a function called analyze_decision_factors() that returns:
- l1_odds_ratios: dict of feature -> odds ratio with confidence interval
- l2_odds_ratios: dict of feature -> odds ratio with confidence interval
- interpretation: human-readable summary of key findings`
  },
  {
    id: 'bias_tests',
    label: 'Bias Detection',
    description: 'Statistical tests to find hidden biases in decisions',
    icon: AlertTriangle,
    color: 'text-amber-400',
    bg: 'bg-amber-500/20',
    prompt: `Generate Python code to perform statistical tests that detect biases in analyst decisions.

The code should:
1. Chi-square tests for categorical variables (geography, customer_segment, channel, l1_region)
2. T-tests for numeric thresholds (do analysts treat amounts differently?)
3. Calculate effect sizes (Cramer's V for categorical, Cohen's d for numeric)
4. Test if analysts override model predictions differently by customer type
5. Compare decision rates across analyst regions and risk appetites
6. Output a summary table with p-values and effect sizes

Output a function called detect_analyst_biases() that returns:
- categorical_biases: list of {variable, chi2, p_value, cramers_v, interpretation}
- numeric_biases: list of {variable, t_stat, p_value, cohens_d, interpretation}
- shadow_rules: list of detected undocumented patterns`
  },
  {
    id: 'accuracy_metrics',
    label: 'Accuracy Analysis',
    description: 'Compare analyst decisions to actual fraud outcomes',
    icon: Target,
    color: 'text-violet-400',
    bg: 'bg-violet-500/20',
    prompt: `Generate Python code to calculate comprehensive accuracy metrics for analyst decisions.

The code should:
1. Build confusion matrices for L1 and L2 decisions vs true_fraud_flag
2. Calculate accuracy, precision, recall, F1-score for each level
3. Compare analyst accuracy to model_risk_score predictions
4. Break down accuracy by analyst (l1_analyst_id, l2_analyst_id) to find best/worst performers
5. Analyze false positive/negative rates by transaction type and amount
6. Calculate potential cost savings from improving accuracy

Output a function called analyze_decision_accuracy() that returns:
- l1_metrics: {accuracy, precision, recall, f1, confusion_matrix}
- l2_metrics: {accuracy, precision, recall, f1, confusion_matrix}
- analyst_rankings: list of {analyst_id, accuracy, case_count} sorted by accuracy
- improvement_opportunities: list of cases where analysts could have done better`
  },
  {
    id: 'complete_model',
    label: 'Complete Analysis',
    description: 'Full mathematical model combining all techniques',
    icon: Brain,
    color: 'text-cyan-400',
    bg: 'bg-cyan-500/20',
    prompt: `Generate a comprehensive Python function that builds a complete mathematical model of analyst decision-making.

The function called build_analyst_decision_model() should:
1. Preprocess data: encode categoricals, handle missing values, scale numerics
2. Train Decision Tree and extract interpretable rules
3. Train Logistic Regression and calculate odds ratios
4. Perform chi-square and t-tests for bias detection
5. Calculate accuracy metrics comparing to true_fraud_flag
6. Cluster analysts by decision patterns using KMeans
7. Identify shadow rules (undocumented decision patterns)
8. Return a structured JSON compatible with the Global JSON schema

The output JSON should include:
- decision_rules: extracted IF-THEN rules
- feature_effects: odds ratios and importance scores
- detected_biases: statistical test results
- accuracy_metrics: L1 and L2 performance
- analyst_clusters: groups of similar decision-makers
- shadow_rules: undocumented patterns found
- recommendations: suggested improvements`
  }
];

export default function CodeAnalyzer({ onComplete }: { onComplete: () => void }) {
  const context = useContext(AppContext);
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(`sess_codegen_${context?.session.id || Math.random().toString(36).substr(2, 9)}`);
  const [copied, setCopied] = useState(false);
  const [customPrompt, setCustomPrompt] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Use messages from context
  const messages = context?.codeAnalyzerMessages || [];
  const setMessages = (newMessages: ChatMessage[]) => {
    context?.setCodeAnalyzerMessages(newMessages);
  };

  // Get data schema info for context
  const dataSchema = context?.dataSchema;
  const hasData = dataSchema !== null;

  // Build schema context for prompts - includes descriptions
  const buildSchemaContext = () => {
    if (!dataSchema) return '';
    
    const features = dataSchema.features || [];
    const featureList = features.map((f: any) => {
      const desc = f.description ? ` - ${f.description}` : '';
      return `- ${f.name} (${f.dtype})${desc}`;
    }).join('\n');
    
    const decisionCols = dataSchema.decision_columns || [];
    const analystCols = dataSchema.analyst_columns || [];
    
    return `
DATA SCHEMA:
Total rows: ${dataSchema.dataset_info?.total_samples || 'Unknown'}
Total columns: ${dataSchema.dataset_info?.total_features || features.length}

COLUMNS (with descriptions):
${featureList}

DECISION COLUMNS (target variables for modeling):
${decisionCols.length > 0 ? decisionCols.map((c: string) => `- ${c}`).join('\n') : '- l1_decision, l2_decision, true_fraud_flag'}

ANALYST ATTRIBUTE COLUMNS:
${analystCols.length > 0 ? analystCols.slice(0, 10).map((c: string) => `- ${c}`).join('\n') : '- l1_analyst_id, l1_tenure_months, l2_analyst_id, etc.'}
`;
  };

  // Get full schema JSON for API
  const getSchemaJson = () => {
    if (!dataSchema) return undefined;
    return JSON.stringify(dataSchema, null, 2);
  };

  const handleGenerateModel = async (modelType: typeof MODEL_TYPES[0]) => {
    if (!hasData) return;

    setLoading(true);
    
    const userMessage = `Generate ${modelType.label} code to understand analyst decisions`;
    const newMessages: ChatMessage[] = [...messages, { type: 'user', content: userMessage }];
    setMessages(newMessages);

    try {
      const fullPrompt = `${modelType.prompt}

DATA SCHEMA CONTEXT:
${buildSchemaContext()}

REQUIREMENTS:
- Use pandas for data manipulation
- Use scikit-learn for modeling (DecisionTreeClassifier, LogisticRegression)
- Use scipy.stats for statistical tests (chi2_contingency, ttest_ind)
- Include clear comments explaining each step
- Handle missing values appropriately (drop or impute)
- Make the code production-ready and well-documented
- The output function should return a dictionary/JSON that can be used in the Global JSON`;

      // Pass the full data schema JSON to the backend
      const res = await guideCode(sessionId, fullPrompt, getSchemaJson());
      
      // Save the generated code to context
      if (res.response) {
        context?.setMlCode(res.response);
      }
      
      setMessages([...newMessages, { type: 'ai', content: res.response }]);
    } catch (e) {
      setMessages([...newMessages, { type: 'ai', content: "Failed to generate code. Please check your connection and try again." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleCustomPrompt = async () => {
    if (!customPrompt.trim() || !hasData) return;

    setLoading(true);
    
    const newMessages: ChatMessage[] = [...messages, { type: 'user', content: customPrompt }];
    setMessages(newMessages);
    setCustomPrompt('');

    try {
      const fullPrompt = `${customPrompt}

CONTEXT: We are building mathematical models to understand how fraud analysts make decisions.
The goal is to use Decision Trees, Logistic Regression, and statistical tests to identify:
- What factors drive L1 and L2 analyst decisions (l1_decision, l2_decision columns)
- Shadow rules (undocumented patterns in analyst behavior)
- Potential biases in analyst decisions
- Accuracy of analyst decisions vs true_fraud_flag

DATA SCHEMA:
${buildSchemaContext()}

Generate Python code that addresses the request above.`;

      // Pass the full data schema JSON to the backend
      const res = await guideCode(sessionId, fullPrompt, getSchemaJson());
      
      if (res.response) {
        context?.setMlCode(res.response);
      }
      
      setMessages([...newMessages, { type: 'ai', content: res.response }]);
    } catch (e) {
      setMessages([...newMessages, { type: 'ai', content: "Failed to generate code. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    context?.setMlCode('');
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const extractCodeBlock = (content: string) => {
    const match = content.match(/```python\n([\s\S]*?)```/);
    return match ? match[1] : null;
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check if we have generated code
  const hasGeneratedCode = context?.mlCode && context.mlCode.length > 0;

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl mb-4 shadow-lg shadow-violet-500/20">
          <Code2 className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-3xl font-bold text-white mb-2">Code Generator</h2>
        <p className="text-slate-400 max-w-lg mx-auto">
          Generate Python code with mathematical models to understand how L1 and L2 analysts make fraud decisions.
        </p>
      </div>

      {/* Data requirement check */}
      {!hasData && (
        <div className="mb-6 p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl">
          <div className="flex items-center gap-2 text-amber-300">
            <AlertTriangle className="w-5 h-5" />
            <span className="font-medium">Data Required</span>
          </div>
          <p className="text-sm text-amber-200/80 mt-1">
            Please upload your fraud alert data first to generate analysis code.
          </p>
        </div>
      )}

      {/* Model Type Selection */}
      {hasData && messages.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h3 className="text-sm font-medium text-slate-400 mb-4 text-center">Choose a Mathematical Model</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {MODEL_TYPES.map((model) => {
              const Icon = model.icon;
              return (
                <button
                  key={model.id}
                  onClick={() => handleGenerateModel(model)}
                  disabled={loading}
                  className={`flex items-start gap-4 p-4 rounded-xl border transition-all text-left hover:scale-[1.02] ${
                    loading ? 'opacity-50 cursor-not-allowed' : 'hover:border-slate-600'
                  } bg-slate-900/50 border-slate-800`}
                >
                  <div className={`p-2 rounded-lg ${model.bg}`}>
                    <Icon className={`w-5 h-5 ${model.color}`} />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-white">{model.label}</div>
                    <div className="text-xs text-slate-500 mt-0.5">{model.description}</div>
                  </div>
                  <Wand2 className="w-4 h-4 text-slate-600" />
                </button>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Chat Interface */}
      <div className="bg-[#0d1117] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
        {/* Chat Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-violet-900/20 to-purple-900/20 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Bot className="w-4 h-4 text-violet-400" />
            <span className="text-sm font-medium text-violet-300">Code Generator</span>
          </div>
          <div className="flex items-center gap-2">
            {hasData && (
              <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
                ✓ {dataSchema?.dataset_info?.total_features || dataSchema?.features?.length || 0} columns
              </span>
            )}
            {messages.length > 0 && (
              <button
                onClick={handleClearChat}
                className="flex items-center gap-1 text-xs text-slate-500 hover:text-white transition-colors"
              >
                <RotateCcw className="w-3 h-3" />
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="h-[400px] overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && hasData && (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <div className="w-16 h-16 bg-violet-500/10 rounded-full flex items-center justify-center mb-4">
                <Sparkles className="w-8 h-8 text-violet-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Ready to Generate Code</h3>
              <p className="text-slate-400 text-sm max-w-xs">
                Select a mathematical model above or describe what you want to understand about analyst decisions.
              </p>
            </div>
          )}

          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${m.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[90%] rounded-xl p-4 text-sm ${
                m.type === 'user'
                  ? 'bg-violet-600/30 border border-violet-500/30 text-violet-100'
                  : 'bg-slate-800/50 border border-slate-700/50 text-slate-200'
              }`}>
                {m.type === 'ai' && extractCodeBlock(m.content) && (
                  <div className="mb-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-violet-400 font-mono">Generated Analysis Code</span>
                      <button
                        onClick={() => copyToClipboard(extractCodeBlock(m.content) || '')}
                        className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors"
                      >
                        {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                        {copied ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                    <pre className="bg-slate-950 p-3 rounded-lg overflow-x-auto text-xs text-violet-300 font-mono max-h-[300px] overflow-y-auto">
                      {extractCodeBlock(m.content)}
                    </pre>
                  </div>
                )}
                <div className="prose prose-invert max-w-none text-sm leading-relaxed">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      code(props: any) {
                        const { node, className, children, ...rest } = props;
                        const match = /language-(\w+)/.exec(className || '');
                        const isInline = !match && !String(children).includes('\n');
                        return !isInline && match ? (
                          <pre className="bg-slate-950 p-3 rounded-lg overflow-x-auto text-xs text-violet-300 font-mono">
                            <code className={className} {...rest}>
                              {children}
                            </code>
                          </pre>
                        ) : (
                          <code className="bg-slate-700/50 text-slate-200 px-1 rounded-sm text-xs" {...rest}>
                            {children}
                          </code>
                        );
                      },
                      a(props: any) {
                        const { node, ...rest } = props;
                        return <a className="text-violet-400 hover:underline" {...rest} />;
                      },
                      p(props: any) {
                        const { node, ...rest } = props;
                        return <p className="mb-2 last:mb-0" {...rest} />;
                      },
                      ul(props: any) {
                        const { node, ...rest } = props;
                        return <ul className="list-disc list-inside mb-2" {...rest} />;
                      },
                      ol(props: any) {
                        const { node, ...rest } = props;
                        return <ol className="list-decimal list-inside mb-2" {...rest} />;
                      },
                      li(props: any) {
                        const { node, ...rest } = props;
                        return <li className="mb-1" {...rest} />;
                      },
                    }}
                  >
                    {m.type === 'ai' ? m.content.replace(/```python\n[\s\S]*?```/g, '[Code shown above]') : m.content}
                  </ReactMarkdown>
                </div>
              </div>
            </motion.div>
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-slate-400 text-sm">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
              <span>Generating analysis code...</span>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Section */}
        {hasData && (
          <div className="p-4 border-t border-slate-800 bg-slate-900/30">
            {/* Quick Actions */}
            {messages.length > 0 && (
              <div className="flex gap-2 mb-3 flex-wrap">
                {MODEL_TYPES.slice(0, 4).map((model) => (
                  <button
                    key={model.id}
                    onClick={() => handleGenerateModel(model)}
                    disabled={loading}
                    className="px-3 py-1 text-xs bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded-lg text-slate-400 hover:text-white transition-colors disabled:opacity-50"
                  >
                    {model.label}
                  </button>
                ))}
              </div>
            )}

            {/* Custom Input */}
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={customPrompt}
                onChange={e => setCustomPrompt(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleCustomPrompt()}
                placeholder="Describe what you want to understand about analyst decisions..."
                className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-white placeholder-slate-500 outline-none focus:border-violet-500/50 transition-colors"
                disabled={loading}
              />
              <button
                onClick={handleCustomPrompt}
                disabled={loading || !customPrompt.trim()}
                className="px-4 py-2.5 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 disabled:opacity-50 rounded-lg text-white transition-all"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Continue Button */}
      {hasGeneratedCode && (
        <div className="flex justify-center mt-6">
          <button
            onClick={onComplete}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 rounded-xl text-white font-medium transition-all shadow-lg shadow-violet-500/20"
          >
            Continue to Patterns
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
