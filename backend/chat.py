import os
import json
from typing import Dict, List, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from models import (
    ExplainGlobalRequest, ExplainTransactionRequest, ChatResponse, TxnChatResponse,
    HumanReadableShadowRule, ShadowRulesConversionResponse, ExtractShadowRulesFromTxnResponse
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHAT_HISTORY_FILE = "chat_memory.json"

# JSON Schema templates for ANALYST DECISION analysis
GLOBAL_JSON_SCHEMA = """
{
  "analysis_version": "string - analysis version identifier",
  "generated_at": "ISO 8601 datetime string",
  "dataset_summary": {
    "total_alerts": number,
    "fraud_rate": 0.0 to 1.0,
    "l1_override_rate": 0.0 to 1.0,
    "l2_override_rate": 0.0 to 1.0
  },
  "l1_decision_factors": [
    {
      "factor": "factor name (e.g., transaction_amount, velocity_score)",
      "importance": 0.0 to 1.0,
      "direction": "increases_fraud_call" or "decreases_fraud_call",
      "threshold_effects": "description of any threshold behavior",
      "notes": "explanation of how this factor influences L1 decisions"
    }
  ],
  "l2_decision_factors": [
    {
      "factor": "factor name",
      "importance": 0.0 to 1.0,
      "direction": "increases_fraud_call" or "decreases_fraud_call",
      "override_influence": "how this factor influences L2 overriding L1",
      "notes": "explanation"
    }
  ],
  "shadow_rules_detected": [
    {
      "rule_description": "Plain English description of undocumented pattern",
      "affected_population": "who is affected (e.g., customers from region X)",
      "evidence": "statistical evidence for this rule",
      "severity": "high", "medium", or "low",
      "potential_bias": "type of bias if applicable"
    }
  ],
  "analyst_consistency": {
    "l1_inter_analyst_variance": 0.0 to 1.0,
    "l2_inter_analyst_variance": 0.0 to 1.0,
    "l1_l2_agreement_rate": 0.0 to 1.0,
    "notable_outlier_analysts": ["analyst_id_1", "analyst_id_2"]
  },
  "accuracy_metrics": {
    "l1_vs_true_fraud": { "accuracy": 0.0-1.0, "precision": 0.0-1.0, "recall": 0.0-1.0 },
    "l2_vs_true_fraud": { "accuracy": 0.0-1.0, "precision": 0.0-1.0, "recall": 0.0-1.0 },
    "system_vs_true_fraud": { "accuracy": 0.0-1.0, "precision": 0.0-1.0, "recall": 0.0-1.0 }
  },
  "key_findings": ["Finding 1", "Finding 2"],
  "recommendations": ["Recommendation 1", "Recommendation 2"]
}
"""

TXN_JSON_SCHEMA = """
{
  "alert_id": "unique alert identifier",
  "analysis_version": "string",
  "generated_at": "ISO 8601 datetime string",
  "alert_context": {
    "transaction_amount": number,
    "channel": "card/ACH/wire/P2P",
    "model_risk_score": 0.0-1.0,
    "expected_action": "block/allow/review"
  },
  "l1_analysis": {
    "analyst_id": "string",
    "decision": "fraud/legit",
    "handling_time_sec": number,
    "override_flag": boolean,
    "key_factors": [
      {
        "factor": "factor name",
        "value": "actual value",
        "influence": "why this likely influenced the decision",
        "compared_to_norm": "above/below/within normal range"
      }
    ],
    "decision_rationale_hypothesis": "Plain English hypothesis of why L1 made this decision"
  },
  "l2_analysis": {
    "analyst_id": "string",
    "decision": "fraud/legit",
    "override_flag": boolean,
    "override_reason": "reason if overridden",
    "key_factors": [...],
    "decision_rationale_hypothesis": "Plain English hypothesis"
  },
  "shadow_rule_indicators": [
    {
      "rule_pattern": "description of potential undocumented rule applied",
      "confidence": 0.0-1.0,
      "evidence": "why we suspect this rule was applied"
    }
  ],
  "discrepancies": [
    {
      "type": "l1_vs_l2 / l1_vs_system / l2_vs_true_fraud",
      "description": "what was different and why"
    }
  ],
  "narrative_plain": ["Plain English explanation of the analyst decisions on this alert"],
  "recommendations": ["What could be improved for similar future cases"]
}
"""

EXAMPLE_EXPLAIN_GLOBAL = '''
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def analyze_analyst_decisions(df, analysis_version="v1"):
    """
    Analyze L1 and L2 analyst decision patterns to understand how they reach fraud conclusions.
    
    Args:
        df: DataFrame with fraud alert data including analyst decisions
        analysis_version: Version string for tracking
    
    Returns:
        dict: Global JSON with analyst decision patterns, shadow rules, and recommendations
    """
    # Dataset summary
    total_alerts = len(df)
    fraud_rate = df['true_fraud_flag'].mean() if 'true_fraud_flag' in df.columns else None
    l1_override_rate = df['l1_override_flag'].mean() if 'l1_override_flag' in df.columns else None
    l2_override_rate = df['l2_override_flag'].mean() if 'l2_override_flag' in df.columns else None
    
    # L1 Decision Factors - use decision tree to find patterns
    l1_features = ['transaction_amount', 'model_risk_score', 'velocity_score', 
                   'device_risk_score', 'geo_mismatch', 'is_night_txn', 'is_new_merchant']
    l1_features = [f for f in l1_features if f in df.columns]
    
    l1_decision_factors = []
    if l1_features and 'l1_decision' in df.columns:
        X_l1 = df[l1_features].fillna(0)
        y_l1 = (df['l1_decision'] == 'Fraud').astype(int)
        
        dt_l1 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50)
        dt_l1.fit(X_l1, y_l1)
        
        importances = dt_l1.feature_importances_
        for i, feat in enumerate(l1_features):
            if importances[i] > 0.01:
                l1_decision_factors.append({
                    "factor": feat,
                    "importance": float(importances[i]),
                    "direction": "increases_fraud_call",
                    "notes": f"Decision tree importance: {importances[i]:.3f}"
                })
        l1_decision_factors.sort(key=lambda x: x['importance'], reverse=True)
    
    # Shadow Rules Detection - find undocumented patterns
    shadow_rules = []
    
    # Check for geographic bias
    if 'merchant_country' in df.columns and 'l1_decision' in df.columns:
        country_fraud_rate = df.groupby('merchant_country')['l1_decision'].apply(
            lambda x: (x == 'Fraud').mean()
        )
        outlier_countries = country_fraud_rate[country_fraud_rate > country_fraud_rate.mean() + 2*country_fraud_rate.std()]
        for country in outlier_countries.index:
            shadow_rules.append({
                "rule_description": f"Transactions from {country} flagged as fraud at unusually high rate",
                "affected_population": f"Merchants in {country}",
                "evidence": f"Fraud call rate: {outlier_countries[country]:.1%} vs avg {country_fraud_rate.mean():.1%}",
                "severity": "high",
                "potential_bias": "geographic"
            })
    
    # Check for amount thresholds
    if 'transaction_amount' in df.columns and 'l1_decision' in df.columns:
        df['amount_bucket'] = pd.qcut(df['transaction_amount'], q=10, duplicates='drop')
        bucket_fraud_rate = df.groupby('amount_bucket')['l1_decision'].apply(lambda x: (x == 'Fraud').mean())
        # Look for sharp jumps
        for i in range(1, len(bucket_fraud_rate)):
            if bucket_fraud_rate.iloc[i] - bucket_fraud_rate.iloc[i-1] > 0.2:
                shadow_rules.append({
                    "rule_description": f"Sharp increase in fraud calls above amount threshold",
                    "affected_population": f"Transactions in {bucket_fraud_rate.index[i]}",
                    "evidence": f"Fraud rate jumps from {bucket_fraud_rate.iloc[i-1]:.1%} to {bucket_fraud_rate.iloc[i]:.1%}",
                    "severity": "medium",
                    "potential_bias": "amount threshold"
                })
                break
    
    # Analyst consistency
    analyst_consistency = {}
    if 'l1_analyst_id' in df.columns and 'l1_decision' in df.columns:
        l1_rates = df.groupby('l1_analyst_id')['l1_decision'].apply(lambda x: (x == 'Fraud').mean())
        analyst_consistency['l1_inter_analyst_variance'] = float(l1_rates.std())
        analyst_consistency['notable_outlier_analysts'] = list(
            l1_rates[abs(l1_rates - l1_rates.mean()) > 2*l1_rates.std()].index
        )
    
    # Accuracy metrics
    accuracy_metrics = {}
    if 'true_fraud_flag' in df.columns:
        y_true = df['true_fraud_flag'].astype(int)
        if 'l1_decision' in df.columns:
            y_l1 = (df['l1_decision'] == 'Fraud').astype(int)
            accuracy_metrics['l1_vs_true_fraud'] = {
                'accuracy': float(accuracy_score(y_true, y_l1)),
                'precision': float(precision_score(y_true, y_l1, zero_division=0)),
                'recall': float(recall_score(y_true, y_l1, zero_division=0))
            }
    
    return {
        "analysis_version": analysis_version,
        "generated_at": datetime.now().isoformat(),
        "dataset_summary": {
            "total_alerts": total_alerts,
            "fraud_rate": fraud_rate,
            "l1_override_rate": l1_override_rate,
            "l2_override_rate": l2_override_rate
        },
        "l1_decision_factors": l1_decision_factors,
        "l2_decision_factors": [],  # Similar analysis for L2
        "shadow_rules_detected": shadow_rules,
        "analyst_consistency": analyst_consistency,
        "accuracy_metrics": accuracy_metrics,
        "key_findings": [
            f"L1 analysts override system {l1_override_rate:.1%} of the time" if l1_override_rate else "Override rate not available",
            f"Top decision factor: {l1_decision_factors[0]['factor']}" if l1_decision_factors else "Decision factors not analyzed"
        ],
        "recommendations": [
            "Review cases where L1 and L2 decisions differ",
            "Investigate geographic patterns for potential bias"
        ]
    }
'''

EXAMPLE_EXPLAIN_TXN = '''
from datetime import datetime
import pandas as pd
import numpy as np

def analyze_single_alert(df, global_patterns, alert_id, analysis_version="v1"):
    """
    Analyze a single fraud alert to understand L1 and L2 analyst decisions.
    
    Args:
        df: Full DataFrame with all alerts (for comparison context)
        global_patterns: Output from analyze_analyst_decisions() for reference
        alert_id: The specific alert_id to analyze
        analysis_version: Version string
    
    Returns:
        dict: Case-level JSON explaining analyst decisions on this specific alert
    """
    # Get the specific alert
    alert = df[df['alert_id'] == alert_id].iloc[0]
    
    # Alert context
    alert_context = {
        "transaction_amount": float(alert.get('transaction_amount', 0)),
        "channel": str(alert.get('channel', 'unknown')),
        "model_risk_score": float(alert.get('model_risk_score', 0)),
        "expected_action": str(alert.get('expected_action', 'unknown'))
    }
    
    # Analyze L1 decision
    l1_key_factors = []
    
    # Compare amount to typical fraud cases
    fraud_avg_amount = df[df['l1_decision'] == 'Fraud']['transaction_amount'].mean()
    if alert['transaction_amount'] > fraud_avg_amount:
        l1_key_factors.append({
            "factor": "transaction_amount",
            "value": float(alert['transaction_amount']),
            "influence": "High amount relative to fraud average may have raised suspicion",
            "compared_to_norm": "above"
        })
    
    # Check velocity score
    if 'velocity_score' in alert and alert['velocity_score'] > df['velocity_score'].quantile(0.75):
        l1_key_factors.append({
            "factor": "velocity_score",
            "value": float(alert['velocity_score']),
            "influence": "High velocity indicates unusual transaction frequency",
            "compared_to_norm": "above"
        })
    
    # Check geographic mismatch
    if alert.get('geo_mismatch', 0) == 1:
        l1_key_factors.append({
            "factor": "geo_mismatch",
            "value": True,
            "influence": "Merchant and customer country mismatch often triggers scrutiny",
            "compared_to_norm": "unusual"
        })
    
    l1_analysis = {
        "analyst_id": str(alert.get('l1_analyst_id', 'unknown')),
        "decision": str(alert.get('l1_decision', 'unknown')),
        "handling_time_sec": int(alert.get('l1_handling_time_sec', 0)),
        "override_flag": bool(alert.get('l1_override_flag', False)),
        "key_factors": l1_key_factors,
        "decision_rationale_hypothesis": _generate_l1_hypothesis(alert, l1_key_factors)
    }
    
    # Analyze L2 decision
    l2_analysis = {
        "analyst_id": str(alert.get('l2_analyst_id', 'unknown')),
        "decision": str(alert.get('l2_decision', 'unknown')),
        "override_flag": bool(alert.get('l2_override_flag', False)),
        "override_reason": str(alert.get('l2_override_reason', '')) if alert.get('l2_override_flag') else None,
        "key_factors": [],  # Similar analysis for L2
        "decision_rationale_hypothesis": _generate_l2_hypothesis(alert)
    }
    
    # Detect shadow rule indicators for this case
    shadow_indicators = []
    
    # Check if decision contradicts model
    if alert.get('expected_action') == 'allow' and alert.get('l1_decision') == 'Fraud':
        shadow_indicators.append({
            "rule_pattern": "Analyst flagged as fraud despite model recommending allow",
            "confidence": 0.7,
            "evidence": f"Model score {alert.get('model_risk_score', 0):.2f} was low but analyst overrode"
        })
    
    # Check for potential geographic bias
    if global_patterns and 'shadow_rules_detected' in global_patterns:
        for rule in global_patterns['shadow_rules_detected']:
            if 'geographic' in rule.get('potential_bias', '').lower():
                merchant_country = alert.get('merchant_country', '')
                if merchant_country in rule.get('affected_population', ''):
                    shadow_indicators.append({
                        "rule_pattern": rule['rule_description'],
                        "confidence": 0.8,
                        "evidence": f"This alert's merchant country matches known geographic bias pattern"
                    })
    
    # Identify discrepancies
    discrepancies = []
    if alert.get('l1_decision') != alert.get('l2_decision'):
        discrepancies.append({
            "type": "l1_vs_l2",
            "description": f"L1 decided {alert.get('l1_decision')} but L2 changed to {alert.get('l2_decision')}"
        })
    
    if alert.get('l1_override_flag'):
        discrepancies.append({
            "type": "l1_vs_system",
            "description": f"L1 overrode system recommendation of {alert.get('expected_action')}"
        })
    
    final_decision = alert.get('l2_decision', alert.get('l1_decision', 'unknown'))
    if 'true_fraud_flag' in alert:
        true_outcome = 'Fraud' if alert['true_fraud_flag'] else 'Legit'
        if final_decision != true_outcome:
            discrepancies.append({
                "type": "decision_vs_true_fraud",
                "description": f"Final decision was {final_decision} but true outcome was {true_outcome}"
            })
    
    return {
        "alert_id": str(alert_id),
        "analysis_version": analysis_version,
        "generated_at": datetime.now().isoformat(),
        "alert_context": alert_context,
        "l1_analysis": l1_analysis,
        "l2_analysis": l2_analysis,
        "shadow_rule_indicators": shadow_indicators,
        "discrepancies": discrepancies,
        "narrative_plain": [
            f"Alert {alert_id}: {alert_context['channel']} transaction for ${alert_context['transaction_amount']:.2f}",
            f"L1 analyst ({l1_analysis['analyst_id']}) decided: {l1_analysis['decision']} in {l1_analysis['handling_time_sec']}s",
            f"L2 analyst ({l2_analysis['analyst_id']}) decided: {l2_analysis['decision']}"
        ],
        "recommendations": _generate_case_recommendations(alert, discrepancies, shadow_indicators)
    }

def _generate_l1_hypothesis(alert, key_factors):
    """Generate a hypothesis for why L1 made their decision."""
    factors = [f["factor"] for f in key_factors]
    if alert.get('l1_decision') == 'Fraud':
        return f"L1 likely flagged as fraud due to: {', '.join(factors) if factors else 'overall risk profile'}"
    else:
        return f"L1 likely cleared the transaction as legitimate despite {', '.join(factors) if factors else 'some risk signals'}"

def _generate_l2_hypothesis(alert):
    """Generate a hypothesis for L2 decision."""
    if alert.get('l2_override_flag'):
        return f"L2 overrode L1 decision. Reason: {alert.get('l2_override_reason', 'Not documented')}"
    else:
        return f"L2 confirmed L1 decision of {alert.get('l1_decision', 'unknown')}"

def _generate_case_recommendations(alert, discrepancies, shadow_indicators):
    """Generate recommendations for this specific case."""
    recs = []
    if discrepancies:
        recs.append("Review decision discrepancies with analysts involved")
    if shadow_indicators:
        recs.append("Investigate potential undocumented decision criteria")
    if alert.get('l1_handling_time_sec', 0) < 30:
        recs.append("Very quick handling time - verify adequate review")
    return recs if recs else ["No specific recommendations for this case"]
'''


class ChatManager:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables.")
        
        self.llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0.7,
            anthropic_api_key=api_key
        )
        self.sessions: Dict[str, ChatMessageHistory] = {}
        self.load_history()

    def load_history(self):
        if os.path.exists(CHAT_HISTORY_FILE):
            try:
                with open(CHAT_HISTORY_FILE, "r") as f:
                    data = json.load(f)
                    for session_id, messages in data.items():
                        history = ChatMessageHistory()
                        for msg in messages:
                            if msg["type"] == "human":
                                history.add_user_message(msg["content"])
                            elif msg["type"] == "ai":
                                history.add_ai_message(msg["content"])
                            elif msg["type"] == "system":
                                history.add_message(SystemMessage(content=msg["content"]))
                        self.sessions[session_id] = history
            except Exception as e:
                logger.error(f"Failed to load chat history: {e}")

    def save_history(self):
        data = {}
        for session_id, history in self.sessions.items():
            messages = []
            for msg in history.messages:
                msg_type = "unknown"
                if isinstance(msg, HumanMessage): msg_type = "human"
                elif isinstance(msg, AIMessage): msg_type = "ai"
                elif isinstance(msg, SystemMessage): msg_type = "system"
                messages.append({"type": msg_type, "content": msg.content})
            data[session_id] = messages
        
        try:
            with open(CHAT_HISTORY_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]

    async def chat(self, session_id: str, user_input: str, context: Dict[str, Any] = None) -> ChatResponse:
        history = self.get_session_history(session_id)
        
        system_content = """You are an expert Data Scientist specializing in understanding human decision-making through mathematical and statistical models.

CONTEXT - FRAUD ALERT WORKFLOW:
In banking fraud detection, the workflow is:
1. ML Model flags transactions as potentially fraudulent (model_risk_score, expected_action)
2. L1 Analysts review flagged transactions and make initial decisions (l1_decision)
3. L2 Analysts review escalated or random-sampled cases (l2_decision)
4. Final outcome is compared against confirmed fraud (true_fraud_flag)

YOUR MISSION:
Help users BUILD MATHEMATICAL MODELS to understand and predict how L1 and L2 analysts make their decisions. Use the data schema to suggest appropriate statistical and ML techniques.

You have access to the user's context which may include:
1. 'ml_code': Python analysis code already generated
2. 'data_schema': Dataset structure with column names, types, and descriptions
3. 'global': Global patterns already discovered in analyst decisions
4. 'txn': Individual case analysis

MATHEMATICAL METHODS TO SUGGEST:

1. **DECISION TREE MODELS** (for interpretable rules):
   - Train DecisionTreeClassifier on l1_decision / l2_decision
   - Extract rules to understand decision boundaries
   - Visualize with tree plots
   - Use: sklearn.tree.DecisionTreeClassifier, export_text, plot_tree

2. **LOGISTIC REGRESSION** (for feature importance):
   - Model P(fraud_decision) as function of features
   - Interpret coefficients as odds ratios
   - Use: sklearn.linear_model.LogisticRegression
   - Show: which features increase/decrease fraud call probability

3. **RANDOM FOREST / GRADIENT BOOSTING** (for complex patterns):
   - Capture non-linear interactions
   - Feature importance rankings
   - Use: sklearn.ensemble.RandomForestClassifier, XGBClassifier
   - Partial dependence plots for feature effects

4. **CLUSTERING** (for analyst behavior grouping):
   - Group analysts by decision patterns (KMeans, DBSCAN)
   - Identify "hawk" vs "dove" analysts
   - Use: sklearn.cluster.KMeans

5. **STATISTICAL TESTS** (for bias detection):
   - Chi-square tests for categorical biases (geography, segment)
   - T-tests for numeric thresholds (amount, velocity)
   - Use: scipy.stats.chi2_contingency, ttest_ind

6. **ASSOCIATION RULES** (for shadow rule discovery):
   - Find frequent patterns: IF condition THEN decision
   - Use: mlxtend.frequent_patterns.apriori, association_rules

7. **SURVIVAL ANALYSIS** (for handling time patterns):
   - Model time-to-decision
   - Identify rushed vs thorough reviews
   - Use: lifelines.KaplanMeierFitter

WHEN GENERATING CODE SUGGESTIONS:
- Always use the DATA SCHEMA to reference correct column names
- Include data preprocessing (handle missing values, encode categoricals)
- Show how to interpret results in business terms
- Calculate accuracy, precision, recall vs true_fraud_flag

WHEN DISCUSSING RESULTS:
- Translate mathematical findings to business insights
- Identify shadow rules: "Analysts flag as fraud when amount > X AND geo_mismatch = True"
- Quantify biases: "Analysts are 2.3x more likely to flag transactions from Region Y"
- Compare analyst accuracy: "L1 precision = 0.72, L2 precision = 0.85"

IMPORTANT - SUGGESTIONS:
Provide actionable code suggestions in global_json_suggestion when you notice:
- A mathematical method that would reveal hidden patterns
- Missing preprocessing for better model accuracy
- Additional metrics to calculate
- Visualization that would clarify findings

Format suggestions as brief, specific code improvements."""

        # Build messages list manually to avoid template parsing issues
        messages = [SystemMessage(content=system_content)]
        
        # Add context as a separate system message if available
        if context:
            context_parts = []
            if context.get('ml_code'):
                code_preview = context['ml_code']
                context_parts.append(f"=== USER'S ML CODE ===\n{code_preview}")
            if context.get('data_schema'):
                schema_preview = json.dumps(context['data_schema'], indent=2)
                context_parts.append(f"=== DATA SCHEMA ===\n{schema_preview}")
            if context.get('global'):
                global_preview = json.dumps(context['global'], indent=2)
                context_parts.append(f"=== GLOBAL EXPLANATIONS ===\n{global_preview}")
            if context.get('txn'):
                txn_data = json.dumps(context['txn'], indent=2)
                context_parts.append(f"=== TRANSACTION EXPLANATION ===\n{txn_data}")
            
            if context_parts:
                context_message = "--- CURRENT CONTEXT ---\n\n" + "\n\n".join(context_parts)
                messages.append(SystemMessage(content=context_message))
        
        # Add conversation history
        for msg in history.messages[-5:]:
            messages.append(msg)
        
        # Add current user input
        messages.append(HumanMessage(content=user_input))
        
        # Use structured output for better response handling
        structured_llm = self.llm.with_structured_output(ChatResponse)
        response: ChatResponse = structured_llm.invoke(messages)
        
        history.add_user_message(user_input)
        history.add_ai_message(response.general_answer)
        self.save_history()

        return response

    async def chat_txn(
        self, 
        session_id: str, 
        user_input: str, 
        context: Dict[str, Any] = None
    ) -> TxnChatResponse:
        """
        Chat method for analyzing wrong predictions (false positives/negatives).
        Uses XGBoost SHAP analysis to understand why analyst decisions were wrong.
        
        Context includes:
        - wrong_predictions: List of false positives and false negatives
        - importance_summary: Feature importance from Random Forest analysis
        - l1_analysis/l2_analysis: XGBoost analysis results
        - prediction_breakdown: TP/FP/TN/FN counts
        """
        history = self.get_session_history(session_id)
        
        system_content = """You are an expert in ML Explainability and Fraud Analysis for banking systems.

YOUR ROLE:
Analyze WRONG PREDICTIONS (false positives and false negatives) to understand why L1/L2 analysts made incorrect decisions.
Use the XGBoost SHAP analysis to explain which features influenced incorrect decisions.
Provide actionable insights for improving analyst training and decision-making.

CONTEXT AVAILABLE:
1. 'wrong_predictions': Cases where analyst decisions did not match the true fraud outcome
   - False Positives (FP): Analyst flagged as fraud, but was actually legitimate
   - False Negatives (FN): Analyst cleared as legitimate, but was actually fraud
2. 'importance_summary': Feature importance showing which features most influence decisions
3. 'l1_analysis': XGBoost model analysis of L1 analyst decisions
4. 'l2_analysis': XGBoost model analysis of L2 analyst decisions  
5. 'prediction_breakdown': Overall accuracy metrics (TP, FP, TN, FN counts)
6. 'data_schema': Feature definitions and column meanings
7. 'guidelines': Bank's official policies and regulatory requirements

YOUR APPROACH FOR ANALYZING WRONG PREDICTIONS:

1. **UNDERSTAND THE ERROR TYPE**:
   - For False Positives: Why did the analyst think this was fraud when it wasn't?
     - What features led them astray?
     - Were they being too cautious? Following a shadow rule?
   - For False Negatives: Why did the analyst clear this when it was actually fraud?
     - What features made the fraud look legitimate?
     - Did they miss key warning signs?

2. **USE SHAP INSIGHTS**:
   - Reference the top SHAP features when explaining decisions
   - Explain how feature values in the wrong cases differed from typical patterns
   - Show which features the XGBoost model found most predictive

3. **IDENTIFY PATTERNS IN ERRORS**:
   - Are there common characteristics in the wrong predictions?
   - Do certain features consistently lead to errors?
   - Are L1 or L2 analysts making more of certain error types?

4. **ROOT CAUSE ANALYSIS**:
   - Was the error due to: Training gap? Guideline ambiguity? Edge case? Time pressure?
   - Could the analyst have caught this with better information?
   - Are there systemic issues causing these errors?

5. **RECOMMENDATIONS**:
   - What training would help prevent these errors?
   - Should guidelines be updated?
   - Are there process improvements to suggest?

WHEN PROVIDING INSIGHTS:
- shadow_rule_detected: If the error stems from an undocumented pattern
- what_if_insight: "If the analyst had noticed X, they would have caught this"
- risk_flag: Concerns about the error (e.g., "This FN was a high-value fraud")

BE SPECIFIC:
- Reference actual values from the wrong prediction cases
- Quote SHAP feature importance when explaining feature effects
- Quantify patterns when possible
- Provide concrete, actionable recommendations"""

        messages = [SystemMessage(content=system_content)]
        
        # Add context as a separate system message
        if context:
            context_parts = []
            
            # Wrong predictions is the primary context (from XGBoost analysis)
            if context.get('wrong_predictions'):
                wrong_preds = context['wrong_predictions']
                if isinstance(wrong_preds, list) and len(wrong_preds) > 0:
                    fp_cases = [p for p in wrong_preds if p.get('case_type') == 'false_positive']
                    fn_cases = [p for p in wrong_preds if p.get('case_type') == 'false_negative']
                    
                    context_parts.append(f"=== WRONG PREDICTIONS ({len(wrong_preds)} cases) ===")
                    context_parts.append(f"False Positives: {len(fp_cases)} | False Negatives: {len(fn_cases)}")
                    
                    # Include case data (limit to first 10 for token efficiency)
                    cases_to_show = wrong_preds[:10]
                    context_parts.append(f"\nCase Details (showing {len(cases_to_show)} of {len(wrong_preds)}):")
                    context_parts.append(json.dumps(cases_to_show, indent=2))
                    
                    if len(wrong_preds) > 10:
                        context_parts.append(f"\n... and {len(wrong_preds) - 10} more cases")
            
            # SHAP summary from XGBoost analysis
            if context.get('importance_summary'):
                context_parts.append(f"=== FEATURE IMPORTANCE ANALYSIS ===\n{context['importance_summary']}")
            
            # L1 Analysis from XGBoost
            if context.get('l1_analysis'):
                l1 = context['l1_analysis']
                l1_summary = {
                    'column': l1.get('column'),
                    'accuracy': l1.get('metrics', {}).get('accuracy'),
                    'precision': l1.get('metrics', {}).get('precision'),
                    'recall': l1.get('metrics', {}).get('recall'),
                    'top_features': list(l1.get('feature_importance', {}).keys())[:5]
                }
                context_parts.append(f"=== L1 DECISION ANALYSIS ===\n{json.dumps(l1_summary, indent=2)}")
            
            # L2 Analysis from XGBoost
            if context.get('l2_analysis'):
                l2 = context['l2_analysis']
                l2_summary = {
                    'column': l2.get('column'),
                    'accuracy': l2.get('metrics', {}).get('accuracy'),
                    'precision': l2.get('metrics', {}).get('precision'),
                    'recall': l2.get('metrics', {}).get('recall'),
                    'top_features': list(l2.get('feature_importance', {}).keys())[:5]
                }
                context_parts.append(f"=== L2 DECISION ANALYSIS ===\n{json.dumps(l2_summary, indent=2)}")
            
            # Prediction breakdown
            if context.get('prediction_breakdown'):
                breakdown = context['prediction_breakdown']
                context_parts.append(f"=== PREDICTION BREAKDOWN ===\n{json.dumps(breakdown, indent=2)}")
            
            # Bank guidelines
            if context.get('guidelines'):
                guidelines_data = context['guidelines']
                if isinstance(guidelines_data, list) and len(guidelines_data) > 0:
                    guidelines_summary = []
                    for g in guidelines_data[:10]:
                        summary = f"- [{g.get('category', 'custom').upper()}] {g.get('title', 'Untitled')}"
                        if g.get('rules'):
                            summary += f"\n  Rules: {', '.join(g['rules'][:3])}"
                        guidelines_summary.append(summary)
                    context_parts.append(f"=== BANK GUIDELINES ===\n" + "\n".join(guidelines_summary))
            
            # Data schema for feature context - with more detail
            if context.get('dataSchema') or context.get('data_schema'):
                schema = context.get('dataSchema') or context.get('data_schema')
                # Include more schema details for better context
                schema_summary = {
                    'total_features': schema.get('dataset_info', {}).get('total_features'),
                    'target': schema.get('target_variable'),
                    'rows': schema.get('dataset_info', {}).get('total_rows'),
                }
                # Include feature descriptions for context
                features_info = []
                for f in schema.get('features', [])[:15]:
                    feat_info = f"{f.get('name')}: {f.get('dtype')}"
                    if f.get('description'):
                        feat_info += f" - {f.get('description')}"
                    features_info.append(feat_info)
                schema_summary['features'] = features_info
                context_parts.append(f"=== DATA SCHEMA ===\n{json.dumps(schema_summary, indent=2)}")
            
            # Decision tree rules from Random Forest analysis
            if context.get('decision_tree_rules'):
                dt_rules = context['decision_tree_rules']
                context_parts.append("=== RANDOM FOREST DECISION TREE RULES ===")
                context_parts.append("NOTE: These rules were learned from the training data. The transactions we're discussing have predictions that DIFFER from these rules.")
                
                if dt_rules.get('l1_decision_rules'):
                    l1_rules = dt_rules['l1_decision_rules'][:5]  # First 5 rules
                    context_parts.append(f"\nL1 Decision Rules (accuracy: {dt_rules.get('l1_accuracy', 'N/A')}):")
                    for i, rule in enumerate(l1_rules, 1):
                        context_parts.append(f"  {i}. {rule[:200]}...")  # Truncate long rules
                
                if dt_rules.get('l2_decision_rules'):
                    l2_rules = dt_rules['l2_decision_rules'][:5]  # First 5 rules
                    context_parts.append(f"\nL2 Decision Rules (accuracy: {dt_rules.get('l2_accuracy', 'N/A')}):")
                    for i, rule in enumerate(l2_rules, 1):
                        context_parts.append(f"  {i}. {rule[:200]}...")
                
                if dt_rules.get('l1_top_features') or dt_rules.get('l2_top_features'):
                    context_parts.append(f"\nTop features for L1: {', '.join(dt_rules.get('l1_top_features', []))}")
                    context_parts.append(f"Top features for L2: {', '.join(dt_rules.get('l2_top_features', []))}")
            
            # Context explanation (why we're chatting about these transactions)
            if context.get('txnJson', {}).get('context_explanation'):
                context_parts.insert(0, f"=== WHY WE'RE ANALYZING THESE TRANSACTIONS ===\n{context['txnJson']['context_explanation']}")
            
            if context_parts:
                context_message = "--- TRANSACTION & ANALYST CONTEXT ---\n\n" + "\n\n".join(context_parts)
                messages.append(SystemMessage(content=context_message))
        
        # Add conversation history
        for msg in history.messages[-3:]:
            messages.append(msg)
        
        # Add current user input
        messages.append(HumanMessage(content=user_input))
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(TxnChatResponse)
        response: TxnChatResponse = structured_llm.invoke(messages)
        
        history.add_user_message(user_input)
        history.add_ai_message(response.general_answer)
        self.save_history()

        return response

    async def extract_shadow_rules_from_transactions(
        self,
        session_id: str,
        selected_transactions: List[Dict[str, Any]],
        data_schema: Dict[str, Any] = None,
        decision_tree_rules: Dict[str, Any] = None,
        chat_history: List[Dict[str, str]] = None
    ) -> ExtractShadowRulesFromTxnResponse:
        """
        Dedicated method for extracting shadow rules from selected wrong predictions.
        
        This method analyzes the selected transactions (false positives/negatives)
        and identifies hidden patterns (shadow rules) that explain why analyst
        decisions differed from the model predictions.
        
        Args:
            session_id: Session identifier
            selected_transactions: List of wrong prediction cases to analyze
            data_schema: Schema information for feature context
            decision_tree_rules: L1/L2 rules and accuracy from Random Forest
            chat_history: Previous conversation for additional context
            
        Returns:
            ExtractShadowRulesFromTxnResponse with discovered shadow rules
        """
        
        system_content = """You are an expert in discovering SHADOW RULES in banking fraud detection systems.

WHAT ARE SHADOW RULES?
Shadow rules are undocumented patterns that analysts follow when making decisions that differ from the ML model's predictions. These are rules that exist in practice but are NOT written in official guidelines.

YOUR TASK:
Analyze the selected WRONG PREDICTIONS (false positives and false negatives) to discover shadow rules that explain:
- Why analysts flagged legitimate transactions as fraud (False Positives)
- Why analysts cleared actual fraud as legitimate (False Negatives)

ANALYSIS APPROACH:

1. **EXAMINE THE DATA**: Look at the feature values in the selected transactions
   - What patterns do the False Positives share?
   - What patterns do the False Negatives share?
   - Are there specific feature thresholds or combinations?

2. **COMPARE TO DECISION TREES**: The Random Forest learned certain rules from the training data.
   If transactions differ from these rules, WHY?
   - Is there a rule analysts follow that the model didn't capture?
   - Is there a business context the model doesn't understand?

3. **IDENTIFY SHADOW RULES**: For each pattern found:
   - Describe it in plain English
   - Specify if it applies to L1 or L2 decisions
   - State what outcome it leads to (block/release/escalate)
   - Explain the reasoning behind why this rule exists
   - List the key features involved

EXAMPLES OF SHADOW RULES:
- "If the customer is a VIP (loyalty_tier = 'platinum'), release even if amount is high"
- "If transaction happens during business hours from a known merchant, be more lenient"
- "If there are multiple small transactions in sequence, escalate regardless of risk score"

BE SPECIFIC:
- Reference actual feature names and values from the transactions
- Quantify thresholds when possible (e.g., "amount > 5000")
- Identify how many of the selected transactions match each rule"""

        messages = [SystemMessage(content=system_content)]
        
        # Build context message
        context_parts = []
        
        # Transaction data (the main focus)
        if selected_transactions:
            fp_cases = [t for t in selected_transactions if t.get('case_type') == 'false_positive']
            fn_cases = [t for t in selected_transactions if t.get('case_type') == 'false_negative']
            
            context_parts.append(f"=== SELECTED TRANSACTIONS TO ANALYZE ({len(selected_transactions)} cases) ===")
            context_parts.append(f"False Positives (analyst said fraud, was actually legitimate): {len(fp_cases)}")
            context_parts.append(f"False Negatives (analyst said legitimate, was actually fraud): {len(fn_cases)}")
            
            # Include all selected transactions (limited to 20 for token efficiency)
            cases_to_show = selected_transactions[:20]
            context_parts.append(f"\nTransaction Details (showing {len(cases_to_show)} of {len(selected_transactions)}):")
            context_parts.append(json.dumps(cases_to_show, indent=2))
        
        # Decision tree context
        if decision_tree_rules:
            context_parts.append("\n=== RANDOM FOREST DECISION TREE RULES ===")
            context_parts.append("These rules were learned from the training data. The transactions above DIFFER from these rules.")
            
            if decision_tree_rules.get('l1_decision_rules'):
                l1_rules = decision_tree_rules['l1_decision_rules'][:5]
                context_parts.append(f"\nL1 Rules (accuracy: {decision_tree_rules.get('l1_accuracy', 'N/A')}):")
                for i, rule in enumerate(l1_rules, 1):
                    context_parts.append(f"  {i}. {rule[:300]}")
            
            if decision_tree_rules.get('l2_decision_rules'):
                l2_rules = decision_tree_rules['l2_decision_rules'][:5]
                context_parts.append(f"\nL2 Rules (accuracy: {decision_tree_rules.get('l2_accuracy', 'N/A')}):")
                for i, rule in enumerate(l2_rules, 1):
                    context_parts.append(f"  {i}. {rule[:300]}")
            
            if decision_tree_rules.get('l1_top_features'):
                context_parts.append(f"\nTop features for L1: {', '.join(decision_tree_rules.get('l1_top_features', []))}")
            if decision_tree_rules.get('l2_top_features'):
                context_parts.append(f"Top features for L2: {', '.join(decision_tree_rules.get('l2_top_features', []))}")
        
        # Schema context
        if data_schema:
            context_parts.append("\n=== DATA SCHEMA ===")
            features_info = []
            for f in data_schema.get('features', [])[:15]:
                feat_info = f"{f.get('name')}: {f.get('dtype')}"
                if f.get('description'):
                    feat_info += f" - {f.get('description')}"
                features_info.append(feat_info)
            context_parts.append("\n".join(features_info))
        
        # Chat history context
        if chat_history and len(chat_history) > 0:
            context_parts.append("\n=== PREVIOUS CONVERSATION ===")
            context_parts.append("The user has already discussed these transactions in chat:")
            for msg in chat_history[-6:]:  # Last 6 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:500]  # Truncate long messages
                context_parts.append(f"[{role.upper()}]: {content}")
        
        if context_parts:
            context_message = "\n".join(context_parts)
            messages.append(SystemMessage(content=context_message))
        
        # The extraction prompt
        extraction_prompt = f"""Analyze the {len(selected_transactions)} selected transactions and extract ALL shadow rules you can identify.

For each shadow rule, provide:
1. A clear, plain English description
2. Whether it applies to L1 or L2 decisions
3. What outcome it leads to (block/release/escalate)
4. The reasoning behind why this rule exists
5. The key features involved
6. How many of the selected transactions match this rule

Discover as many distinct shadow rules as possible from the data."""

        messages.append(HumanMessage(content=extraction_prompt))
        
        # Use structured output for clean parsing
        structured_llm = self.llm.with_structured_output(ExtractShadowRulesFromTxnResponse)
        response: ExtractShadowRulesFromTxnResponse = structured_llm.invoke(messages)
        
        # Set transactions_analyzed count
        response.transactions_analyzed = len(selected_transactions)
        
        logger.info(f"Extracted {len(response.shadow_rules)} shadow rules from {len(selected_transactions)} transactions")
        
        return response

    async def guide_code_to_global_json(self, session_id: str, user_code: str, data_schema: str = None) -> str:
        """
        Generate mathematical model code to understand analyst decisions.
        Uses data schema to build appropriate ML/statistical models.
        """
        history = self.get_session_history(session_id)
        
        # Include data schema context if available
        schema_context = ""
        if data_schema:
            schema_context = f"""
DATA SCHEMA AVAILABLE:
{data_schema}

IMPORTANT: Use the exact column names from the schema above. The schema includes:
- Column types (numeric, categorical, boolean)
- Descriptions explaining what each column means
- Decision columns (l1_decision, l2_decision, etc.)
- Analyst columns (l1_analyst_id, l1_tenure_months, etc.)
"""
        
        system_content = f"""You are an expert Data Scientist who builds mathematical models to understand human decision-making.

CONTEXT - FRAUD ALERT WORKFLOW:
In banking fraud detection:
1. ML Model flags transactions (model_risk_score, expected_action)
2. L1 Analysts review and make initial decisions (l1_decision)
3. L2 Analysts review escalated/sampled cases (l2_decision)  
4. Results compared against confirmed fraud (true_fraud_flag)
{schema_context}
YOUR TASK:
Generate Python code that builds MATHEMATICAL MODELS to understand how analysts make decisions.

REQUIRED MATHEMATICAL APPROACHES:

1. **DECISION TREE MODEL** (Primary - for interpretable rules):
```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder

# Encode target: l1_decision or l2_decision
le = LabelEncoder()
y = le.fit_transform(df['l1_decision'])

# Select features (use schema columns)
features = ['transaction_amount', 'velocity_score', 'model_risk_score', 'geo_mismatch', ...]
X = df[features].fillna(0)

# Train interpretable decision tree
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50)
dt.fit(X, y)

# Extract human-readable rules
rules = export_text(dt, feature_names=features)
print("L1 ANALYST DECISION RULES:")
print(rules)
```

2. **LOGISTIC REGRESSION** (for odds ratios):
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)

# Interpret as odds ratios
odds_ratios = np.exp(lr.coef_[0])
for feat, odds in zip(features, odds_ratios):
    print(f"{{feat}}: {{odds:.2f}}x more likely to flag as fraud per unit increase")
```

3. **STATISTICAL BIAS TESTS**:
```python
from scipy.stats import chi2_contingency, ttest_ind

# Test geographic bias
contingency = pd.crosstab(df['merchant_country'], df['l1_decision'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"Geographic bias p-value: {{p_value:.4f}}")

# Test amount threshold
fraud_amounts = df[df['l1_decision'] == 'Fraud']['transaction_amount']
legit_amounts = df[df['l1_decision'] == 'Legit']['transaction_amount']
t_stat, p_value = ttest_ind(fraud_amounts, legit_amounts)
```

4. **ACCURACY METRICS**:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Compare analyst decisions to true outcome
y_true = df['true_fraud_flag'].astype(int)
y_l1 = (df['l1_decision'] == 'Fraud').astype(int)

accuracy = accuracy_score(y_true, y_l1)
precision = precision_score(y_true, y_l1)
recall = recall_score(y_true, y_l1)
```

OUTPUT FORMAT:
Generate a complete Python function called `build_analyst_decision_model()` that:
1. Preprocesses data using the schema column names
2. Trains a Decision Tree to extract interpretable rules
3. Calculates feature importance and odds ratios
4. Tests for statistical biases
5. Computes accuracy metrics vs true_fraud_flag
6. Returns a structured JSON matching this schema:

{GLOBAL_JSON_SCHEMA}

EXAMPLE build_analyst_decision_model() function:
```python
{EXAMPLE_EXPLAIN_GLOBAL}
```

Generate Python code that will help uncover how analysts reach their fraud decisions."""

        messages = [SystemMessage(content=system_content)]
        for msg in history.messages[-4:]:
            messages.append(msg)
        messages.append(HumanMessage(content=user_code))
        
        response_msg = self.llm.invoke(messages)
        
        history.add_user_message(f"Analyst analysis request: {user_code[:100]}...")
        history.add_ai_message(response_msg.content)
        self.save_history()
        
        return response_msg.content

    async def guide_code_to_txn_json(self, session_id: str, user_code: str, global_json_context: str = None) -> str:
        """
        Generate analyze_single_alert function for case-level analyst decision analysis.
        Focused on understanding L1/L2 decisions for a specific fraud alert.
        """
        history = self.get_session_history(session_id)
        
        # Build context section if global patterns are provided
        global_context = ""
        if global_json_context:
            global_context = f"""
GLOBAL PATTERNS CONTEXT:
The user has already analyzed global analyst patterns:
{global_json_context}

Use these global patterns to identify when this specific case follows or deviates from typical behavior.
"""
        
        system_content = f"""You are an expert in Fraud Analyst Behavior Analysis who helps users understand individual case decisions.

CONTEXT - FRAUD ALERT WORKFLOW:
For each fraud alert:
1. ML Model calculates risk score and recommends action (model_risk_score, expected_action)
2. L1 Analyst reviews and decides (l1_decision, l1_override_flag)
3. L2 Analyst may override or confirm (l2_decision, l2_override_flag, l2_override_reason)
4. True outcome is eventually confirmed (true_fraud_flag)

YOUR TASK:
Generate a complete Python function called `analyze_single_alert()` that:
1. Analyzes why L1 and L2 analysts made their decisions for a specific alert
2. Identifies key factors that likely influenced each analyst
3. Detects if any "shadow rules" were applied
4. Compares decisions against model recommendations and true outcome
5. Returns a structured JSON with case-level analysis
{global_context}
CASE-LEVEL JSON SCHEMA:
{TXN_JSON_SCHEMA}

IMPLEMENTATION GUIDANCE:
- Compare the alert's features to population averages and typical fraud cases
- Identify what makes this case stand out (unusual amount, geo mismatch, etc.)
- Analyze if L1 overrode the system and why
- Analyze if L2 overrode L1 and why  
- Check if the final decision matched true_fraud_flag
- Generate hypotheses for why analysts made their decisions

EXAMPLE analyze_single_alert() function:
```python
{EXAMPLE_EXPLAIN_TXN}
```

Generate Python code to analyze a specific fraud alert's analyst decisions."""

        messages = [SystemMessage(content=system_content)]
        for msg in history.messages[-4:]:
            messages.append(msg)
        messages.append(HumanMessage(content=user_code))
        
        response_msg = self.llm.invoke(messages)
        
        history.add_user_message(f"Case analysis request: {user_code[:100]}...")
        history.add_ai_message(response_msg.content)
        self.save_history()

        return response_msg.content

    async def guide_schema_analysis(self, session_id: str, user_code: str) -> str:
        """
        Generate code to analyze data schema - features, target, and feature engineering.
        """
        history = self.get_session_history(session_id)
        
        system_content = """You are an expert Data Scientist who helps users understand their data structure.

YOUR TASK:
Analyze the user's ML code and generate a complete, runnable Python function called `analyze_data_schema()` that extracts information about the data.

OUTPUT FORMAT:
Always output a complete Python function that:
1. Analyzes X (features) - column names, data types, missing values, statistics
2. Analyzes y (target) - type, distribution, class balance
3. Detects feature engineering - one-hot encoded, datetime-derived, computed columns
4. Returns a structured dictionary with all this information

EXPECTED OUTPUT SCHEMA:
```json
{
  "dataset_info": {
    "total_samples": 10000,
    "total_features": 25,
    "memory_usage_mb": 2.5
  },
  "features": [
    {
      "name": "age",
      "dtype": "int64",
      "missing_count": 0,
      "missing_pct": 0.0,
      "unique_count": 80,
      "stats": {"min": 18, "max": 95, "mean": 45.2, "std": 15.3}
    },
    {
      "name": "category_encoded",
      "dtype": "object",
      "missing_count": 10,
      "missing_pct": 0.1,
      "unique_count": 5,
      "top_values": [{"value": "A", "count": 5000}, {"value": "B", "count": 3000}]
    }
  ],
  "target": {
    "name": "is_fraud",
    "dtype": "int64",
    "task_type": "binary_classification",
    "class_distribution": {"0": 9500, "1": 500},
    "class_balance": "imbalanced"
  },
  "feature_engineering": {
    "one_hot_encoded": [
      {"original": "category", "columns": ["category_A", "category_B", "category_C"]}
    ],
    "datetime_derived": [
      {"original": "timestamp", "columns": ["hour", "day_of_week", "month"]}
    ],
    "computed": [
      {"name": "age_income_ratio", "likely_formula": "age / income"}
    ],
    "normalized": ["income_scaled", "age_normalized"],
    "binned": ["age_group", "income_bracket"]
  },
  "data_quality": {
    "total_missing": 150,
    "missing_pct": 0.15,
    "potential_leakage": ["future_date_column"],
    "high_cardinality": ["user_id", "transaction_id"]
  }
}
```

IMPLEMENTATION GUIDANCE:
- Use pandas DataFrame methods (.info(), .describe(), .dtypes)
- Detect one-hot encoding by finding columns with common prefixes ending in _A, _B, etc.
- Detect datetime-derived features by names like 'hour', 'day', 'month', 'weekday'
- Identify potential computed features by names with 'ratio', 'diff', 'avg', 'sum'
- Flag high-cardinality categorical features (>50 unique values)
- Check for class imbalance in classification targets

EXAMPLE FUNCTION:
```python
import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_data_schema(X, y=None, feature_names=None, target_name=None):
    \"\"\"
    Analyze data schema including features, target, and detect feature engineering.
    
    Args:
        X: Feature DataFrame or array
        y: Target variable (optional)
        feature_names: List of feature names if X is array
        target_name: Name of target variable
    
    Returns:
        dict: Structured schema information
    \"\"\"
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if feature_names:
            X = pd.DataFrame(X, columns=feature_names)
        else:
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    schema = {
        'dataset_info': {
            'total_samples': len(X),
            'total_features': len(X.columns),
            'memory_usage_mb': round(X.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        },
        'features': [],
        'target': None,
        'feature_engineering': {
            'one_hot_encoded': [],
            'datetime_derived': [],
            'computed': [],
            'normalized': [],
            'binned': []
        },
        'data_quality': {
            'total_missing': int(X.isnull().sum().sum()),
            'missing_pct': round(X.isnull().sum().sum() / (len(X) * len(X.columns)) * 100, 2),
            'potential_leakage': [],
            'high_cardinality': []
        }
    }
    
    # Analyze each feature
    for col in X.columns:
        feat_info = {
            'name': col,
            'dtype': str(X[col].dtype),
            'missing_count': int(X[col].isnull().sum()),
            'missing_pct': round(X[col].isnull().sum() / len(X) * 100, 2),
            'unique_count': int(X[col].nunique())
        }
        
        if pd.api.types.is_numeric_dtype(X[col]):
            feat_info['stats'] = {
                'min': float(X[col].min()) if not pd.isna(X[col].min()) else None,
                'max': float(X[col].max()) if not pd.isna(X[col].max()) else None,
                'mean': round(float(X[col].mean()), 3) if not pd.isna(X[col].mean()) else None,
                'std': round(float(X[col].std()), 3) if not pd.isna(X[col].std()) else None
            }
        else:
            top_vals = X[col].value_counts().head(5)
            feat_info['top_values'] = [
                {'value': str(v), 'count': int(c)} 
                for v, c in top_vals.items()
            ]
            
            # Check for high cardinality
            if feat_info['unique_count'] > 50:
                schema['data_quality']['high_cardinality'].append(col)
        
        schema['features'].append(feat_info)
    
    # Detect feature engineering patterns
    # One-hot encoding detection
    col_prefixes = defaultdict(list)
    for col in X.columns:
        if '_' in col:
            prefix = '_'.join(col.split('_')[:-1])
            col_prefixes[prefix].append(col)
    
    for prefix, cols in col_prefixes.items():
        if len(cols) >= 3:  # Likely one-hot encoded
            schema['feature_engineering']['one_hot_encoded'].append({
                'original': prefix,
                'columns': cols
            })
    
    # Datetime-derived detection
    datetime_keywords = ['hour', 'day', 'month', 'year', 'weekday', 'week', 'quarter']
    for col in X.columns:
        if any(kw in col.lower() for kw in datetime_keywords):
            schema['feature_engineering']['datetime_derived'].append(col)
    
    # Computed feature detection
    computed_keywords = ['ratio', 'diff', 'avg', 'sum', 'mean', 'log', 'sqrt']
    for col in X.columns:
        if any(kw in col.lower() for kw in computed_keywords):
            schema['feature_engineering']['computed'].append({'name': col})
    
    # Analyze target if provided
    if y is not None:
        if isinstance(y, pd.Series):
            y_arr = y
        else:
            y_arr = pd.Series(y)
        
        target_info = {
            'name': target_name or 'target',
            'dtype': str(y_arr.dtype),
            'unique_count': int(y_arr.nunique())
        }
        
        if y_arr.nunique() == 2:
            target_info['task_type'] = 'binary_classification'
            class_dist = y_arr.value_counts().to_dict()
            target_info['class_distribution'] = {str(k): int(v) for k, v in class_dist.items()}
            majority = max(class_dist.values())
            minority = min(class_dist.values())
            target_info['class_balance'] = 'balanced' if minority/majority > 0.4 else 'imbalanced'
        elif y_arr.nunique() <= 20:
            target_info['task_type'] = 'multiclass_classification'
            class_dist = y_arr.value_counts().to_dict()
            target_info['class_distribution'] = {str(k): int(v) for k, v in class_dist.items()}
        else:
            target_info['task_type'] = 'regression'
            target_info['stats'] = {
                'min': float(y_arr.min()),
                'max': float(y_arr.max()),
                'mean': round(float(y_arr.mean()), 3),
                'std': round(float(y_arr.std()), 3)
            }
        
        schema['target'] = target_info
    
    return schema
```

Now analyze the user's code and provide a tailored implementation."""

        messages = [SystemMessage(content=system_content)]
        for msg in history.messages[-4:]:
            messages.append(msg)
        messages.append(HumanMessage(content=user_code))
        
        response_msg = self.llm.invoke(messages)
        
        history.add_user_message(f"Schema analysis request: {user_code[:100]}...")
        history.add_ai_message(response_msg.content)
        self.save_history()
        
        return response_msg.content

    # Keep the old method for backward compatibility (routes to global by default)
    async def guide_code_to_json(self, session_id: str, user_code: str) -> str:
        """
        Legacy method - routes to global JSON generation.
        """
        return await self.guide_code_to_global_json(session_id, user_code)

    async def convert_rules_to_human_readable(
        self,
        l1_rules: List[str],
        l2_rules: List[str],
        data_schema: Dict[str, Any] = None,
        existing_rules: List[str] = None
    ) -> ShadowRulesConversionResponse:
        """
        Convert decision tree rules to simple human-readable language using LLM.
        Avoids duplicates by checking against existing rules.
        """
        existing_rules = existing_rules or []
        existing_set = set(r.lower().strip() for r in existing_rules)
        
        # Filter out rules that already exist
        new_l1_rules = [r for r in l1_rules if r.lower().strip() not in existing_set]
        new_l2_rules = [r for r in l2_rules if r.lower().strip() not in existing_set]
        
        if not new_l1_rules and not new_l2_rules:
            return ShadowRulesConversionResponse(rules=[])
        
        # Build schema context
        schema_context = ""
        if data_schema:
            if 'columns' in data_schema:
                schema_context = f"\nData columns available: {', '.join(data_schema['columns'][:20])}"
            if 'detected_columns' in data_schema:
                detected = data_schema['detected_columns']
                if detected.get('l1_decision'):
                    schema_context += f"\nL1 Decision Column: {detected['l1_decision']}"
                if detected.get('l2_decision'):
                    schema_context += f"\nL2 Decision Column: {detected['l2_decision']}"
        
        system_prompt = f"""You are an expert at converting technical decision tree rules into simple, plain English that anyone can understand.

Your task is to convert each decision tree rule into a simple rule that:
1. Uses everyday language - NO mathematical symbols like <=, >=, <, >, ==
2. Uses words like "less than", "more than", "at least", "at most", "equals"
3. Explains what the rule means in practical terms
4. Is easy for a non-technical person to understand
5. Captures the essence of when to block, release, or escalate a transaction

{schema_context}

For each rule, extract:
- The simple human-readable version
- The original technical rule (keep exactly as provided)
- Which decision it applies to (L1 or L2)
- The predicted outcome (block, release, or escalate)
- Key factors involved (in plain English)
- Confidence level based on sample size (high if >100 samples, medium if 20-100, low if <20)

Examples of good conversions:
- Technical: "IF transaction_amount <= 500 AND time_of_day >= 22 THEN block"
- Simple: "Block transactions when the amount is $500 or less AND it happens at night (after 10 PM)"

- Technical: "IF velocity_24h > 5 AND is_new_customer == 1 THEN escalate"  
- Simple: "Escalate for review when a new customer makes more than 5 transactions in 24 hours"

- Technical: "IF merchant_risk_score <= 0.3 AND transaction_amount <= 1000 THEN release"
- Simple: "Release transactions when the merchant is low-risk and the amount is $1,000 or less"

IMPORTANT: Make the rules sound natural and conversational, like explaining to a colleague."""

        # Prepare the rules for conversion
        rules_to_convert = []
        for rule in new_l1_rules:
            rules_to_convert.append({"rule": rule, "target": "l1"})
        for rule in new_l2_rules:
            rules_to_convert.append({"rule": rule, "target": "l2"})
        
        user_prompt = f"""Convert these {len(rules_to_convert)} decision tree rules to simple human-readable language:

{json.dumps(rules_to_convert, indent=2)}

Return ALL rules converted. Each rule should have a clear, simple explanation."""

        try:
            # Use structured output
            structured_llm = self.llm.with_structured_output(ShadowRulesConversionResponse)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = structured_llm.invoke(messages)
            
            # Ensure original rules are preserved
            for i, converted_rule in enumerate(response.rules):
                if i < len(rules_to_convert):
                    converted_rule.original_rule = rules_to_convert[i]["rule"]
                    converted_rule.target_decision = rules_to_convert[i]["target"]
            
            return response
            
        except Exception as e:
            logger.error(f"Error converting rules: {e}")
            # Fallback: return rules with basic conversion
            fallback_rules = []
            for item in rules_to_convert:
                rule_text = item["rule"]
                target = item["target"]
                
                # Basic extraction
                outcome = "unknown"
                if "block" in rule_text.lower():
                    outcome = "block"
                elif "release" in rule_text.lower():
                    outcome = "release"
                elif "escalate" in rule_text.lower():
                    outcome = "escalate"
                
                # Extract sample count
                samples = 0
                import re
                samples_match = re.search(r'samples?\s*[=:]\s*(\d+)', rule_text, re.IGNORECASE)
                if samples_match:
                    samples = int(samples_match.group(1))
                
                fallback_rules.append(HumanReadableShadowRule(
                    simple_rule=f"[Auto-converted] {rule_text[:200]}",
                    original_rule=rule_text,
                    target_decision=target,
                    predicted_outcome=outcome,
                    key_factors=[],
                    confidence_level="medium" if samples > 20 else "low",
                    samples_affected=samples
                ))
            
            return ShadowRulesConversionResponse(rules=fallback_rules)

chat_manager = ChatManager()
