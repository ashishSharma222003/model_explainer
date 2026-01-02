import os
import json
from typing import Dict, List, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from models import ExplainGlobalRequest, ExplainTransactionRequest, ChatResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHAT_HISTORY_FILE = "chat_memory.json"

# JSON Schema templates for reference
GLOBAL_JSON_SCHEMA = """
{
  "model_version": "string - your model version identifier",
  "generated_at": "ISO 8601 datetime string",
  "global_importance": [
    {
      "feature_or_group": "feature name or group name",
      "importance": 0.0 to 1.0 (normalized importance score),
      "direction": "positive" or "negative",
      "notes": "optional explanation"
    }
  ],
  "global_trends": [
    {
      "feature": "feature name",
      "numeric_trends": [
        { "bin_start": 0, "bin_end": 10, "avg_score": 0.5 }
      ],
      "categorical_trends": [
        { "category": "category_value", "avg_score": 0.5 }
      ]
    }
  ],
  "reliability": {
    "sample_size": number of samples used,
    "stability_score": 0.0 to 1.0
  },
  "limits": ["Known limitation 1", "Known limitation 2"]
}
"""

TXN_JSON_SCHEMA = """
{
  "txn_id": "unique transaction identifier",
  "model_version": "string",
  "generated_at": "ISO 8601 datetime string",
  "prediction": {
    "score": 0.0 to 1.0,
    "threshold": 0.5,
    "label": "predicted class label",
    "calibrated": false
  },
  "local_contributions": [
    {
      "feature": "feature name",
      "value": "the actual value for this transaction",
      "contribution": -1.0 to 1.0,
      "direction": "positive" or "negative",
      "confidence": optional 0.0 to 1.0
    }
  ],
  "narrative_plain": ["Plain English explanation of the prediction"],
  "narrative_compliance": ["Compliance-focused explanation"]
}
"""

EXAMPLE_EXPLAIN_GLOBAL = '''
from datetime import datetime
import numpy as np

def explain_global(model, X, feature_names, model_version="v1"):
    # Get feature importances
    importances = model.feature_importances_
    
    # Create sorted importance list
    sorted_idx = np.argsort(importances)[::-1]
    global_importance = []
    for idx in sorted_idx:
        global_importance.append({
            "feature_or_group": feature_names[idx],
            "importance": float(importances[idx]),
            "direction": "positive"  # Requires SHAP for accurate direction
        })
    
    # Compute trends for top features
    global_trends = []
    for feat_idx in sorted_idx[:3]:
        feat_name = feature_names[feat_idx]
        values = X[:, feat_idx] if isinstance(X, np.ndarray) else X.iloc[:, feat_idx]
        bins = np.percentile(values, [0, 25, 50, 75, 100])
        trends = []
        for i in range(len(bins)-1):
            mask = (values >= bins[i]) & (values < bins[i+1])
            if mask.sum() > 0:
                avg_pred = model.predict_proba(X[mask])[:, 1].mean() if hasattr(model, 'predict_proba') else model.predict(X[mask]).mean()
                trends.append({
                    "bin_start": float(bins[i]),
                    "bin_end": float(bins[i+1]),
                    "avg_score": float(avg_pred)
                })
        global_trends.append({
            "feature": feat_name,
            "numeric_trends": trends
        })
    
    return {
        "model_version": model_version,
        "generated_at": datetime.now().isoformat(),
        "global_importance": global_importance,
        "global_trends": global_trends,
        "reliability": {
            "sample_size": len(X),
            "stability_score": 0.9
        },
        "limits": [
            "Feature directions require SHAP analysis for accuracy",
            "Trends computed on training data"
        ]
    }
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
        
        system_content = """You are an expert ML Engineer and Model Critic. Your role is to help users understand their ML models deeply.

You have access to the user's context which may include:
1. 'ml_code': The user's actual model source code
2. 'global': Global model explanations (feature importance, trends, reliability metrics)
3. 'txn': Individual transaction/prediction explanations (local contributions, specific predictions)

YOUR APPROACH:
- Be analytical and insightful. Don't just describe what you see - interpret it.
- Connect code to behavior: "You see high importance for feature X because line 45 applies log transform..."
- Be critical but constructive. Point out potential issues like data leakage, feature engineering problems, etc.
- Answer concisely but thoroughly.

WHEN DISCUSSING GLOBAL EXPLANATIONS:
- Explain what the feature importance rankings mean for model behavior
- Discuss if the importance rankings make domain sense
- Highlight any red flags (unexpected features, potential leakage signals)
- Explain reliability metrics and limitations

WHEN DISCUSSING TRANSACTION PREDICTIONS:
- Explain why this specific prediction was made
- Compare local contributions to global patterns
- Assess prediction reliability
- Suggest what changes would alter the prediction

IMPORTANT - SUGGESTIONS:
When you notice opportunities to improve the explain_global() function based on the discussion, provide a brief actionable suggestion in the global_json_suggestion field. Examples of when to suggest:
- The global JSON is missing important features that should be tracked
- Trend calculations could be improved or added for certain features
- Additional metadata or reliability metrics would be valuable
- Feature groupings or directions need adjustment
Only provide suggestions when there's a concrete, actionable improvement. Keep suggestions concise (1-2 sentences)."""

        # Build messages list manually to avoid template parsing issues
        messages = [SystemMessage(content=system_content)]
        
        # Add context as a separate system message if available
        if context:
            context_parts = []
            if context.get('ml_code'):
                code_preview = context['ml_code']
                context_parts.append(f"=== USER'S ML CODE ===\n{code_preview}")
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

    async def guide_code_to_json(self, session_id: str, user_code: str) -> str:
        """
        Analyze user code and generate explain_global or explain_txn function.
        """
        history = self.get_session_history(session_id)
        
        system_content = f"""You are an expert ML Engineer who helps users create explainability functions for their models.

YOUR TASK:
Analyze the user's ML code and generate a complete, runnable Python function that creates explanation JSON.

OUTPUT FORMAT:
Always output a complete Python function that:
1. Takes appropriate inputs (model, data, features, etc.)
2. Extracts/computes the required explanation data
3. Returns a dictionary matching the expected JSON schema
4. Includes helpful comments

GLOBAL JSON SCHEMA (for explain_global):
{GLOBAL_JSON_SCHEMA}

TRANSACTION JSON SCHEMA (for explain_txn):
{TXN_JSON_SCHEMA}

IMPLEMENTATION GUIDANCE:

For explain_global():
- Use model.feature_importances_ for tree-based models
- Use permutation_importance() for any model
- Use SHAP for more detailed explanations
- Compute trends by binning features and averaging predictions
- Include datetime.now().isoformat() for generated_at

For explain_txn():
- Use SHAP values for local contributions
- Or use model-specific methods (e.g., treeExplainer)
- Include the actual feature values
- Generate plain English narrative

EXAMPLE explain_global() for sklearn RandomForest:
```python
{EXAMPLE_EXPLAIN_GLOBAL}
```

Now analyze the user's code and provide a tailored implementation."""

        # Build messages manually
        messages = [SystemMessage(content=system_content)]
        
        # Add recent history
        for msg in history.messages[-4:]:
            messages.append(msg)
        
        # Add user's code
        messages.append(HumanMessage(content=user_code))
        
        # Invoke LLM directly
        response_msg = self.llm.invoke(messages)
        
        history.add_user_message(f"Code analysis request: {user_code[:100]}...")
        history.add_ai_message(response_msg.content)
        self.save_history()
        
        return response_msg.content

chat_manager = ChatManager()
