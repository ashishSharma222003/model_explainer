from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

# --- Specs ---

class PreprocessingStep(BaseModel):
    step_type: str  # e.g., "imputer", "scaler", "onehot", "custom"
    params: Dict[str, Any] = {}

class TransformSpec(BaseModel):
    normalization: Optional[Dict[str, Any]] = None  # type + params
    encoding: Optional[Dict[str, str]] = None # feature -> encoding type
    missing_handling: Optional[Dict[str, Any]] = None
    feature_engineering: List[PreprocessingStep] = []
    feature_selection_order: List[str] = []

class FeatureSchema(BaseModel):
    name: str
    type: str  # numeric, categorical, bool, text, embedding
    allowed_range: Optional[List[float]] = None
    units: Optional[str] = None
    actionable: bool = True
    sensitivity: str = "none" # none, pii, restricted
    grouping: Optional[str] = None

class SchemaSpec(BaseModel):
    features: List[FeatureSchema]

# --- Global Explanation ---

class ImportanceItem(BaseModel):
    feature_or_group: str
    importance: float
    direction: Optional[str] = None
    notes: Optional[str] = None

class TrendBin(BaseModel):
    bin_start: float
    bin_end: float
    avg_score: float

class TrendCategory(BaseModel):
    category: str
    avg_score: float

class GlobalTrend(BaseModel):
    feature: str
    numeric_trends: Optional[List[TrendBin]] = None
    categorical_trends: Optional[List[TrendCategory]] = None

class ReliabilityMeta(BaseModel):
    sample_size: int
    num_model_calls: int
    stability_score: Optional[float] = None
    known_failure_modes: List[str] = []

class GlobalExplanation(BaseModel):
    # Meta
    model_version: str
    transform_version: Optional[str] = None
    schema_version: Optional[str] = None
    data_slice: str = "default"
    method: str = "mock_explainer"
    generated_at: str

    # Core Content
    global_importance: List[ImportanceItem]
    global_trends: List[GlobalTrend]
    
    # Reliability & Limits
    reliability: ReliabilityMeta
    limits: List[str] = []

# --- Transaction Explanation ---

class PredictionSummary(BaseModel):
    score: float
    threshold: Optional[float] = None
    label: str
    calibrated: bool = False

class Contribution(BaseModel):
    feature: str
    value: Any
    contribution: float
    direction: str # "positive", "negative", "neutral"
    confidence: Optional[float] = None

class EvidenceSanityCheck(BaseModel):
    feature: str
    change: str
    score_before: float
    score_after: float
    passed: bool

class Counterfactual(BaseModel):
    changes: Dict[str, Any] # feature -> new_value
    new_score: float
    new_label: str
    realism_score: float

class NearestCase(BaseModel):
    case_id: str
    similarity: float
    label: str
    key_differences: Dict[str, Any]

class TransactionExplanation(BaseModel):
    # Meta
    txn_id: str
    model_version: str
    transform_version: Optional[str] = None
    schema_version: Optional[str] = None
    generated_at: str
    mode: str = "fast" # fast, standard, audit
    num_model_calls: int
    seed: Optional[int] = None

    # Prediction
    prediction: PredictionSummary

    # Contributions
    local_contributions: List[Contribution]

    # Evidence
    sanity_checks: List[EvidenceSanityCheck] = []
    counterfactuals: List[Counterfactual] = []
    nearest_cases: List[NearestCase] = []

    # Reliability
    stability_score: Optional[float] = None
    correlation_warnings: List[str] = []
    ood_flags: List[str] = []

    # Narrative
    narrative_plain: List[str] = []
    narrative_analyst: List[str] = []
    narrative_compliance: List[str] = []

# --- Request Models ---

class ExplainGlobalRequest(BaseModel):
    model_version: str
    schema_spec: SchemaSpec
    # Note: data and functions would technically be needed but for API we might pass IDs or small samples

class ExplainTransactionRequest(BaseModel):
    txn_id: str
    features: Dict[str, Any]
    model_version: str
    schema_spec: SchemaSpec


# --- Chat Response Models ---

class ChatResponse(BaseModel):
    """Structured chat response with optional suggestion for global JSON function generation."""
    general_answer: str = Field(
        ..., 
        description="The main response to the user's question about their ML model"
    )
    global_json_suggestion: Optional[str] = Field(
        None,
        description="If the conversation reveals improvements needed for the explain_global() function (e.g., missing features, better trend calculation, additional metadata), provide a brief actionable suggestion here. Only populate if there's a concrete improvement to make."
    )


class TxnChatResponse(BaseModel):
    """Structured chat response for transaction-level discussions in banking fraud analysis."""
    general_answer: str = Field(
        ..., 
        description="The main response explaining the transaction prediction and analyst behavior"
    )
    txn_json_suggestion: Optional[str] = Field(
        None,
        description="If the conversation reveals improvements needed for the explain_txn() function (e.g., better narratives, additional local contributions, counterfactual suggestions), provide a brief actionable suggestion here."
    )
    what_if_insight: Optional[str] = Field(
        None,
        description="If applicable, provide a 'what-if' insight: what changes to the transaction would alter the prediction significantly."
    )
    risk_flag: Optional[str] = Field(
        None,
        description="Flag any concerns about this specific prediction (e.g., edge case, low confidence, unusual feature values)."
    )
    shadow_rule_detected: Optional[str] = Field(
        None,
        description="If a pattern is detected where analyst behavior differs from official guidelines or model predictions, describe the potential shadow rule here. E.g., 'Analyst tends to override model for transactions under $500' or 'Late-night transactions are frequently escalated regardless of score'."
    )
    guideline_reference: Optional[str] = Field(
        None,
        description="Reference to specific bank guidelines that apply to this transaction. Include guideline title and how it applies."
    )
    compliance_note: Optional[str] = Field(
        None,
        description="Any regulatory or compliance considerations that should be noted for this transaction."
    )


# --- Shadow Rule Models ---

class HumanReadableShadowRule(BaseModel):
    """A single shadow rule converted to human-readable language."""
    simple_rule: str = Field(
        ...,
        description="The rule in simple, plain English that anyone can understand. No mathematical symbols or technical jargon. Example: 'If the transaction amount is very high and happens at night, block it'"
    )
    original_rule: str = Field(
        ...,
        description="The original decision tree rule exactly as it was"
    )
    target_decision: str = Field(
        ...,
        description="Which decision this rule applies to: 'l1' or 'l2'"
    )
    predicted_outcome: str = Field(
        ...,
        description="What the rule predicts: 'block', 'release', or 'escalate'"
    )
    key_factors: List[str] = Field(
        default_factory=list,
        description="List of the main factors/features involved in this rule in plain English"
    )
    confidence_level: str = Field(
        ...,
        description="How confident this rule is: 'high', 'medium', or 'low'"
    )
    samples_affected: int = Field(
        default=0,
        description="Number of samples this rule applies to"
    )


class ShadowRulesConversionRequest(BaseModel):
    """Request to convert decision tree rules to human-readable format."""
    l1_rules: List[str] = Field(default_factory=list)
    l2_rules: List[str] = Field(default_factory=list)
    data_schema: Optional[Dict[str, Any]] = None
    existing_rules: List[str] = Field(
        default_factory=list,
        description="List of existing rule texts to avoid duplicates"
    )


class ShadowRulesConversionResponse(BaseModel):
    """Response containing converted shadow rules."""
    rules: List[HumanReadableShadowRule] = Field(default_factory=list)


# --- Semantic Search / Deduplication Models ---

class SimilarShadowRule(BaseModel):
    """A similar shadow rule found via semantic search."""
    rule_id: str = Field(..., description="Unique identifier of the similar rule")
    rule_text: str = Field(..., description="The text that was embedded for similarity")
    simple_rule: str = Field(..., description="Human-readable version of the rule")
    similarity_score: float = Field(..., description="Cosine similarity score (0-1)")
    source_analysis: str = Field(..., description="Source: 'decision-tree', 'chat-discovered', or 'manual'")
    target_decision: str = Field(default="", description="Which decision: 'l1' or 'l2'")
    predicted_outcome: str = Field(default="", description="Predicted outcome: 'block', 'release', 'escalate'")
    is_duplicate: bool = Field(..., description="True if similarity >= threshold")


class DeduplicationResult(BaseModel):
    """Result of checking a shadow rule for duplicates."""
    is_duplicate: bool = Field(..., description="True if a duplicate was found")
    similar_rules: List[SimilarShadowRule] = Field(
        default_factory=list,
        description="List of similar rules found, sorted by similarity"
    )
    suggested_action: str = Field(
        ...,
        description="Suggested action: 'save_new', 'use_existing', or 'review'"
    )


class VectorStoreStats(BaseModel):
    """Statistics about a session's shadow rules vector store."""
    session_id: str = Field(..., description="Session ID")
    total_rules: int = Field(..., description="Total number of rules in the index")
    decision_tree_rules: int = Field(default=0, description="Rules from decision tree analysis")
    chat_discovered_rules: int = Field(default=0, description="Rules discovered via chat")
    manual_rules: int = Field(default=0, description="Manually added rules")
    index_size: int = Field(default=0, description="Number of vectors in FAISS index")


class AddRuleToIndexRequest(BaseModel):
    """Request to add a shadow rule to the vector index."""
    rule_id: str = Field(..., description="Unique identifier for the rule")
    rule_text: str = Field(..., description="Text to embed (usually the simple_rule)")
    source_analysis: str = Field(
        default="manual",
        description="Source: 'decision-tree', 'chat-discovered', or 'manual'"
    )
    simple_rule: str = Field(default="", description="Human-readable version")
    target_decision: str = Field(default="", description="'l1' or 'l2'")
    predicted_outcome: str = Field(default="", description="'block', 'release', 'escalate'")
    confidence_level: str = Field(default="", description="'high', 'medium', 'low'")
    samples_affected: int = Field(default=0, description="Number of samples")


class AddRulesBulkRequest(BaseModel):
    """Request to add multiple shadow rules at once."""
    rules: List[AddRuleToIndexRequest] = Field(
        ...,
        description="List of rules to add"
    )


class CheckDuplicateRequest(BaseModel):
    """Request to check if a rule is a duplicate."""
    rule_text: str = Field(..., description="Rule text to check for duplicates")
    threshold: float = Field(
        default=0.95,
        description="Similarity threshold (0-1) for marking as duplicate"
    )


# --- Shadow Rule Extraction Models ---

class ExtractedShadowRule(BaseModel):
    """A shadow rule extracted by LLM from transaction analysis."""
    rule_description: str = Field(
        ...,
        description="Plain English description of the shadow rule pattern"
    )
    target_decision: str = Field(
        ...,
        description="Which decision level this applies to: 'l1' or 'l2'"
    )
    predicted_outcome: str = Field(
        ...,
        description="What outcome this rule leads to: 'block', 'release', or 'escalate'"
    )
    reasoning: str = Field(
        ...,
        description="Why this rule exists - the reasoning behind the pattern"
    )
    confidence: str = Field(
        default="medium",
        description="How confident we are in this rule: 'high', 'medium', or 'low'"
    )
    affected_transactions: int = Field(
        default=0,
        description="How many of the selected transactions match this rule"
    )
    key_features: List[str] = Field(
        default_factory=list,
        description="List of key features/factors involved in this rule"
    )


class ExtractShadowRulesFromTxnResponse(BaseModel):
    """Structured response for shadow rule extraction from transactions."""
    shadow_rules: List[ExtractedShadowRule] = Field(
        default_factory=list,
        description="List of shadow rules discovered from analyzing the transactions"
    )
    summary: str = Field(
        ...,
        description="Brief summary of what patterns were found"
    )
    transactions_analyzed: int = Field(
        default=0,
        description="Number of transactions that were analyzed"
    )


class ExtractShadowRulesRequest(BaseModel):
    """Request to extract shadow rules from selected transactions."""
    session_id: str = Field(..., description="Session ID")
    selected_transactions: List[Dict[str, Any]] = Field(
        ...,
        description="List of selected wrong predictions to analyze"
    )
    data_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Schema information for feature context"
    )
    decision_tree_rules: Optional[Dict[str, Any]] = Field(
        None,
        description="Decision tree rules for context (L1/L2 rules, accuracy)"
    )
    chat_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Previous chat conversation for additional context"
    )

