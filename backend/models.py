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
