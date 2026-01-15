"""
Random Forest Analyzer Module
Automatically analyzes fraud analyst decisions using Random Forest.
Includes hyperparameter tuning and decision tree visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel
import json
import logging
from datetime import datetime
import base64
import io

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier

# For tree visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available for tree visualization")

logger = logging.getLogger(__name__)


class RandomForestHyperparameters(BaseModel):
    """Configurable hyperparameters for Random Forest model."""
    n_estimators: int = 100          # Number of trees in the forest
    max_depth: int = 5               # Maximum depth of each tree
    min_samples_split: int = 10      # Minimum samples required to split a node
    min_samples_leaf: int = 5        # Minimum samples required at a leaf node
    max_features: str = "sqrt"       # Number of features to consider for best split
    bootstrap: bool = True           # Whether to use bootstrap samples
    random_state: int = 42
    
    class Config:
        extra = "allow"


class TreeNode(BaseModel):
    """A node in the decision tree."""
    node_id: int
    feature: Optional[str] = None  # None for leaf nodes
    threshold: Optional[float] = None
    decision: str  # "< threshold" or leaf class
    samples: int = 0
    left_child: Optional[int] = None
    right_child: Optional[int] = None
    is_leaf: bool = False
    leaf_value: Optional[float] = None


class DecisionPath(BaseModel):
    """A path through the decision tree for a specific prediction."""
    nodes: List[Dict[str, Any]]
    final_decision: str
    confidence: float


class GeneralPurposeRule(BaseModel):
    """A Simplified, human-readable rule for general understanding."""
    original_rule: str
    description: str
    target: str  # "Fraud" or "Legit"
    accuracy: float
    coverage: float
    confidence: str


class AnalysisResult(BaseModel):
    """Result of XGBoost analysis on analyst decisions."""
    success: bool
    error: Optional[str] = None
    
    # Dataset info
    total_samples: int = 0
    features_used: List[str] = []
    
    # Hyperparameters used
    hyperparameters: Optional[Dict[str, Any]] = None
    
    # L1 Decision Analysis
    l1_analysis: Optional[Dict[str, Any]] = None
    
    # L2 Decision Analysis
    l2_analysis: Optional[Dict[str, Any]] = None
    
    # Prediction Breakdown
    prediction_breakdown: Optional[Dict[str, Any]] = None
    
    # SHAP values summary (for LLM context)
    importance_summary: Optional[str] = None
    
    # Wrong predictions for chat
    wrong_predictions: Optional[List[Dict[str, Any]]] = None
    
    # Tree visualization (base64 encoded PNG)
    l1_tree_image: Optional[str] = None
    l2_tree_image: Optional[str] = None
    
    # Decision rules extracted from trees (human-readable paths)
    l1_decision_rules: Optional[List[str]] = None
    l2_decision_rules: Optional[List[str]] = None
    
    # Full decision tree as text (sklearn export_text format)
    l1_tree_text: Optional[str] = None
    l2_tree_text: Optional[str] = None
    
    # Tree structure for interactive display
    l1_tree_structure: Optional[List[Dict[str, Any]]] = None
    l2_tree_structure: Optional[List[Dict[str, Any]]] = None
    
    # === NEW: Segment Analysis (RF Leaf-based patterns) ===
    # Combined target distribution (block_1, block_0, release_1, etc.)
    combined_target_distribution: Optional[Dict[str, int]] = None
    
    # All segment analysis results
    segment_analysis: Optional[List[Dict[str, Any]]] = None
    
    # Top segments with high false positive rates
    top_fp_segments: Optional[List[Dict[str, Any]]] = None
    
    # Top segments with high false negative rates
    top_fn_segments: Optional[List[Dict[str, Any]]] = None
    
    # Summary of segment patterns for LLM/reports
    segment_summary: Optional[str] = None
    
    # === NEW: General Purpose Rules ===
    general_purpose_rules: Optional[List[GeneralPurposeRule]] = None
    
    analyzed_at: str = ""


class RandomForestAnalyzer:
    """Analyzes fraud analyst decisions using Random Forest."""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.l1_model = None
        self.l2_model = None
        self.feature_names: List[str] = []
        self.hyperparams: RandomForestHyperparameters = RandomForestHyperparameters()
    
    def get_best_segments(self, segments: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
        """Filter and return the top N most interesting segments."""
        if not segments:
            return []
            
        # Score segments by (Accuracy * log(Samples)) to balance precision and coverage
        # Focus on high confidence rules
        scored_segments = []
        for seg in segments:
            # We want rules that are highly predictive of either class
            validity = seg.get('accuracy', 0)
            n_samples = seg.get('n_samples', 0)
            
            # Skip very small segments
            if n_samples < 5:
                continue
                
            # Score
            score = validity * np.log1p(n_samples)
            scored_segments.append((score, seg))
            
        # Sort desc
        scored_segments.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N
        return [s[1] for s in scored_segments[:n]]
    
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Update hyperparameters for the Random Forest model."""
        self.hyperparams = RandomForestHyperparameters(**params)
        logger.info(f"Hyperparameters updated: {self.hyperparams.model_dump()}")
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return self.hyperparams.model_dump()
    
    def _generate_tree_image(
        self, 
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_names: List[str],
        max_depth: int = 4,
        title: str = "Decision Tree"
    ) -> Optional[str]:
        """
        Generate a decision tree visualization using sklearn's DecisionTreeClassifier.
        This is much easier to visualize than XGBoost trees and doesn't require Graphviz.
        We train a simple decision tree on the same data to show interpretable rules.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for tree visualization")
            return None
        
        try:
            logger.info(f"Generating decision tree image: {title}, max_depth={max_depth}, samples={len(X_train)}")
            
            # Train a simple decision tree for visualization purposes
            # This gives us an interpretable view of the decision process
            viz_tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=max(10, len(X_train) // 50),
                min_samples_leaf=max(5, len(X_train) // 100),
                random_state=42
            )
            viz_tree.fit(X_train, y_train)
            
            # Store the tree for later rule extraction
            self._last_viz_tree = viz_tree
            self._last_viz_feature_names = feature_names
            
            # Get actual class names from the trained tree
            n_classes = len(viz_tree.classes_)
            class_names = [str(c) for c in viz_tree.classes_]
            logger.info(f"Tree has {n_classes} classes: {class_names}")
            
            # Create the visualization with a larger figure for better readability
            fig, ax = plt.subplots(figsize=(28, 18))
            
            # Convert feature names to list if needed
            feature_name_list = list(feature_names) if not isinstance(feature_names, list) else feature_names
            
            plot_tree(
                viz_tree,
                feature_names=feature_name_list,
                class_names=class_names,  # Use actual class names from the tree
                filled=True,
                rounded=True,
                ax=ax,
                fontsize=9,
                proportion=True,
                impurity=False
            )
            
            ax.set_title(f'{title}\n(Decision Tree - max depth {max_depth})', 
                        fontsize=18, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save to bytes with higher DPI for better quality
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            plt.close(fig)
            
            image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            logger.info(f"Decision tree image generated successfully, size: {len(image_data)} chars")
            
            # Encode as base64
            return image_data
        except Exception as e:
            logger.error(f"Error generating tree image: {e}")
            import traceback
            traceback.print_exc()
            # Return None to trigger fallback
            return None
    
    def _extract_sklearn_tree_rules(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_names: List[str],
        max_depth: int = 5
    ) -> Tuple[str, List[str]]:
        """
        Extract human-readable decision rules from a sklearn DecisionTree.
        Returns both the full tree text and a list of individual rules.
        """
        try:
            # Train a decision tree
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=max(10, len(X_train) // 50),
                min_samples_leaf=max(5, len(X_train) // 100),
                random_state=42
            )
            tree.fit(X_train, y_train)
            
            # Get actual class names from the tree
            class_names = [str(c) for c in tree.classes_]
            
            # Get full tree as text
            tree_text = export_text(
                tree, 
                feature_names=list(feature_names),
                class_names=class_names,  # Use actual class names
                show_weights=True
            )
            
            # Extract individual decision paths
            rules = self._extract_paths_from_tree(tree, feature_names)
            
            return tree_text, rules
        except Exception as e:
            logger.error(f"Error extracting sklearn tree rules: {e}")
            return "", []
    
    def _extract_paths_from_tree(
        self,
        tree: DecisionTreeClassifier,
        feature_names: List[str]
    ) -> List[str]:
        """Extract decision paths from a sklearn decision tree."""
        rules = []
        
        try:
            tree_ = tree.tree_
            # Get actual class names from the tree
            class_names = [str(c) for c in tree.classes_]
            
            feature_name = [
                feature_names[i] if i != -2 and i < len(feature_names) else "undefined!"
                for i in tree_.feature
            ]
            
            def recurse(node, path):
                if tree_.feature[node] != -2:  # Not a leaf
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    
                    # Left branch (<=)
                    left_path = path + [f"{name} <= {threshold:.2f}"]
                    recurse(tree_.children_left[node], left_path)
                    
                    # Right branch (>)
                    right_path = path + [f"{name} > {threshold:.2f}"]
                    recurse(tree_.children_right[node], right_path)
                else:
                    # Leaf node - determine class
                    value = tree_.value[node][0]
                    total = sum(value)
                    if total > 0:
                        class_idx = np.argmax(value)
                        # Use actual class name from the tree
                        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
                        confidence = value[class_idx] / total * 100
                        samples = int(total)
                        
                        if path:  # Only add if there's a path
                            rule = " AND ".join(path) + f" → {class_name} ({confidence:.0f}% conf, {samples} samples)"
                            rules.append(rule)
            
            recurse(0, [])
            
            # Sort by number of samples (most common paths first)
            return rules[:20]  # Limit to top 20 rules
        except Exception as e:
            logger.error(f"Error extracting paths from tree: {e}")
            return []
    
    def _generate_feature_importance_chart(
        self, 
        model: Any,
        feature_names: List[str],
        title: str = "Feature Importance"
    ) -> Optional[str]:
        """Generate a feature importance bar chart."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Get feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create horizontal bar chart
            y_pos = np.arange(len(indices))
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(indices)))[::-1]
            ax.barh(y_pos, importances[indices], align='center', color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f'f{i}' for i in indices])
            ax.invert_yaxis()  # Top feature at top
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title(f'{title}\n(XGBoost Model)', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
                ax.text(imp + 0.005, i, f'{imp:.3f}', va='center', fontsize=9)
            
            # Add grid
            ax.xaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save to bytes
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            plt.close(fig)
            
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error generating feature importance chart: {e}")
            return None
    
    def _extract_tree_structure(
        self, 
        model: Any,
        feature_names: List[str],
        tree_index: int = 0
    ) -> List[Dict[str, Any]]:
        """Extract tree structure as JSON for interactive display."""
        # Random Forest doesn't need XGBoost-style tree structure extraction
        # The visualization is handled by sklearn's plot_tree instead
        return []
    
    def _extract_decision_rules(
        self, 
        model: Any,
        feature_names: List[str],
        max_rules: int = 10
    ) -> List[str]:
        """Extract human-readable decision rules from the model."""
        if not XGBOOST_AVAILABLE:
            return []
        
        try:
            booster = model.get_booster()
            trees = booster.get_dump(with_stats=False)
            
            rules = []
            
            # Parse first few trees
            for tree_idx, tree_str in enumerate(trees[:3]):
                # Build paths through tree
                paths = self._build_tree_paths(tree_str, feature_names)
                for path in paths[:max_rules // 3]:
                    rules.append(path)
            
            return rules[:max_rules]
        except Exception as e:
            logger.error(f"Error extracting decision rules: {e}")
            return []
    
    def _build_tree_paths(
        self, 
        tree_str: str, 
        feature_names: List[str]
    ) -> List[str]:
        """Build human-readable paths through a tree."""
        paths = []
        
        try:
            lines = tree_str.strip().split('\n')
            
            # Simple extraction of leaf paths
            current_path = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                indent = len(line) - len(line.lstrip('\t'))
                line = line.lstrip('\t')
                
                # Trim path to current depth
                current_path = current_path[:indent]
                
                if 'leaf=' in line:
                    # This is a leaf - create rule
                    leaf_val = float(line.split('leaf=')[1].split(',')[0])
                    decision = 'FRAUD' if leaf_val > 0 else 'LEGIT'
                    if current_path:
                        rule = ' AND '.join(current_path) + f' → {decision}'
                        paths.append(rule)
                elif '[' in line and ']' in line:
                    # This is a split
                    condition = line.split('[')[1].split(']')[0]
                    if '<' in condition:
                        feature_idx = condition.split('<')[0]
                        threshold = condition.split('<')[1]
                        
                        try:
                            if feature_idx.startswith('f'):
                                idx = int(feature_idx[1:])
                                feature_name = feature_names[idx] if idx < len(feature_names) else feature_idx
                            else:
                                feature_name = feature_idx
                        except:
                            feature_name = feature_idx
                        
                        current_path.append(f"{feature_name} < {float(threshold):.2f}")
            
            return paths
        except Exception as e:
            logger.error(f"Error building tree paths: {e}")
            return []
    
    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Detect relevant columns in the dataset."""
        columns = df.columns.tolist()
        
        detected = {
            'l1_decision': None,
            'l2_decision': None,
            'true_fraud': None,
            'transaction_id': None,
            'transaction_amount': None,
            'verification_outcome': None,  # Should be excluded from training
        }
        
        for col in columns:
            col_lower = col.lower()
            
            # L1 decision
            if 'l1' in col_lower and 'decision' in col_lower:
                detected['l1_decision'] = col
            elif col_lower in ['l1_decision', 'l1decision', 'level1_decision']:
                detected['l1_decision'] = col
            
            # L2 decision
            if 'l2' in col_lower and 'decision' in col_lower:
                detected['l2_decision'] = col
            elif col_lower in ['l2_decision', 'l2decision', 'level2_decision']:
                detected['l2_decision'] = col
            
            # True fraud flag
            if 'true' in col_lower and 'fraud' in col_lower:
                detected['true_fraud'] = col
            elif col_lower in ['true_fraud_flag', 'actual_fraud', 'is_fraud', 'fraud_flag']:
                detected['true_fraud'] = col
            
            # Transaction ID
            if 'alert' in col_lower and 'id' in col_lower:
                detected['transaction_id'] = col
            elif col_lower in ['txn_id', 'transaction_id', 'alert_id', 'case_id', 'id']:
                detected['transaction_id'] = col
            
            # Transaction Amount
            if 'amount' in col_lower:
                detected['transaction_amount'] = col
            elif 'transaction' in col_lower and 'amt' in col_lower:
                detected['transaction_amount'] = col
            elif col_lower in ['txn_amount', 'transaction_amount', 'amount', 'txn_amt']:
                detected['transaction_amount'] = col
            
            # Verification Outcome (should be excluded - it's a result, not a predictor)
            if 'verification' in col_lower and 'outcome' in col_lower:
                detected['verification_outcome'] = col
            elif col_lower in ['verification_outcome', 'verificationoutcome', 'verification_result']:
                detected['verification_outcome'] = col
        
        return detected
    
    def _preprocess_data(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        exclude_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Preprocess data for XGBoost training."""
        
        # Remove target and excluded columns
        feature_cols = [c for c in df.columns if c not in exclude_cols and c != target_col]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
            self.label_encoders[target_col] = le
        
        # Process features
        for col in X.columns:
            if X[col].dtype == 'object':
                # Encode categorical
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('missing'))
                self.label_encoders[col] = le
            else:
                # Fill numeric NaN
                X[col] = X[col].fillna(X[col].median())
        
        return X, y, feature_cols
    
    # =========================================================================
    # NEW: Segment Analysis Methods for RF Pattern Discovery
    # =========================================================================
    
    def _create_combined_target(
        self, 
        df: pd.DataFrame, 
        l1_col: str, 
        fraud_col: str
    ) -> pd.Series:
        """
        Create combined target: L1_decision + fraud_flag
        Results in up to 8 classes: block_1, block_0, release_1, release_0, etc.
        """
        # Normalize L1 decisions
        l1_normalized = df[l1_col].astype(str).str.lower().str.strip()
        
        # Normalize fraud flags to 0/1
        fraud_values = df[fraud_col].astype(str).str.lower().str.strip()
        fraud_binary = fraud_values.isin(['true', '1', '1.0', 'yes', 'fraud', 'positive']).astype(int)
        
        # Create combined target
        combined = l1_normalized + "_" + fraud_binary.astype(str)
        
        return combined
    
    def _extract_leaf_segments(
        self,
        model: RandomForestClassifier,
        X: pd.DataFrame,
        df_original: pd.DataFrame,
        l1_col: str,
        fraud_col: str,
        feature_names: List[str],
        max_segments: int = 500  # Increased to allow more segments
    ) -> List[Dict[str, Any]]:
        """
        Extract leaf segments using rf.apply() and compute metrics per segment.
        
        Each segment is a group of transactions that followed the same path
        through a specific tree (identified by tree_id + leaf_id).
        """
        try:
            # Get leaf node IDs for all samples in all trees
            # Shape: (n_samples, n_trees)
            leaf_ids = model.apply(X)
            
            # Get tree count
            n_trees = leaf_ids.shape[1]
            logger.info(f"Extracting segments from {n_trees} trees")
            
            # Prepare normalized columns for metrics
            l1_normalized = df_original[l1_col].astype(str).str.lower().str.strip()
            fraud_values = df_original[fraud_col].astype(str).str.lower().str.strip()
            fraud_binary = fraud_values.isin(['true', '1', '1.0', 'yes', 'fraud', 'positive'])
            
            all_segments = []
            
            # Analyze more trees for comprehensive coverage
            trees_to_analyze = min(25, n_trees)  # Analyze 25 trees for better coverage
            
            for tree_idx in range(trees_to_analyze):
                # Get unique leaf IDs in this tree
                unique_leaves = np.unique(leaf_ids[:, tree_idx])
                
                for leaf_id in unique_leaves:
                    # Get mask for this segment
                    mask = leaf_ids[:, tree_idx] == leaf_id
                    segment_size = mask.sum()
                    
                    # Skip very small segments
                    if segment_size < 5:
                        continue
                    
                    # Get segment data
                    segment_l1 = l1_normalized[mask]
                    segment_fraud = fraud_binary[mask]
                    
                    # Compute TP/FP/TN/FN
                    metrics = self._compute_segment_metrics(segment_l1, segment_fraud)
                    
                    # Get combined target distribution
                    combined = segment_l1 + "_" + segment_fraud.astype(int).astype(str)
                    class_dist = combined.value_counts().to_dict()
                    
                    # Determine dominant L1 decision
                    dominant_decision = segment_l1.value_counts().idxmax() if len(segment_l1) > 0 else 'unknown'
                    
                    # Calculate fraud rate in segment
                    fraud_rate = segment_fraud.sum() / len(segment_fraud) if len(segment_fraud) > 0 else 0
                    
                    # Extract rule text for this segment (approximation)
                    rule_text = self._extract_rule_for_leaf(
                        model.estimators_[tree_idx], 
                        feature_names, 
                        leaf_id
                    )
                    
                    segment = {
                        'segment_id': f"tree_{tree_idx}_leaf_{leaf_id}",
                        'tree_id': tree_idx,
                        'leaf_id': int(leaf_id),
                        'transaction_count': int(segment_size),
                        'class_distribution': class_dist,
                        'TP': metrics['TP'],
                        'FP': metrics['FP'],
                        'TN': metrics['TN'],
                        'FN': metrics['FN'],
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'fraud_rate': round(fraud_rate, 4),
                        'dominant_l1_decision': dominant_decision,
                        'rule_text': rule_text,
                        'fp_rate': metrics['fp_rate'],
                        'fn_rate': metrics['fn_rate']
                    }
                    
                    all_segments.append(segment)
            
            # Sort by transaction count (most impactful first)
            all_segments.sort(key=lambda x: x['transaction_count'], reverse=True)
            
            logger.info(f"Extracted {len(all_segments)} segments")
            return all_segments[:max_segments]
            
        except Exception as e:
            logger.error(f"Error extracting leaf segments: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _compute_segment_metrics(
        self, 
        l1_decisions: pd.Series, 
        fraud_flags: pd.Series
    ) -> Dict[str, Any]:
        """
        Compute TP/FP/TN/FN for a segment.
        
        Binary interpretation:
        - block = predicted fraud
        - release = predicted not fraud
        - verify/escalate = excluded from binary metrics
        """
        # Binary decisions only
        is_block = l1_decisions.isin(['block', 'blocked'])
        is_release = l1_decisions.isin(['release', 'released'])
        is_fraud = fraud_flags
        
        # Calculate metrics
        TP = int(((is_block) & (is_fraud)).sum())  # Block + Was fraud = Correct
        FP = int(((is_block) & (~is_fraud)).sum())  # Block + Was legit = Wrong
        TN = int(((is_release) & (~is_fraud)).sum())  # Release + Was legit = Correct
        FN = int(((is_release) & (is_fraud)).sum())  # Release + Was fraud = Wrong
        
        # Binary total (excluding verify/escalate)
        binary_total = TP + FP + TN + FN
        
        # Derived metrics
        accuracy = (TP + TN) / binary_total if binary_total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        fp_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
        fn_rate = FN / (FN + TP) if (FN + TP) > 0 else 0
        
        return {
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'fp_rate': round(fp_rate, 4),
            'fn_rate': round(fn_rate, 4)
        }
    
    def _extract_rule_for_leaf(
        self,
        tree: DecisionTreeClassifier,
        feature_names: List[str],
        target_leaf_id: int
    ) -> str:
        """
        Extract the decision path (rule) that leads to a specific leaf node.
        """
        try:
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != -2 and i < len(feature_names) else "?"
                for i in tree_.feature
            ]
            
            # Find path to target leaf
            path = []
            
            def find_path(node, current_path):
                if node == target_leaf_id:
                    return current_path
                
                if tree_.feature[node] == -2:  # Leaf node
                    return None
                
                # Check left child
                left_result = find_path(
                    tree_.children_left[node],
                    current_path + [(feature_name[node], "<=", tree_.threshold[node])]
                )
                if left_result is not None:
                    return left_result
                
                # Check right child
                right_result = find_path(
                    tree_.children_right[node],
                    current_path + [(feature_name[node], ">", tree_.threshold[node])]
                )
                return right_result
            
            path = find_path(0, [])
            
            if path:
                # Format as readable rule
                conditions = [f"{feat} {op} {thresh:.2f}" for feat, op, thresh in path]
                return " AND ".join(conditions)
            else:
                return "Complex path"
                
        except Exception as e:
            logger.error(f"Error extracting rule for leaf: {e}")
            return "Error extracting rule"
    
    def _generate_segment_summary(self, segments: List[Dict[str, Any]]) -> str:
        """Generate a text summary of segment patterns for reports/LLM."""
        if not segments:
            return "No segments analyzed."
        
        # Calculate totals
        total_txn = sum(s['transaction_count'] for s in segments)
        total_fp = sum(s['FP'] for s in segments)
        total_fn = sum(s['FN'] for s in segments)
        
        # Find problem segments
        high_fp_segments = [s for s in segments if s['fp_rate'] > 0.3 and s['transaction_count'] >= 10]
        high_fn_segments = [s for s in segments if s['fn_rate'] > 0.3 and s['transaction_count'] >= 10]
        
        summary_lines = [
            f"Analyzed {len(segments)} decision segments covering {total_txn} transactions.",
            f"Total False Positives: {total_fp}, Total False Negatives: {total_fn}",
            "",
            f"High False Positive Segments ({len(high_fp_segments)}):"
        ]
        
        for seg in high_fp_segments[:5]:
            summary_lines.append(
                f"  - {seg['segment_id']}: FP rate {seg['fp_rate']:.0%}, "
                f"{seg['transaction_count']} txns, Rule: {seg['rule_text'][:80]}..."
            )
        
        summary_lines.append(f"\nHigh False Negative Segments ({len(high_fn_segments)}):")
        for seg in high_fn_segments[:5]:
            summary_lines.append(
                f"  - {seg['segment_id']}: FN rate {seg['fn_rate']:.0%}, "
                f"{seg['transaction_count']} txns, Rule: {seg['rule_text'][:80]}..."
            )
        
        return "\n".join(summary_lines)
    
    def _calculate_confusion_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray,
        decision_col: str
    ) -> Dict[str, Any]:
        """Calculate confusion matrix and related metrics."""
        
        cm = confusion_matrix(y_true, y_pred)
        n_classes = len(cm)
        
        # Calculate metrics that work for both binary and multi-class
        accuracy = float(accuracy_score(y_true, y_pred))
        
        # Use weighted average for multi-class to get meaningful precision/recall/F1
        precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # For binary classification, also calculate TP/FP/TN/FN
        if n_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_classes': n_classes,
            }
        else:
            # Multi-class - calculate per-class metrics and aggregate
            # Sum diagonal for correct predictions
            correct = int(np.trace(cm))
            total = int(cm.sum())
            wrong = total - correct
            
            metrics = {
                'true_positives': correct,  # Total correct predictions
                'false_positives': wrong // 2,  # Approximate split
                'true_negatives': 0,  # Not applicable for multi-class
                'false_negatives': wrong // 2,  # Approximate split
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_classes': n_classes,
                'confusion_matrix': cm.tolist(),
                'per_class_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            }
        
        return metrics
    
    def _get_feature_importance(
        self, 
        model: RandomForestClassifier, 
        feature_names: List[str],
        top_n: int = 10
    ) -> Tuple[Dict[str, float], str]:
        """Get feature importance from Random Forest model."""
        
        try:
            # Get feature importances from Random Forest (built-in)
            importances = model.feature_importances_
            
            # Create feature importance dict
            feature_importance = {}
            for i, name in enumerate(feature_names):
                if i < len(importances):
                    feature_importance[name] = float(importances[i])
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Generate text summary for LLM
            summary_lines = ["Top factors influencing decisions:"]
            for i, (feature, importance) in enumerate(sorted_features[:top_n]):
                summary_lines.append(f"{i+1}. {feature}: importance score {importance:.4f}")
            
            return dict(sorted_features[:top_n]), "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}, f"Error calculating importance: {str(e)}"
    
    def _identify_wrong_predictions(
        self,
        df: pd.DataFrame,
        l1_decision_col: str,
        true_fraud_col: str,
        l2_decision_col: Optional[str],
        id_col: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Identify ALL cases with their classification status.
        
        L1 Decision meanings:
        - block = L1 thinks it's fraud
        - escalate = L1 is unsure, needs review
        - release = L1 thinks it's legitimate
        
        Returns all transactions with their classification type based on L1 and L2 decisions.
        """
        
        all_predictions = []
        
        # Log unique values to help debug
        l1_unique = df[l1_decision_col].unique().tolist() if l1_decision_col in df.columns else []
        fraud_unique = df[true_fraud_col].unique().tolist() if true_fraud_col in df.columns else []
        logger.info(f"Unique L1 decisions: {l1_unique}")
        logger.info(f"Unique true_fraud values: {fraud_unique}")
        
        for idx, row in df.iterrows():
            l1_decision_raw = str(row.get(l1_decision_col, ''))
            l1_decision = l1_decision_raw.lower().strip()
            true_fraud_raw = str(row.get(true_fraud_col, ''))
            true_fraud = true_fraud_raw.lower().strip()
            
            # Get L2 decision if available
            l2_decision_raw = str(row.get(l2_decision_col, '')) if l2_decision_col else ''
            l2_decision = l2_decision_raw.lower().strip()
            
            # Determine if L1 classified as fraud
            # block = fraud, escalate = unsure, release = not fraud
            l1_is_block = l1_decision in ['block', 'blocked']
            l1_is_escalate = l1_decision in ['escalate', 'escalated', 'review', 'pending']
            l1_is_release = l1_decision in ['release', 'released']
            
            # Determine actual fraud status - true_fraud_flag = True/1 means it WAS fraud
            # Check for various formats: True, true, 1, 1.0, yes, fraud, etc.
            actual_is_fraud = true_fraud in ['true', '1', '1.0', 'yes', 'fraud', 'positive', 'confirmed']
            
            # Determine L2 decision if available
            l2_is_block = l2_decision in ['block', 'blocked', 'confirm', 'confirmed'] if l2_decision else False
            l2_is_release = l2_decision in ['release', 'released', 'clear', 'cleared'] if l2_decision else False
            
            # Classify based on L1 decision vs actual fraud
            if l1_is_block:
                if actual_is_fraud:
                    l1_case_type = 'true_positive'  # L1 blocked, was fraud - CORRECT
                else:
                    l1_case_type = 'false_positive'  # L1 blocked, was legit - WRONG
            elif l1_is_escalate:
                l1_case_type = 'escalated_fraud' if actual_is_fraud else 'escalated_legit'
            elif l1_is_release:
                if actual_is_fraud:
                    l1_case_type = 'false_negative'  # L1 released, was fraud - WRONG
                else:
                    l1_case_type = 'true_negative'  # L1 released, was legit - CORRECT
            else:
                # Unknown L1 decision - classify based on actual fraud
                l1_case_type = 'unknown_fraud' if actual_is_fraud else 'unknown_legit'
            
            # Also classify based on L2 if available
            l2_case_type = None
            if l2_decision_col and l2_decision:
                if l2_is_block:
                    l2_case_type = 'l2_true_positive' if actual_is_fraud else 'l2_false_positive'
                elif l2_is_release:
                    l2_case_type = 'l2_false_negative' if actual_is_fraud else 'l2_true_negative'
            
            case_data = {
                'index': int(idx) if not isinstance(idx, int) else idx,
                'case_id': str(row.get(id_col, idx)) if id_col else str(idx),
                'case_type': l1_case_type,
                'l1_decision': l1_decision_raw,
                'true_fraud': true_fraud_raw,
                'is_correct': l1_case_type in ['true_positive', 'true_negative'],
                'is_wrong': l1_case_type in ['false_positive', 'false_negative'],
                'is_escalated': l1_case_type in ['escalated_fraud', 'escalated_legit'],
            }
            
            if l2_decision_col:
                case_data['l2_decision'] = l2_decision_raw
                case_data['l2_case_type'] = l2_case_type
                case_data['l2_is_correct'] = l2_case_type in ['l2_true_positive', 'l2_true_negative'] if l2_case_type else None
            
            # Add other relevant columns (transaction data)
            for col in df.columns[:20]:  # Include more columns
                if col not in case_data:
                    val = row.get(col, '')
                    case_data[col] = str(val) if pd.notna(val) else ''
            
            all_predictions.append(case_data)
        
        # Log detailed summary
        tp = sum(1 for p in all_predictions if p['case_type'] == 'true_positive')
        fp = sum(1 for p in all_predictions if p['case_type'] == 'false_positive')
        tn = sum(1 for p in all_predictions if p['case_type'] == 'true_negative')
        fn = sum(1 for p in all_predictions if p['case_type'] == 'false_negative')
        esc_fraud = sum(1 for p in all_predictions if p['case_type'] == 'escalated_fraud')
        esc_legit = sum(1 for p in all_predictions if p['case_type'] == 'escalated_legit')
        unknown = sum(1 for p in all_predictions if 'unknown' in p['case_type'])
        
        logger.info(f"Classification breakdown:")
        logger.info(f"  TP (Block→Fraud, correct): {tp}")
        logger.info(f"  FP (Block→Legit, wrong): {fp}")
        logger.info(f"  TN (Release→Legit, correct): {tn}")
        logger.info(f"  FN (Release→Fraud, wrong): {fn}")
        logger.info(f"  Escalated (fraud): {esc_fraud}")
        logger.info(f"  Escalated (legit): {esc_legit}")
        logger.info(f"  Unknown: {unknown}")
        logger.info(f"  Total: {len(all_predictions)}")
        
        return all_predictions
    
    async def analyze(self, csv_data: List[Dict[str, Any]]) -> AnalysisResult:
        """
        Perform full Random Forest analysis on the dataset.
        
        Args:
            csv_data: List of dictionaries representing CSV rows
            
        Returns:
            AnalysisResult with model insights and feature importance
        """
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(csv_data)
            
            if df.empty:
                return AnalysisResult(
                    success=False,
                    error="No data provided",
                    analyzed_at=datetime.now().isoformat()
                )
            
            # Detect columns
            detected = self._detect_columns(df)
            
            l1_col = detected['l1_decision']
            l2_col = detected['l2_decision']
            true_fraud_col = detected['true_fraud']
            id_col = detected['transaction_id']
            
            if not l1_col:
                return AnalysisResult(
                    success=False,
                    error="Could not detect L1 decision column. Expected column name containing 'l1' and 'decision'.",
                    analyzed_at=datetime.now().isoformat()
                )
            
            if not true_fraud_col:
                return AnalysisResult(
                    success=False,
                    error="Could not detect true fraud column. Expected column name containing 'true' and 'fraud'.",
                    analyzed_at=datetime.now().isoformat()
                )
            
            # Get verification_outcome column (should be excluded - it's a result, not a predictor)
            verification_outcome_col = detected.get('verification_outcome')
            
            # Columns to exclude from L1 features:
            # - Target columns (l1_decision, true_fraud, id)
            # - ALL L1-related columns (l1_analyst, l1_tenure, l1_region, l1_risk_appetite, etc.)
            # - ALL L2-related columns
            # - verification_outcome and expected_action (these are results, not predictors)
            l1_exclude_cols = [l1_col, true_fraud_col, id_col]
            if l2_col:
                l1_exclude_cols.append(l2_col)
            if verification_outcome_col:
                l1_exclude_cols.append(verification_outcome_col)
            
            # Exclude ALL L1-related columns (analyst info should not predict their own decision)
            # L1 decision meanings: block=fraud, escalate=unsure, release=not fraud
            for col in df.columns:
                col_lower = col.lower()
                # Exclude L1 analyst-related columns (these describe the analyst, not the transaction)
                if col_lower.startswith('l1_') and col != l1_col:
                    l1_exclude_cols.append(col)
                elif 'l1' in col_lower and col != l1_col and col not in l1_exclude_cols:
                    # Also catch columns like "l1analyst", "l1_tenure", etc.
                    l1_exclude_cols.append(col)
                # Exclude L2-related columns
                if 'l2' in col_lower and col not in l1_exclude_cols:
                    l1_exclude_cols.append(col)
                # Exclude verification-related columns (results, not predictors)
                if 'verification' in col_lower and col not in l1_exclude_cols:
                    l1_exclude_cols.append(col)
                # Exclude expected_action (this is a result/label, not a predictor)
                if col_lower == 'expected_action' or 'expected' in col_lower:
                    if col not in l1_exclude_cols:
                        l1_exclude_cols.append(col)
                # Exclude customer_response (this happens after the decision, not before)
                if 'customer_response' in col_lower or col_lower == 'customer_response':
                    if col not in l1_exclude_cols:
                        l1_exclude_cols.append(col)
            
            l1_exclude_cols = [c for c in l1_exclude_cols if c is not None]
            
            logger.info(f"L1 analysis - Excluding columns: {l1_exclude_cols}")
            
            # Log which columns WILL be used for training
            used_cols = [c for c in df.columns if c not in l1_exclude_cols and c != l1_col]
            logger.info(f"L1 analysis - Using columns for training: {used_cols}")
            
            result = AnalysisResult(
                success=True,
                total_samples=len(df),
                analyzed_at=datetime.now().isoformat()
            )
            
            # Analyze L1 Decisions
            logger.info(f"Analyzing L1 decisions using column: {l1_col}")
            X_l1, y_l1, feature_names = self._preprocess_data(df, l1_col, l1_exclude_cols)
            result.features_used = feature_names
            
            # Train Random Forest for L1 with configurable hyperparameters
            X_train, X_test, y_train, y_test = train_test_split(
                X_l1, y_l1, test_size=0.2, random_state=self.hyperparams.random_state
            )
            
            self.l1_model = RandomForestClassifier(
                n_estimators=self.hyperparams.n_estimators,
                max_depth=self.hyperparams.max_depth,
                min_samples_split=self.hyperparams.min_samples_split,
                min_samples_leaf=self.hyperparams.min_samples_leaf,
                max_features=self.hyperparams.max_features,
                bootstrap=self.hyperparams.bootstrap,
                random_state=self.hyperparams.random_state,
                n_jobs=-1  # Use all CPU cores
            )
            self.l1_model.fit(X_train, y_train)
            
            # Store hyperparameters used
            result.hyperparameters = self.hyperparams.model_dump()
            
            # Predictions and metrics
            l1_preds = self.l1_model.predict(X_test)
            l1_metrics = self._calculate_confusion_metrics(y_test, l1_preds, l1_col)
            
            # Get feature importance from Random Forest (built-in, no SHAP needed)
            l1_feature_importance, l1_importance_summary = self._get_feature_importance(
                self.l1_model, feature_names
            )
            
            # Generate tree visualization for L1 (using sklearn DecisionTree for easy visualization)
            l1_tree_image = self._generate_tree_image(
                X_train, y_train, feature_names, 
                max_depth=min(self.hyperparams.max_depth, 5),
                title="L1 Decision Analysis"
            )
            # Also generate feature importance chart
            l1_importance_image = self._generate_feature_importance_chart(
                self.l1_model, feature_names, title="L1 Feature Importance"
            )
            # Use tree image if available, otherwise use importance chart
            if not l1_tree_image:
                l1_tree_image = l1_importance_image
            
            # Extract decision rules from sklearn tree (more readable than XGBoost)
            l1_tree_text, l1_decision_rules = self._extract_sklearn_tree_rules(
                X_train, y_train, feature_names,
                max_depth=min(self.hyperparams.max_depth, 5)
            )
            l1_tree_structure = self._extract_tree_structure(self.l1_model, feature_names)
            
            result.l1_analysis = {
                'column': l1_col,
                'metrics': l1_metrics,
                'feature_importance': l1_feature_importance,
                'importance_summary': l1_importance_summary,
            }
            result.l1_tree_image = l1_tree_image
            result.l1_tree_structure = l1_tree_structure
            result.l1_decision_rules = l1_decision_rules
            result.l1_tree_text = l1_tree_text
            
            # ===================================================================
            # NEW: Segment Analysis - Extract leaf segments with TP/FP/TN/FN
            # ===================================================================
            logger.info("Starting segment analysis...")
            
            # Create combined target distribution
            combined_target = self._create_combined_target(df, l1_col, true_fraud_col)
            combined_dist = combined_target.value_counts().to_dict()
            result.combined_target_distribution = combined_dist
            logger.info(f"Combined target distribution: {combined_dist}")
            
            # Extract leaf segments from RF model (using full dataset)
            # Need to preprocess full dataset for apply()
            X_full, _, _ = self._preprocess_data(df.copy(), l1_col, l1_exclude_cols)
            
            segments = self._extract_leaf_segments(
                model=self.l1_model,
                X=X_full,
                df_original=df,
                l1_col=l1_col,
                fraud_col=true_fraud_col,
                feature_names=feature_names,
                max_segments=50
            )
            
            result.segment_analysis = segments
            
            # Identify top FP and FN segments
            if segments:
                # Sort by FP rate (descending) and filter significant ones
                result.top_fp_segments = sorted(
                    [s for s in segments if s['FP'] > 0 and s['transaction_count'] >= 10],
                    key=lambda x: x['fp_rate'],
                    reverse=True
                )[:10]
                
                # Sort by FN rate (descending) and filter significant ones
                result.top_fn_segments = sorted(
                    [s for s in segments if s['FN'] > 0 and s['transaction_count'] >= 10],
                    key=lambda x: x['fn_rate'],
                    reverse=True
                )[:10]
                
                # Generate summary
                result.segment_summary = self._generate_segment_summary(segments)
                logger.info(f"Segment analysis complete: {len(segments)} segments, "
                           f"{len(result.top_fp_segments)} high-FP, {len(result.top_fn_segments)} high-FN")
            
            # ===================================================================
            
            # Analyze L2 Decisions if available
            if l2_col and l2_col in df.columns:
                logger.info(f"Analyzing L2 decisions using column: {l2_col}")
                
                # Filter to non-null L2 decisions
                df_l2 = df[df[l2_col].notna()].copy()
                
                if len(df_l2) > 50:  # Need enough samples
                    # For L2: Include L1 decision as a feature (L2 reviews L1's work)
                    # Exclude: L2 target, true fraud, ID, verification_outcome, expected_action
                    # Keep: L1 decision (important for L2 to know what L1 decided)
                    # Also exclude L2 analyst-related columns (l2_analyst, l2_tenure, etc.)
                    l2_exclude_cols = [l2_col, true_fraud_col, id_col]
                    if verification_outcome_col:
                        l2_exclude_cols.append(verification_outcome_col)
                    
                    # Exclude L2 analyst-related columns and other non-predictors
                    for col in df_l2.columns:
                        col_lower = col.lower()
                        # Exclude L2 analyst-related columns (these describe the analyst, not the transaction)
                        if col_lower.startswith('l2_') and col != l2_col:
                            l2_exclude_cols.append(col)
                        elif 'l2' in col_lower and col != l2_col and col not in l2_exclude_cols:
                            l2_exclude_cols.append(col)
                        # Exclude verification-related columns
                        if 'verification' in col_lower and col not in l2_exclude_cols:
                            l2_exclude_cols.append(col)
                        # Exclude expected_action
                        if col_lower == 'expected_action' or 'expected' in col_lower:
                            if col not in l2_exclude_cols:
                                l2_exclude_cols.append(col)
                        # Exclude customer_response (this happens after the decision, not before)
                        if 'customer_response' in col_lower or col_lower == 'customer_response':
                            if col not in l2_exclude_cols:
                                l2_exclude_cols.append(col)
                    
                    l2_exclude_cols = [c for c in l2_exclude_cols if c is not None]
                    
                    logger.info(f"L2 analysis - Excluding columns: {l2_exclude_cols}")
                    logger.info(f"L2 analysis - L1 decision column '{l1_col}' will be INCLUDED as feature")
                    
                    # Log which columns WILL be used for training
                    used_cols_l2 = [c for c in df_l2.columns if c not in l2_exclude_cols and c != l2_col]
                    logger.info(f"L2 analysis - Using columns for training: {used_cols_l2}")
                    
                    X_l2, y_l2, l2_feature_names = self._preprocess_data(df_l2, l2_col, l2_exclude_cols)
                    
                    X_train_l2, X_test_l2, y_train_l2, y_test_l2 = train_test_split(
                        X_l2, y_l2, test_size=0.2, random_state=self.hyperparams.random_state
                    )
                    
                    self.l2_model = RandomForestClassifier(
                        n_estimators=self.hyperparams.n_estimators,
                        max_depth=self.hyperparams.max_depth,
                        min_samples_split=self.hyperparams.min_samples_split,
                        min_samples_leaf=self.hyperparams.min_samples_leaf,
                        max_features=self.hyperparams.max_features,
                        bootstrap=self.hyperparams.bootstrap,
                        random_state=self.hyperparams.random_state,
                        n_jobs=-1
                    )
                    self.l2_model.fit(X_train_l2, y_train_l2)
                    
                    l2_preds = self.l2_model.predict(X_test_l2)
                    l2_metrics = self._calculate_confusion_metrics(y_test_l2, l2_preds, l2_col)
                    
                    # Get feature importance from Random Forest
                    l2_feature_importance, l2_importance_summary = self._get_feature_importance(
                        self.l2_model, l2_feature_names
                    )
                    
                    # Generate tree visualization for L2 (using sklearn DecisionTree for easy visualization)
                    l2_tree_image = self._generate_tree_image(
                        X_train_l2, y_train_l2, l2_feature_names,
                        max_depth=min(self.hyperparams.max_depth, 5),
                        title="L2 Decision Analysis (includes L1 decision)"
                    )
                    # Also generate feature importance chart as fallback
                    l2_importance_image = self._generate_feature_importance_chart(
                        self.l2_model, l2_feature_names, title="L2 Feature Importance"
                    )
                    if not l2_tree_image:
                        l2_tree_image = l2_importance_image
                    
                    # Extract decision rules from sklearn tree (more readable than XGBoost)
                    l2_tree_text, l2_decision_rules = self._extract_sklearn_tree_rules(
                        X_train_l2, y_train_l2, l2_feature_names,
                        max_depth=min(self.hyperparams.max_depth, 5)
                    )
                    l2_tree_structure = self._extract_tree_structure(self.l2_model, l2_feature_names)
                    
                    result.l2_analysis = {
                        'column': l2_col,
                        'metrics': l2_metrics,
                        'feature_importance': l2_feature_importance,
                        'importance_summary': l2_importance_summary,
                    }
                    result.l2_tree_image = l2_tree_image
                    result.l2_tree_structure = l2_tree_structure
                    result.l2_decision_rules = l2_decision_rules
                    result.l2_tree_text = l2_tree_text
            
            # Calculate prediction breakdown (L1 vs True Fraud)
            wrong_preds = self._identify_wrong_predictions(
                df, l1_col, true_fraud_col, l2_col, id_col
            )
            
            fp_count = sum(1 for p in wrong_preds if p['case_type'] == 'false_positive')
            fn_count = sum(1 for p in wrong_preds if p['case_type'] == 'false_negative')
            
            result.prediction_breakdown = {
                'total_cases': len(df),
                'false_positives': fp_count,
                'false_negatives': fn_count,
                'correct_predictions': len(df) - fp_count - fn_count,
                'accuracy_rate': (len(df) - fp_count - fn_count) / len(df) if len(df) > 0 else 0,
            }
            
            # Store wrong predictions for chat (limit to 50)
            result.wrong_predictions = wrong_preds[:50]
            
            # Generate overall feature importance summary for LLM
            summary_parts = [
                f"Analysis of {len(df)} fraud alert cases:",
                f"\nL1 Decision Analysis:",
                result.l1_analysis['importance_summary'],
            ]
            
            if result.l2_analysis:
                summary_parts.extend([
                    f"\nL2 Decision Analysis:",
                    result.l2_analysis['importance_summary'],
                ])
            
            summary_parts.extend([
                f"\nPrediction Accuracy:",
                f"- Correct: {result.prediction_breakdown['correct_predictions']} ({result.prediction_breakdown['accuracy_rate']:.1%})",
                f"- False Positives: {fp_count}",
                f"- False Negatives: {fn_count}",
            ])
            
            result.importance_summary = "\n".join(summary_parts)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Random Forest analysis: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                success=False,
                error=str(e),
                analyzed_at=datetime.now().isoformat()
            )


# Global instance (keeping name for backward compatibility with imports)
xgboost_analyzer = RandomForestAnalyzer()

