"""
Report Generation Module
Generates downloadable reports from chat sessions and context.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import os
import logging

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    FULL_EXPORT = "full_export"
    SHADOW_REPORT = "shadow_report"  # Shadow rule detection report (requires txn chat)


class ReportFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class ReportSection(BaseModel):
    title: str
    content: str
    section_type: str = "text"  # text, code, finding, recommendation


class GeneratedReport(BaseModel):
    """Structured report output from LLM."""
    title: str = Field(..., description="Report title")
    executive_summary: str = Field(..., description="2-3 sentence non-technical summary")
    key_findings: List[str] = Field(..., description="Bullet points of main findings")
    risk_assessment: Optional[str] = Field(None, description="Any risks or concerns identified")
    recommendations: List[str] = Field(..., description="Actionable next steps")
    technical_details: Optional[str] = Field(None, description="Technical analysis for developer reports")


class AnalystBehaviorReport(BaseModel):
    """Structured report for analyst behavior analysis and shadow rule detection."""
    title: str = Field(..., description="Report title")
    executive_summary: str = Field(..., description="High-level summary for management")
    
    # Shadow Rules Section
    shadow_rules_detected: List[str] = Field(
        ..., 
        description="List of undocumented patterns/rules detected in analyst behavior that differ from official guidelines"
    )
    shadow_rule_severity: str = Field(
        ..., 
        description="Overall severity: 'low' (minor deviations), 'medium' (notable patterns), 'high' (significant compliance concerns)"
    )
    
    # Bias Analysis
    potential_biases: List[str] = Field(
        ..., 
        description="Identified biases in analyst decisions (e.g., demographic, temporal, transaction type)"
    )
    
    # Guideline Compliance
    guideline_compliance_score: str = Field(
        ..., 
        description="Estimated compliance level: 'high', 'medium', 'low' with explanation"
    )
    guideline_gaps: List[str] = Field(
        ..., 
        description="Guidelines that are frequently ignored or inconsistently applied"
    )
    
    # Decision Patterns
    override_patterns: List[str] = Field(
        ..., 
        description="Patterns in when analysts override model predictions"
    )
    consistency_issues: List[str] = Field(
        ..., 
        description="Inconsistencies in decision-making across similar cases"
    )
    
    # Recommendations
    training_recommendations: List[str] = Field(
        ..., 
        description="Suggested training topics to address identified issues"
    )
    process_improvements: List[str] = Field(
        ..., 
        description="Recommended changes to the review process"
    )
    guideline_updates: List[str] = Field(
        ..., 
        description="Suggested updates to official guidelines based on shadow rules that may be valid"
    )
    
    # Risk Assessment
    compliance_risks: List[str] = Field(
        ..., 
        description="Regulatory or compliance risks identified"
    )
    
    # Evidence
    supporting_examples: List[str] = Field(
        ..., 
        description="Specific examples from the analysis that support findings"
    )


class ReportRequest(BaseModel):
    session_id: str
    report_type: ReportType = ReportType.EXECUTIVE
    format: ReportFormat = ReportFormat.MARKDOWN
    include_code: bool = False
    include_schema: bool = True
    include_chat_history: bool = True
    include_json_data: bool = False


class ReportGenerator:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0.3,
            anthropic_api_key=api_key
        )

    def _summarize_chat(self, messages: List[Dict]) -> str:
        """Summarize chat messages for report context."""
        if not messages:
            return "No chat history available."
        
        summary_parts = []
        for msg in messages[-10:]:  # Last 10 messages
            role = "User" if msg.get('type') == 'user' else "AI"
            content = msg.get('content', '')[:200]
            if len(msg.get('content', '')) > 200:
                content += "..."
            summary_parts.append(f"**{role}**: {content}")
        
        return "\n\n".join(summary_parts)

    def _summarize_schema(self, schema: Dict[str, Any]) -> str:
        """Summarize data schema for report context."""
        if not schema:
            return "No data schema available."
        
        parts = []
        
        # Dataset info
        dataset_info = schema.get('dataset_info', {})
        if dataset_info:
            parts.append(f"**Dataset**: {dataset_info.get('total_samples', 'Unknown')} samples, {dataset_info.get('total_features', 'Unknown')} features")
        
        # Decision columns (L1/L2 decisions, true fraud flag)
        decision_cols = schema.get('decision_columns', [])
        if decision_cols:
            parts.append(f"**Decision Columns**: {', '.join(decision_cols)}")
        
        # Analyst columns
        analyst_cols = schema.get('analyst_columns', [])
        if analyst_cols:
            parts.append(f"**Analyst Columns**: {', '.join(analyst_cols[:8])}")
        
        # Top features
        features = schema.get('features', [])
        if features:
            # Show features with descriptions
            feature_info = []
            for f in features[:10]:
                name = f.get('name', '')
                desc = f.get('description', '')
                if desc:
                    feature_info.append(f"`{name}` - {desc}")
                else:
                    feature_info.append(f"`{name}`")
            parts.append(f"**Key Features** ({len(features)} total):\n" + "\n".join([f"  - {fi}" for fi in feature_info]))
            
            # Missing values summary
            missing_features = [f for f in features if f.get('missing_count', 0) > 0]
            if missing_features:
                parts.append(f"**Data Quality**: {len(missing_features)} features with missing values")
        
        # Feature engineering
        fe = schema.get('feature_engineering', {})
        if fe:
            fe_parts = []
            if fe.get('one_hot_encoded'):
                fe_parts.append(f"{len(fe['one_hot_encoded'])} one-hot encoded")
            if fe.get('datetime_derived'):
                fe_parts.append(f"{len(fe['datetime_derived'])} datetime-derived")
            if fe.get('computed'):
                fe_parts.append(f"{len(fe['computed'])} computed features")
            if fe_parts:
                parts.append(f"**Feature Engineering**: {', '.join(fe_parts)}")
        
        # Data quality
        quality = schema.get('data_quality', {})
        if quality:
            if quality.get('high_cardinality'):
                parts.append(f"**High Cardinality Columns**: {', '.join(quality['high_cardinality'][:5])}")
            if quality.get('potential_leakage'):
                parts.append(f"⚠️ **Potential Leakage**: {', '.join(quality['potential_leakage'])}")
        
        return "\n".join(parts) if parts else "Schema data available but minimal details."

    def _get_report_prompt(self, report_type: ReportType) -> str:
        """Get the appropriate prompt based on report type."""
        
        base_prompt = """You are generating a professional report for a fraud detection model analysis session.

Based on the provided context, generate a structured report.

IMPORTANT: This report is for business stakeholders. Do NOT include any code, technical implementation details, or programming-related content unless explicitly a technical report."""

        if report_type == ReportType.EXECUTIVE:
            return base_prompt + """

TARGET AUDIENCE: Executives, Board Members, Senior Management
TONE: Clear, concise, business-focused, easy to understand
FOCUS ON:
- What the model does and how well it performs
- Key risk factors the model identifies
- Business impact and operational implications
- High-level concerns or areas needing attention
- Clear, actionable recommendations

AVOID:
- Technical jargon, code, or implementation details
- Statistical formulas or ML terminology
- Feature names without plain English explanations
- Anything that requires technical background to understand

Keep the executive summary under 3 sentences.
Write as if explaining to someone unfamiliar with machine learning."""

        elif report_type == ReportType.TECHNICAL:
            return base_prompt + """

TARGET AUDIENCE: Data scientists, ML engineers, Technical team
TONE: Technical, detailed, precise
FOCUS ON:
- Feature importance analysis and interpretation
- Potential data leakage or model issues
- Model architecture and methodology insights
- Code quality observations and improvements
- Statistical considerations and validation
- Data preprocessing and feature engineering review

This is the ONLY report type where code and technical details are appropriate.
Include specific technical observations about the model implementation."""

        elif report_type == ReportType.SHADOW_REPORT:
            return """You are an expert in analyzing human analyst behavior in fraud detection workflows.

Your task is to analyze the transaction review discussions and identify:
1. SHADOW RULES: Undocumented patterns in analyst decisions that differ from official guidelines
2. BIASES: Systematic biases in decision-making (demographic, temporal, amount-based, etc.)
3. INCONSISTENCIES: Cases where similar transactions received different treatment
4. COMPLIANCE GAPS: Areas where guidelines are being ignored or misinterpreted

TARGET AUDIENCE: Risk Management, Compliance, Operations Management
TONE: Clear, evidence-based, actionable - written for business professionals

IMPORTANT:
- Cite specific examples from the discussions when possible
- Distinguish between potentially valid shadow rules (that should become official) and problematic ones
- Prioritize findings by severity and frequency
- Provide concrete, actionable recommendations
- Write in plain business language, avoid technical jargon

The goal is to help the bank improve their fraud review process, ensure compliance, and reduce bias."""

        else:  # FULL_EXPORT
            return base_prompt + """

TARGET AUDIENCE: Operations teams, Process managers, Quality assurance
TONE: Comprehensive but accessible, written in plain business language
FOCUS ON:
- Complete summary of the model analysis session
- Key findings about fraud detection patterns
- All recommendations from the conversation
- Process improvement opportunities
- Risk factors and areas of concern

AVOID:
- Code snippets or technical implementation details
- Complex statistical terminology
- ML jargon that requires technical background

Write in clear business language that operations staff can understand and act upon."""

    async def generate_report(
        self,
        session_data: Dict[str, Any],
        request: ReportRequest
    ) -> Dict[str, Any]:
        """Generate a report from session data."""
        
        # For shadow reports, use specialized method
        if request.report_type == ReportType.SHADOW_REPORT:
            return await self._generate_analyst_behavior_report(session_data, request)
        
        # Extract relevant data
        ml_code = session_data.get('mlCode', '')
        data_schema = session_data.get('dataSchema')
        global_json = session_data.get('globalJson')
        txn_json = session_data.get('txnJson')
        chat_history = session_data.get('chatHistory', {})
        
        # Combine all chat messages
        all_messages = []
        all_messages.extend(chat_history.get('codeAnalyzer', []))
        all_messages.extend(chat_history.get('globalChat', []))
        all_messages.extend(chat_history.get('txnChat', []))
        
        # Build context for LLM
        context_parts = []
        
        if ml_code and request.include_code:
            code_preview = ml_code[:2000] + ("..." if len(ml_code) > 2000 else "")
            context_parts.append(f"## ML Code\n```python\n{code_preview}\n```")
        
        if data_schema and request.include_schema:
            # Summarize schema for report
            schema_summary = self._summarize_schema(data_schema)
            context_parts.append(f"## Data Schema\n{schema_summary}")
        
        if global_json and request.include_json_data:
            context_parts.append(f"## Global Model Explanation\n```json\n{json.dumps(global_json, indent=2)[:1500]}\n```")
        
        if txn_json and request.include_json_data:
            # New format: selected cases with filters
            selected_cases = txn_json.get('selectedCases', [])
            active_filters = txn_json.get('filters', [])
            
            txn_summary = f"**Selected Cases**: {len(selected_cases)}\n"
            if active_filters:
                filter_labels = [f.get('label', '') for f in active_filters]
                txn_summary += f"**Filters**: {', '.join(filter_labels)}\n"
            
            # Show sample of selected cases
            if selected_cases:
                sample = selected_cases[:3]
                txn_summary += "\n**Sample Cases**:\n```json\n" + json.dumps(sample, indent=2)[:800] + "\n```"
            
            context_parts.append(f"## Selected Cases for Analysis\n{txn_summary}")
        
        if all_messages and request.include_chat_history:
            chat_summary = self._summarize_chat(all_messages)
            context_parts.append(f"## Conversation Summary\n{chat_summary}")
        
        context_text = "\n\n".join(context_parts) if context_parts else "Limited context available."
        
        # Build the prompt
        system_prompt = self._get_report_prompt(request.report_type)
        
        # Get schema stats for session info
        feature_count = 0
        target_name = "Unknown"
        if data_schema:
            feature_count = len(data_schema.get('features', [])) or data_schema.get('dataset_info', {}).get('total_features', 0)
            target_name = data_schema.get('target', {}).get('name', 'Unknown')
        
        # Get selected cases count
        selected_cases_count = len(txn_json.get('selectedCases', [])) if txn_json else 0
        
        user_prompt = f"""Generate a {request.report_type.value} report based on this analyst decision analysis session:

{context_text}

Session Info:
- Model Version: {global_json.get('model_version', 'Unknown') if global_json else 'Not provided'}
- Code Length: {len(ml_code)} characters
- Features: {feature_count} features
- Chat Messages: {len(all_messages)}
- Selected Cases: {selected_cases_count}

Generate a comprehensive report following the structured format."""

        # Generate report using structured output
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            structured_llm = self.llm.with_structured_output(GeneratedReport)
            report: GeneratedReport = structured_llm.invoke(messages)
            
            # Format based on requested format
            if request.format == ReportFormat.MARKDOWN:
                return self._format_as_markdown(report, session_data, request)
            else:
                return self._format_as_json(report, session_data, request)
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    async def _generate_analyst_behavior_report(
        self,
        session_data: Dict[str, Any],
        request: ReportRequest
    ) -> Dict[str, Any]:
        """Generate a specialized analyst behavior/shadow rule detection report."""
        
        # Extract all relevant data
        chat_history = session_data.get('chatHistory', {})
        txn_chat = chat_history.get('txnChat', [])
        global_chat = chat_history.get('globalChat', [])
        
        # Data schema for context
        data_schema = session_data.get('dataSchema')
        
        # Global patterns from code analysis
        global_json = session_data.get('globalJson')
        
        # Selected cases with filters (new format)
        txn_json = session_data.get('txnJson')
        selected_cases = txn_json.get('selectedCases', []) if txn_json else []
        active_filters = txn_json.get('filters', []) if txn_json else []
        
        # Guidelines context
        guidelines = session_data.get('guidelines', [])
        
        # Build context
        context_parts = []
        
        # Data Schema Summary
        if data_schema:
            schema_summary = self._summarize_schema(data_schema)
            context_parts.append(f"## Data Schema\n{schema_summary}")
        
        # Global Patterns (from code analysis)
        if global_json:
            patterns_text = []
            if global_json.get('model_version'):
                patterns_text.append(f"**Model Version**: {global_json.get('model_version')}")
            if global_json.get('global_importance'):
                top_features = global_json.get('global_importance', [])[:5]
                patterns_text.append(f"**Top Features**: {', '.join([f.get('feature', '') for f in top_features])}")
            if global_json.get('decision_rules'):
                patterns_text.append(f"**Decision Rules Extracted**: {len(global_json.get('decision_rules', []))}")
            if global_json.get('detected_biases'):
                patterns_text.append(f"**Detected Biases**: {', '.join(global_json.get('detected_biases', []))}")
            if patterns_text:
                context_parts.append(f"## Global Patterns (from Code Analysis)\n" + "\n".join(patterns_text))
        
        # Selected Cases Summary
        if selected_cases:
            context_parts.append(f"## Selected Cases for Analysis")
            context_parts.append(f"**Total Cases**: {len(selected_cases)}")
            if active_filters:
                filter_labels = [f.get('label', f"{f.get('column')} {f.get('operator')} {f.get('value')}") for f in active_filters]
                context_parts.append(f"**Filters Applied**: {', '.join(filter_labels)}")
            # Sample of selected cases
            sample_cases = selected_cases[:5]
            cases_text = []
            for case in sample_cases:
                case_info = f"- Alert {case.get('alert_id', 'N/A')}: L1={case.get('l1_decision', 'N/A')}, L2={case.get('l2_decision', 'N/A')}, True Fraud={case.get('true_fraud_flag', 'N/A')}"
                if case.get('l1_override_flag') or case.get('l2_override_flag'):
                    case_info += " [OVERRIDE]"
                cases_text.append(case_info)
            context_parts.append("**Sample Cases**:\n" + "\n".join(cases_text))
        
        # Guidelines context
        if guidelines:
            guidelines_text = []
            for g in guidelines:
                g_text = f"**{g.get('title')}** ({g.get('category', 'custom')}): {g.get('description', '')}"
                if g.get('rules'):
                    g_text += f"\n  Rules: {', '.join(g['rules'][:5])}"
                guidelines_text.append(g_text)
            context_parts.append(f"## Bank Guidelines\n" + "\n".join(guidelines_text))
        
        # Chat History - combine global and txn chat for full context
        all_chat = []
        if global_chat:
            all_chat.append("### Global Analysis Discussion")
            for msg in global_chat[-10:]:  # Last 10 messages
                role = "User" if msg.get('type') == 'user' else "AI"
                content = msg.get('content', '')[:500]
                if len(msg.get('content', '')) > 500:
                    content += "..."
                all_chat.append(f"**{role}**: {content}")
        
        if txn_chat:
            all_chat.append("\n### Case Analysis Discussion")
            for msg in txn_chat:  # Full txn chat history
                role = "User" if msg.get('type') == 'user' else "AI"
                content = msg.get('content', '')
                all_chat.append(f"**{role}**: {content}")
        
        if all_chat:
            context_parts.append(f"## Chat History\n" + "\n\n".join(all_chat))
        
        context_text = "\n\n".join(context_parts) if context_parts else "Limited context available. Please analyze more transactions to detect patterns."
        
        system_prompt = self._get_report_prompt(ReportType.SHADOW_REPORT)
        
        user_prompt = f"""Analyze the following fraud review session and generate a Shadow Report.

{context_text}

Session Stats:
- Data Schema: {len(data_schema.get('features', [])) if data_schema else 0} features
- Selected Cases: {len(selected_cases)}
- Filters Applied: {len(active_filters)}
- Guidelines Configured: {len(guidelines)}
- Chat Messages: {len(global_chat) + len(txn_chat)}

Based on this information:
1. Identify shadow rules (undocumented patterns) in analyst behavior
2. Detect potential biases in how cases were handled
3. Assess guideline compliance based on the selected cases
4. Note inconsistencies in decision-making
5. Provide actionable recommendations

Generate a comprehensive shadow report."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            structured_llm = self.llm.with_structured_output(AnalystBehaviorReport)
            report: AnalystBehaviorReport = structured_llm.invoke(messages)
            
            # Format as markdown
            return self._format_analyst_report_as_markdown(report, session_data, request)
                
        except Exception as e:
            logger.error(f"Error generating analyst behavior report: {e}")
            raise

    def _format_analyst_report_as_markdown(
        self,
        report: AnalystBehaviorReport,
        session_data: Dict,
        request: ReportRequest
    ) -> Dict[str, Any]:
        """Format analyst behavior report as Markdown."""
        
        md_parts = []
        
        # Header
        md_parts.append(f"# {report.title}")
        md_parts.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        md_parts.append(f"*Report Type: Analyst Behavior Analysis*\n")
        
        # Executive Summary
        md_parts.append("## Executive Summary")
        md_parts.append(report.executive_summary)
        md_parts.append("")
        
        # Shadow Rules Section
        md_parts.append("## 🔍 Shadow Rules Detected")
        md_parts.append(f"**Severity: {report.shadow_rule_severity.upper()}**\n")
        if report.shadow_rules_detected:
            for rule in report.shadow_rules_detected:
                md_parts.append(f"- {rule}")
        else:
            md_parts.append("- No significant shadow rules detected")
        md_parts.append("")
        
        # Bias Analysis
        md_parts.append("## ⚠️ Potential Biases")
        if report.potential_biases:
            for bias in report.potential_biases:
                md_parts.append(f"- {bias}")
        else:
            md_parts.append("- No significant biases identified")
        md_parts.append("")
        
        # Guideline Compliance
        md_parts.append("## 📋 Guideline Compliance")
        md_parts.append(f"**Compliance Score: {report.guideline_compliance_score}**\n")
        if report.guideline_gaps:
            md_parts.append("**Gaps Identified:**")
            for gap in report.guideline_gaps:
                md_parts.append(f"- {gap}")
        md_parts.append("")
        
        # Override Patterns
        md_parts.append("## 🔄 Model Override Patterns")
        if report.override_patterns:
            for pattern in report.override_patterns:
                md_parts.append(f"- {pattern}")
        else:
            md_parts.append("- No significant override patterns detected")
        md_parts.append("")
        
        # Consistency Issues
        if report.consistency_issues:
            md_parts.append("## ⚖️ Consistency Issues")
            for issue in report.consistency_issues:
                md_parts.append(f"- {issue}")
            md_parts.append("")
        
        # Compliance Risks
        if report.compliance_risks:
            md_parts.append("## 🚨 Compliance Risks")
            for risk in report.compliance_risks:
                md_parts.append(f"- {risk}")
            md_parts.append("")
        
        # Recommendations
        md_parts.append("## 📚 Training Recommendations")
        for rec in report.training_recommendations:
            md_parts.append(f"- {rec}")
        md_parts.append("")
        
        md_parts.append("## 🔧 Process Improvements")
        for improvement in report.process_improvements:
            md_parts.append(f"- {improvement}")
        md_parts.append("")
        
        md_parts.append("## 📝 Suggested Guideline Updates")
        for update in report.guideline_updates:
            md_parts.append(f"- {update}")
        md_parts.append("")
        
        # Supporting Examples
        if report.supporting_examples:
            md_parts.append("## 📌 Supporting Examples")
            for example in report.supporting_examples:
                md_parts.append(f"- {example}")
            md_parts.append("")
        
        # Footer
        md_parts.append("---")
        md_parts.append(f"*Session: {session_data.get('name', 'Unnamed')}*")
        
        content = "\n".join(md_parts)
        
        return {
            "success": True,
            "report": {
                "title": report.title,
                "content": content,
                "format": "markdown",
                "report_type": "analyst_behavior",
                "generated_at": datetime.now().isoformat(),
                "shadow_rule_severity": report.shadow_rule_severity,
                "shadow_rules_count": len(report.shadow_rules_detected),
                "compliance_score": report.guideline_compliance_score,
            }
        }

    def _format_as_markdown(
        self, 
        report: GeneratedReport, 
        session_data: Dict, 
        request: ReportRequest
    ) -> Dict[str, Any]:
        """Format report as Markdown."""
        
        md_parts = []
        
        # Header
        md_parts.append(f"# {report.title}")
        md_parts.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        md_parts.append(f"*Report Type: {request.report_type.value.title()}*\n")
        
        # Executive Summary
        md_parts.append("## Executive Summary")
        md_parts.append(report.executive_summary)
        md_parts.append("")
        
        # Key Findings
        md_parts.append("## Key Findings")
        for finding in report.key_findings:
            md_parts.append(f"- {finding}")
        md_parts.append("")
        
        # Risk Assessment
        if report.risk_assessment:
            md_parts.append("## Risk Assessment")
            md_parts.append(report.risk_assessment)
            md_parts.append("")
        
        # Recommendations
        md_parts.append("## Recommendations")
        for i, rec in enumerate(report.recommendations, 1):
            md_parts.append(f"{i}. {rec}")
        md_parts.append("")
        
        # Technical Details (for technical reports)
        if report.technical_details and request.report_type == ReportType.TECHNICAL:
            md_parts.append("## Technical Details")
            md_parts.append(report.technical_details)
            md_parts.append("")
        
        # Appendix with raw data if requested
        if request.include_json_data and session_data.get('globalJson'):
            md_parts.append("## Appendix: Model Data")
            md_parts.append("### Global Explanation JSON")
            md_parts.append(f"```json\n{json.dumps(session_data['globalJson'], indent=2)}\n```")
        
        content = "\n".join(md_parts)
        
        return {
            "format": "markdown",
            "content": content,
            "filename": f"report_{request.report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            "title": report.title
        }

    def _format_as_json(
        self, 
        report: GeneratedReport, 
        session_data: Dict, 
        request: ReportRequest
    ) -> Dict[str, Any]:
        """Format report as JSON."""
        
        return {
            "format": "json",
            "content": {
                "title": report.title,
                "generated_at": datetime.now().isoformat(),
                "report_type": request.report_type.value,
                "executive_summary": report.executive_summary,
                "key_findings": report.key_findings,
                "risk_assessment": report.risk_assessment,
                "recommendations": report.recommendations,
                "technical_details": report.technical_details,
                "session_info": {
                    "session_id": request.session_id,
                    "model_version": session_data.get('globalJson', {}).get('model_version') if session_data.get('globalJson') else None,
                    "code_length": len(session_data.get('mlCode', '')),
                }
            },
            "filename": f"report_{request.report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "title": report.title
        }


# Global instance
report_generator = ReportGenerator()

