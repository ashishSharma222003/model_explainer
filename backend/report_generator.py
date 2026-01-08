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


class ReportRequest(BaseModel):
    session_id: str
    report_type: ReportType = ReportType.EXECUTIVE
    format: ReportFormat = ReportFormat.MARKDOWN
    include_code: bool = False
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

    def _get_report_prompt(self, report_type: ReportType) -> str:
        """Get the appropriate prompt based on report type."""
        
        base_prompt = """You are generating a professional report for an ML model analysis session.

Based on the provided context, generate a structured report."""

        if report_type == ReportType.EXECUTIVE:
            return base_prompt + """

TARGET AUDIENCE: Executives, non-technical stakeholders
TONE: Clear, concise, business-focused
FOCUS ON:
- Business implications of the model
- High-level risks and concerns
- Clear recommendations without technical jargon
- Bottom-line impact

Keep the executive summary under 3 sentences.
Make recommendations actionable and clear."""

        elif report_type == ReportType.TECHNICAL:
            return base_prompt + """

TARGET AUDIENCE: Data scientists, ML engineers
TONE: Technical, detailed, precise
FOCUS ON:
- Feature importance analysis
- Potential data leakage or issues
- Model architecture insights
- Code quality observations
- Statistical considerations

Include technical details and specific observations about the model."""

        else:  # FULL_EXPORT
            return base_prompt + """

TARGET AUDIENCE: Mixed (technical and non-technical)
TONE: Comprehensive, balanced
FOCUS ON:
- Complete analysis summary
- Both business and technical perspectives
- Full recommendations list
- All findings from the conversation"""

    async def generate_report(
        self,
        session_data: Dict[str, Any],
        request: ReportRequest
    ) -> Dict[str, Any]:
        """Generate a report from session data."""
        
        # Extract relevant data
        ml_code = session_data.get('mlCode', '')
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
        
        if global_json and request.include_json_data:
            context_parts.append(f"## Global Model Explanation\n```json\n{json.dumps(global_json, indent=2)[:1500]}\n```")
        
        if txn_json and request.include_json_data:
            context_parts.append(f"## Transaction Explanation\n```json\n{json.dumps(txn_json, indent=2)[:1000]}\n```")
        
        if all_messages and request.include_chat_history:
            chat_summary = self._summarize_chat(all_messages)
            context_parts.append(f"## Conversation Summary\n{chat_summary}")
        
        context_text = "\n\n".join(context_parts) if context_parts else "Limited context available."
        
        # Build the prompt
        system_prompt = self._get_report_prompt(request.report_type)
        
        user_prompt = f"""Generate a {request.report_type.value} report based on this ML model analysis session:

{context_text}

Session Info:
- Model Version: {global_json.get('model_version', 'Unknown') if global_json else 'Not provided'}
- Code Length: {len(ml_code)} characters
- Chat Messages: {len(all_messages)}
- Has Transaction Data: {'Yes' if txn_json else 'No'}

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

