import os
import json
from typing import Dict, List, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from models import ExplainGlobalRequest, ExplainTransactionRequest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHAT_HISTORY_FILE = "chat_memory.json"

class ChatManager:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables.")
        
        self.llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
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

    async def chat(self, session_id: str, user_input: str, context: Dict[str, Any] = None) -> str:
        history = self.get_session_history(session_id)
        
        system_prompt = (
            "You are an expert AI Model Explainer Assistant. "
            "Your goal is to help users understand their machine learning models.\n"
            "You have access to 'Global Explanations' and 'Transaction Explanations' provided in the context.\n"
            "When answering, refer to specific features, scores, and contributions from the JSON data.\n"
            "If the user asks for more evidence (e.g., 'What if income was higher?'), "
            "suggest that they run a drilldown/perturbation analysis (simulated response for now)."
        )

        # Include context if available
        if context:
             system_prompt += f"\n\nCURRENT CONTEXT:\n{json.dumps(context, indent=2)}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm
        
        # Manually manage history for simple control + persistence
        # (In a full LangChain app we might use RunnableWithMessageHistory, but this gives us direct control for saving)
        response_msg = chain.invoke({
            "history": history.messages,
            "input": user_input
        })
        
        history.add_user_message(user_input)
        history.add_ai_message(response_msg.content)
        self.save_history()

        return response_msg.content

    async def guide_code_to_json(self, session_id: str, user_code: str) -> str:
        """
        Special helper to guide the user on how to implement explain_global/explain_txn
        based on their provided code.
        """
        history = self.get_session_history(session_id)
        
        system_prompt = (
            "You are an expert ML Engineer. The user has provided their model training/inference code.\n"
            "Your task is to analyze their code and provide specific Python code snippets implementing two functions:\n"
            "1. `explain_global(...) -> GlobalExplanation`\n"
            "2. `explain_txn(...) -> TransactionExplanation`\n"
            "Map their specific model object, variables, and feature names to the required JSON structure.\n"
            "Assume they have the `models.py` definitions available."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Here is my code:\n\n{input}"),
        ])
        
        chain = prompt | self.llm
        
        response_msg = chain.invoke({
            "history": history.messages[-1:], # Keep context light for code analysis
            "input": user_code
        })
        
        history.add_user_message(f"Code analysis request for: {user_code[:50]}...")
        history.add_ai_message(response_msg.content)
        self.save_history()
        
        return response_msg.content

chat_manager = ChatManager()
