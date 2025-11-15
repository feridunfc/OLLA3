# -*- coding: utf-8 -*-
import json
import logging
from typing import Dict, Any, List

from pydantic import BaseModel, Field
from datetime import datetime

from ..base_llm_agent import BaseLLMAgent, LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


# ============================================================
# Pydantic Structured Output
# ============================================================
class CriticOutput(BaseModel):
    security_issues: int = Field(..., ge=0, le=10)
    quality_issues: int = Field(..., ge=0, le=10)
    criticality: str = Field(..., pattern="^(low|medium|high)$")
    suggestions: List[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str


# ============================================================
# OpenAI Critic Agent
# ============================================================
class OpenAICriticAgent(BaseLLMAgent):

    def __init__(self):
        super().__init__(
            LLMConfig(
                provider="openai",
                model="gpt-4o",              # primary
                max_tokens=2000,
                temperature=0.1,
                max_retries=3,
                timeout=30
            )
        )

        # fallback chain
        self.fallback_models = ["gpt-4o-mini", "gpt-4-turbo"]

    # --------------------------------------------------------
    async def run(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified interface for orchestrator
        """
        try:
            messages = self._build_messages(goal, context)

            # 1) primary model call
            response = await self.invoke(
                messages,
                response_format={"type": "json_object"}
            )

            if not response.success:
                logger.warning(f"Primary model failed: {response.error}")
                return await self._try_fallback(messages)

            output = self._parse_json(response.content)
            return self._success(output, response)

        except Exception as e:
            logger.error(f"CriticAgent fatal error: {e}")
            return self._error(str(e))

    # --------------------------------------------------------
    async def _try_fallback(self, messages):
        """
        Try alternative OpenAI models automatically.
        """

        for model in self.fallback_models:
            logger.warning(f"Falling back to: {model}")
            self.config.model = model

            try:
                response = await self.invoke(
                    messages,
                    response_format={"type": "json_object"}
                )

                if response.success:
                    output = self._parse_json(response.content)
                    return self._success(output, response)  # return asap

            except Exception as e:
                logger.error(f"Fallback {model} failed: {e}")

        # All fallbacks failed
        return self._error("All fallback models failed")

    # --------------------------------------------------------
    def _build_messages(self, goal: str, context: Dict[str, Any]):
        """
        Build high-quality prompt with context-awareness
        """
        domain = context.get("domain", "general")
        priority = context.get("priority", "medium")
        files = context.get("files", [])

        system_msg = """
You are a senior code reviewer and security auditor.
You MUST output ONLY valid JSON according to the schema.
Be strict, concise, and actionable.
"""

        user_msg = f"""
CODE REVIEW REQUEST — {domain.upper()}

GOAL:
{goal}

CONTEXT:
- Priority: {priority}
- Files: {files}

REQUIREMENTS:
1. Rate security issues (0-10)
2. Rate code quality issues (0-10)
3. Predict criticality: low / medium / high
4. Provide up to 5 actionable suggestions
5. Confidence score 0.0 - 1.0
6. Provide a short summary

OUTPUT JSON SCHEMA:
{{
  "security_issues": number,
  "quality_issues": number,
  "criticality": "low|medium|high",
  "suggestions": ["text1", "text2"],
  "confidence": 0.0,
  "summary": "text"
}}
"""

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg.strip()}
        ]

    # --------------------------------------------------------
    def _parse_json(self, raw: str) -> CriticOutput:
        """
        Extract JSON safely from model output
        """

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1

            if start == -1 or end <= 0:
                raise ValueError("No JSON found")

            data = json.loads(raw[start:end])
            return CriticOutput(**data)

        except Exception as e:
            logger.error(f"JSON parse failed: {e}")

            return CriticOutput(
                security_issues=0,
                quality_issues=0,
                criticality="low",
                suggestions=["Failed to parse JSON"],
                confidence=0.1,
                summary="Model output could not be parsed."
            )

    # --------------------------------------------------------
    def _success(self, output: CriticOutput, llm: LLMResponse):
        return {
            "status": "success",
            "agent": "openai_critic",
            "data": output.dict(),
            "metrics": {
                "model": self.config.model,
                "tokens_used": llm.total_tokens,
                "prompt_tokens": llm.prompt_tokens,
                "completion_tokens": llm.completion_tokens,
            },
            "confidence": output.confidence,
            "timestamp": datetime.now().isoformat()
        }

    # --------------------------------------------------------
    def _error(self, error: str):
        return {
            "status": "failed",
            "agent": "openai_critic",
            "error": error,
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }
