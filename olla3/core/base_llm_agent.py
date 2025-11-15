# -*- coding: utf-8 -*-
import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import ollama

from .enhanced_metrics import track_llm_metrics

logger = logging.getLogger(__name__)


# ----------------------------------------------------
# LLM Configuration Model
# ----------------------------------------------------
class LLMConfig(BaseModel):
    provider: str = Field(..., description="LLM provider: openai | ollama")
    model: str = Field(..., description="Model name")
    max_tokens: int = Field(default=2048, ge=1, le=32000)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=30, description="Timeout in seconds")


# ----------------------------------------------------
# Standardized LLM Response
# ----------------------------------------------------
class LLMResponse(BaseModel):
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error: Optional[str] = None


# ----------------------------------------------------
# Base Agent Abstraction Layer
# ----------------------------------------------------
class BaseLLMAgent(ABC):

    def __init__(self, config: LLMConfig):
        self.config = config
        self._validate_env()

        # OpenAI async client oluştur
        if config.provider == "openai":
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=config.timeout
            )
        else:
            self.client = None

    # ------------------------------------------------
    # Environment Validation
    # ------------------------------------------------
    def _validate_env(self):
        if self.config.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY environment variable missing.")

        if self.config.provider == "ollama":
            # Ollama için env zorunlu değil, ancak health check konabilir
            pass

    # ------------------------------------------------
    # Public Interface With Retry
    # ------------------------------------------------
    async def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                return await self._call_llm(messages, **kwargs)

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM attempt {attempt + 1}/{self.config.max_retries} failed: {e}")

                if attempt < self.config.max_retries - 1:
                    delay = 2 ** attempt
                    await asyncio.sleep(delay)

        return LLMResponse(
            content="",
            success=False,
            error=f"All retries failed. Last error: {last_error}"
        )

    # ------------------------------------------------
    # Provider Dispatch
    # ------------------------------------------------
    @track_llm_metrics(
        get_provider=lambda self: self.config.provider,
        get_model=lambda self: self.config.model
    )
    async def _call_llm(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:

        if self.config.provider == "openai":
            return await self._call_openai(messages, **kwargs)

        if self.config.provider == "ollama":
            return await self._call_ollama(messages, **kwargs)

        raise ValueError(f"Unsupported provider: {self.config.provider}")

    # ------------------------------------------------
    # OpenAI Implementation
    # ------------------------------------------------
    async def _call_openai(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            **kwargs,
        )

        msg = response.choices[0].message.content or ""

        return LLMResponse(
            content=msg,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    # ------------------------------------------------
    # Ollama Implementation
    # ------------------------------------------------
    async def _call_ollama(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:

        def _sync():
            return ollama.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
                **kwargs
            )

        try:
            result = await asyncio.to_thread(_sync)

            content = result["message"]["content"]
            pt = result.get("prompt_eval_count", 0)
            ct = result.get("eval_count", 0)

            return LLMResponse(
                content=content,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
            )

        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return LLMResponse(content="", success=False, error=str(e))

    # ------------------------------------------------
    # Abstract interface for agents
    # ------------------------------------------------
    @abstractmethod
    async def run(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    # ------------------------------------------------
    # Utility
    # ------------------------------------------------
    def get_config(self):
        return self.config.dict()
