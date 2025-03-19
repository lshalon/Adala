import logging
import traceback
from typing import Any, Dict, List, Optional, Type

import instructor
from openai import AsyncOpenAI, OpenAI  # If you still need fallback to openai
from pydantic import BaseModel, ConfigDict, field_validator
from tenacity import AsyncRetrying, Retrying, stop_after_attempt

from adala.runtimes.base import CostEstimate
from adala.utils.internal_data import InternalDataFrame
from adala.utils.llm_utils import (
    arun_instructor_with_payloads,
    run_instructor_with_payload,
)
from adala.utils.model_info_utils import (
    NoModelsFoundError,
    _estimate_cost,
    match_model_provider_string,
)
from adala.utils.parse import MessageChunkType, partial_str_format

from .base import AsyncRuntime, Runtime

logger = logging.getLogger(__name__)

# Disabling multi-attempt retries for clarity. Adjust as needed.
RETRY_POLICY = dict(stop=stop_after_attempt(1))
retries = Retrying(**RETRY_POLICY)
async_retries = AsyncRetrying(**RETRY_POLICY)


class OpenRouterClientMixin(BaseModel):
    """
    Simple example: we treat the 'OpenRouter' client
    as a standard OpenAI client, but point base_url
    to 'https://openrouter.ai/api/v1'.
    """

    instructor_mode: str = "tool_call"
    provider: Optional[str] = None
    api_key: Optional[str] = None

    # By default, we can override to the openrouter endpoint:
    base_url: str = "https://openrouter.ai/api/v1"

    model: str = "anthropic/claude-3.5-haiku"
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.0
    seed: Optional[int] = 47

    model_config = ConfigDict(extra="allow")

    @property
    def client(self):
        """
        We pass our base_url to the standard OpenAI client,
        which will talk to openrouter.ai via the completely
        OpenAI-compatible endpoint. Then we wrap it
        in 'instructor' for usage.
        """
        # You can adjust which openai client you use
        # (sync vs. async) based on your runtime class.
        openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return instructor.from_openai(
            openai_client, mode=instructor.Mode(self.instructor_mode)
        )

    def _check_client(self):
        """
        Send a small test prompt to confirm connectivity.
        """
        resp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from OpenRouter!"}],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            response_model=None,
            max_retries=retries,
        )
        return resp

    def get_canonical_model_provider_string(self, model: str) -> str:
        """
        Optionally look up the model / provider via a known map.
        """
        try:
            return match_model_provider_string(model)
        except NoModelsFoundError:
            logger.warning(f"Model {model} not found in any known map.")
            return model
        except Exception:
            logger.exception(f"Failed to get canonical model string for {model}")
            return model


class OpenRouterChatRuntime(OpenRouterClientMixin, Runtime):
    """
    Sync usage of the 'openrouter' approach, purely as
    a base_url override of openai + instructor.
    """

    def init_runtime(self) -> "Runtime":
        try:
            self._check_client()
        except Exception as e:
            logger.exception(
                f'Failed to check availability of "{self.model}": {e}\n{traceback.format_exc()}'
            )
            raise ValueError(f'Failed to check availability of "{self.model}": {e}')
        return self

    def record_to_record(
        self,
        record: Dict[str, Any],
        input_template: str,
        instructions_template: str,
        response_model: Type[BaseModel],
        output_template: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = False,
    ) -> Dict[str, Any]:
        """
        Use the instructor library to handle templated prompts and responses.
        """
        if not response_model:
            raise ValueError("You must pass a `response_model` to record_to_record().")

        return run_instructor_with_payload(
            client=self.client,
            payload=record,
            user_prompt_template=input_template,
            response_model=response_model,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            retries=retries,
            extra_fields=extra_fields or {},
            instructions_first=instructions_first,
            instructions_template=instructions_template,
        )

    def get_cost_estimate(
        self,
        prompt: str,
        substitutions: List[Dict],
        output_fields: Optional[List[str]],
        provider: str,
    ) -> CostEstimate:
        """
        Example cost estimator, paralleling the logic used in the LiteLLM runtimes.
        Update with custom or real pricing if you have a pricing schedule from OpenRouter.
        """
        try:
            user_prompts = [
                partial_str_format(prompt, **substitution)
                for substitution in substitutions
            ]
            cumulative_prompt_cost = 0
            cumulative_completion_cost = 0
            cumulative_total_cost = 0
            model = self.get_canonical_model_provider_string(self.model)
            for user_prompt in user_prompts:
                prompt_cost, completion_cost, total_cost = _estimate_cost(
                    user_prompt=user_prompt,
                    model=model,
                    output_fields=output_fields,
                    provider=provider,
                )
                cumulative_prompt_cost += prompt_cost
                cumulative_completion_cost += completion_cost
                cumulative_total_cost += total_cost
            return CostEstimate(
                prompt_cost_usd=cumulative_prompt_cost,
                completion_cost_usd=cumulative_completion_cost,
                total_cost_usd=cumulative_total_cost,
            )
        except Exception as e:
            logger.error("Failed to estimate cost: %s", e)
            return CostEstimate(
                is_error=True,
                error_type=type(e).__name__,
                error_message=str(e),
            )

    def get_llm_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Simple single-call usage for the LLM, mirroring the 'get_llm_response'
        style function in _litellm.py, but adapted to use OpenRouter.
        """
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            # We don't specify a response model here, if you just want raw text
            response_model=None,
            max_retries=retries,
        )
        # Return the content of the first choice
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content
        return ""


class AsyncOpenRouterChatRuntime(OpenRouterClientMixin, AsyncRuntime):
    """
    Async usage of the 'openrouter' approach, purely as
    an async base_url override of openai + instructor.
    """

    concurrency: Optional[int] = None

    @property
    def client(self):
        """
        Example of using the async OpenAI client with a custom base_url.
        """
        openai_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return instructor.from_openai(
            openai_client, mode=instructor.Mode(self.instructor_mode)
        )

    @field_validator("concurrency", mode="before")
    def check_concurrency(cls, value) -> int:
        value = value or -1
        if value < 1:
            raise NotImplementedError(
                "You must explicitly specify the number of concurrent clients. "
                "For example, set `AsyncOpenRouterChatRuntime(concurrency=10, ...)`."
            )
        return value

    async def init_runtime(self) -> "AsyncRuntime":
        """
        For the async usage, you might choose to skip the connectivity test or
        do it synchronously (like re-instantiating a sync client). Example:
        """
        try:
            # Reuse the sync check from the parent
            # by building a temporary sync client
            temp = OpenAI(api_key=self.api_key, base_url=self.base_url)
            sync_cli = instructor.from_openai(temp, mode=instructor.Mode(self.instructor_mode))

            sync_cli.chat.completions.create(
                messages=[{"role": "user", "content": "Hello from OpenRouter-Async!"}],
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
                response_model=None,
                max_retries=retries,
            )
        except Exception as e:
            logger.exception(
                f'Async check for model "{self.model}" failed: {e}\n{traceback.format_exc()}'
            )
            raise ValueError(f'Async check for model "{self.model}" failed: {e}')
        return self

    async def batch_to_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        instructions_template: str,
        response_model: Type[BaseModel],
        output_template: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> InternalDataFrame:
        if not response_model:
            raise ValueError("Must specify `response_model`.")

        payloads = batch.to_dict(orient="records")

        df_data = await arun_instructor_with_payloads(
            client=self.client,
            payloads=payloads,
            user_prompt_template=input_template,
            response_model=response_model,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            retries=async_retries,
            extra_fields=extra_fields or {},
            instructions_first=instructions_first,
            instructions_template=instructions_template,
        )
        return InternalDataFrame(df_data).set_index(batch.index)

    async def record_to_record(
        self,
        record: Dict[str, Any],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        input_df = InternalDataFrame([record])

        output_df = await self.batch_to_batch(
            batch=input_df,
            input_template=input_template,
            instructions_template=instructions_template,
            response_model=response_model,
            output_template=output_template,
            extra_fields=extra_fields,
            field_schema=field_schema,
            instructions_first=instructions_first,
        )
        return output_df.iloc[0].to_dict()

    def get_llm_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Simple single-call usage for the LLM, mirroring the 'get_llm_response'
        style function in _litellm.py, but adapted to use OpenRouter.
        """
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            # We don't specify a response model here, if you just want raw text
            response_model=None,
            max_retries=retries,
        )
        # Return the content of the first choice
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content
        return ""


class AsyncOpenRouterVisionRuntime(AsyncOpenRouterChatRuntime):
    """
    Example for an OpenRouter-based vision model runtime, paralleling the
    AsyncLiteLLMVisionRuntime from _litellm.py.
    """

    def init_runtime(self) -> "Runtime":
        super().init_runtime()
        # Example check: if openrouter supports a "supports_vision" function
        # you could do something like this:
        if not hasattr(openrouter, "supports_vision"):
            logger.warning("OpenRouter doesn't appear to have a 'supports_vision' check.")
        else:
            if not openrouter.supports_vision(self.model):
                raise ValueError(f"Model {self.model} does not support vision")
        return self

    async def batch_to_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        instructions_template: str,
        response_model: Type[BaseModel],
        output_template: Optional[str] = None,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
        input_field_types: Optional[Dict[str, MessageChunkType]] = None,
    ) -> InternalDataFrame:
        """
        Async vision batch request. Optionally handle multi-image contexts
        by splitting into chunks if necessary, etc.
        """
        if not response_model:
            raise ValueError("You must explicitly specify the `response_model` in runtime.")

        extra_fields = extra_fields or {}
        input_field_types = input_field_types or {}
        records = batch.to_dict(orient="records")

        # Example approach: if you have large image lists, you can split them up:
        ensure_messages_fit_in_context_window = any(
            input_field_types.get(field) == MessageChunkType.IMAGE_URLS
            for field in input_field_types
        )

        df_data = await arun_instructor_with_payloads(
            client=self.client,
            payloads=records,
            user_prompt_template=input_template,
            response_model=response_model,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            retries=async_retries,
            split_into_chunks=True,
            input_field_types=input_field_types,
            instructions_first=instructions_first,
            instructions_template=instructions_template,
            extra_fields=extra_fields,
            ensure_messages_fit_in_context_window=ensure_messages_fit_in_context_window,
        )

        output_df = InternalDataFrame(df_data)
        return output_df.set_index(batch.index)