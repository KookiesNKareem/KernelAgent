"""Meta's models: Muse Spark over the Meta Model API, Muse Spark Open over Baseten."""

from typing import Any

from dotenv import load_dotenv

from .base import LLMResponse
from .openai_base import OpenAICompatibleProvider

load_dotenv()

REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


class _MuseProvider(OpenAICompatibleProvider):
    default_reasoning_effort: str | None = None

    def _build_api_params(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": min(kwargs.get("max_tokens", 16000), self.get_max_tokens_limit(model_name)),
        }
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        effort = kwargs.get("reasoning_effort", self.default_reasoning_effort)
        if kwargs.get("high_reasoning_effort"):
            effort = "high"
        if effort:
            if effort not in REASONING_EFFORTS:
                raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
            params["reasoning_effort"] = effort
        if kwargs.get("tools"):
            params["tools"] = kwargs["tools"]
        return params

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        kwargs.pop("n", None)
        return [self.get_response(model_name, messages, **kwargs) for _ in range(n)]

    def get_max_tokens_limit(self, model_name: str) -> int:
        return 32000

    def supports_multiple_completions(self) -> bool:
        # The Meta API rejects n > 1, so the base class samples one call per completion.
        return False


class MetaProvider(_MuseProvider):
    default_reasoning_effort = "high"

    def __init__(self):
        super().__init__(api_key_env="MODEL_API_KEY", base_url="https://api.meta.ai/v1")

    @property
    def name(self) -> str:
        return "meta"


class BasetenProvider(_MuseProvider):
    def __init__(self):
        super().__init__(api_key_env="BASETEN_API_KEY", base_url="https://inference.baseten.co/v1")

    @property
    def name(self) -> str:
        return "baseten"
