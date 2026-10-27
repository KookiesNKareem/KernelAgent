"""Meta's models: Muse Spark over the Meta Model API, Muse Spark Open over Baseten."""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .base import LLMResponse
from .budget import BudgetExceeded, reserve, settle
from .openai_base import OpenAICompatibleProvider

load_dotenv()

REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")

# USD per million tokens (input, cached input, output); reasoning tokens bill as output.
META_PRICES = {
    "muse-wonder": (4.00, 0.20, 20.00),
    "muse-spark-1.3-contributor": (0.10, 0.002, 0.20),
    "muse-spark-1.2-contributor": (0.10, 0.002, 0.20),
    "muse-spark": (1.25, 0.15, 4.25),
    "meta-models/muse-spark-open": (1.25, 0.15, 4.25),
}
LEDGER = Path(os.environ.get("META_SPEND_LEDGER", Path.home() / ".kernelagent_meta_spend.json"))
BASETEN_LEDGER = Path.home() / ".kernelagent_baseten_spend.json"


def spent(ledger: Path = LEDGER) -> float:
    try:
        return json.loads(ledger.read_text())["usd"]
    except (FileNotFoundError, KeyError, ValueError):
        return 0.0




class _MuseProvider(OpenAICompatibleProvider):
    """Refuses to call once its ledger plus the call's worst case would pass the budget."""

    default_reasoning_effort: str | None = None
    ledger: Path = LEDGER
    budget: float = 120.0

    def _initialize_client(self):
        super()._initialize_client()
        if self.client is not None:
            self.client = self.client.with_options(max_retries=0)

    def get_response(self, model_name: str, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        # One UTF-8 byte per token plus framing is a conservative reservation, not a tokenizer estimate.
        price_in, _, price_out = next(p for k, p in META_PRICES.items() if model_name.startswith(k))
        prompt_bound = sum(len((m.get("content") or "").encode()) + 128 for m in messages)
        worst = (prompt_bound * price_in + min(kwargs.get("max_tokens", 16000),
                                             self.get_max_tokens_limit(model_name)) * price_out) / 1e6
        remaining = kwargs.pop("call_budget_usd", self.budget)
        if worst > remaining:
            raise BudgetExceeded(f"Next call reservation ${worst:.2f} exceeds run remainder ${remaining:.2f}")
        token = reserve(self.ledger, worst, self.budget)
        # Leave the reservation in place on ambiguous API failure: a timed-out request may have billed.
        resp = super().get_response(model_name, messages, **kwargs)
        usage = resp.usage
        actual = None
        if usage and "prompt_tokens" in usage and "completion_tokens" in usage:
            pi, pc, po = next(p for k, p in META_PRICES.items() if model_name.startswith(k))
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            actual = ((usage["prompt_tokens"] - cached) * pi + cached * pc + usage["completion_tokens"] * po) / 1e6
        settle(self.ledger, token, actual)
        return resp

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
        return 128000

    def supports_multiple_completions(self) -> bool:
        # The Meta API rejects n > 1, so the base class samples one call per completion.
        return False


class MetaProvider(_MuseProvider):
    default_reasoning_effort = "high"

    def __init__(self):
        super().__init__(api_key_env="MODEL_API_KEY", base_url="https://api.meta.ai/v1")
        self.ledger = Path(os.environ.get("MUSE_EXPERIMENT_LEDGER", str(LEDGER)))
        self.budget = float(os.environ.get("MUSE_EXPERIMENT_BUDGET_USD", os.environ.get("META_BUDGET_USD", 120)))

    @property
    def name(self) -> str:
        return "meta"


class BasetenProvider(_MuseProvider):
    ledger = BASETEN_LEDGER

    def __init__(self):
        super().__init__(api_key_env="BASETEN_API_KEY", base_url="https://inference.baseten.co/v1")
        self.ledger = Path(os.environ.get("MUSE_EXPERIMENT_LEDGER", str(BASETEN_LEDGER)))
        self.budget = float(os.environ.get("MUSE_EXPERIMENT_BUDGET_USD", os.environ.get("BASETEN_BUDGET_USD", 40)))

    @property
    def name(self) -> str:
        return "baseten"
