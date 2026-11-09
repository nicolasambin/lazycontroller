"""Drop-in ``openai`` clients: a fast decision model answers the structured calls it is sure of."""

from .client import AsyncOpenAI, OpenAI
from .lazy import LazyConfig, Route
from .shadow import ShadowReport

__all__ = ["AsyncOpenAI", "LazyConfig", "OpenAI", "Route", "ShadowReport"]

__version__ = "0.1.0"
