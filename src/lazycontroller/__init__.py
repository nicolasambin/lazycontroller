"""Drop-in ``openai`` clients: a fast decision model answers the structured calls it is sure of."""

from .client import AsyncOpenAI, OpenAI
from .lazy import FastConfig, Route
from .shadow import ShadowReport

__all__ = ["AsyncOpenAI", "FastConfig", "OpenAI", "Route", "ShadowReport"]

__version__ = "0.1.0"
