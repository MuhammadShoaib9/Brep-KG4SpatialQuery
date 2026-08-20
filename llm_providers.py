# -*- coding: utf-8 -*-
"""
Central registry of LLM providers used by
the Understanding Agent and Response Agent.

Purpose: keep all provider-specific details
(API endpoint, API key, model name) in ONE
place, external to the agent classes and the
evaluation script. Adding a new provider only
requires adding one entry to PROVIDERS below -
no changes needed anywhere else in the
codebase.

All four providers below expose an
OpenAI-compatible chat completions endpoint,
so the same OpenAI Python client works for
all of them via the base_url parameter:
  - OpenAI    : native endpoint (base_url=None)
  - Groq      : OpenAI-compatible
  - Mistral   : OpenAI-compatible
  - Gemini    : OpenAI-compatible endpoint at
                .../v1beta/openai/
"""

import os


PROVIDERS = {

    "gpt-4": {
        "base_url"   : None,  # OpenAI default
        "api_key_env": "OPENAI_API_KEY",
        "model_name" : "gpt-4",
    },

    "llama-3.3-70b": {
        "base_url"   : "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_name" : "llama-3.3-70b-versatile",
    },

    "mistral-small": {
        "base_url"   : "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "model_name" : "mistral-small-latest",
    },

    "qwen2.5-72b": {
        "base_url"   : "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_name" : "qwen/qwen-2.5-72b-instruct",
    },

    "gemini-2.5-flash": {
        "base_url"   :
            "https://generativelanguage."
            "googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model_name" : "gemini-3-flash-preview",
    },

}


def get_provider_config(provider_key):
    """
    Look up a provider by its registry key
    (e.g. "llama-3.3-70b") and return a dict
    ready to be unpacked directly into
    UnderstandingAgent / ResponseAgent:

        provider = get_provider_config(
            "llama-3.3-70b")
        agent = UnderstandingAgent(**provider)

    Returns:
        {
            "api_key" : <resolved from .env>,
            "base_url": <endpoint URL or None>,
            "model"   : <model name string>
        }

    Raises:
        ValueError if provider_key is not
        registered in PROVIDERS.
        RuntimeError if the required API key
        environment variable is not set (empty
        or missing), so a missing key fails
        loudly at startup rather than causing
        a confusing authentication error deep
        inside an evaluation run.
    """
    cfg = PROVIDERS.get(provider_key)
    if cfg is None:
        available = ", ".join(
            sorted(PROVIDERS.keys()))
        raise ValueError(
            f"Unknown provider '{provider_key}'. "
            f"Available providers: {available}")

    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Environment variable "
            f"'{cfg['api_key_env']}' is not "
            f"set. Add it to your .env file "
            f"before running provider "
            f"'{provider_key}'.")

    return {
        "api_key" : api_key,
        "base_url": cfg["base_url"],
        "model"   : cfg["model_name"],
    }


def list_providers():
    """
    Return the list of all registered
    provider keys. Useful for the evaluation
    orchestrator to loop over every provider
    without hardcoding the list elsewhere.
    """
    return list(PROVIDERS.keys())
