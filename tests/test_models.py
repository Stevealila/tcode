"""The /model catalogue lookup and fuzzy resolver (src/models.py)."""

from __future__ import annotations

import httpx
import pytest

from src import models
from src.models import GROQ_STATIC_MODELS, list_groq_models, resolve_model_choice

_CAT = [
    "groq/compound",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
]


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("openai/gpt-oss-120b", "openai/gpt-oss-120b"),  # exact
        ("OPENAI/GPT-OSS-120B", "openai/gpt-oss-120b"),  # case-insensitive exact
        ("120b", "openai/gpt-oss-120b"),                 # unique last-segment
        ("mini", "groq/compound-mini"),                  # unique last-segment
        ("compound", "groq/compound"),                   # bare-name beats -mini sibling
        ("safeguard", "openai/gpt-oss-safeguard-20b"),   # unique substring
        ("gpt-oss-20b", "openai/gpt-oss-20b"),           # unique substring
        ("google:gemini-3-pro", "google:gemini-3-pro"),  # explicit provider -> literal
        ("some/new-model", "some/new-model"),            # explicit org/name -> literal
    ],
)
def test_resolve_unambiguous(arg, expected):
    chosen, _ = resolve_model_choice(arg, _CAT)
    assert chosen == expected


@pytest.mark.parametrize(
    "arg,n_candidates",
    [
        ("20b", 2),      # gpt-oss-20b and gpt-oss-safeguard-20b
        ("oss", 3),
        ("gpt", 3),
    ],
)
def test_resolve_ambiguous_returns_candidates(arg, n_candidates):
    chosen, candidates = resolve_model_choice(arg, _CAT)
    assert chosen is None
    assert len(candidates) == n_candidates


@pytest.mark.parametrize("arg", ["", "   ", "totally-made-up", "qwen"])
def test_resolve_no_match(arg):
    chosen, candidates = resolve_model_choice(arg, _CAT)
    assert chosen is None
    assert candidates == []


def _fake_resp(payload, status=200):
    request = httpx.Request("GET", models._GROQ_MODELS_URL)
    return httpx.Response(status, json=payload, request=request)


def test_list_groq_models_parses_and_filters(monkeypatch):
    payload = {
        "data": [
            {"id": "openai/gpt-oss-120b", "context_window": 131072},
            {"id": "openai/gpt-oss-20b", "context_window": 131072},
            {"id": "qwen/qwen3.8-27b", "context_window": 131072},      # banned
            {"id": "whisper-large-v3", "context_window": 448},         # non-chat
            {"id": "allam-2-7b", "context_window": 4096},              # too small
            {"id": "meta-llama/llama-prompt-guard-2-86m", "context_window": 512},
        ]
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_resp(payload))
    ids, err = list_groq_models("key")
    assert err is None
    assert ids == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def test_list_groq_models_network_error_falls_back(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "get", boom)
    ids, err = list_groq_models("key")
    assert ids == list(GROQ_STATIC_MODELS)
    assert "ConnectError" in err


def test_list_groq_models_bad_status_falls_back(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_resp({"error": "nope"}, status=500))
    ids, err = list_groq_models("key")
    assert ids == list(GROQ_STATIC_MODELS)
    assert err


def test_list_groq_models_weird_body_falls_back(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_resp({"not_data": []}))
    ids, err = list_groq_models("key")
    assert ids == list(GROQ_STATIC_MODELS)
    assert "shape" in err


def test_list_groq_models_empty_data_falls_back(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_resp({"data": []}))
    ids, _ = list_groq_models("key")
    assert ids == list(GROQ_STATIC_MODELS)
