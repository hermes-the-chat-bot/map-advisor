"""Tests for the LLM abstraction layer — mock client behavior and factory."""

from __future__ import annotations

import pytest

from map_advisor.errors import GuardrailError
from map_advisor.llm import LLMClient, LLMResponse, MockLLMClient, OpenAIClient, make_client


@pytest.mark.llm
class TestMockLLMClient:
    def test_scripted_responses_fifo(self) -> None:
        c = MockLLMClient(responses=["first", "second"])
        assert c.generate([{"role": "user", "content": "hi"}]).text == "first"
        assert c.generate([{"role": "user", "content": "hi"}]).text == "second"

    def test_pattern_response_matches_last_user(self) -> None:
        c = MockLLMClient(patterns={r"cpu": "Capacity draft here."})
        r = c.generate([{"role": "user", "content": "What's our CPU headroom?"}])
        assert r.text == "Capacity draft here."

    def test_pattern_response_no_match_uses_default(self) -> None:
        c = MockLLMClient(patterns={r"cpu": "x"})
        r = c.generate([{"role": "user", "content": "Tell me a joke."}])
        assert "[mock]" in r.text

    def test_callback_takes_precedence(self) -> None:
        c = MockLLMClient(patterns={r"cpu": "p"}, responses=["r"])

        def cb(messages, meta):
            content = messages[-1]["content"]
            if "special" in content:
                return "from callback"
            return None

        c.respond_with(cb)
        assert c.generate([{"role": "user", "content": "special please"}]).text == "from callback"
        # Non-special still hits scripted.
        assert c.generate([{"role": "user", "content": "other"}]).text == "r"

    def test_call_log_records_messages(self) -> None:
        c = MockLLMClient(responses=["x"])
        c.generate([{"role": "user", "content": "hello"}], temperature=0.5)
        assert len(c.call_log) == 1
        assert c.call_log[0]["messages"][0]["content"] == "hello"
        assert c.call_log[0]["meta"]["temperature"] == 0.5

    def test_response_is_typed(self) -> None:
        r = MockLLMClient(responses=["x"]).generate([{"role": "user", "content": "y"}])
        assert isinstance(r, LLMResponse)
        assert isinstance(r, type(r))

    def test_missing_user_message_falls_back(self) -> None:
        c = MockLLMClient(patterns={r"cpu": "should not match role-free text"})
        r = c.generate([{"role": "system", "content": "cpu"}])
        assert "[mock]" in r.text  # no user message → no pattern match


@pytest.mark.llm
class TestLLMBaseContract:
    def test_base_generate_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            LLMClient().generate([{"role": "user", "content": "x"}])

    def test_openai_client_lazy_import_error(self, monkeypatch) -> None:
        # Force ImportError of 'openai' inside generate().
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "openai" or name.startswith("openai."):
                raise ImportError("no openai")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        client = OpenAIClient(client=None)  # no injected client → import attempted
        with pytest.raises(ImportError) as exc:
            client.generate([{"role": "user", "content": "hi"}])
        assert "openai" in str(exc.value).lower()


@pytest.mark.llm
class TestMakeClientFactory:
    def test_mock_default(self) -> None:
        c = make_client("mock")
        assert isinstance(c, MockLLMClient)

    def test_openai(self) -> None:
        c = make_client("openai", model="gpt-x")
        assert isinstance(c, OpenAIClient)
        assert c.model == "gpt-x"

    def test_decode_json_patterns(self) -> None:
        c = make_client("mock", patterns=r'{"cpu": "yes"}')
        assert isinstance(c, MockLLMClient)
        r = c.generate([{"role": "user", "content": "cpu?"}])
        assert r.text == "yes"

    def test_decode_json_responses(self) -> None:
        c = make_client("mock", responses=r'["a", "b"]')
        assert c.generate([{"role": "user", "content": "x"}]).text == "a"
        assert c.generate([{"role": "user", "content": "x"}]).text == "b"

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(GuardrailError):
            make_client("bogus")
