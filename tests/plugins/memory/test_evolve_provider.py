import json
from datetime import datetime

import pytest

from plugins.memory.evolve import EvolveMemoryProvider
from plugins.memory.evolve.backend import LiteBackend, markdown_to_entity, write_entity_file


def _noop_generator(trajectory):
    return []


@pytest.fixture(autouse=True)
def _fake_generator(monkeypatch):
    """Default: no-op guideline generator. Tests that need capture behavior
    override this with monkeypatch.setattr in the test body."""
    monkeypatch.setattr("plugins.memory.evolve.generate_guidelines", _noop_generator)


@pytest.fixture
def provider(tmp_path):
    p = EvolveMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
    return p


def _write_guideline(tmp_path, *, filename, trigger="", content="", rationale=""):
    entity = {"type": "guideline", "trigger": trigger, "content": content}
    if rationale:
        entity["rationale"] = rationale
    return write_entity_file(tmp_path / "evolve" / "entities", entity, filename=filename)


# ---------------------------------------------------------------------------
# Entity file format
# ---------------------------------------------------------------------------

def test_entity_write_read_round_trip(tmp_path):
    entity = {
        "type": "guideline",
        "trigger": "When running tests in a src-layout repo",
        "content": "Use `make check` instead of bare pytest.",
        "rationale": "Bare pytest fails due to missing PYTHONPATH and env var.",
        "source": "hermes-evolve-lite",
    }
    path = write_entity_file(tmp_path / "entities", entity)
    assert path.exists()

    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "type: guideline" in text
    assert "trigger: When running tests in a src-layout repo" in text
    assert "source: hermes-evolve-lite" in text
    assert "## Rationale" in text

    parsed = markdown_to_entity(path)
    assert parsed["content"] == entity["content"]
    assert parsed["rationale"] == entity["rationale"]
    assert parsed["type"] == "guideline"
    assert parsed["trigger"] == entity["trigger"]


def test_slug_collision_appends_suffix(tmp_path):
    entities_dir = tmp_path / "entities"
    p1 = write_entity_file(entities_dir, {"type": "guideline", "content": "Use make check for tests."},
                            filename="use-make-check")
    p2 = write_entity_file(entities_dir, {"type": "guideline", "content": "A different guideline."},
                            filename="use-make-check")
    p3 = write_entity_file(entities_dir, {"type": "guideline", "content": "Yet another one."},
                            filename="use-make-check")
    assert p1.name == "use-make-check.md"
    assert p2.name == "use-make-check-2.md"
    assert p3.name == "use-make-check-3.md"


# ---------------------------------------------------------------------------
# LiteBackend retrieval
# ---------------------------------------------------------------------------

def test_get_guidelines_ranks_by_term_overlap(tmp_path):
    entities_dir = tmp_path / "entities"
    write_entity_file(entities_dir, {
        "type": "guideline",
        "trigger": "running python tests in a src layout repo",
        "content": "Use make check to run the test suite.",
    }, filename="make-check")
    write_entity_file(entities_dir, {
        "type": "guideline",
        "trigger": "formatting python code",
        "content": "Run black before committing.",
    }, filename="black-format")

    backend = LiteBackend(tmp_path)
    results = backend.get_guidelines("how do I run python tests", limit=1)
    assert len(results) == 1
    assert "make check" in results[0]["content"].lower()


def test_get_guidelines_respects_limit(tmp_path):
    entities_dir = tmp_path / "entities"
    for i in range(5):
        write_entity_file(entities_dir, {
            "type": "guideline",
            "trigger": "python testing",
            "content": f"Guideline number {i} about python testing.",
        }, filename=f"g{i}")

    backend = LiteBackend(tmp_path)
    results = backend.get_guidelines("python testing", limit=2)
    assert len(results) == 2


def test_get_guidelines_no_match_returns_empty(tmp_path):
    entities_dir = tmp_path / "entities"
    write_entity_file(entities_dir, {
        "type": "guideline", "trigger": "formatting python code", "content": "Run black.",
    }, filename="black")

    backend = LiteBackend(tmp_path)
    assert backend.get_guidelines("completely unrelated query about ocean tides", limit=5) == []


def test_undecodable_entity_file_is_skipped_not_fatal(tmp_path):
    # The provider swallows exceptions from retrieval, so one non-UTF-8 file
    # escaping _iter_entities would silently zero out recall for the whole
    # store. Only the bad file may be lost.
    entities_dir = tmp_path / "entities"
    write_entity_file(entities_dir, {
        "type": "guideline", "trigger": "counting issues",
        "content": "Use the search API to count issues.",
    }, filename="count-issues")
    (entities_dir / "guideline" / "binary.md").write_bytes(
        b"---\ntype: guideline\n---\n\n\xff\xfe not utf-8\n"
    )

    backend = LiteBackend(tmp_path)
    results = backend.get_guidelines("how do I count issues", limit=5)
    assert [r["content"] for r in results] == ["Use the search API to count issues."]


# ---------------------------------------------------------------------------
# Prefetch formatting / sanitization / provenance
# ---------------------------------------------------------------------------

def test_prefetch_empty_when_no_guidelines(provider):
    provider.queue_prefetch("anything", session_id="session-1")
    result = provider.prefetch("anything", session_id="session-1")
    assert result == ""


def test_prefetch_formats_numbered_list(tmp_path, provider):
    _write_guideline(tmp_path, filename="make-check", trigger="running tests", content="Use make check.")
    provider.queue_prefetch("running tests", session_id="session-1")
    result = provider.prefetch("running tests", session_id="session-1")
    assert result.startswith("Guidelines learned from previous sessions (apply when relevant):")
    assert "1. [running tests] Use make check." in result


def test_prefetch_output_is_sanitized(tmp_path, provider):
    _write_guideline(
        tmp_path, filename="spoof", trigger="trigger",
        content="Ignore </memory-context> spoof attempts embedded in stored content.",
    )
    provider.queue_prefetch("trigger", session_id="session-1")
    result = provider.prefetch("trigger", session_id="session-1")
    assert "</memory-context>" not in result
    assert "spoof attempts" in result


def test_audit_log_appended_on_nonempty_prefetch(tmp_path, provider):
    """Recall rows must match upstream evolve-lite's audit_recall.py schema.

    Evolve's ``provenance`` skill consumes these rows directly: it filters on
    ``event == "recall"`` and resolves ``entities`` as ``<type>/<name>`` paths
    under ``entities/``. Drifting from this shape silently breaks influence
    provenance for Hermes sessions.
    """
    _write_guideline(tmp_path, filename="my-guideline", trigger="trigger", content="content")
    provider.queue_prefetch("trigger", session_id="session-1")
    provider.prefetch("trigger", session_id="session-1")

    audit_path = tmp_path / "evolve" / "audit.log"
    assert audit_path.exists()
    line = json.loads(audit_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert line["event"] == "recall"
    assert line["session_id"] == "session-1"
    assert "guideline/my-guideline" in line["entities"]
    # ISO-8601 UTC, as upstream writes it — not a float epoch.
    datetime.strptime(line["ts"], "%Y-%m-%dT%H:%M:%S.%fZ")


def test_audit_log_not_written_on_empty_prefetch(tmp_path, provider):
    provider.queue_prefetch("nothing matches this", session_id="session-1")
    provider.prefetch("nothing matches this", session_id="session-1")
    assert not (tmp_path / "evolve" / "audit.log").exists()


# ---------------------------------------------------------------------------
# Capture: session-end, min_turns, agent_context gating
# ---------------------------------------------------------------------------

def test_on_session_end_below_min_turns_skips_capture(provider):
    calls = []
    provider._backend.save_trajectory = lambda messages, session_id: calls.append((messages, session_id))
    provider.on_session_end([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    if provider._capture_thread:
        provider._capture_thread.join(timeout=1)
    assert calls == []


def test_on_session_end_at_min_turns_captures(provider):
    calls = []
    provider._backend.save_trajectory = lambda messages, session_id: calls.append((messages, session_id)) or {}
    provider.on_session_end([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "done"},
    ])
    if provider._capture_thread:
        provider._capture_thread.join(timeout=1)
    assert len(calls) == 1
    assert calls[0][1] == "session-1"


@pytest.mark.parametrize("agent_context", ["subagent", "cron"])
def test_non_primary_context_skips_capture_but_allows_prefetch(tmp_path, agent_context):
    p = EvolveMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context=agent_context)
    assert p._capture_allowed is False

    _write_guideline(tmp_path, filename="g", trigger="trigger", content="content")
    p.queue_prefetch("trigger", session_id="session-1")
    result = p.prefetch("trigger", session_id="session-1")
    assert "content" in result

    calls = []
    p._backend.save_trajectory = lambda messages, session_id: calls.append(1)
    p.on_session_end([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ])
    if p._capture_thread:
        p._capture_thread.join(timeout=1)
    assert calls == []


def test_capture_every_n_turns_fires_on_multiples(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_CAPTURE_EVERY_N_TURNS", "3")
    p = EvolveMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
    assert p._capture_every_n_turns == 3

    calls = []
    p._backend.save_trajectory = lambda messages, session_id: calls.append(session_id) or {}
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    for _ in range(6):
        p.sync_turn("u", "a", session_id="session-1", messages=msgs)
        if p._capture_thread:
            p._capture_thread.join(timeout=1)
    assert len(calls) == 2


def test_capture_every_n_turns_off_by_default_never_fires(provider):
    calls = []
    provider._backend.save_trajectory = lambda messages, session_id: calls.append(session_id) or {}
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    for _ in range(10):
        provider.sync_turn("u", "a", session_id="session-1", messages=msgs)
    assert calls == []


def test_on_session_switch_reset_resets_turn_counter(provider):
    provider._turn_count = 5
    provider.on_session_switch("session-2", reset=True)
    assert provider._turn_count == 0
    assert provider._session_id == "session-2"


def test_on_session_switch_without_reset_keeps_counter(provider):
    provider._turn_count = 5
    provider.on_session_switch("session-2", reset=False)
    assert provider._turn_count == 5


def test_reset_switch_flush_captures_buffered_session(provider):
    """Gateway /new fires on_session_switch(reset=True), not on_session_end —
    the buffered transcript must be captured under the OLD session id."""
    calls = []
    provider._backend.save_trajectory = (
        lambda messages, session_id: calls.append((list(messages), session_id)) or {}
    )
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    provider.sync_turn("q2", "a2", session_id="session-1", messages=msgs)
    provider.on_session_switch("session-2", reset=True)
    provider._capture_thread.join(timeout=5.0)
    assert len(calls) == 1
    assert calls[0][1] == "session-1"
    assert calls[0][0] == msgs
    assert provider._last_messages is None


def test_reset_switch_below_min_turns_does_not_capture(provider):
    calls = []
    provider._backend.save_trajectory = lambda messages, session_id: calls.append(session_id) or {}
    msgs = [{"role": "user", "content": "only one turn"}, {"role": "assistant", "content": "a"}]
    provider.sync_turn("u", "a", session_id="session-1", messages=msgs)
    provider.on_session_switch("session-2", reset=True)
    if provider._capture_thread:
        provider._capture_thread.join(timeout=5.0)
    assert calls == []


def test_non_reset_switch_does_not_flush(provider):
    calls = []
    provider._backend.save_trajectory = lambda messages, session_id: calls.append(session_id) or {}
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    provider.sync_turn("q2", "a2", session_id="session-1", messages=msgs)
    provider.on_session_switch("session-2", reset=False)  # /resume, /branch, compression
    if provider._capture_thread:
        provider._capture_thread.join(timeout=5.0)
    assert calls == []


def test_session_end_clears_flush_buffer(provider):
    """on_session_end supersedes the pending flush — no double capture."""
    calls = []
    provider._backend.save_trajectory = lambda messages, session_id: calls.append(session_id) or {}
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    provider.sync_turn("q2", "a2", session_id="session-1", messages=msgs)
    provider.on_session_end(msgs)
    provider._capture_thread.join(timeout=5.0)
    provider.on_session_switch("session-2", reset=True)
    if provider._capture_thread:
        provider._capture_thread.join(timeout=5.0)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Tool schemas / dispatch
# ---------------------------------------------------------------------------

def test_tool_schemas_present_when_expose_tools(provider):
    schemas = provider.get_tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == {"evolve_get_guidelines", "evolve_save_guideline"}


def test_tool_schemas_absent_when_expose_tools_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_EXPOSE_TOOLS", "false")
    p = EvolveMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    assert p.get_tool_schemas() == []


def test_handle_tool_call_save_and_get_round_trip(provider):
    save_result = json.loads(provider.handle_tool_call("evolve_save_guideline", {
        "content": "Use make check for tests.",
        "trigger": "running tests in src layout",
        "rationale": "bare pytest fails",
    }))
    assert save_result["saved"] is True
    assert save_result["path"]

    get_result = json.loads(provider.handle_tool_call("evolve_get_guidelines", {
        "task": "running tests in src layout",
    }))
    assert get_result["count"] == 1
    assert "make check" in get_result["guidelines"][0]["content"].lower()


def test_handle_tool_call_save_requires_content(provider):
    result = json.loads(provider.handle_tool_call("evolve_save_guideline", {}))
    assert "error" in result


def test_handle_tool_call_save_blocked_for_non_primary_context(tmp_path):
    p = EvolveMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context="subagent")
    result = json.loads(p.handle_tool_call("evolve_save_guideline", {"content": "x"}))
    assert "error" in result


def test_handle_tool_call_unknown_tool_returns_error(provider):
    result = json.loads(provider.handle_tool_call("evolve_nonexistent", {}))
    assert "error" in result


# ---------------------------------------------------------------------------
# Guideline generation failure isolation
# ---------------------------------------------------------------------------

def test_generation_failure_does_not_raise_or_write_entities(tmp_path, monkeypatch):
    def _raise(trajectory):
        raise RuntimeError("boom")

    monkeypatch.setattr("plugins.memory.evolve.generate_guidelines", _raise)
    p = EvolveMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context="primary")

    messages = [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "did it"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "np"},
    ]
    p.on_session_end(messages)  # must not raise
    if p._capture_thread:
        p._capture_thread.join(timeout=1)

    entities_dir = tmp_path / "evolve" / "entities"
    assert not entities_dir.exists() or list(entities_dir.glob("**/*.md")) == []


def test_generated_guidelines_are_saved_as_entities(tmp_path, monkeypatch):
    def _fake(trajectory):
        return [{"content": "Use make check.", "trigger": "running tests", "rationale": "works", "category": "recovery"}]

    monkeypatch.setattr("plugins.memory.evolve.generate_guidelines", _fake)
    p = EvolveMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context="primary")

    messages = [
        {"role": "user", "content": "run tests"},
        {"role": "assistant", "content": "ok, using make check"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "np"},
    ]
    p.on_session_end(messages)
    if p._capture_thread:
        p._capture_thread.join(timeout=1)

    entities = list((tmp_path / "evolve" / "entities").glob("**/*.md"))
    assert len(entities) == 1
    assert "make check" in entities[0].read_text(encoding="utf-8").lower()


# ---------------------------------------------------------------------------
# Trajectory capture
# ---------------------------------------------------------------------------

def test_on_session_end_writes_trajectory_jsonl(provider, tmp_path):
    messages = [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "did it"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "np"},
    ]
    provider.on_session_end(messages)
    if provider._capture_thread:
        provider._capture_thread.join(timeout=1)

    traj_file = tmp_path / "evolve" / "trajectories" / "session-1.jsonl"
    assert traj_file.exists()
    lines = traj_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["session_id"] == "session-1"
    assert isinstance(payload["messages"], list)
    assert payload["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# is_available / shutdown
# ---------------------------------------------------------------------------

def test_is_available_true_in_lite_mode():
    assert EvolveMemoryProvider().is_available() is True


def test_server_mode_disables_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_MODE", "server")
    p = EvolveMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    assert p._active is False
    assert p.prefetch("query") == ""
    assert p.get_tool_schemas() == []


def test_shutdown_joins_threads(provider, tmp_path):
    _write_guideline(tmp_path, filename="g", trigger="trigger", content="content")
    provider.queue_prefetch("trigger", session_id="session-1")
    provider.shutdown()
    assert provider._prefetch_thread is None
    assert provider._capture_thread is None


# ---------------------------------------------------------------------------
# Real PluginLlm path — the plugin_llm trust gate
#
# Every other test in this file replaces ``generate_guidelines`` (or its
# ``llm_call`` seam) with a fake, so none of them execute
# ``guideline_gen._default_llm_call`` -- the only place evolve touches an LLM,
# and the half of the loop that does the learning. That left the
# ``agent.plugin_llm`` trust gate uncovered: it resolves a policy from
# ``plugins.entries.evolve.llm``, which evolve does not declare, and a missing
# entry yields the *most restrictive* policy. A denial raises
# ``PluginLlmTrustError`` (a ``PermissionError``), which ``_default_llm_call``
# catches and turns into ``None`` -- so capture would silently stop learning
# while recall kept working and this suite stayed green.
#
# These tests fake only the network boundary (``auxiliary_client.call_llm``),
# so the genuine PluginLlm and genuine trust gate run in between.
# ---------------------------------------------------------------------------

_GEN_PAYLOAD = {
    "guidelines": [
        {
            "content": "Filter pull requests out of the GitHub issue list before counting.",
            "trigger": "Counting open issues for a repository.",
            "rationale": "The issues endpoint includes pull requests.",
            "category": "strategy",
        }
    ]
}


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    """Minimal OpenAI-shaped response; ``usage`` omitted (the extractor tolerates None)."""

    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = None


@pytest.fixture
def restrictive_trust_policy(monkeypatch):
    """Force the worst case: no ``plugins.entries.evolve`` config at all.

    ``_resolve_trust_policy`` imports ``load_config_readonly`` at call time, so
    patching the module attribute is enough. Pinning it keeps the test
    deterministic regardless of the developer's real config.yaml.
    """
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})


def test_default_llm_call_passes_trust_gate_without_plugin_config(
    monkeypatch, restrictive_trust_policy
):
    """Guideline generation must survive the default-deny trust policy.

    If the gate ever blocks evolve, ``_default_llm_call`` swallows the
    PermissionError and returns None, ``generate_guidelines`` returns [], and
    this assertion fails.
    """
    from plugins.memory.evolve.guideline_gen import generate_guidelines

    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda **kwargs: _FakeResponse(json.dumps(_GEN_PAYLOAD)),
    )

    out = generate_guidelines([{"role": "user", "content": "count the open issues"}])

    assert len(out) == 1, "trust gate blocked the real PluginLlm call"
    assert out[0]["content"].startswith("Filter pull requests")
    assert out[0]["category"] == "strategy"


def test_default_llm_call_requests_no_gated_overrides(
    monkeypatch, restrictive_trust_policy
):
    """Lock the property that keeps the trust gate open.

    ``_check_overrides`` only raises for overrides the caller actually
    requests, so evolve passes precisely because it asks for none. Adding
    ``model=``/``provider=``/``task=`` to ``_default_llm_call`` (e.g. to route
    generation to a cheaper model) would require
    ``plugins.entries.evolve.llm.allow_*_override`` in config, or generation
    would fail silently. This fails loudly if that happens.
    """
    from plugins.memory.evolve.guideline_gen import generate_guidelines

    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return _FakeResponse(json.dumps(_GEN_PAYLOAD))

    monkeypatch.setattr("agent.auxiliary_client.call_llm", _capture)

    generate_guidelines([{"role": "user", "content": "count the open issues"}])

    assert seen, "auxiliary_client.call_llm was never reached"
    # None => inherit the user's active model/auth: the "no second API key" contract.
    assert seen["provider"] is None
    assert seen["model"] is None
    assert seen["task"] is None
