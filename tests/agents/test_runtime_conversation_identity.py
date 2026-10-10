"""Stable conversation metadata must preserve per-run lifecycle correlation.

Room storage is tested in code_puppy_core_plugins. These core contract tests
also run against published plugins that do not yet consume conversation_id.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from code_puppy import callbacks, config
from code_puppy.agents import _runtime
from code_puppy.agents.agent_code_puppy import CodePuppyAgent
from code_puppy.tools.subagent_context import (
    reset_conversation_root_id,
    set_conversation_root_id,
)


@pytest.fixture
def memory_runtime(monkeypatch):
    """Exercise the real runtime/hooks with an offline model."""
    monkeypatch.setattr(config, "_CURRENT_AUTOSAVE_ID", "")
    monkeypatch.setattr(_runtime, "reset_pause_state_at_run_start", Mock())
    monkeypatch.setattr(_runtime, "inject_interrupted_subagent_notes", Mock())
    monkeypatch.setattr(_runtime, "on_user_prompt_submit", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "code_puppy.model_switching.resolve_run_model_selection", Mock()
    )
    monkeypatch.setattr(_runtime, "_should_prepend_system_prompt", lambda _, p: p)
    monkeypatch.setattr(_runtime, "sigint_fallback_cancels", lambda: True)
    monkeypatch.setattr(_runtime, "get_enable_streaming", lambda: False)
    monkeypatch.setattr(_runtime, "should_render_fallback", lambda *a, **kw: False)
    callbacks.clear_callbacks("agent_run_start")
    callbacks.clear_callbacks("agent_run_end")
    starts, ends, outcomes = [], [], []

    def record_start(agent_name, model_name, session_id):
        starts.append(session_id)

    def record_end(*args):
        ends.append((args[2], args[6]))
        outcomes.append((args[3], args[5]))

    callbacks.register_callback("agent_run_start", record_start)
    callbacks.register_callback("agent_run_end", record_end)

    def make_agent():
        agent = CodePuppyAgent()
        monkeypatch.setattr(agent, "get_model_name", lambda: "offline-model")

        async def run(prompt, **options):
            await asyncio.sleep(0)
            return SimpleNamespace(
                output="An offline assistant reply recorded by the real kennel. " * 3,
                all_messages=lambda: [],
            )

        agent._code_generation_agent = SimpleNamespace(run=run)
        return agent

    return SimpleNamespace(
        make_agent=make_agent, starts=starts, ends=ends, outcomes=outcomes
    )


async def test_multiple_turns_share_conversation_and_keep_distinct_run_ids(
    memory_runtime,
):
    env = memory_runtime
    agent = env.make_agent()
    for prompt in ("first", "second", "third"):
        await agent.run_with_mcp(prompt)

    assert len(env.ends) == 3
    assert len(set(env.starts)) == 3
    assert env.starts == [session_id for session_id, _ in env.ends]
    assert len({metadata["conversation_id"] for _, metadata in env.ends}) == 1
    assert all(metadata["conversation_id"] != run_id for run_id, metadata in env.ends)


async def test_new_auto_session_gets_a_new_conversation(memory_runtime):
    env = memory_runtime
    agent = env.make_agent()
    await agent.run_with_mcp("first session")
    config.rotate_session_name()
    agent.clear_message_history()
    await agent.run_with_mcp("new session")

    assert len(env.ends) == 2
    assert env.ends[0][1]["conversation_id"] != env.ends[1][1]["conversation_id"]


async def test_resumed_named_session_reuses_identity_with_a_new_agent(memory_runtime):
    env = memory_runtime
    config.pin_current_session_name("conversation-alpha")
    await env.make_agent().run_with_mcp("first session")
    config.pin_current_session_name("conversation-beta")
    await env.make_agent().run_with_mcp("other session")
    config.pin_current_session_name("conversation-alpha")
    await env.make_agent().run_with_mcp("resume first session")

    assert len(env.ends) == 3
    ids = [metadata["conversation_id"] for _, metadata in env.ends]
    assert ids[0] == ids[2] != ids[1]


async def test_concurrent_acp_sessions_keep_separate_identities(memory_runtime):
    env = memory_runtime

    async def conversation(root_id):
        token = set_conversation_root_id(root_id)
        try:
            agent = env.make_agent()
            for prompt in ("first", "second"):
                await agent.run_with_mcp(prompt)
        finally:
            reset_conversation_root_id(token)

    await asyncio.gather(conversation("acp-alpha"), conversation("acp-beta"))
    identities = [metadata["conversation_id"] for _, metadata in env.ends]
    assert len(set(identities)) == 2
    assert all(identities.count(identity) == 2 for identity in set(identities))
    assert sorted(env.starts) == sorted(session_id for session_id, _ in env.ends)


async def test_session_identity_is_captured_before_run(memory_runtime):
    env = memory_runtime
    agent = env.make_agent()
    await agent.run_with_mcp("initial turn")
    run = agent._code_generation_agent.run

    async def rotate_during_run(*args, **kwargs):
        config.rotate_session_name()
        return await run(*args, **kwargs)

    agent._code_generation_agent.run = rotate_during_run
    await agent.run_with_mcp("finish in the session that started the run")
    assert env.ends[0][1]["conversation_id"] == env.ends[1][1]["conversation_id"]
    assert env.starts == [session_id for session_id, _ in env.ends]
    await agent.run_with_mcp("next session")
    assert env.ends[2][1]["conversation_id"] != env.ends[1][1]["conversation_id"]


@pytest.fixture
def enabled_notifications(memory_runtime, monkeypatch):
    from code_puppy_core_plugins.completion_notification import (
        register_callbacks as notifications,
    )

    monkeypatch.setattr(notifications, "is_enabled", lambda: True)
    monkeypatch.setattr(notifications, "get_sound", lambda: "")
    monkeypatch.setattr(notifications, "_prompts", {})
    monkeypatch.setattr(notifications, "_run_depth", 0)
    worker = Mock()
    monkeypatch.setattr(notifications, "_start_notification_worker", worker)
    monkeypatch.setattr(
        _runtime, "on_user_prompt_submit", callbacks.on_user_prompt_submit
    )
    callbacks.clear_callbacks("user_prompt_submit")
    callbacks.register_callback(
        "user_prompt_submit", notifications._on_user_prompt_submit
    )
    callbacks.register_callback("agent_run_start", notifications._on_agent_run_start)
    callbacks.register_callback("agent_run_end", notifications._on_agent_run_end)
    return SimpleNamespace(worker=worker, plugin=notifications)


async def test_completion_notification_keeps_prompt_and_clears_cache(
    memory_runtime, enabled_notifications
):
    await memory_runtime.make_agent().run_with_mcp("Keep the user's prompt")
    assert enabled_notifications.worker.call_args.args[1] == "Keep the user's prompt"
    assert enabled_notifications.plugin._prompts == {}


async def test_nested_run_does_not_consume_outer_notification_prompt(
    memory_runtime, enabled_notifications
):
    outer, inner = memory_runtime.make_agent(), memory_runtime.make_agent()
    original_run = outer._code_generation_agent.run

    async def run_with_nested_call(*args, **kwargs):
        await inner.run_with_mcp("An internal assessment")
        return await original_run(*args, **kwargs)

    outer._code_generation_agent.run = run_with_nested_call
    await outer.run_with_mcp("The user's outer request")
    enabled_notifications.worker.assert_called_once()
    assert enabled_notifications.worker.call_args.args[1] == "The user's outer request"
    assert enabled_notifications.plugin._prompts == {}
    assert enabled_notifications.plugin._run_depth == 0


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_failed_and_cancelled_runs_keep_identity_and_lifecycle_pairing(
    memory_runtime, enabled_notifications, failure
):
    env = memory_runtime
    agent = env.make_agent()
    original_run = agent._code_generation_agent.run

    async def fail(*args, **kwargs):
        raise failure("stop this run")

    agent._code_generation_agent.run = fail
    if failure is ValueError:
        with pytest.raises(ValueError):
            await agent.run_with_mcp("do not record this failed turn")
    else:
        await agent.run_with_mcp("do not record this cancelled turn")
    # Cancellation is swallowed by the existing runtime and returns no text.
    # The recorder rejects empty responses regardless of the success flag.
    assert len(env.outcomes) == 1
    assert env.outcomes[0][1] == ""
    if failure is ValueError:
        assert env.outcomes[0][0] is False
    assert env.starts == [run_id for run_id, _ in env.ends]
    assert enabled_notifications.plugin._prompts == {}
    enabled_notifications.worker.assert_not_called()
    agent._code_generation_agent.run = original_run
    await agent.run_with_mcp("a later successful turn")
    assert env.outcomes[1][0] is True
    assert env.outcomes[1][1]
    assert env.ends[0][1]["conversation_id"] == env.ends[1][1]["conversation_id"]


async def test_cli_and_acp_identifiers_have_separate_namespaces(memory_runtime):
    env = memory_runtime
    config.pin_current_session_name("identical-session-name")
    await env.make_agent().run_with_mcp("CLI conversation")
    token = set_conversation_root_id("identical-session-name")
    try:
        await env.make_agent().run_with_mcp("ACP conversation")
    finally:
        reset_conversation_root_id(token)
    assert env.ends[0][1]["conversation_id"] != env.ends[1][1]["conversation_id"]


async def test_oauth_heartbeat_start_and_end_still_pair(memory_runtime, monkeypatch):
    from code_puppy_core_plugins.claude_code_oauth import register_callbacks as oauth
    from code_puppy_core_plugins.claude_code_oauth import token_refresh_heartbeat

    heartbeat = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), refresh_count=0)
    monkeypatch.setattr(
        token_refresh_heartbeat, "TokenRefreshHeartbeat", lambda: heartbeat
    )
    monkeypatch.setattr(oauth, "_active_heartbeats", {})
    callbacks.register_callback("agent_run_start", oauth._on_agent_run_start)
    callbacks.register_callback("agent_run_end", oauth._on_agent_run_end)
    agent = memory_runtime.make_agent()
    monkeypatch.setattr(agent, "get_model_name", lambda: "claude-code-offline")
    await agent.run_with_mcp("first OAuth run")
    await agent.run_with_mcp("second OAuth run")
    assert heartbeat.start.await_count == heartbeat.stop.await_count == 2
    assert oauth._active_heartbeats == {}
