# Evolve memory provider (Phase 0 -- lite backend)

[ALTK-Evolve](https://github.com/AgentToolkit/altk-evolve) integration for
hermes-agent, wired in through the `MemoryProvider` ABC
(`agent/memory_provider.py`). Learns structured, single-rule task
**guidelines** from session trajectories and recalls the relevant ones
before each turn -- a middle tier between the char-capped built-in memory
snapshot (`MEMORY.md`/`USER.md`) and name-triggered skills.

> **This is a frozen reference snapshot, not a maintained install target.**
> It exists to show what the integration looks like when it lives *inside*
> hermes-agent, as a bundled provider discovered alongside `mem0`, `honcho`,
> and the rest. The version that is actually maintained and released lives in
> [altk-evolve](https://github.com/AgentToolkit/altk-evolve) under
> `plugin-source/_hermes/`, is rendered into
> `platform-integrations/hermes/`, and installs into
> `$HERMES_HOME/plugins/evolve/` from outside the Hermes tree — see
> [the Hermes integration docs](https://github.com/AgentToolkit/altk-evolve/blob/main/docs/integrations/hermes.md).
> The two diverge: this copy re-implements the entity format locally, where
> the maintained one imports the shared `entity_io` module. Nothing here is
> kept in sync with it.

## Lite vs. server mode

Phase 0 ships **lite mode only**: a filesystem backend, zero extra
dependencies, no server process, no MCP client. Guideline generation runs
in-plugin via `agent.plugin_llm.PluginLlm`, using the user's active model
and auth -- no second API key.

**Server mode is a Phase 1 stub.** Setting `EVOLVE_MODE=server` disables
the provider (it logs a warning and returns inactive) rather than crashing,
since the MCP-client backend (conflict resolution, semantic retrieval via
pgvector/Milvus) is not implemented yet.

| | Lite (Phase 0, default) | Server (Phase 1, not yet implemented) |
|---|---|---|
| Retrieval | case-insensitive term-overlap | semantic (with pgvector/Milvus) |
| Conflict resolution on write | none (duplicates accumulate) | LLM-based, per write |
| Infra | none | `evolve-mcp` server, own venv |
| Guideline generation | in-plugin (`PluginLlm`) | server-side |

## How it works

- **Recall**: `queue_prefetch()` runs `LiteBackend.get_guidelines()` on a
  background thread after each turn; `prefetch()` returns the cached
  result on the next turn, formatted as a numbered list under
  `Guidelines learned from previous sessions (apply when relevant):`.
  Output is passed through `agent.memory_manager.sanitize_context` before
  injection (defense against fence-tag spoofing from stored content).
- **Capture**: at session end (`on_session_end`), if the session had at
  least `min_turns` user turns and `agent_context == "primary"`, the
  conversation is converted via `trajectory_adapter.to_openai_trajectory`,
  appended to `trajectories/<session_id>.jsonl`, and passed to
  `guideline_gen.generate_guidelines()` -- a single structured LLM call
  (`PluginLlm.complete_structured`) that ports the capture criteria and
  exclusion lists from evolve-lite's `learn` skill. Each returned
  guideline is saved as a markdown entity. Optional
  `capture_every_n_turns` triggers the same capture mid-session, for
  long-lived sessions that rarely hit a clean end.
- **Non-primary contexts** (`subagent`, `cron`, `flush`) get recall but
  never write -- forks and background jobs must not pollute the
  namespace.
- **Provenance**: every prefetch that returns entries appends
  `{ts, session_id, guideline_slugs}` as a JSON line to
  `$HERMES_HOME/evolve/audit.log`. This is free in Phase 0 because the
  provider is the injector; judging whether a guideline actually changed
  the outcome is a later, LLM-analysis pass (Phase 2).
- **Tools**: `evolve_get_guidelines(task)` for explicit on-demand recall,
  `evolve_save_guideline(content, trigger, rationale)` for explicit
  capture. Gated by `expose_tools` (default on).

## Storage layout

```
$HERMES_HOME/evolve/
  config.json              # optional, see below
  audit.log                # recall provenance (JSON lines)
  entities/
    guideline/
      use-make-check-for-tests.md
  trajectories/
    <session_id>.jsonl
```

Entity files are markdown with YAML frontmatter, compatible with upstream
evolve-lite's format (`type`, `trigger`, `trajectory`, `owner`, `source`,
`native_path`, `visibility`, `published_at` -- only non-empty keys are
written), body = guideline content, optional `## Rationale` section. See
`plugins/memory/evolve/backend.py` for the lean re-implementation (this
plugin does not vendor `altk-evolve`'s `entity_io.py`).

## Config

Env vars take precedence; unset ones fall back to
`$HERMES_HOME/evolve/config.json`.

| Env var | config.json key | Default | Meaning |
|---|---|---|---|
| `EVOLVE_MODE` | `mode` | `lite` | `lite` or `server` (Phase 1 stub) |
| `EVOLVE_DIR` | `dir` | `$HERMES_HOME/evolve` | Storage root override |
| `EVOLVE_PREFETCH_LIMIT` | `prefetch_limit` | `5` | Max guidelines recalled per turn |
| `EVOLVE_CAPTURE_EVERY_N_TURNS` | `capture_every_n_turns` | `0` (off) | Periodic mid-session capture cadence |
| `EVOLVE_MIN_TURNS` | `min_turns` | `2` | Minimum user turns before session-end capture fires |
| `EVOLVE_EXPOSE_TOOLS` | `expose_tools` | `true` | Expose the `evolve_*` tools to the model |

## Enable it

```
hermes config set memory.provider evolve
```

(Or run `hermes memory setup` and pick `evolve` -- `get_config_schema()`
walks through the fields above.)

## Division of labor (vs. skills, vs. built-in memory)

- **Evolve guidelines**: fine-grained, per-task-class rules and
  error-recovery steps -- single actionable statements, not multi-step
  docs.
- **Skills**: class-level how-to documentation with support files
  (scripts, references) for multi-step workflows.
- **Built-in memory (`MEMORY.md`/`USER.md`)**: user/persona/environment
  facts, not task procedure.

Phase 0 has no cross-store reconciliation -- a lesson can end up captured
as both a guideline and a skill/memory entry. Phase 1.5 (a routing-rubric
prompt change to `agent/background_review.py`, out of scope here) is the
planned mitigation.

## Known Phase 0 limitations

- No conflict resolution: duplicate/near-duplicate guidelines accumulate
  over time.
- Retrieval is lexical term-overlap, not semantic. That is the honest
  baseline for a dependency-free backend: semantic retrieval needs a vector
  index, which is server-only.
- No lite-to-server entity migration path yet; upgrading means starting
  fresh or writing an import script later.
