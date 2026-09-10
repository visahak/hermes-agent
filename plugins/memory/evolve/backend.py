"""Backend seam for the Evolve memory provider.

The provider (``plugins/memory/evolve/__init__.py``) talks to a 3-method
interface — ``get_guidelines``, ``save_guideline``, ``save_trajectory`` —
with two implementations:

- ``LiteBackend`` (Phase 0): markdown entities on the local filesystem,
  case-insensitive term-overlap retrieval, in-plugin guideline generation.
  No server, no network, no extra dependency.
- ``ServerBackend`` (Phase 1): MCP client to a running ``evolve-mcp``
  server. Not implemented in Phase 0 — every method raises
  ``NotImplementedError`` so a misconfigured ``EVOLVE_MODE=server`` fails
  loudly instead of silently doing nothing.

Entity file format is compatible with upstream evolve-lite
(``altk-evolve/plugin-source/lib/entity_io.py``): markdown files under
``entities/{type}/{slug}.md`` with YAML frontmatter (a fixed key order,
only non-empty keys written) and an optional ``## Rationale`` section.
This module is a lean, dependency-free re-implementation — it does not
vendor or import the upstream library.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Frontmatter keys, in the order upstream evolve-lite writes them. Only
# non-empty values are emitted — this keeps the format compatible with
# entity_io.py's parser (and its own writer, for round-tripping).
_FRONTMATTER_KEYS = (
    "type", "trigger", "trajectory", "owner", "source",
    "native_path", "visibility", "published_at",
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is",
    "are", "with", "this", "that", "it", "be", "as", "at", "by", "from",
    "was", "were", "will", "would", "should", "can", "could", "not",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens, stopwords and single-char tokens dropped."""
    return [w for w in _WORD_RE.findall((text or "").lower())
            if w not in _STOPWORDS and len(w) > 1]


# ---------------------------------------------------------------------------
# Entity file format (lean re-implementation of entity_io.py)
# ---------------------------------------------------------------------------

def slugify(text: str, max_length: int = 60) -> str:
    """Convert *text* to a filesystem-safe slug."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if len(text) > max_length:
        text = text[:max_length].rsplit("-", 1)[0]
    return text or "entity"


def _sanitize_type(text: str) -> str:
    """Sanitize an entity ``type`` into a filesystem-safe subdirectory name."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def entity_to_markdown(entity: Dict[str, Any]) -> str:
    """Serialize an entity dict to markdown with YAML frontmatter."""
    lines = ["---"]
    for key in _FRONTMATTER_KEYS:
        val = entity.get(key)
        if val:
            lines.append(f"{key}: {val}")
    lines.append("---")
    lines.append("")
    lines.append(entity.get("content", ""))

    rationale = entity.get("rationale")
    if rationale:
        lines.append("")
        lines.append("## Rationale")
        lines.append("")
        lines.append(rationale)

    lines.append("")
    return "\n".join(lines)


def markdown_to_entity(path: Any) -> Dict[str, Any]:
    """Parse a markdown entity file back into a dict."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    entity: Dict[str, Any] = {}

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2]
            for line in frontmatter.splitlines():
                line = line.strip()
                if not line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if key and value:
                    entity[key] = value
        else:
            body = text
    else:
        body = text

    body = body.strip()
    m = re.search(r"^## Rationale", body, re.MULTILINE)
    if m:
        content = body[: m.start()].strip()
        rationale = body[m.end():].strip()
        if rationale:
            entity["rationale"] = rationale
    else:
        content = body

    if content:
        entity["content"] = content

    return entity


def _unique_filename(directory: Path, slug: str) -> Path:
    """Return a Path that doesn't collide with existing files in *directory*."""
    candidate = directory / f"{slug}.md"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = directory / f"{slug}-{n}.md"
        if not candidate.exists():
            return candidate
        n += 1


def write_entity_file(directory: Any, entity: Dict[str, Any], filename: Optional[str] = None) -> Path:
    """Write a single entity as a markdown file under *directory*.

    The file is placed in a ``{type}/`` subdirectory. Uses an atomic write
    (unique temp file, then ``os.replace``) and appends a ``-2``/``-3``
    suffix on slug collision.
    """
    entity_type = _sanitize_type(entity.get("type", "guideline")) or "guideline"
    entity = dict(entity)
    entity["type"] = entity_type
    type_dir = Path(directory) / entity_type
    type_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(filename) if filename else slugify(entity.get("content", "entity"))
    content = entity_to_markdown(entity)

    fd, tmp_path = tempfile.mkstemp(dir=type_dir, suffix=".tmp", prefix=slug)
    target: Optional[Path] = None
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd = None

        while True:
            target = _unique_filename(type_dir, slug)
            try:
                claim_fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(claim_fd)
                break
            except FileExistsError:
                continue

        os.replace(tmp_path, target)
        return target
    except BaseException:
        if fd is not None:
            os.close(fd)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if target and os.path.exists(str(target)) and os.path.getsize(str(target)) == 0:
            os.unlink(str(target))
        raise


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class EvolveBackend:
    """Abstract backend interface consumed by ``EvolveMemoryProvider``."""

    def get_guidelines(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return up to *limit* guideline entity dicts relevant to *query*."""
        raise NotImplementedError

    def save_guideline(self, content: str, trigger: str = "", rationale: str = "",
                        type: str = "guideline") -> str:
        """Persist a guideline; return an id/path identifying it."""
        raise NotImplementedError

    def save_trajectory(self, messages: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        """Persist a trajectory and (if configured) generate + save guidelines from it.

        Returns a summary dict, e.g. ``{"trajectory_path": ..., "guidelines": [...]}``.
        """
        raise NotImplementedError


class ServerBackend(EvolveBackend):
    """Phase 1 stub — MCP client to a running ``evolve-mcp`` server.

    Not implemented in Phase 0. Every method raises ``NotImplementedError``
    so ``EVOLVE_MODE=server`` fails loudly rather than silently no-op'ing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_guidelines(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "ServerBackend is a Phase 1 stub (see plans/evolve-memory-provider.md). "
            "Set EVOLVE_MODE=lite (the default) to use the filesystem backend."
        )

    def save_guideline(self, content: str, trigger: str = "", rationale: str = "",
                        type: str = "guideline") -> str:
        raise NotImplementedError(
            "ServerBackend is a Phase 1 stub; use EVOLVE_MODE=lite."
        )

    def save_trajectory(self, messages: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "ServerBackend is a Phase 1 stub; use EVOLVE_MODE=lite."
        )


class LiteBackend(EvolveBackend):
    """Phase 0 filesystem backend.

    Storage layout under *root* (default: ``$HERMES_HOME/evolve``,
    overridable via ``EVOLVE_DIR``):

      entities/{type}/{slug}.md        -- markdown entities (evolve-lite compatible)
      trajectories/{session_id}.jsonl  -- captured trajectories, one JSON line each

    Retrieval is case-insensitive term-overlap scoring of query tokens
    against each entity's ``trigger`` + ``content`` — no vector index, no
    server. ``save_trajectory`` writes the trajectory JSONL then invokes an
    injected ``guideline_generator(messages) -> list[dict]`` callable and
    saves each returned guideline as an entity. Generation failures never
    propagate — capture must not break a session.
    """

    def __init__(
        self,
        root: Any,
        *,
        guideline_generator: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    ) -> None:
        self.root = Path(root)
        self.entities_dir = self.root / "entities"
        self.trajectories_dir = self.root / "trajectories"
        self._guideline_generator = guideline_generator
        self._write_lock = threading.Lock()

    # -- retrieval ------------------------------------------------------

    def _iter_entities(self) -> List[Dict[str, Any]]:
        if not self.entities_dir.is_dir():
            return []
        entities = []
        for md in sorted(self.entities_dir.glob("**/*.md")):
            try:
                entity = markdown_to_entity(md)
            except (OSError, ValueError):
                # ValueError covers UnicodeDecodeError, which is what a non-UTF-8
                # file under entities/ raises. Skip that one file: the caller
                # swallows exceptions from here, so letting it escape would zero
                # out recall for every guideline in the store, not just this one.
                continue
            if entity.get("content"):
                entity["_path"] = str(md)
                entity["_slug"] = md.stem
                entities.append(entity)
        return entities

    def get_guidelines(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        limit = max(0, int(limit or 0))
        entities = self._iter_entities()
        if limit == 0:
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            # No usable query terms — return entities in stable (path) order.
            return entities[:limit]

        scored = []
        for entity in entities:
            haystack = f"{entity.get('trigger', '')} {entity.get('content', '')}"
            score = len(query_tokens & set(_tokenize(haystack)))
            if score > 0:
                scored.append((score, entity))
        # Stable sort by score desc; ties keep filesystem (path) order.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entity for _, entity in scored[:limit]]

    # -- writes -----------------------------------------------------------

    def save_guideline(self, content: str, trigger: str = "", rationale: str = "",
                        type: str = "guideline") -> str:
        content = (content or "").strip()
        if not content:
            raise ValueError("content is required")
        entity = {
            "type": type or "guideline",
            "trigger": (trigger or "").strip(),
            "content": content,
            "rationale": (rationale or "").strip(),
            "source": "hermes-evolve-lite",
        }
        with self._write_lock:
            path = write_entity_file(self.entities_dir, entity)
        return str(path)

    def save_trajectory(self, messages: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        try:
            from .trajectory_adapter import to_openai_trajectory
        except Exception:
            logger.warning("evolve: trajectory_adapter unavailable", exc_info=True)
            return {"trajectory_path": "", "guidelines": []}

        trajectory = to_openai_trajectory(messages)

        traj_path: Optional[Path] = None
        try:
            self.trajectories_dir.mkdir(parents=True, exist_ok=True)
            safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "session")
            traj_path = self.trajectories_dir / f"{safe_session}.jsonl"
            line = json.dumps(
                {"ts": time.time(), "session_id": session_id, "messages": trajectory},
                ensure_ascii=False,
            )
            with self._write_lock:
                with open(traj_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            logger.warning("evolve: failed to write trajectory for session %s", session_id, exc_info=True)
            traj_path = None

        saved_guidelines: List[Dict[str, Any]] = []
        if self._guideline_generator is not None and trajectory:
            try:
                guidelines = self._guideline_generator(trajectory) or []
            except Exception:
                logger.warning("evolve: guideline generation failed", exc_info=True)
                guidelines = []

            for g in guidelines:
                if not isinstance(g, dict):
                    continue
                content = str(g.get("content") or "").strip()
                if not content:
                    continue
                try:
                    path = self.save_guideline(
                        content=content,
                        trigger=str(g.get("trigger") or ""),
                        rationale=str(g.get("rationale") or ""),
                        type="guideline",
                    )
                    saved_guidelines.append({"path": path, "content": content})
                except Exception:
                    logger.warning("evolve: failed to save generated guideline", exc_info=True)

        return {
            "trajectory_path": str(traj_path) if traj_path else "",
            "guidelines": saved_guidelines,
        }
