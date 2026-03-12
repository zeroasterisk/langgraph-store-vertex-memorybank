"""Vertex AI Memory Bank store implementation for LangGraph.

This module implements LangGraph's BaseStore interface backed by Vertex AI
Agent Engine Memory Bank, using the official ``google-cloud-aiplatform`` SDK.

Namespace Mapping (A+C pattern):
    LangGraph namespaces map to Memory Bank scopes, with optional topic filtering:

    - ``("memories", "user_id", "alice")`` → scope ``{"user_id": "alice"}``
    - ``("memories", "user_id", "alice", "topic", "preferences")`` → scope
      ``{"user_id": "alice"}`` with topic filter ``"preferences"``

    The first element is the prefix (configurable, default ``"memories"``).
    Remaining elements are alternating key-value pairs forming the scope,
    except a final ``("topic", "<name>")`` pair which filters by Memory Bank topic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import vertexai
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

logger = logging.getLogger(__name__)


# ── Namespace / Scope helpers ───────────────────────────────────────────


def _parse_namespace(
    namespace: tuple[str, ...],
) -> tuple[dict[str, str], str | None]:
    """Parse a namespace tuple into a scope dict and optional topic filter.

    Format: ``(prefix, k1, v1, k2, v2, ..., ["topic", topic_name])``

    The first element is skipped (prefix). Remaining elements are key-value
    pairs. If the last pair is ``("topic", name)``, it's extracted as the
    topic filter and excluded from the scope.

    Args:
        namespace: LangGraph namespace tuple.

    Returns:
        Tuple of (scope_dict, topic_name_or_None).

    Raises:
        ValueError: If namespace format is invalid.
    """
    if len(namespace) < 3:
        raise ValueError(
            f"Namespace must have >= 3 elements (prefix, key, value), got: {namespace}"
        )
    pairs = namespace[1:]
    if len(pairs) % 2 != 0:
        raise ValueError(
            f"Namespace key-value pairs must be even-length, got {len(pairs)}: {pairs}"
        )

    scope: dict[str, str] = {}
    topic: str | None = None

    for i in range(0, len(pairs), 2):
        k, v = pairs[i], pairs[i + 1]
        if k == "topic":
            topic = v
        else:
            scope[k] = v

    if not scope:
        raise ValueError(
            f"Namespace must contain at least one non-topic key-value pair: {namespace}"
        )

    return scope, topic


def _scope_to_namespace(
    scope: dict[str, str],
    prefix: str = "memories",
    topic: str | None = None,
) -> tuple[str, ...]:
    """Convert a scope dict (and optional topic) to a namespace tuple.

    Args:
        scope: Memory Bank scope dictionary.
        prefix: Namespace prefix.
        topic: Optional topic name to append.

    Returns:
        LangGraph namespace tuple.
    """
    parts: list[str] = [prefix]
    for k, v in sorted(scope.items()):
        parts.extend([k, v])
    if topic:
        parts.extend(["topic", topic])
    return tuple(parts)


def _extract_memory_id(resource_name: str) -> str:
    """Extract the short memory ID from a full resource name."""
    return resource_name.rsplit("/", 1)[-1]


def _distance_to_score(distance: float | None) -> float | None:
    """Convert Memory Bank distance to similarity score.

    Memory Bank returns Euclidean distance (lower = more similar).
    We convert: ``score = 1.0 / (1.0 + distance)``.
    """
    if distance is None:
        return None
    return 1.0 / (1.0 + distance)


def _sdk_memory_to_item(
    memory: Any,
    namespace: tuple[str, ...],
) -> Item:
    """Convert an SDK Memory object to a LangGraph Item.

    Args:
        memory: A ``vertexai`` Memory object with fields like name, fact, scope, etc.
        namespace: LangGraph namespace tuple.
    """
    resource_name = memory.name or ""
    key = _extract_memory_id(resource_name)
    fact = memory.fact or ""
    scope = dict(memory.scope) if memory.scope else {}

    # Convert SDK metadata to plain dict
    metadata: dict[str, Any] = {}
    if memory.metadata:
        for mk, mv in memory.metadata.items():
            if hasattr(mv, "string_value") and mv.string_value is not None:
                metadata[mk] = mv.string_value
            elif hasattr(mv, "double_value") and mv.double_value is not None:
                metadata[mk] = mv.double_value
            elif hasattr(mv, "bool_value") and mv.bool_value is not None:
                metadata[mk] = mv.bool_value
            else:
                metadata[mk] = str(mv)

    now = datetime.now(timezone.utc)
    created_at = memory.create_time if memory.create_time else now
    updated_at = memory.update_time if memory.update_time else now

    return Item(
        value={
            "fact": fact,
            "metadata": metadata,
            "scope": scope,
            "resource_name": resource_name,
        },
        key=key,
        namespace=namespace,
        created_at=created_at,
        updated_at=updated_at,
    )


def _sdk_retrieved_to_search_item(
    retrieved: Any,
    namespace: tuple[str, ...],
) -> SearchItem:
    """Convert an SDK RetrieveMemoriesResponseRetrievedMemory to a SearchItem.

    Args:
        retrieved: SDK retrieved memory object with ``memory`` and ``distance`` fields.
        namespace: LangGraph namespace tuple.
    """
    memory = retrieved.memory
    resource_name = memory.name or ""
    key = _extract_memory_id(resource_name)
    fact = memory.fact or ""
    scope = dict(memory.scope) if memory.scope else {}

    metadata: dict[str, Any] = {}
    if memory.metadata:
        for mk, mv in memory.metadata.items():
            if hasattr(mv, "string_value") and mv.string_value is not None:
                metadata[mk] = mv.string_value
            elif hasattr(mv, "double_value") and mv.double_value is not None:
                metadata[mk] = mv.double_value
            elif hasattr(mv, "bool_value") and mv.bool_value is not None:
                metadata[mk] = mv.bool_value
            else:
                metadata[mk] = str(mv)

    now = datetime.now(timezone.utc)
    created_at = memory.create_time if memory.create_time else now
    updated_at = memory.update_time if memory.update_time else now

    return SearchItem(
        namespace=namespace,
        key=key,
        value={
            "fact": fact,
            "metadata": metadata,
            "scope": scope,
            "resource_name": resource_name,
        },
        created_at=created_at,
        updated_at=updated_at,
        score=_distance_to_score(retrieved.distance),
    )


class VertexMemoryBankStore(BaseStore):
    """LangGraph BaseStore backed by Vertex AI Agent Engine Memory Bank.

    Uses the official ``google-cloud-aiplatform`` SDK for all API calls,
    getting automatic auth refresh, retries, endpoint routing, and pagination.

    Unlike ``InMemoryStore`` which loses data on restart, or vector stores that
    just embed raw text, Memory Bank uses an LLM to extract meaningful facts
    and consolidates them over time.

    Features beyond standard BaseStore:
        - ``generate_memories()``: LLM-powered fact extraction from conversations
        - Topic-based namespace scoping via the A+C pattern
        - Automatic memory consolidation and deduplication
        - Similarity-score filtering with configurable threshold

    Args:
        project_id: Google Cloud project ID.
        location: Google Cloud region (e.g., ``"us-central1"``).
        reasoning_engine_id: Agent Engine (Reasoning Engine) instance ID.
        client: Optional pre-configured ``vertexai.Client``. If provided,
            ``project_id`` and ``location`` are ignored for client creation.
        namespace_prefix: Prefix for namespace tuples (default: ``"memories"``).
        default_top_k: Default number of results for search (default: 10).
        min_similarity_score: Minimum score threshold for search results.
            Memories below this score are filtered out (default: 0.3).
        revision_labels: Optional labels passed to ``generate_memories()`` calls.
        topics: Default topics for memory extraction. If None, uses Memory Bank
            defaults (USER_PERSONAL_INFO, USER_PREFERENCES, etc.).
        disable_consolidation: If True, skip consolidation during generation.

    Example::

        store = VertexMemoryBankStore(
            project_id="my-project",
            location="us-central1",
            reasoning_engine_id="123456",
            default_top_k=10,
            min_similarity_score=0.3,
            topics=["preferences", "facts", "instructions"],
        )

        # Or with an existing client:
        client = vertexai.Client(project="my-project", location="us-central1")
        store = VertexMemoryBankStore(
            project_id="my-project",
            location="us-central1",
            reasoning_engine_id="123456",
            client=client,
        )
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        reasoning_engine_id: str,
        client: vertexai.Client | None = None,
        namespace_prefix: str = "memories",
        default_top_k: int = 10,
        min_similarity_score: float = 0.3,
        revision_labels: dict[str, str] | None = None,
        topics: list[str] | None = None,
        disable_consolidation: bool = False,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.reasoning_engine_id = reasoning_engine_id
        self.namespace_prefix = namespace_prefix
        self.default_top_k = default_top_k
        self.min_similarity_score = min_similarity_score
        self.revision_labels = revision_labels
        self.topics = topics
        self.disable_consolidation = disable_consolidation

        self._client = client or vertexai.Client(
            project=project_id, location=location
        )
        self._known_scopes: set[tuple[tuple[str, str], ...]] = set()

        # Build the resource name used by all SDK calls
        self._engine_name = (
            f"projects/{project_id}/locations/{location}"
            f"/reasoningEngines/{reasoning_engine_id}"
        )

    @property
    def _memories(self) -> Any:
        """Shortcut to the SDK memories sub-resource."""
        return self._client.agent_engines.memories

    @property
    def _amemories(self) -> Any:
        """Shortcut to the async SDK memories sub-resource."""
        return self._client.aio.agent_engines.memories

    # ── BaseStore interface ─────────────────────────────────────────────

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations synchronously."""
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._handle_get(op))
            elif isinstance(op, SearchOp):
                results.append(self._handle_search(op))
            elif isinstance(op, PutOp):
                self._handle_put(op)
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            else:
                raise ValueError(f"Unsupported operation: {type(op)}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously."""
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(await self._ahandle_get(op))
            elif isinstance(op, SearchOp):
                results.append(await self._ahandle_search(op))
            elif isinstance(op, PutOp):
                await self._ahandle_put(op)
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(await self._ahandle_list_namespaces(op))
            else:
                raise ValueError(f"Unsupported operation: {type(op)}")
        return results

    # ── Sync operation handlers ─────────────────────────────────────────

    def _handle_get(self, op: GetOp) -> Item | None:
        """Fetch a single memory by key (memory ID)."""
        memory_name = f"{self._engine_name}/memories/{op.key}"
        try:
            memory = self._memories.get(name=memory_name)
            return _sdk_memory_to_item(memory, op.namespace)
        except Exception as e:
            if "404" in str(e) or "NOT_FOUND" in str(e):
                return None
            raise

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        """Search memories via Memory Bank's retrieve endpoint."""
        try:
            scope, topic = _parse_namespace(op.namespace_prefix)
        except ValueError:
            logger.debug("Cannot extract scope from namespace: %s", op.namespace_prefix)
            return []

        self._known_scopes.add(tuple(sorted(scope.items())))

        kwargs: dict[str, Any] = {
            "name": self._engine_name,
            "scope": scope,
        }

        if op.query:
            top_k = op.limit or self.default_top_k
            kwargs["similarity_search_params"] = {
                "search_query": op.query,
                "top_k": top_k,
            }

        # Topic filter via config
        if topic:
            kwargs.setdefault("config", {})
            kwargs["config"]["filter"] = f'topic = "{topic}"'

        try:
            results_iter = self._memories.retrieve(**kwargs)
            retrieved_list = list(results_iter)
        except Exception as e:
            logger.warning("Failed to retrieve memories: %s", e)
            return []

        namespace = _scope_to_namespace(scope, self.namespace_prefix, topic)

        items: list[SearchItem] = []
        for retrieved in retrieved_list:
            item = _sdk_retrieved_to_search_item(retrieved, namespace)

            # Filter by minimum similarity score
            if item.score is not None and item.score < self.min_similarity_score:
                continue

            items.append(item)

        # Apply offset/limit
        items = items[op.offset : op.offset + op.limit]

        # Client-side metadata filtering
        if op.filter:
            items = [
                item for item in items
                if all(
                    item.value.get("metadata", {}).get(k) == v
                    or item.value.get(k) == v
                    for k, v in op.filter.items()
                )
            ]

        return items

    def _handle_put(self, op: PutOp) -> None:
        """Create or delete a memory."""
        if op.value is None:
            # Delete
            memory_name = f"{self._engine_name}/memories/{op.key}"
            try:
                self._memories.delete(name=memory_name)
            except Exception as e:
                if "404" not in str(e) and "NOT_FOUND" not in str(e):
                    raise
            return

        try:
            scope, _topic = _parse_namespace(op.namespace)
        except ValueError as e:
            raise ValueError(f"Cannot put without a valid scope namespace: {e}") from e

        self._known_scopes.add(tuple(sorted(scope.items())))

        fact = op.value.get("fact", "")
        if not fact:
            fact = json.dumps(op.value)

        self._memories.create(
            name=self._engine_name,
            fact=fact,
            scope=scope,
        )

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """List known namespaces, discovering from API when possible."""
        namespaces: set[tuple[str, ...]] = set()

        # Add locally-known scopes
        for scope_items in self._known_scopes:
            ns = _scope_to_namespace(dict(scope_items), self.namespace_prefix)
            namespaces.add(ns)

        # Try listing memories from API for discovery
        try:
            for mem in self._memories.list(name=self._engine_name):
                scope = dict(mem.scope) if mem.scope else {}
                if scope:
                    ns = _scope_to_namespace(scope, self.namespace_prefix)
                    namespaces.add(ns)
                    self._known_scopes.add(tuple(sorted(scope.items())))
        except Exception:
            logger.debug("Could not list memories for namespace discovery")

        result = list(namespaces)

        if op.match_conditions:
            result = _filter_namespaces(result, op.match_conditions)

        if op.max_depth is not None:
            result = list({ns[: op.max_depth] for ns in result})

        result.sort()
        return result[op.offset : op.offset + op.limit]

    # ── Async operation handlers ────────────────────────────────────────

    async def _ahandle_get(self, op: GetOp) -> Item | None:
        """Async fetch a single memory by key."""
        memory_name = f"{self._engine_name}/memories/{op.key}"
        try:
            memory = await self._amemories.get(name=memory_name)
            return _sdk_memory_to_item(memory, op.namespace)
        except Exception as e:
            if "404" in str(e) or "NOT_FOUND" in str(e):
                return None
            raise

    async def _ahandle_search(self, op: SearchOp) -> list[SearchItem]:
        """Async search memories."""
        try:
            scope, topic = _parse_namespace(op.namespace_prefix)
        except ValueError:
            logger.debug("Cannot extract scope from namespace: %s", op.namespace_prefix)
            return []

        self._known_scopes.add(tuple(sorted(scope.items())))

        kwargs: dict[str, Any] = {
            "name": self._engine_name,
            "scope": scope,
        }

        if op.query:
            top_k = op.limit or self.default_top_k
            kwargs["similarity_search_params"] = {
                "search_query": op.query,
                "top_k": top_k,
            }

        if topic:
            kwargs.setdefault("config", {})
            kwargs["config"]["filter"] = f'topic = "{topic}"'

        try:
            results_pager = await self._amemories.retrieve(**kwargs)
            retrieved_list: list[Any] = []
            async for item in results_pager:
                retrieved_list.append(item)
        except Exception as e:
            logger.warning("Failed to retrieve memories: %s", e)
            return []

        namespace = _scope_to_namespace(scope, self.namespace_prefix, topic)

        items: list[SearchItem] = []
        for retrieved in retrieved_list:
            item = _sdk_retrieved_to_search_item(retrieved, namespace)
            if item.score is not None and item.score < self.min_similarity_score:
                continue
            items.append(item)

        items = items[op.offset : op.offset + op.limit]

        if op.filter:
            items = [
                item for item in items
                if all(
                    item.value.get("metadata", {}).get(k) == v
                    or item.value.get(k) == v
                    for k, v in op.filter.items()
                )
            ]

        return items

    async def _ahandle_put(self, op: PutOp) -> None:
        """Async create or delete a memory."""
        if op.value is None:
            memory_name = f"{self._engine_name}/memories/{op.key}"
            try:
                await self._amemories.delete(name=memory_name)
            except Exception as e:
                if "404" not in str(e) and "NOT_FOUND" not in str(e):
                    raise
            return

        try:
            scope, _topic = _parse_namespace(op.namespace)
        except ValueError as e:
            raise ValueError(f"Cannot put without a valid scope namespace: {e}") from e

        self._known_scopes.add(tuple(sorted(scope.items())))

        fact = op.value.get("fact", "")
        if not fact:
            fact = json.dumps(op.value)

        await self._amemories.create(
            name=self._engine_name,
            fact=fact,
            scope=scope,
        )

    async def _ahandle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Async list namespaces."""
        namespaces: set[tuple[str, ...]] = set()

        for scope_items in self._known_scopes:
            ns = _scope_to_namespace(dict(scope_items), self.namespace_prefix)
            namespaces.add(ns)

        try:
            pager = await self._amemories.list(name=self._engine_name)
            async for mem in pager:
                scope = dict(mem.scope) if mem.scope else {}
                if scope:
                    ns = _scope_to_namespace(scope, self.namespace_prefix)
                    namespaces.add(ns)
                    self._known_scopes.add(tuple(sorted(scope.items())))
        except Exception:
            logger.debug("Could not list memories for namespace discovery")

        result = list(namespaces)

        if op.match_conditions:
            result = _filter_namespaces(result, op.match_conditions)

        if op.max_depth is not None:
            result = list({ns[: op.max_depth] for ns in result})

        result.sort()
        return result[op.offset : op.offset + op.limit]

    # ── Memory Bank extensions (beyond BaseStore) ───────────────────────

    def generate_memories(
        self,
        scope: dict[str, str],
        events: list[dict[str, Any]],
        *,
        topics: list[str] | None = None,
        wait_for_completion: bool = True,
        disable_consolidation: bool | None = None,
        revision_labels: dict[str, str] | None = None,
    ) -> Any:
        """Generate memories from conversation events via LLM extraction.

        Memory Bank's core feature: uses an LLM to extract meaningful facts from
        conversation turns and consolidates them with existing memories.

        Args:
            scope: Scope dict for generated memories (e.g., ``{"user_id": "alice"}``).
            events: Conversation events, each with a ``"content"`` dict containing
                ``"role"`` and ``"parts"`` (Vertex AI Content format).
            topics: Topics to extract into. Overrides constructor default.
            wait_for_completion: Whether to poll the LRO until done (default True).
                Note: The SDK handles LRO polling automatically when accessing
                the operation result.
            disable_consolidation: Override constructor setting for this call.
            revision_labels: Override constructor labels for this call.

        Returns:
            The ``AgentEngineGenerateMemoriesOperation`` from the SDK.
            Access ``.response.generated_memories`` for extracted facts.

        Example::

            events = [
                {"content": {"role": "user", "parts": [{"text": "I live in Portland"}]}},
                {"content": {"role": "model", "parts": [{"text": "Nice city!"}]}},
            ]
            op = store.generate_memories(
                scope={"user_id": "alice"},
                events=events,
                topics=["preferences", "facts"],
            )
        """
        self._known_scopes.add(tuple(sorted(scope.items())))

        kwargs: dict[str, Any] = {
            "name": self._engine_name,
            "direct_contents_source": {"events": events},
            "scope": scope,
        }

        # Build config dict
        config: dict[str, Any] = {}

        should_disable = (
            disable_consolidation
            if disable_consolidation is not None
            else self.disable_consolidation
        )
        if should_disable:
            config["disable_consolidation"] = True

        labels = revision_labels or self.revision_labels
        if labels:
            config["revision_labels"] = labels

        effective_topics = topics or self.topics
        if effective_topics:
            config["memory_topics"] = [
                {"custom_memory_topic": {"label": t}} for t in effective_topics
            ]

        if config:
            kwargs["config"] = config

        return self._memories.generate(**kwargs)

    async def agenerate_memories(
        self,
        scope: dict[str, str],
        events: list[dict[str, Any]],
        *,
        topics: list[str] | None = None,
        wait_for_completion: bool = True,
        disable_consolidation: bool | None = None,
        revision_labels: dict[str, str] | None = None,
    ) -> Any:
        """Async version of ``generate_memories``."""
        self._known_scopes.add(tuple(sorted(scope.items())))

        kwargs: dict[str, Any] = {
            "name": self._engine_name,
            "direct_contents_source": {"events": events},
            "scope": scope,
        }

        config: dict[str, Any] = {}

        should_disable = (
            disable_consolidation
            if disable_consolidation is not None
            else self.disable_consolidation
        )
        if should_disable:
            config["disable_consolidation"] = True

        labels = revision_labels or self.revision_labels
        if labels:
            config["revision_labels"] = labels

        effective_topics = topics or self.topics
        if effective_topics:
            config["memory_topics"] = [
                {"custom_memory_topic": {"label": t}} for t in effective_topics
            ]

        if config:
            kwargs["config"] = config

        return await self._amemories.generate(**kwargs)

    # ── Utilities ───────────────────────────────────────────────────────

    def scope_for_namespace(self, namespace: tuple[str, ...]) -> dict[str, str]:
        """Convert a namespace to a scope dict.

        Useful when calling ``generate_memories()`` with a scope derived from
        a namespace you're already using.
        """
        scope, _topic = _parse_namespace(namespace)
        return scope

    def namespace_for_scope(
        self,
        scope: dict[str, str],
        topic: str | None = None,
    ) -> tuple[str, ...]:
        """Convert a scope dict to a namespace tuple."""
        return _scope_to_namespace(scope, self.namespace_prefix, topic)


# ── Module-level helpers ────────────────────────────────────────────────


def _filter_namespaces(
    namespaces: list[tuple[str, ...]],
    conditions: tuple[MatchCondition, ...],
) -> list[tuple[str, ...]]:
    """Filter namespaces by match conditions (prefix/suffix with wildcards)."""
    result = namespaces
    for cond in conditions:
        if cond.match_type == "prefix":
            prefix = cond.path
            result = [
                ns for ns in result
                if len(ns) >= len(prefix)
                and all(p == "*" or p == n for p, n in zip(prefix, ns))
            ]
        elif cond.match_type == "suffix":
            suffix = cond.path
            result = [
                ns for ns in result
                if len(ns) >= len(suffix)
                and all(
                    p == "*" or p == n
                    for p, n in zip(reversed(suffix), reversed(ns))
                )
            ]
    return result
