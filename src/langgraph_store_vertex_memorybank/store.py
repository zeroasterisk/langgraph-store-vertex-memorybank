"""Vertex AI Memory Bank store for LangGraph.

A thin ``BaseStore`` implementation backed by Vertex AI Agent Engine Memory
Bank. Drop it into any LangGraph agent (including ``langmem`` tools) for
fully-managed, LLM-powered long-term memory.

Namespace Mapping (A+C pattern)::

    ("memories", "user_id", "alice")
        → scope {"user_id": "alice"}

    ("memories", "user_id", "alice", "topic", "preferences")
        → scope {"user_id": "alice"}, topic filter "preferences"
"""

from __future__ import annotations

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
    """Parse ``(prefix, k1, v1, …, ["topic", name])`` into scope + topic."""
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
        raise ValueError(f"Namespace must contain at least one non-topic pair: {namespace}")
    return scope, topic


def _scope_to_namespace(
    scope: dict[str, str],
    prefix: str = "memories",
    topic: str | None = None,
) -> tuple[str, ...]:
    """Convert a scope dict (and optional topic) to a namespace tuple."""
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
    """Convert Memory Bank distance to similarity: ``1 / (1 + distance)``."""
    if distance is None:
        return None
    return 1.0 / (1.0 + distance)


def _sdk_memory_to_item(memory: Any, namespace: tuple[str, ...]) -> Item:
    """Convert an SDK Memory object to a LangGraph ``Item``."""
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
    return Item(
        value={"fact": fact, "metadata": metadata, "scope": scope, "resource_name": resource_name},
        key=key,
        namespace=namespace,
        created_at=memory.create_time or now,
        updated_at=memory.update_time or now,
    )


def _sdk_retrieved_to_search_item(retrieved: Any, namespace: tuple[str, ...]) -> SearchItem:
    """Convert an SDK retrieved-memory to a LangGraph ``SearchItem``."""
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
    return SearchItem(
        namespace=namespace,
        key=key,
        value={"fact": fact, "metadata": metadata, "scope": scope, "resource_name": resource_name},
        created_at=memory.create_time or now,
        updated_at=memory.update_time or now,
        score=_distance_to_score(retrieved.distance),
    )


def _filter_namespaces(
    namespaces: list[tuple[str, ...]],
    conditions: tuple[MatchCondition, ...],
) -> list[tuple[str, ...]]:
    """Filter namespaces by prefix/suffix match conditions (with wildcards)."""
    result = namespaces
    for cond in conditions:
        if cond.match_type == "prefix":
            result = [
                ns for ns in result
                if len(ns) >= len(cond.path)
                and all(p == "*" or p == n for p, n in zip(cond.path, ns))
            ]
        elif cond.match_type == "suffix":
            result = [
                ns for ns in result
                if len(ns) >= len(cond.path)
                and all(p == "*" or p == n for p, n in zip(reversed(cond.path), reversed(ns)))
            ]
    return result


# ── Store ───────────────────────────────────────────────────────────────


class VertexMemoryBankStore(BaseStore):
    """LangGraph ``BaseStore`` backed by Vertex AI Agent Engine Memory Bank.

    Unlike ``InMemoryStore`` (lost on restart) or ``AsyncPostgresStore``
    (DIY infrastructure), Memory Bank is fully managed with LLM-powered
    extraction, consolidation, and semantic recall.

    Works with any LangGraph agent, including ``langmem`` tools::

        from langmem import create_manage_memory_tool, create_search_memory_tool

        store = VertexMemoryBankStore(
            project_id="my-project",
            location="us-central1",
            reasoning_engine_id="123456",
        )
        agent = create_react_agent(
            "gemini-2.5-flash",
            tools=[
                create_manage_memory_tool(namespace=("memories",)),
                create_search_memory_tool(namespace=("memories",)),
            ],
            store=store,
        )

    Args:
        project_id: Google Cloud project ID.
        location: Google Cloud region (e.g. ``"us-central1"``).
        reasoning_engine_id: Agent Engine instance ID.
        client: Optional pre-configured ``vertexai.Client``.
        namespace_prefix: Prefix for namespace tuples (default ``"memories"``).
        default_top_k: Default result count for search (default 10).
        min_similarity_score: Floor for search results (default 0.3).
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
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.reasoning_engine_id = reasoning_engine_id
        self.namespace_prefix = namespace_prefix
        self.default_top_k = default_top_k
        self.min_similarity_score = min_similarity_score

        self._client = client or vertexai.Client(project=project_id, location=location)
        self._known_scopes: set[tuple[tuple[str, str], ...]] = set()
        self._engine_name = (
            f"projects/{project_id}/locations/{location}"
            f"/reasoningEngines/{reasoning_engine_id}"
        )

    @property
    def _memories(self) -> Any:
        return self._client.agent_engines.memories

    @property
    def _amemories(self) -> Any:
        return self._client.aio.agent_engines.memories

    # ── BaseStore interface ─────────────────────────────────────────────

    def batch(self, ops: Iterable[Op]) -> list[Result]:
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

    # ── Sync handlers ───────────────────────────────────────────────────

    def _handle_get(self, op: GetOp) -> Item | None:
        memory_name = f"{self._engine_name}/memories/{op.key}"
        try:
            memory = self._memories.get(name=memory_name)
            return _sdk_memory_to_item(memory, op.namespace)
        except Exception as e:
            if "404" in str(e) or "NOT_FOUND" in str(e):
                return None
            raise

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        try:
            scope, topic = _parse_namespace(op.namespace_prefix)
        except ValueError:
            logger.debug("Cannot extract scope from namespace: %s", op.namespace_prefix)
            return []

        self._known_scopes.add(tuple(sorted(scope.items())))

        kwargs: dict[str, Any] = {"name": self._engine_name, "scope": scope}

        if op.query:
            kwargs["similarity_search_params"] = {
                "search_query": op.query,
                "top_k": op.limit or self.default_top_k,
            }

        if topic:
            kwargs.setdefault("config", {})
            kwargs["config"]["filter"] = f'topic = "{topic}"'

        try:
            retrieved_list = list(self._memories.retrieve(**kwargs))
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
                    item.value.get("metadata", {}).get(k) == v or item.value.get(k) == v
                    for k, v in op.filter.items()
                )
            ]
        return items

    def _handle_put(self, op: PutOp) -> None:
        if op.value is None:
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

        self._memories.create(name=self._engine_name, fact=fact, scope=scope)

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces: set[tuple[str, ...]] = set()
        for scope_items in self._known_scopes:
            namespaces.add(_scope_to_namespace(dict(scope_items), self.namespace_prefix))

        try:
            for mem in self._memories.list(name=self._engine_name):
                scope = dict(mem.scope) if mem.scope else {}
                if scope:
                    namespaces.add(_scope_to_namespace(scope, self.namespace_prefix))
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

    # ── Async handlers ──────────────────────────────────────────────────

    async def _ahandle_get(self, op: GetOp) -> Item | None:
        memory_name = f"{self._engine_name}/memories/{op.key}"
        try:
            memory = await self._amemories.get(name=memory_name)
            return _sdk_memory_to_item(memory, op.namespace)
        except Exception as e:
            if "404" in str(e) or "NOT_FOUND" in str(e):
                return None
            raise

    async def _ahandle_search(self, op: SearchOp) -> list[SearchItem]:
        try:
            scope, topic = _parse_namespace(op.namespace_prefix)
        except ValueError:
            logger.debug("Cannot extract scope from namespace: %s", op.namespace_prefix)
            return []

        self._known_scopes.add(tuple(sorted(scope.items())))
        kwargs: dict[str, Any] = {"name": self._engine_name, "scope": scope}

        if op.query:
            kwargs["similarity_search_params"] = {
                "search_query": op.query,
                "top_k": op.limit or self.default_top_k,
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
                    item.value.get("metadata", {}).get(k) == v or item.value.get(k) == v
                    for k, v in op.filter.items()
                )
            ]
        return items

    async def _ahandle_put(self, op: PutOp) -> None:
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

        await self._amemories.create(name=self._engine_name, fact=fact, scope=scope)

    async def _ahandle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces: set[tuple[str, ...]] = set()
        for scope_items in self._known_scopes:
            namespaces.add(_scope_to_namespace(dict(scope_items), self.namespace_prefix))

        try:
            pager = await self._amemories.list(name=self._engine_name)
            async for mem in pager:
                scope = dict(mem.scope) if mem.scope else {}
                if scope:
                    namespaces.add(_scope_to_namespace(scope, self.namespace_prefix))
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

    # ── Utilities ───────────────────────────────────────────────────────

    def scope_for_namespace(self, namespace: tuple[str, ...]) -> dict[str, str]:
        """Convert a namespace to a scope dict."""
        scope, _topic = _parse_namespace(namespace)
        return scope

    def namespace_for_scope(
        self, scope: dict[str, str], topic: str | None = None
    ) -> tuple[str, ...]:
        """Convert a scope dict to a namespace tuple."""
        return _scope_to_namespace(scope, self.namespace_prefix, topic)
