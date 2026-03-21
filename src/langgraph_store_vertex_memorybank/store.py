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
        namespace_ttl: Optional mapping of namespace prefixes to TTL in seconds.
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
        namespace_ttl: dict[tuple[str, ...], float] | None = None,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.reasoning_engine_id = reasoning_engine_id
        self.namespace_prefix = namespace_prefix
        self.default_top_k = default_top_k
        self.min_similarity_score = min_similarity_score
        self.namespace_ttl = namespace_ttl or {}

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
        """Process multiple operations, batching PutOps by namespace for efficiency."""
        ops_list = list(ops)
        results: list[Result | None] = [None] * len(ops_list)
        
        # maintain order of operations
        i = 0
        while i < len(ops_list):
            op = ops_list[i]
            # Batch consecutive PutOps to the same namespace (memories only, no keys/deletes)
            if isinstance(op, PutOp) and op.value is not None and op.key is None:
                batch_start = i
                namespace = op.namespace
                batch_ops = [op]
                
                j = i + 1
                while (
                    j < len(ops_list) 
                    and isinstance(ops_list[j], PutOp) 
                    and ops_list[j].value is not None 
                    and ops_list[j].key is None
                    and ops_list[j].namespace == namespace
                ):
                    batch_ops.append(ops_list[j])
                    j += 1
                
                if len(batch_ops) > 1:
                    self._handle_put_batch(batch_ops)
                    for k in range(batch_start, j):
                        results[k] = None
                    i = j
                    continue

            # Default: handle single op
            if isinstance(op, GetOp):
                results[i] = self._handle_get(op)
            elif isinstance(op, SearchOp):
                results[i] = self._handle_search(op)
            elif isinstance(op, PutOp):
                self._handle_put(op)
                results[i] = None
            elif isinstance(op, ListNamespacesOp):
                results[i] = self._handle_list_namespaces(op)
            else:
                raise ValueError(f"Unsupported operation: {type(op)}")
            i += 1
            
        return results

    def _handle_put_batch(self, ops: list[PutOp]) -> None:
        """Process a batch of PutOps to the same namespace using generate_memories."""
        if not ops:
            return
            
        namespace = ops[0].namespace
        try:
            scope, _topic = _parse_namespace(namespace)
        except ValueError:
            for op in ops:
                self._handle_put(op)
            return

        events = []
        for op in ops:
            fact = op.value.get("fact", "") if op.value else ""
            if not fact and op.value:
                fact = json.dumps(op.value)
            
            if fact:
                events.append({
                    "content": {
                        "role": "model", 
                        "parts": [{"text": f"FACT: {fact}"}]
                    }
                })

        if not events:
            return

        self.generate_memories(scope=scope, events=events)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Async version of batch."""
        ops_list = list(ops)
        results: list[Result | None] = [None] * len(ops_list)
        
        i = 0
        while i < len(ops_list):
            op = ops_list[i]
            if isinstance(op, PutOp) and op.value is not None and op.key is None:
                batch_start = i
                namespace = op.namespace
                batch_ops = [op]
                
                j = i + 1
                while (
                    j < len(ops_list) 
                    and isinstance(ops_list[j], PutOp) 
                    and ops_list[j].value is not None 
                    and ops_list[j].key is None
                    and ops_list[j].namespace == namespace
                ):
                    batch_ops.append(ops_list[j])
                    j += 1
                
                if len(batch_ops) > 1:
                    await self._ahandle_put_batch(batch_ops)
                    for k in range(batch_start, j):
                        results[k] = None
                    i = j
                    continue

            if isinstance(op, GetOp):
                results[i] = await self._ahandle_get(op)
            elif isinstance(op, SearchOp):
                results[i] = await self._ahandle_search(op)
            elif isinstance(op, PutOp):
                await self._ahandle_put(op)
                results[i] = None
            elif isinstance(op, ListNamespacesOp):
                results[i] = await self._ahandle_list_namespaces(op)
            else:
                raise ValueError(f"Unsupported operation: {type(op)}")
            i += 1
            
        return results

    async def _ahandle_put_batch(self, ops: list[PutOp]) -> None:
        """Async version of _handle_put_batch."""
        if not ops:
            return
            
        namespace = ops[0].namespace
        try:
            scope, _topic = _parse_namespace(namespace)
        except ValueError:
            for op in ops:
                await self._ahandle_put(op)
            return

        events = []
        for op in ops:
            fact = op.value.get("fact", "") if op.value else ""
            if not fact and op.value:
                fact = json.dumps(op.value)
            if fact:
                events.append({
                    "content": {
                        "role": "model", 
                        "parts": [{"text": f"FACT: {fact}"}]
                    }
                })

        if not events:
            return

        await self.agenerate_memories(scope=scope, events=events)

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

        ttl = self._get_ttl(op.namespace, getattr(op, "ttl", None))
        kwargs = {"name": self._engine_name, "fact": fact, "scope": scope}
        if ttl is not None:
            kwargs["config"] = {"ttl": ttl}

        self._memories.create(**kwargs)

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        # NOTE: We rely on _known_scopes (populated via search/put ops) rather than
        # calling memories.list(), which is being deprecated in favour of retrieve()
        # (retrieve() requires an explicit scope, so bulk scanning isn't possible).
        # Track scopes proactively in _handle_search / _handle_put instead.
        namespaces: set[tuple[str, ...]] = set()
        for scope_items in self._known_scopes:
            namespaces.add(_scope_to_namespace(dict(scope_items), self.namespace_prefix))

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

        ttl = self._get_ttl(op.namespace, getattr(op, "ttl", None))
        kwargs = {"name": self._engine_name, "fact": fact, "scope": scope}
        if ttl is not None:
            kwargs["config"] = {"ttl": ttl}

        await self._amemories.create(**kwargs)

    async def _ahandle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        # NOTE: mirrors _handle_list_namespaces — relies on _known_scopes only.
        # memories.list() is being deprecated; retrieve() requires explicit scope.
        namespaces: set[tuple[str, ...]] = set()
        for scope_items in self._known_scopes:
            namespaces.add(_scope_to_namespace(dict(scope_items), self.namespace_prefix))

        result = list(namespaces)
        if op.match_conditions:
            result = _filter_namespaces(result, op.match_conditions)
        if op.max_depth is not None:
            result = list({ns[: op.max_depth] for ns in result})
        result.sort()
        return result[op.offset : op.offset + op.limit]

    # ── Memory Generation (extension, not part of BaseStore) ───────────

    def generate_memories(
        self,
        scope: dict[str, str],
        events: list[dict[str, Any]],
        *,
        wait_for_completion: bool = True,
    ) -> Any:
        """Generate memories from conversation events using Memory Bank's LLM extraction.

        This is Memory Bank's killer feature: an LLM extracts meaningful facts
        from conversation turns and consolidates them with existing memories —
        deduplicating, updating, and deleting contradicted information
        automatically.

        Unlike ``BaseStore.put()`` (which stores exactly what you give it),
        ``generate_memories()`` uses an LLM to decide what's worth remembering.

        Not part of the ``BaseStore`` interface — this is a Memory Bank–specific
        extension. Use it directly or via :func:`create_capture_node`.

        Args:
            scope: Memory isolation scope, e.g. ``{"user_id": "alice"}``.
            events: Conversation events in Vertex AI format::

                [{"content": {"role": "user", "parts": [{"text": "I live in Portland"}]}},
                 {"content": {"role": "model", "parts": [{"text": "Nice city!"}]}}]

            wait_for_completion: If True, wait for the LRO to complete.

        Returns:
            The operation result from the SDK. Access generated memories via
            the response object.

        Example::

            events = [
                {"content": {"role": "user", "parts": [{"text": "I love hiking"}]}},
                {"content": {"role": "model", "parts": [{"text": "Great hobby!"}]}},
            ]
            result = store.generate_memories(
                scope={"user_id": "alice"},
                events=events,
            )
        """
        self._known_scopes.add(tuple(sorted(scope.items())))
        return self._memories.generate(
            name=self._engine_name,
            direct_contents_source={"events": events},
            scope=scope,
            config={"wait_for_completion": wait_for_completion},
        )

    async def agenerate_memories(
        self,
        scope: dict[str, str],
        events: list[dict[str, Any]],
        *,
        wait_for_completion: bool = True,
    ) -> Any:
        """Async version of :meth:`generate_memories`."""
        self._known_scopes.add(tuple(sorted(scope.items())))
        return await self._amemories.generate(
            name=self._engine_name,
            direct_contents_source={"events": events},
            scope=scope,
            config={"wait_for_completion": wait_for_completion},
        )

    # ── Utilities ───────────────────────────────────────────────────────

    # ── Memory Revisions (extension, not part of BaseStore) ────────────

    def list_revisions(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> list[Any]:
        """List revisions for a memory.

        Not part of the ``BaseStore`` interface — this is a Memory Bank–specific extension.

        Args:
            namespace: The namespace of the memory.
            key: The key (ID) of the memory.

        Returns:
            A list of memory revision objects from the SDK.
        """
        memory_name = f"{self._engine_name}/memories/{key}"
        return list(self._memories.revisions.list(name=memory_name))

    def get_revision(
        self,
        namespace: tuple[str, ...],
        key: str,
        revision_id: str,
    ) -> Any:
        """Get a specific memory revision.

        Not part of the ``BaseStore`` interface — this is a Memory Bank–specific extension.

        Args:
            namespace: The namespace of the memory.
            key: The key (ID) of the memory.
            revision_id: The ID of the revision.

        Returns:
            The memory revision object from the SDK.
        """
        memory_name = f"{self._engine_name}/memories/{key}/revisions/{revision_id}"
        return self._memories.revisions.get(name=memory_name)

    def rollback(
        self,
        namespace: tuple[str, ...],
        key: str,
        revision_id: str,
    ) -> Any:
        """Rollback a memory to a specific revision.

        Not part of the ``BaseStore`` interface — this is a Memory Bank–specific extension.

        Args:
            namespace: The namespace of the memory.
            key: The key (ID) of the memory.
            revision_id: The target revision ID.

        Returns:
            The operation result from the SDK.
        """
        memory_name = f"{self._engine_name}/memories/{key}"
        return self._memories.rollback(name=memory_name, target_revision_id=revision_id)

    async def alist_revisions(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> list[Any]:
        """Async version of :meth:`list_revisions`."""
        memory_name = f"{self._engine_name}/memories/{key}"
        pager = await self._amemories.revisions.list(name=memory_name)
        revisions = []
        async for r in pager:
            revisions.append(r)
        return revisions

    async def aget_revision(
        self,
        namespace: tuple[str, ...],
        key: str,
        revision_id: str,
    ) -> Any:
        """Async version of :meth:`get_revision`."""
        memory_name = f"{self._engine_name}/memories/{key}/revisions/{revision_id}"
        return await self._amemories.revisions.get(name=memory_name)

    async def arollback(
        self,
        namespace: tuple[str, ...],
        key: str,
        revision_id: str,
    ) -> Any:
        """Async version of :meth:`rollback`."""
        memory_name = f"{self._engine_name}/memories/{key}"
        return await self._amemories.rollback(name=memory_name, target_revision_id=revision_id)


    def _get_ttl(self, namespace: tuple[str, ...], put_ttl: float | None) -> str | None:
        if put_ttl is not None:
            return f"{int(put_ttl)}s"
        
        if self.namespace_ttl:
            best_match_len = -1
            best_ttl = None
            for prefix, ttl in self.namespace_ttl.items():
                if len(namespace) >= len(prefix) and namespace[:len(prefix)] == prefix:
                    if len(prefix) > best_match_len:
                        best_match_len = len(prefix)
                        best_ttl = ttl
            
            if best_ttl is not None:
                return f"{int(best_ttl)}s"
        
        return None

    def scope_for_namespace(self, namespace: tuple[str, ...]) -> dict[str, str]:
        """Convert a namespace to a scope dict."""
        scope, _topic = _parse_namespace(namespace)
        return scope

    def namespace_for_scope(
        self, scope: dict[str, str], topic: str | None = None
    ) -> tuple[str, ...]:
        """Convert a scope dict to a namespace tuple."""
        return _scope_to_namespace(scope, self.namespace_prefix, topic)

    def extract_memories_anthropic(
        self,
        conversation: list[dict[str, Any]],
        namespace: tuple[str, ...],
        model_name: str = "claude-3-haiku-20240307",
    ) -> None:
        """Extract facts from a conversation and save them to the memory bank.

        Args:
            conversation: A list of messages in the conversation.
            namespace: The namespace to save the memories to.
            model_name: The Anthropic model to use for fact extraction.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic SDK not found. Please install it with `pip install anthropic`."
            )

        client = anthropic.Anthropic()
        scope, _ = _parse_namespace(namespace)

        prompt = (
            "You are a memory extraction agent. Your task is to extract key facts from the "
            "conversation and represent them as concise, self-contained statements. "
            "Format the output as a JSON list of strings.\n\n"
            f"Conversation:\n{json.dumps(conversation, indent=2)}\n\n"
            "Extracted facts:"
        )

        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            facts_text = response.content[0].text
            facts = json.loads(facts_text)
            for fact in facts:
                self._memories.create(name=self._engine_name, fact=fact, scope=scope)
        except Exception as e:
            logger.error(f"Failed to generate memories: {e}")


def create_capture_node(store: VertexMemoryBankStore, namespace: tuple[str, ...]):
    """Create a LangGraph node that captures memories from the conversation.

    Args:
        store: The VertexMemoryBankStore instance.
        namespace: The namespace to save the memories to.

    Returns:
        A LangGraph node.
    """

    def capture_node(state: dict) -> dict:
        if "messages" in state:
            store.extract_memories_anthropic(state["messages"], namespace)
        return {}

    return capture_node
