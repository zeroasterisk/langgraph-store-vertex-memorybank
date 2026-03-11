"""Vertex AI Memory Bank store implementation for LangGraph.

This module implements LangGraph's BaseStore interface backed by Vertex AI
Agent Engine Memory Bank. It maps LangGraph's namespace/key model to Memory
Bank's scope-based memory isolation and provides access to Memory Bank's
LLM-powered memory extraction and consolidation.

Namespace Mapping:
    LangGraph uses tuples like ("memories", "user_id", "alice") as namespaces.
    Memory Bank uses scope dicts like {"user_id": "alice"}.

    This store maps between them:
    - Namespace ("memories", "k1", "v1", "k2", "v2") → scope {"k1": "v1", "k2": "v2"}
    - The first element ("memories") is the namespace prefix (configurable)
    - Remaining elements are alternating key-value pairs forming the scope
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from google.auth import default as google_auth_default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request
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


def _namespace_to_scope(namespace: tuple[str, ...]) -> dict[str, str]:
    """Convert a LangGraph namespace tuple to a Memory Bank scope dict.

    The namespace is expected to have the format:
        ("memories", key1, val1, key2, val2, ...)

    The first element is skipped (it's the prefix). The remaining elements
    are treated as alternating key-value pairs.

    Args:
        namespace: LangGraph namespace tuple.

    Returns:
        Scope dictionary for Memory Bank.

    Raises:
        ValueError: If namespace has fewer than 3 elements or odd key-value pairs.
    """
    if len(namespace) < 3:
        raise ValueError(
            f"Namespace must have at least 3 elements (prefix, key, value), got: {namespace}"
        )
    pairs = namespace[1:]  # skip prefix
    if len(pairs) % 2 != 0:
        raise ValueError(
            f"Namespace key-value pairs must be even-length, got {len(pairs)} elements: {pairs}"
        )
    return {pairs[i]: pairs[i + 1] for i in range(0, len(pairs), 2)}


def _scope_to_namespace(scope: dict[str, str], prefix: str = "memories") -> tuple[str, ...]:
    """Convert a Memory Bank scope dict to a LangGraph namespace tuple.

    Args:
        scope: Memory Bank scope dictionary.
        prefix: Namespace prefix (default: "memories").

    Returns:
        LangGraph namespace tuple.
    """
    parts: list[str] = [prefix]
    for k, v in sorted(scope.items()):
        parts.extend([k, v])
    return tuple(parts)


def _memory_resource_name(
    project_id: str, location: str, engine_id: str, memory_id: str
) -> str:
    """Build the full resource name for a memory."""
    return (
        f"projects/{project_id}/locations/{location}"
        f"/reasoningEngines/{engine_id}/memories/{memory_id}"
    )


def _extract_memory_id(resource_name: str) -> str:
    """Extract the short memory ID from a full resource name."""
    return resource_name.rsplit("/", 1)[-1]


def _memory_to_item(
    memory: dict[str, Any],
    namespace: tuple[str, ...],
) -> Item:
    """Convert a Memory Bank memory dict to a LangGraph Item."""
    resource_name = memory.get("name", "")
    key = _extract_memory_id(resource_name)
    fact = memory.get("fact", "")
    metadata = memory.get("metadata", {})
    scope = memory.get("scope", {})

    # Parse timestamps
    create_time = memory.get("createTime", memory.get("create_time", ""))
    update_time = memory.get("updateTime", memory.get("update_time", ""))
    now = datetime.now(timezone.utc)

    created_at = datetime.fromisoformat(create_time) if create_time else now
    updated_at = datetime.fromisoformat(update_time) if update_time else now

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


def _memory_to_search_item(
    memory_result: dict[str, Any],
    namespace: tuple[str, ...],
) -> SearchItem:
    """Convert a Memory Bank retrieval result to a LangGraph SearchItem."""
    memory = memory_result.get("memory", memory_result)
    resource_name = memory.get("name", "")
    key = _extract_memory_id(resource_name)
    fact = memory.get("fact", "")
    metadata = memory.get("metadata", {})
    scope = memory.get("scope", {})
    distance = memory_result.get("distance")

    # Convert distance to similarity score (lower distance = higher score)
    score = 1.0 / (1.0 + distance) if distance is not None else None

    create_time = memory.get("createTime", memory.get("create_time", ""))
    update_time = memory.get("updateTime", memory.get("update_time", ""))
    now = datetime.now(timezone.utc)

    created_at = datetime.fromisoformat(create_time) if create_time else now
    updated_at = datetime.fromisoformat(update_time) if update_time else now

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
        score=score,
    )


class VertexMemoryBankStore(BaseStore):
    """LangGraph BaseStore backed by Vertex AI Agent Engine Memory Bank.

    This store provides persistent, semantic long-term memory using Google Cloud's
    managed Memory Bank service. It maps LangGraph's namespace/key model to Memory
    Bank's scope-based memory isolation.

    Memory Bank's killer feature is LLM-powered memory extraction: instead of
    storing raw text, it uses an LLM to extract meaningful facts from conversations
    and consolidates them with existing memories over time. This is exposed via
    the ``generate_memories()`` method (beyond the BaseStore interface).

    Args:
        project_id: Google Cloud project ID.
        location: Google Cloud region (e.g., "us-central1").
        reasoning_engine_id: The Agent Engine (Reasoning Engine) instance ID.
        credentials: Optional Google auth credentials. If None, uses ADC.
        namespace_prefix: Prefix for namespaces (default: "memories").

    Example:
        >>> store = VertexMemoryBankStore(
        ...     project_id="my-project",
        ...     location="us-central1",
        ...     reasoning_engine_id="1234567890",
        ... )
        >>> # Store a memory
        >>> store.put(
        ...     ("memories", "user_id", "alice"),
        ...     "pref-1",
        ...     {"fact": "Alice prefers dark mode"},
        ... )
        >>> # Search memories
        >>> results = store.search(
        ...     ("memories", "user_id", "alice"),
        ...     query="What UI preferences does the user have?",
        ... )
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        reasoning_engine_id: str,
        credentials: Credentials | None = None,
        namespace_prefix: str = "memories",
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.reasoning_engine_id = reasoning_engine_id
        self.namespace_prefix = namespace_prefix
        self._credentials = credentials
        self._known_scopes: set[tuple[tuple[str, str], ...]] = set()

        # Build base URL for REST API
        self._base_url = (
            f"https://{location}-aiplatform.googleapis.com/v1beta1"
            f"/projects/{project_id}/locations/{location}"
            f"/reasoningEngines/{reasoning_engine_id}"
        )

    def _get_credentials(self) -> Credentials:
        """Get or refresh credentials."""
        if self._credentials is None:
            self._credentials, _ = google_auth_default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return self._credentials

    def _get_headers(self) -> dict[str, str]:
        """Get authorization headers."""
        creds = self._get_credentials()
        return {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
        }

    def _make_request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the Memory Bank API.

        Args:
            method: HTTP method (GET, POST, DELETE).
            path: URL path relative to base URL.
            json_body: Optional JSON request body.
            params: Optional query parameters.

        Returns:
            Response JSON as dict.
        """
        import json
        import urllib.request

        url = f"{self._base_url}{path}"
        if params:
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query_string}"

        headers = self._get_headers()
        data = json.dumps(json_body).encode() if json_body else None

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req) as response:
                body = response.read().decode()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            logger.error(
                "Memory Bank API error: %s %s → %d: %s",
                method, path, e.code, error_body,
            )
            raise RuntimeError(
                f"Memory Bank API error {e.code}: {error_body}"
            ) from e

    async def _amake_request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Async version of _make_request using asyncio.to_thread."""
        return await asyncio.to_thread(
            self._make_request, method, path, json_body, params
        )

    # ── BaseStore abstract methods ──────────────────────────────────────

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
                raise ValueError(f"Unsupported operation type: {type(op)}")
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
                raise ValueError(f"Unsupported operation type: {type(op)}")
        return results

    # ── Operation handlers (sync) ───────────────────────────────────────

    def _handle_get(self, op: GetOp) -> Item | None:
        """Handle a GetOp by fetching a single memory by ID."""
        resource_name = _memory_resource_name(
            self.project_id, self.location, self.reasoning_engine_id, op.key
        )
        try:
            memory = self._make_request("GET", f"/memories/{op.key}")
            return _memory_to_item(memory, op.namespace)
        except RuntimeError as e:
            if "404" in str(e):
                return None
            raise

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        """Handle a SearchOp by retrieving memories from Memory Bank."""
        try:
            scope = _namespace_to_scope(op.namespace_prefix)
        except ValueError:
            # If namespace is too short for scope extraction, return empty
            logger.debug("Cannot extract scope from namespace: %s", op.namespace_prefix)
            return []

        # Track known scopes
        self._known_scopes.add(tuple(sorted(scope.items())))

        body: dict[str, Any] = {"scope": scope}

        if op.query:
            # Semantic similarity search
            body["similaritySearchParams"] = {
                "searchQuery": op.query,
                "topK": op.limit,
            }
        # If no query, retrieves all memories for scope

        try:
            response = self._make_request("POST", "/memories:retrieve", json_body=body)
        except RuntimeError as e:
            logger.warning("Failed to retrieve memories: %s", e)
            return []

        # Parse response - may be paginated
        retrieved = response.get("retrievedMemories", response.get("memories", []))

        items: list[SearchItem] = []
        namespace = _scope_to_namespace(scope, self.namespace_prefix)

        for mem_result in retrieved:
            items.append(_memory_to_search_item(mem_result, namespace))

        # Apply offset and limit (API may not support offset natively)
        items = items[op.offset : op.offset + op.limit]

        # Apply filter if provided (client-side filtering on metadata)
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
        """Handle a PutOp by creating or deleting a memory."""
        if op.value is None:
            # Delete
            try:
                self._make_request("DELETE", f"/memories/{op.key}")
            except RuntimeError as e:
                if "404" not in str(e):
                    raise
            return

        try:
            scope = _namespace_to_scope(op.namespace)
        except ValueError as e:
            raise ValueError(f"Cannot put without a valid scope namespace: {e}") from e

        # Track known scopes
        self._known_scopes.add(tuple(sorted(scope.items())))

        fact = op.value.get("fact", "")
        if not fact:
            # If no fact provided, serialize the entire value as the fact
            import json
            fact = json.dumps(op.value)

        body: dict[str, Any] = {
            "scope": scope,
            "fact": fact,
        }

        # Include metadata if present
        metadata = op.value.get("metadata")
        if metadata:
            body["metadata"] = metadata

        self._make_request("POST", "/memories", json_body=body)

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Handle a ListNamespacesOp.

        Memory Bank doesn't have a native "list scopes" API, so we return
        the set of scopes we've seen during this session. For a more complete
        listing, we also try to list memories and extract unique scopes.
        """
        namespaces: set[tuple[str, ...]] = set()

        # Add known scopes
        for scope_items in self._known_scopes:
            ns = _scope_to_namespace(dict(scope_items), self.namespace_prefix)
            namespaces.add(ns)

        # Try listing some memories to discover more scopes
        try:
            response = self._make_request("GET", "/memories", params={"pageSize": "100"})
            memories = response.get("memories", [])
            for mem in memories:
                scope = mem.get("scope", {})
                if scope:
                    ns = _scope_to_namespace(scope, self.namespace_prefix)
                    namespaces.add(ns)
                    self._known_scopes.add(tuple(sorted(scope.items())))
        except RuntimeError:
            logger.debug("Could not list memories for namespace discovery")

        result = list(namespaces)

        # Apply match conditions
        if op.match_conditions:
            result = self._filter_namespaces(result, op.match_conditions)

        # Apply max_depth
        if op.max_depth is not None:
            result = [ns[:op.max_depth] for ns in result]
            result = list(set(result))

        # Sort for deterministic output
        result.sort()

        # Apply pagination
        return result[op.offset : op.offset + op.limit]

    # ── Operation handlers (async) ──────────────────────────────────────

    async def _ahandle_get(self, op: GetOp) -> Item | None:
        """Async handle for GetOp."""
        try:
            memory = await self._amake_request("GET", f"/memories/{op.key}")
            return _memory_to_item(memory, op.namespace)
        except RuntimeError as e:
            if "404" in str(e):
                return None
            raise

    async def _ahandle_search(self, op: SearchOp) -> list[SearchItem]:
        """Async handle for SearchOp."""
        return await asyncio.to_thread(self._handle_search, op)

    async def _ahandle_put(self, op: PutOp) -> None:
        """Async handle for PutOp."""
        return await asyncio.to_thread(self._handle_put, op)

    async def _ahandle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Async handle for ListNamespacesOp."""
        return await asyncio.to_thread(self._handle_list_namespaces, op)

    # ── Memory Bank-specific methods (beyond BaseStore) ─────────────────

    def generate_memories(
        self,
        scope: dict[str, str],
        events: list[dict[str, Any]],
        *,
        wait_for_completion: bool = True,
        disable_consolidation: bool = False,
    ) -> list[dict[str, Any]]:
        """Generate memories from conversation events using Memory Bank's LLM extraction.

        This is Memory Bank's killer feature: it uses an LLM to extract meaningful
        facts from conversation turns and consolidates them with existing memories.
        Unlike simple vector stores, this produces clean, deduplicated factual memories.

        Args:
            scope: Scope dict for the generated memories (e.g., {"user_id": "alice"}).
            events: List of conversation events, each with a "content" dict containing
                "role" and "parts" (matching Vertex AI Content format).
            wait_for_completion: Whether to wait for the operation to complete.
            disable_consolidation: If True, skip consolidation with existing memories.

        Returns:
            List of generated memory dicts (each with "memory" and "action" keys).

        Example:
            >>> events = [
            ...     {"content": {"role": "user", "parts": [{"text": "I live in Portland"}]}},
            ...     {"content": {"role": "model", "parts": [{"text": "Nice! Portland is great."}]}},
            ... ]
            >>> results = store.generate_memories(
            ...     scope={"user_id": "alice"},
            ...     events=events,
            ... )
        """
        # Track scope
        self._known_scopes.add(tuple(sorted(scope.items())))

        body: dict[str, Any] = {
            "directContentsSource": {
                "events": events,
            },
            "scope": scope,
        }
        if disable_consolidation:
            body["disableConsolidation"] = True

        response = self._make_request("POST", "/memories:generate", json_body=body)

        # The REST API returns a long-running operation (LRO)
        if response.get("done"):
            inner = response.get("response", {})
            return inner.get("generatedMemories", [])
        elif "generatedMemories" in response:
            return response["generatedMemories"]

        # LRO not done yet — if caller wants to wait, poll it
        if wait_for_completion and "name" in response:
            return self._poll_operation(response["name"])

        # Return operation info for background tracking
        return [{"operation": response}]

    async def agenerate_memories(
        self,
        scope: dict[str, str],
        events: list[dict[str, Any]],
        *,
        wait_for_completion: bool = True,
        disable_consolidation: bool = False,
    ) -> list[dict[str, Any]]:
        """Async version of generate_memories."""
        return await asyncio.to_thread(
            self.generate_memories,
            scope,
            events,
            wait_for_completion=wait_for_completion,
            disable_consolidation=disable_consolidation,
        )

    def _poll_operation(
        self,
        operation_name: str,
        max_attempts: int = 30,
        delay_seconds: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Poll a long-running operation until completion.

        Args:
            operation_name: Full resource name of the operation.
            max_attempts: Maximum poll attempts.
            delay_seconds: Delay between polls.

        Returns:
            List of generated memories.
        """
        import time

        # The operation name is relative to the base Vertex AI URL
        # e.g., projects/p/locations/l/operations/op-id
        # We need to call the operations endpoint
        op_url = f"https://{self.location}-aiplatform.googleapis.com/v1beta1/{operation_name}"

        for _ in range(max_attempts):
            import json
            import urllib.request

            headers = self._get_headers()
            req = urllib.request.Request(op_url, headers=headers, method="GET")

            try:
                with urllib.request.urlopen(req) as resp:
                    result = json.loads(resp.read().decode())
                    if result.get("done"):
                        inner = result.get("response", {})
                        return inner.get("generatedMemories", [])
            except Exception:
                pass

            time.sleep(delay_seconds)

        logger.warning("Operation %s did not complete in time", operation_name)
        return []

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _filter_namespaces(
        namespaces: list[tuple[str, ...]],
        conditions: tuple[MatchCondition, ...],
    ) -> list[tuple[str, ...]]:
        """Filter namespaces by match conditions."""
        result = namespaces
        for cond in conditions:
            if cond.match_type == "prefix":
                prefix = cond.path
                result = [
                    ns for ns in result
                    if len(ns) >= len(prefix)
                    and all(
                        p == "*" or p == n
                        for p, n in zip(prefix, ns)
                    )
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

    def scope_for_namespace(self, namespace: tuple[str, ...]) -> dict[str, str]:
        """Utility to convert a namespace to a scope dict.

        Useful when you need to call generate_memories() with a scope
        derived from a namespace you're already using.
        """
        return _namespace_to_scope(namespace)

    def namespace_for_scope(self, scope: dict[str, str]) -> tuple[str, ...]:
        """Utility to convert a scope dict to a namespace tuple."""
        return _scope_to_namespace(scope, self.namespace_prefix)
