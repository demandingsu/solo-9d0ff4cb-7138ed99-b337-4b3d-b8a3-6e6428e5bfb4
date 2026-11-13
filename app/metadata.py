"""Reconcile machine-readable JSON-LD metadata against notice body text.

The verifier walks the same paginated notice index used by the deadline
checker and collects every JSON-LD ``@graph`` published on either the index
pages or the detail pages.  Project nodes identify a lot through
``hasPart`` references (by ``@id``); lot nodes carry the machine-readable
``lotNo`` and ``bidDeadline``.

The comparison is deliberately conservative:

* a lot is only compared when *both* sides identify the same project and
  lot (a body observation matched by project number + lot number, linked
  to a lot node reached through that project's ``hasPart``);
* conflicting machine-readable declarations for the same lot -- or a
  ``hasPart`` reference that resolves to no node -- make the lot
  *pending review* even when one of the declarations happens to agree;
* any fetch/parse failure marks the whole round *incomplete*: confirmed
  differences are still reported, but the round can never be advertised
  as fully consistent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from .crawler import (
    MAX_DETAIL_PAGES,
    MAX_LIST_PAGES,
    USER_AGENT,
    FetchIssue,
    _fetch,
    canonical_project,
    canonical_section,
    extract_links,
    format_dt,
    normalize_space,
    normalize_url,
    parse_datetime,
    parse_notice_html,
    project_matches,
    section_matches,
)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class JsonLdNode:
    """A single node collected from a JSON-LD ``@graph``."""

    node_id: str | None
    types: list[str]
    project_no: str | None
    lot_no: str | None
    bid_deadline_raw: str | None
    bid_deadline: datetime | None
    bid_deadline_has_time: bool
    has_part: list[str]
    source_url: str
    graph_index: int

    @property
    def is_project(self) -> bool:
        return bool(self.project_no)

    @property
    def is_lot(self) -> bool:
        return bool(self.lot_no or self.bid_deadline_raw is not None)


@dataclass
class GraphDoc:
    url: str
    stage: str  # list | detail
    node_count: int
    ok: bool
    error: str | None = None


@dataclass
class RegistryEntry:
    node: JsonLdNode
    occurrences: list[JsonLdNode] = field(default_factory=list)


@dataclass
class LotDeclaration:
    """One machine-readable deadline declaration for a resolved lot node."""

    node_id: str | None
    lot_no: str | None
    deadline_raw: str | None
    deadline: datetime | None
    has_time: bool
    source_url: str


@dataclass
class BodyObservation:
    """The deadline stated in the human-readable body of an original notice."""

    url: str
    title: str
    project_no: str
    section_no: str
    deadline_raw: str | None
    deadline: datetime | None
    has_time: bool


@dataclass
class MetadataRow:
    lot_no: str | None
    status: str  # consistent | difference | pending
    status_label: str
    reason: str
    body: BodyObservation | None
    declarations: list[LotDeclaration]
    project_id: str | None
    project_sources: list[str]
    reference_ids: list[str]
    body_source_url: str | None
    machine_values: list[str]
    body_value: str | None
    node_source_urls: list[str]


@dataclass
class MetadataOutcome:
    list_url: str
    project_no: str
    complete: bool
    rows: list[MetadataRow]
    failures: list[FetchIssue]
    warnings: list[str]
    searched_pages: list[str]
    checked_detail_urls: list[str]
    graph_docs: list[GraphDoc]
    detail_page_count: int

    # ------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        result = {"consistent": 0, "difference": 0, "pending": 0}
        for row in self.rows:
            result[row.status] = result.get(row.status, 0) + 1
        return result

    def as_dict(self, public_url_fn=None) -> dict[str, Any]:
        def pub(url: str | None) -> str | None:
            if not url:
                return None
            return public_url_fn(url) if public_url_fn else url

        def row_dict(row: MetadataRow) -> dict[str, Any]:
            return {
                "lot_no": row.lot_no,
                "status": row.status,
                "status_label": row.status_label,
                "reason": row.reason,
                "body_value": row.body_value,
                "machine_values": row.machine_values,
                "body_source_url": pub(row.body_source_url),
                "node_source_urls": [pub(url) or url for url in row.node_source_urls],
                "project_sources": [pub(url) or url for url in row.project_sources],
                "project_id": row.project_id,
                "reference_ids": row.reference_ids,
                "declarations": [
                    {
                        "node_id": decl.node_id,
                        "lot_no": decl.lot_no,
                        "deadline_raw": decl.deadline_raw,
                        "deadline": format_dt(decl.deadline, decl.has_time),
                        "source_url": pub(decl.source_url),
                    }
                    for decl in row.declarations
                ],
                "body": (
                    None
                    if row.body is None
                    else {
                        "url": pub(row.body.url),
                        "title": row.body.title,
                        "project_no": row.body.project_no,
                        "section_no": row.body.section_no,
                        "deadline_raw": row.body.deadline_raw,
                        "deadline": format_dt(row.body.deadline, row.body.has_time),
                    }
                ),
            }

        return {
            "input_url": self.list_url,
            "project_no": self.project_no,
            "complete": self.complete,
            "summary": self.counts(),
            "rows": [row_dict(row) for row in self.rows],
            "failures": [
                issue.as_dict() | {"url": pub(issue.url) or issue.url}
                for issue in self.failures
            ],
            "warnings": self.warnings,
            "searched_pages": [pub(url) or url for url in self.searched_pages],
            "checked_detail_urls": [pub(url) or url for url in self.checked_detail_urls],
            "graph_docs": [
                {
                    "url": pub(doc.url) or doc.url,
                    "stage": doc.stage,
                    "node_count": doc.node_count,
                    "ok": doc.ok,
                    "error": doc.error,
                }
                for doc in self.graph_docs
            ],
            "detail_page_count": self.detail_page_count,
        }


# ---------------------------------------------------------------------------
# JSON-LD extraction
# ---------------------------------------------------------------------------


def _node_types(node: dict[str, Any]) -> list[str]:
    raw = node.get("@type")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _string_or_first(value: Any) -> str | None:
    """JSON-LD values may be plain strings or ``{"@value": ...}`` objects."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = value.get("@value")
        if nested is not None:
            return str(nested)
    if isinstance(value, list) and value:
        return _string_or_first(value[0])
    return None


def parse_iso_deadline(value: str | None) -> tuple[datetime | None, bool]:
    """Parse an ISO-8601-ish machine deadline into a naive wall-clock datetime.

    Returns ``(datetime, had_clock_time)``.  Time zone designators are
    accepted but discarded: comparisons use the page's local wall clock,
    matching how body-text dates are parsed elsewhere in the application.
    """

    if not value:
        return None, False
    text = normalize_space(value)
    if not text:
        return None, False
    has_time = bool(re.search(r"T?\s*\d{1,2}\s*[:：]\d{2}", text))
    iso_text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        fallback = parse_datetime(text)
        if fallback is None:
            return None, False
        return fallback[0], fallback[1]
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed, has_time


def extract_graph_nodes(html: str, page_url: str) -> tuple[list[JsonLdNode], list[GraphDoc]]:
    """Collect every node described by JSON-LD blocks on one page.

    A malformed JSON-LD block does not abort the round: the block is
    reported as a graph document with its error so the caller can flag the
    round instead of silently ignoring the broken metadata.
    """

    soup = BeautifulSoup(html, "html.parser")
    nodes: list[JsonLdNode] = []
    docs: list[GraphDoc] = []

    for index, script in enumerate(
        soup.find_all("script", attrs={"type": True})
    ):
        if "application/ld+json" not in str(script.get("type") or "").lower():
            continue
        raw_text = script.string or script.get_text() or ""
        if not raw_text.strip():
            docs.append(GraphDoc(url=page_url, stage="", node_count=0, ok=False, error="空的 JSON-LD 脚本块"))
            continue
        try:
            payload = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            docs.append(
                GraphDoc(
                    url=page_url,
                    stage="",
                    node_count=0,
                    ok=False,
                    error=f"JSON-LD 解析失败：{exc.__class__.__name__}",
                )
            )
            continue

        graph = payload.get("@graph") if isinstance(payload, dict) else None
        entries = graph if isinstance(graph, list) else (
            payload if isinstance(payload, list) else [payload]
        )
        page_nodes: list[JsonLdNode] = []
        for graph_index, item in enumerate(entries):
            if not isinstance(item, dict):
                continue
            node_id = _string_or_first(item.get("@id"))
            if node_id:
                node_id = urljoin(page_url, node_id)
            deadline_raw = _string_or_first(item.get("bidDeadline"))
            deadline, has_time = parse_iso_deadline(deadline_raw)
            has_part = []
            for ref in _as_list(item.get("hasPart")):
                if isinstance(ref, dict):
                    ref_id = _string_or_first(ref.get("@id"))
                else:
                    ref_id = ref if isinstance(ref, str) else None
                if ref_id:
                    has_part.append(urljoin(page_url, ref_id))
            node = JsonLdNode(
                node_id=node_id,
                types=_node_types(item),
                project_no=_string_or_first(item.get("projectNo")),
                lot_no=_string_or_first(item.get("lotNo")),
                bid_deadline_raw=deadline_raw,
                bid_deadline=deadline,
                bid_deadline_has_time=has_time,
                has_part=has_part,
                source_url=page_url,
                graph_index=graph_index,
            )
            page_nodes.append(node)
        nodes.extend(page_nodes)
        docs.append(
            GraphDoc(url=page_url, stage="", node_count=len(page_nodes), ok=True)
        )
    return nodes, docs


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _deadlines_equal(left: datetime, left_has_time: bool, right: datetime, right_has_time: bool) -> bool:
    if left_has_time and right_has_time:
        return left == right
    if not left_has_time and not right_has_time:
        return left.date() == right.date()
    # One side carries a clock time and the other only a date: compare the
    # calendar day so a date-only statement is not falsely reported as a
    # difference on its missing (midnight-defaulted) time component.
    return left.date() == right.date()


def _declaration(node: JsonLdNode) -> LotDeclaration:
    return LotDeclaration(
        node_id=node.node_id,
        lot_no=node.lot_no,
        deadline_raw=node.bid_deadline_raw,
        deadline=node.bid_deadline,
        has_time=node.bid_deadline_has_time,
        source_url=node.source_url,
    )


async def check_metadata(
    list_url: str,
    project_no: str,
    *,
    client_factory=None,
) -> MetadataOutcome:
    """Crawl the index and reconcile JSON-LD deadlines with body text."""

    try:
        start_url = normalize_url(list_url)
    except ValueError as exc:
        return MetadataOutcome(
            list_url=list_url,
            project_no=project_no,
            complete=False,
            rows=[],
            failures=[FetchIssue(url=list_url, reason=str(exc), stage="validation")],
            warnings=[],
            searched_pages=[],
            checked_detail_urls=[],
            graph_docs=[],
            detail_page_count=0,
        )

    target_project = canonical_project(project_no)
    issues: list[FetchIssue] = []
    warnings: list[str] = []
    searched_pages: list[str] = []
    checked_detail_urls: list[str] = []
    graph_docs: list[GraphDoc] = []
    all_nodes: list[JsonLdNode] = []
    body_observations: list[BodyObservation] = []
    seen_pages: set[str] = set()
    seen_details: set[str] = set()
    queue: list[str] = [start_url]
    detail_count = 0
    incomplete = False

    timeout = httpx.Timeout(15, connect=8)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)

    if client_factory is None:
        client = httpx.AsyncClient(
            follow_redirects=True, timeout=timeout, headers=headers, limits=limits
        )
        owns_client = True
    else:
        client = client_factory()
        owns_client = False

    try:
        while queue:
            if len(searched_pages) >= MAX_LIST_PAGES:
                incomplete = True
                warnings.append(f"公告列表超过 {MAX_LIST_PAGES} 页，已停止继续翻页，本轮核对不完整。")
                break
            page_url = queue.pop(0)
            if page_url in seen_pages:
                continue
            seen_pages.add(page_url)
            searched_pages.append(page_url)

            html, error, final_page_url = await _fetch(client, page_url, "list")
            if html is None:
                issues.append(
                    FetchIssue(url=page_url, reason=f"公告列表页抓取失败：{error}", stage="list")
                )
                incomplete = True
                continue
            if final_page_url != page_url:
                seen_pages.add(final_page_url)
                if final_page_url not in searched_pages:
                    searched_pages.append(final_page_url)

            try:
                page_nodes, page_docs = extract_graph_nodes(html, final_page_url)
                links = extract_links(html, final_page_url)
            except Exception as exc:
                issues.append(
                    FetchIssue(
                        url=page_url,
                        reason=f"公告列表页解析失败：{exc.__class__.__name__}",
                        stage="list",
                    )
                )
                incomplete = True
                continue
            for doc in page_docs:
                doc.stage = "list"
                if not doc.ok:
                    warnings.append(f"列表页 {final_page_url} 存在无法解析的 JSON-LD：{doc.error}")
            graph_docs.extend(page_docs)
            all_nodes.extend(page_nodes)

            for next_url in links.pagination:
                if next_url not in seen_pages and next_url not in queue:
                    queue.append(next_url)

            candidates: list[str] = []
            for detail_url in links.detail_urls:
                if detail_url in seen_details or detail_url in candidates:
                    continue
                candidates.append(detail_url)

            if detail_count + len(candidates) > MAX_DETAIL_PAGES:
                incomplete = True
                warnings.append(
                    f"公告详情超过 {MAX_DETAIL_PAGES} 页，已停止继续核对，本轮核对不完整。"
                )
                candidates = candidates[: max(0, MAX_DETAIL_PAGES - detail_count)]

            for detail_url in candidates:
                seen_details.add(detail_url)
                checked_detail_urls.append(detail_url)
                detail_count += 1
                detail_html, detail_error, final_detail_url = await _fetch(
                    client, detail_url, "detail"
                )
                if detail_html is None:
                    issues.append(
                        FetchIssue(
                            url=detail_url,
                            reason=f"公告详情页抓取失败：{detail_error}",
                            stage="detail",
                        )
                    )
                    incomplete = True
                    continue
                source_url = final_detail_url or detail_url
                try:
                    detail_nodes, detail_docs = extract_graph_nodes(detail_html, source_url)
                    parsed = parse_notice_html(detail_html, source_url)
                except Exception as exc:
                    issues.append(
                        FetchIssue(
                            url=detail_url,
                            reason=f"公告详情页解析失败：{exc.__class__.__name__}",
                            stage="detail",
                        )
                    )
                    incomplete = True
                    continue
                for doc in detail_docs:
                    doc.stage = "detail"
                    if not doc.ok:
                        warnings.append(f"详情页 {source_url} 存在无法解析的 JSON-LD：{doc.error}")
                graph_docs.extend(detail_docs)
                all_nodes.extend(detail_nodes)

                # Only single-lot original notices establish the body side
                # of the comparison; follow-up notices describe changes and
                # are outside this metadata check.
                if parsed.notice_type != "original":
                    continue
                if not project_matches(project_no, parsed.project_candidates):
                    continue
                if not parsed.section_candidates:
                    issues.append(
                        FetchIssue(
                            url=source_url,
                            reason="原公告未解析出标段号，无法与机读标段节点对应",
                            stage="parse",
                        )
                    )
                    incomplete = True
                    continue
                section_value = parsed.section_candidates[0]
                body_observations.append(
                    BodyObservation(
                        url=source_url,
                        title=parsed.title,
                        project_no=parsed.project_candidates[0] if parsed.project_candidates else project_no,
                        section_no=section_value,
                        deadline_raw=parsed.deadline.raw,
                        deadline=parsed.deadline.value,
                        has_time=parsed.deadline.has_time,
                    )
                )
    finally:
        if owns_client:
            await client.aclose()

    rows = _reconcile(target_project, all_nodes, body_observations)

    if not rows and not issues:
        issues.append(
            FetchIssue(
                url=start_url,
                reason="已检索全部列表页，但未找到该项目可核对的标段数据",
                stage="missing_project",
            )
        )
        incomplete = True

    return MetadataOutcome(
        list_url=start_url,
        project_no=project_no,
        complete=not incomplete,
        rows=rows,
        failures=issues,
        warnings=warnings,
        searched_pages=searched_pages,
        checked_detail_urls=checked_detail_urls,
        graph_docs=graph_docs,
        detail_page_count=detail_count,
    )


def _reconcile(
    target_project: str,
    nodes: Iterable[JsonLdNode],
    bodies: list[BodyObservation],
) -> list[MetadataRow]:
    # ---- Index nodes by @id and by canonical project number -------------
    registry: dict[str, RegistryEntry] = {}
    project_nodes: list[JsonLdNode] = []
    lot_nodes: list[JsonLdNode] = []

    for node in nodes:
        if node.node_id and node.node_id in registry:
            registry[node.node_id].occurrences.append(node)
        elif node.node_id:
            registry[node.node_id] = RegistryEntry(node=node, occurrences=[node])
        if node.project_no and canonical_project(node.project_no) == target_project:
            project_nodes.append(node)
        if node.lot_no is not None or node.bid_deadline_raw is not None:
            lot_nodes.append(node)

    rows: list[MetadataRow] = []
    handled_node_ids: set[str] = set()
    handled_lot_keys: set[str] = set()
    handled_refs: set[tuple[str, str]] = set()

    # De-duplicate project nodes: the same project @id may be repeated in
    # the @graph of several index pages, but its hasPart graph is processed
    # only once.
    unique_projects: list[JsonLdNode] = []
    seen_project_ids: set[str] = set()
    for project in project_nodes:
        key = project.node_id or f"anon:{project.source_url}:{project.graph_index}"
        if key in seen_project_ids:
            continue
        seen_project_ids.add(key)
        unique_projects.append(project)

    # Map a canonical lot number to every node id carrying that lotNo, so
    # nodes that model the same lot under different @ids are reconciled as
    # one machine-readable statement.
    lot_ids_by_number: dict[str, set[str]] = {}
    for lot in lot_nodes:
        key = canonical_section(lot.lot_no or "")
        if key and lot.node_id:
            lot_ids_by_number.setdefault(key, set()).add(lot.node_id)

    # ---- Rows driven by the project's hasPart graph ---------------------
    for project in unique_projects:
        entry = registry.get(project.node_id) if project.node_id else None
        project_sources = (
            list(dict.fromkeys(item.source_url for item in entry.occurrences))
            if entry is not None
            else [project.source_url]
        )

        for ref_id in project.has_part:
            if (project.node_id or "", ref_id) in handled_refs:
                continue
            handled_refs.add((project.node_id or "", ref_id))
            entry = registry.get(ref_id)
            if entry is None:
                # A dangling reference is itself a pending-review item:
                # nothing it might agree with can make the graph consistent.
                rows.append(
                    MetadataRow(
                        lot_no=None,
                        status="pending",
                        status_label="待核",
                        reason=f"项目节点的 hasPart 引用 {ref_id} 在全部列表页与详情页的 @graph 中均无对应标段节点（引用悬空）。",
                        body=None,
                        declarations=[],
                        project_id=project.node_id,
                        project_sources=list(dict.fromkeys(project_sources)),
                        reference_ids=[ref_id],
                        body_source_url=None,
                        machine_values=[],
                        body_value=None,
                        node_source_urls=[],
                    )
                )
                continue

            lot = entry.node
            handled_node_ids.add(ref_id)
            lot_key = canonical_section(lot.lot_no or "")
            # Several referenced node ids may model the same lot number:
            # fold them into the row already built for that lot instead of
            # emitting duplicate rows for one lot.
            existing_row = (
                next(
                    (
                        row
                        for row in rows
                        if lot_key and canonical_section(row.lot_no or "") == lot_key
                    ),
                    None,
                )
                if lot_key
                else None
            )
            if lot_key and lot_key in handled_lot_keys and existing_row is not None:
                existing_row.reference_ids.append(ref_id)
                existing_row.declarations.extend(
                    _declaration(item) for item in entry.occurrences
                )
                existing_row.declarations = list(
                    {
                        (decl.node_id, decl.deadline_raw, decl.source_url): decl
                        for decl in existing_row.declarations
                    }.values()
                )
                existing_row.node_source_urls = list(
                    dict.fromkeys(
                        existing_row.node_source_urls
                        + [item.source_url for item in entry.occurrences]
                    )
                )
                existing_row.machine_values = [
                    format_dt(decl.deadline, decl.has_time)
                    if decl.deadline
                    else (decl.deadline_raw or "（无法解析）")
                    for decl in existing_row.declarations
                ]
                # Re-evaluate the merged row now that declarations changed.
                merged = _build_row_for_lot(
                    lot=lot,
                    declarations=existing_row.declarations,
                    project=project,
                    project_sources=project_sources,
                    reference_ids=existing_row.reference_ids,
                    bodies=bodies,
                )
                existing_row.status = merged.status
                existing_row.status_label = merged.status_label
                existing_row.reason = merged.reason
                continue
            handled_lot_keys.add(lot_key)
            declarations = [_declaration(item) for item in entry.occurrences]
            # The same lot number may be modelled under several @id nodes:
            # merge only the sibling nodes this same project references, so
            # a cross-node conflict can never be hidden -- without letting
            # another project's same-number lot leak into the statement.
            project_ref_ids = set(project.has_part)
            for sibling_id in lot_ids_by_number.get(lot_key, set()):
                if sibling_id == ref_id or sibling_id not in project_ref_ids:
                    continue
                sibling_entry = registry.get(sibling_id)
                if sibling_entry is not None:
                    declarations.extend(
                        _declaration(item) for item in sibling_entry.occurrences
                    )
                    handled_node_ids.add(sibling_id)
            declarations = list(
                {
                    (
                        decl.node_id,
                        decl.deadline_raw,
                        decl.source_url,
                    ): decl
                    for decl in declarations
                }.values()
            )
            row = _build_row_for_lot(
                lot=lot,
                declarations=declarations,
                project=project,
                project_sources=project_sources,
                reference_ids=[ref_id],
                bodies=bodies,
            )
            rows.append(row)

    # ---- Lot nodes that no project node references ----------------------
    referenced_ids: set[str] = set()
    for node in nodes:
        if node.project_no:
            referenced_ids.update(node.has_part)

    for lot in lot_nodes:
        if lot.node_id and lot.node_id in handled_node_ids:
            continue
        # A lot that any project node -- including another project's --
        # references belongs to that project's graph and must not leak into
        # this project's result.  Only genuinely orphaned nodes are kept.
        if lot.node_id and lot.node_id in referenced_ids:
            continue
        lot_key = canonical_section(lot.lot_no or "")
        if not lot_key or lot_key in handled_lot_keys:
            continue
        related_body = next(
            (
                body
                for body in bodies
                if section_matches(body.section_no, [lot.lot_no or ""])
            ),
            None,
        )
        # Without a body observation for the requested project's lot there
        # is no evidence the orphan node belongs to this project at all.
        if related_body is None:
            continue
        handled_lot_keys.add(lot_key)
        if lot.node_id:
            handled_node_ids.add(lot.node_id)
        entry = registry.get(lot.node_id) if lot.node_id else None
        declarations = (
            [_declaration(item) for item in entry.occurrences]
            if entry is not None
            else [_declaration(lot)]
        )
        rows.append(
            _build_row_for_lot(
                lot=lot,
                declarations=declarations,
                project=None,
                project_sources=[],
                reference_ids=[],
                bodies=bodies,
                unlinked=True,
            )
        )

    # ---- Body observations that the graph never links to ----------------
    for body in bodies:
        body_key = canonical_section(body.section_no)
        if any(
            row.body is not None
            and canonical_section(row.body.section_no) == body_key
            for row in rows
        ):
            continue
        rows.append(
            MetadataRow(
                lot_no=body.section_no,
                status="pending",
                status_label="待核",
                reason="原公告正文载明了该标段，但项目节点的 hasPart 未引用任何对应的机读标段节点，无法进行两侧核对。",
                body=body,
                declarations=[],
                project_id=None,
                project_sources=[],
                reference_ids=[],
                body_source_url=body.url,
                machine_values=[],
                body_value=format_dt(body.deadline, body.has_time) if body.deadline else None,
                node_source_urls=[],
            )
        )

    rows.sort(key=lambda row: (row.status != "difference", row.status != "pending", row.lot_no or "~"))
    return rows


def _build_row_for_lot(
    *,
    lot: JsonLdNode,
    declarations: list[LotDeclaration],
    project: JsonLdNode | None,
    project_sources: list[str],
    reference_ids: list[str],
    bodies: list[BodyObservation],
    unlinked: bool = False,
) -> MetadataRow:
    lot_no = lot.lot_no
    machine_values = [
        format_dt(decl.deadline, decl.has_time) if decl.deadline else (decl.deadline_raw or "（无法解析）")
        for decl in declarations
    ]
    node_sources = list(dict.fromkeys(decl.source_url for decl in declarations))

    matching_bodies = [
        body for body in bodies if section_matches(body.section_no, [lot_no or ""])
    ]
    body = matching_bodies[0] if matching_bodies else None
    body_value = (
        format_dt(body.deadline, body.has_time) if body and body.deadline else None
    )

    # Duplicate nodes for the same @id with conflicting declarations: the
    # machine-readable side itself is unreliable, so the lot stays pending
    # even if one of the duplicates matches the body.
    comparable = [decl for decl in declarations if decl.deadline is not None]
    distinct_deadlines = {decl.deadline for decl in comparable}
    conflict = len(distinct_deadlines) > 1
    unparseable = any(decl.deadline is None for decl in declarations)

    if unlinked:
        status = "pending"
        reason = "该机读标段节点未被目标项目节点的 hasPart 引用，跨页关联不成立，不能据此判定一致。"
    elif conflict:
        status = "pending"
        reason = (
            "同一标段号在 @graph 中存在互相冲突的 bidDeadline 声明（同一 @id 重复出现或被建模为多个节点），"
            "机读侧结论不可靠，需人工核实发布内容。"
        )
    elif unparseable:
        status = "pending"
        reason = "机读标段节点的 bidDeadline 无法解析为有效时间，无法与正文时间比较。"
    elif body is None:
        status = "pending"
        reason = "机读标段节点存在，但未找到同项目同标段号的原公告正文，缺少正文侧截止时间用于核对。"
    elif body.deadline is None:
        status = "pending"
        reason = "原公告正文未解析出明确的投标截止时间，无法与机读声明比较。"
    elif not comparable:
        status = "pending"
        reason = "机读标段节点未提供可解析的 bidDeadline。"
    else:
        if any(
            _deadlines_equal(
                body.deadline, body.has_time, decl.deadline, decl.has_time
            )
            for decl in comparable
        ) and len(distinct_deadlines) == 1:
            status = "consistent"
            reason = "项目节点经 hasPart 关联到该标段节点，且机读 bidDeadline 与原公告正文载明的截止时间一致。"
        else:
            status = "difference"
            reason = "项目与标段均对应，但机读 bidDeadline 与原公告正文载明的截止时间不一致，请修正发布内容。"

    return MetadataRow(
        lot_no=lot_no,
        status=status,
        status_label={"consistent": "一致", "difference": "差异", "pending": "待核"}[status],
        reason=reason,
        body=body,
        declarations=declarations,
        project_id=project.node_id if project else None,
        project_sources=list(dict.fromkeys(project_sources)),
        reference_ids=reference_ids,
        body_source_url=body.url if body else None,
        machine_values=machine_values,
        body_value=body_value,
        node_source_urls=node_sources,
    )
