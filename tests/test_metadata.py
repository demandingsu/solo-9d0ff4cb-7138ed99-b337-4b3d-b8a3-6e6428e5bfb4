"""Regression tests for JSON-LD metadata reconciliation.

Run with:  python -m unittest discover -s tests -v
Uses httpx's in-process mock transport, so no network or demo container is
required.
"""

from __future__ import annotations

import unittest

import httpx

from app.metadata import check_metadata, extract_graph_nodes, parse_iso_deadline


def project_graph(project_no: str, lot_ids: list[str], dangling: list[str] | None = None) -> str:
    parts = ",".join(f'{{"@id": "{ref}"}}' for ref in [*lot_ids, *(dangling or [])])
    return (
        '<script type="application/ld+json">'
        '{"@context":"https://schema.org","@graph":['
        f'{{"@id":"/p/{project_no}#project","@type":"Project",'
        f'"projectNo":"{project_no}","hasPart":[{parts}]}}'
        "]} </script>"
    )


def lot_graph(lot_id: str, lot_no: str, deadlines: list[str]) -> str:
    nodes = ",".join(
        f'{{"@id":"{lot_id}","@type":"TenderLot","lotNo":"{lot_no}",'
        f'"bidDeadline":"{deadline}"}}'
        for deadline in deadlines
    )
    return f'<script type="application/ld+json">{{"@graph":[{nodes}]}}</script>'


def original_detail(project: str, section: str, deadline_text: str, graph: str = "") -> str:
    return f"""
    <html><body>{graph}
    <article class="notice-detail" data-project-no="{project}" data-section-no="{section}"
             data-notice-type="original" data-publish-time="2026-10-01 09:00">
      <h1>{project} 标段 {section} 招标公告</h1>
      <p>项目编号：{project}；标段编号：{section}。</p>
      <p>投标截止时间：{deadline_text}。</p>
    </article></body></html>
    """


def listing(rows: str, next_href: str | None = None, graph: str = "") -> str:
    pager = f'<a rel="next" href="{next_href}">下一页</a>' if next_href else ""
    return f"<html><body>{graph}<main><ul>{rows}</ul>{pager}</main></body></html>"


def detail_link(href: str, project: str, section: str) -> str:
    return (
        f'<li><a class="notice-link" data-project-no="{project}" '
        f'data-section-no="{section}" href="{href}">公告</a></li>'
    )


LOT_1 = "/d/p1-lot1#lot-01"
LOT_2 = "/d/p1-lot2#lot-02"


class MetadataCrawlTests(unittest.IsolatedAsyncioTestCase):
    def make_factory(self, routes: dict[str, str], failing: set[str] | None = None):
        failing = failing or set()

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path in failing:
                return httpx.Response(503, text="service unavailable")
            if path not in routes:
                return httpx.Response(404, text="not found")
            return httpx.Response(
                200, text=routes[path], headers={"content-type": "text/html"}
            )

        def factory():
            return httpx.AsyncClient(
                transport=httpx.MockTransport(handler), follow_redirects=True
            )

        return factory

    async def test_cross_page_reference_resolves_and_matches(self):
        # Project node on the index page; lot node only on a detail page.
        routes = {
            "/list": listing(
                detail_link("/d/lot1", "P-1", "01") + detail_link("/d/lot2", "P-1", "02"),
                graph=project_graph("P-1", [LOT_1, LOT_2]),
            ),
            "/d/lot1": original_detail(
                "P-1", "01", "2026年11月02日 09:30",
                lot_graph(LOT_1, "01", ["2026-11-02T09:30:00+08:00"]),
            ),
            "/d/lot2": original_detail(
                "P-1", "02", "2026年11月03日 09:30",
                lot_graph(LOT_2, "02", ["2026-11-03T09:30:00+08:00"]),
            ),
        }
        outcome = await check_metadata(
            "http://test/list", "P-1", client_factory=self.make_factory(routes)
        )
        result = outcome.as_dict()
        self.assertTrue(result["complete"])
        self.assertEqual(result["summary"], {"consistent": 2, "difference": 0, "pending": 0})
        self.assertEqual({row["lot_no"] for row in result["rows"]}, {"01", "02"})

    async def test_deadline_difference_is_reported_with_both_sources(self):
        routes = {
            "/list": listing(
                detail_link("/d/lot1", "P-1", "01"),
                graph=project_graph("P-1", [LOT_1]),
            ),
            "/d/lot1": original_detail(
                "P-1", "01", "2026年11月02日 09:30",
                lot_graph(LOT_1, "01", ["2026-11-05T10:00:00+08:00"]),
            ),
        }
        result = (
            await check_metadata(
                "http://test/list", "P-1", client_factory=self.make_factory(routes)
            )
        ).as_dict()
        self.assertTrue(result["complete"])
        self.assertEqual(result["summary"], {"consistent": 0, "difference": 1, "pending": 0})
        row = result["rows"][0]
        self.assertEqual(row["machine_values"], ["2026-11-05 10:00"])
        self.assertEqual(row["body_value"], "2026-11-02 09:30")
        self.assertTrue(row["body_source_url"])
        self.assertTrue(row["node_source_urls"])

    async def test_conflicting_machine_declarations_stay_pending_even_if_one_matches(self):
        routes = {
            "/list": listing(
                detail_link("/d/lot1", "P-1", "01"),
                graph=project_graph("P-1", [LOT_1]),
            ),
            "/d/lot1": original_detail(
                "P-1", "01", "2026年11月02日 09:30",
                # One duplicate matches the body, one does not: still pending.
                lot_graph(LOT_1, "01", ["2026-11-02T09:30:00+08:00", "2026-11-09T09:30:00+08:00"]),
            ),
        }
        result = (
            await check_metadata(
                "http://test/list", "P-1", client_factory=self.make_factory(routes)
            )
        ).as_dict()
        self.assertEqual(result["summary"], {"consistent": 0, "difference": 0, "pending": 1})
        self.assertIn("冲突", result["rows"][0]["reason"])

    async def test_conflicting_nodes_with_distinct_ids_stay_pending(self):
        # The same lot number is modelled under two @id nodes, both
        # referenced by the project's hasPart, with conflicting deadlines.
        alt_lot = "/d/lot1-alt#lot-01"
        routes = {
            "/list": listing(
                detail_link("/d/lot1", "P-1", "01"),
                graph=project_graph("P-1", [LOT_1, alt_lot]),
            ),
            "/d/lot1": original_detail(
                "P-1", "01", "2026年11月02日 09:30",
                lot_graph(LOT_1, "01", ["2026-11-02T09:30:00+08:00"])
                + lot_graph(alt_lot, "01", ["2026-11-20T09:30:00+08:00"]),
            ),
        }
        result = (
            await check_metadata(
                "http://test/list", "P-1", client_factory=self.make_factory(routes)
            )
        ).as_dict()
        self.assertEqual(result["summary"], {"consistent": 0, "difference": 0, "pending": 1})
        # The single lot row carries both conflicting declarations and both refs.
        row = result["rows"][0]
        self.assertEqual(len(row["declarations"]), 2)
        self.assertEqual(len(row["reference_ids"]), 2)
        self.assertIn("冲突", row["reason"])

    async def test_dangling_haspart_is_pending(self):
        routes = {
            "/list": listing(
                detail_link("/d/lot1", "P-1", "01"),
                graph=project_graph("P-1", [LOT_1], dangling=["/d/missing#lot-99"]),
            ),
            "/d/lot1": original_detail(
                "P-1", "01", "2026年11月02日 09:30",
                lot_graph(LOT_1, "01", ["2026-11-02T09:30:00+08:00"]),
            ),
        }
        result = (
            await check_metadata(
                "http://test/list", "P-1", client_factory=self.make_factory(routes)
            )
        ).as_dict()
        self.assertEqual(result["summary"]["pending"], 1)
        self.assertIn("悬空", result["rows"][0]["reason"])
        # A confirmed-consistent lot can coexist with the dangling row.
        self.assertEqual(result["summary"]["consistent"], 1)

    async def test_detail_fetch_failure_makes_round_incomplete_but_keeps_differences(self):
        routes = {
            "/list": listing(
                detail_link("/d/lot1", "P-1", "01") + detail_link("/d/lot2", "P-1", "02"),
                graph=project_graph("P-1", [LOT_1, LOT_2]),
            ),
            "/d/lot1": original_detail(
                "P-1", "01", "2026年11月02日 09:30",
                lot_graph(LOT_1, "01", ["2026-11-05T10:00:00+08:00"]),
            ),
            "/d/lot2": original_detail(
                "P-1", "02", "2026年11月03日 09:30",
                lot_graph(LOT_2, "02", ["2026-11-03T09:30:00+08:00"]),
            ),
        }
        outcome = await check_metadata(
            "http://test/list",
            "P-1",
            client_factory=self.make_factory(routes, failing={"/d/lot2"}),
        )
        result = outcome.as_dict()
        self.assertFalse(result["complete"])
        # The difference found on lot 01 survives...
        self.assertEqual(result["summary"]["difference"], 1)
        # ...but the round must never be advertised as fully consistent.
        self.assertGreater(result["summary"]["pending"] + result["summary"]["difference"], 0)
        failed = [f["url"] for f in result["failures"]]
        self.assertIn("http://test/d/lot2", failed)

    async def test_other_projects_lot_nodes_do_not_leak_into_result(self):
        other_lot = "/d/p2-lot#lot-A"
        routes = {
            "/list": listing(
                detail_link("/d/lot1", "P-1", "01"),
                graph=(
                    project_graph("P-1", [LOT_1])
                    + project_graph("P-2", [other_lot])
                ),
            ),
            "/d/lot1": original_detail(
                "P-1", "01", "2026年11月02日 09:30",
                lot_graph(LOT_1, "01", ["2026-11-02T09:30:00+08:00"])
                + lot_graph(other_lot, "A", ["2026-12-01T09:30:00+08:00"]),
            ),
        }
        result = (
            await check_metadata(
                "http://test/list", "P-1", client_factory=self.make_factory(routes)
            )
        ).as_dict()
        self.assertEqual(result["summary"], {"consistent": 1, "difference": 0, "pending": 0})
        self.assertEqual([row["lot_no"] for row in result["rows"]], ["01"])

    async def test_body_without_machine_node_is_pending(self):
        routes = {
            "/list": listing(
                detail_link("/d/lot1", "P-1", "01"),
                graph=project_graph("P-1", []),
            ),
            "/d/lot1": original_detail("P-1", "01", "2026年11月02日 09:30"),
        }
        result = (
            await check_metadata(
                "http://test/list", "P-1", client_factory=self.make_factory(routes)
            )
        ).as_dict()
        self.assertEqual(result["summary"], {"consistent": 0, "difference": 0, "pending": 1})
        self.assertIn("hasPart", result["rows"][0]["reason"])

    async def test_malformed_jsonld_does_not_abort_round(self):
        broken = '<script type="application/ld+json">{"@graph": [oops</script>'
        routes = {
            "/list": listing(
                detail_link("/d/lot1", "P-1", "01"),
                graph=broken + project_graph("P-1", [LOT_1]),
            ),
            "/d/lot1": original_detail(
                "P-1", "01", "2026年11月02日 09:30",
                lot_graph(LOT_1, "01", ["2026-11-02T09:30:00+08:00"]),
            ),
        }
        outcome = await check_metadata(
            "http://test/list", "P-1", client_factory=self.make_factory(routes)
        )
        result = outcome.as_dict()
        self.assertTrue(result["complete"])
        self.assertTrue(any(not doc["ok"] for doc in result["graph_docs"]))
        self.assertEqual(result["summary"]["consistent"], 1)

    async def test_unknown_project_is_incomplete(self):
        routes = {
            "/list": listing(
                detail_link("/d/lot1", "P-1", "01"),
                graph=project_graph("P-1", [LOT_1]),
            ),
            "/d/lot1": original_detail(
                "P-1", "01", "2026年11月02日 09:30",
                lot_graph(LOT_1, "01", ["2026-11-02T09:30:00+08:00"]),
            ),
        }
        result = (
            await check_metadata(
                "http://test/list", "P-NOPE", client_factory=self.make_factory(routes)
            )
        ).as_dict()
        self.assertFalse(result["complete"])
        self.assertEqual(result["rows"], [])
        self.assertTrue(result["failures"])


class MetadataParseTests(unittest.TestCase):
    def test_iso_deadline_timezone_is_naive_wallclock(self):
        dt, has_time = parse_iso_deadline("2026-11-02T09:30:00+08:00")
        self.assertIsNotNone(dt)
        self.assertIsNone(dt.tzinfo)
        self.assertEqual((dt.year, dt.month, dt.day, dt.hour, dt.minute), (2026, 11, 2, 9, 30))
        self.assertTrue(has_time)

    def test_iso_date_only_has_no_clock_time(self):
        dt, has_time = parse_iso_deadline("2026-11-02")
        self.assertIsNotNone(dt)
        self.assertFalse(has_time)

    def test_graph_node_ids_are_resolved_against_page_url(self):
        html = (
            '<script type="application/ld+json">{"@graph":['
            '{"@id":"/n/1#lot","@type":"TenderLot","lotNo":"01",'
            '"bidDeadline":"2026-11-02T09:30:00Z"}'
            "]}</script>"
        )
        nodes, docs = extract_graph_nodes(html, "http://test/notices?page=2")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].node_id, "http://test/n/1#lot")
        self.assertEqual(nodes[0].lot_no, "01")
        self.assertTrue(docs[0].ok)

    def test_relative_haspart_is_resolved(self):
        html = (
            '<script type="application/ld+json">{"@graph":['
            '{"@id":"/p#project","@type":"Project","projectNo":"P-1",'
            '"hasPart":[{"@id":"/n/1#lot"}]}'
            "]}</script>"
        )
        nodes, _ = extract_graph_nodes(html, "http://test/notices")
        self.assertEqual(nodes[0].has_part, ["http://test/n/1#lot"])


if __name__ == "__main__":
    unittest.main()
