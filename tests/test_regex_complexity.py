"""Regression tests for GHSA-3r75-qcwc-78f2.

Regexes that run over attacker-controlled input must not backtrack
quadratically. Each hostile input below took 6-17 s before the fix and takes
a few milliseconds after it, so a one second budget separates the two without
depending on machine speed.
"""

import time

import pytest

from fastfeedparser import parse
from fastfeedparser.main import _extract_meta_refresh_url, _fix_malformed_xml_bytes

_BUDGET_SECONDS = 1.0

# A utf-16 declaration on bytes that are not utf-16 routes the whole document
# through _fix_malformed_xml_bytes.
_UTF16_DECL = b'<?xml version="1.0" encoding="utf-16"?>'


def _meta_refresh_page(content: str) -> str:
    return (
        f'<html><head><meta http-equiv="refresh" content="{content}">'
        "</head><body></body></html>"
    )


def _parse_ignoring_invalid(source):
    try:
        parse(source)
    except ValueError:
        pass


_HOSTILE_INPUTS = [
    pytest.param(
        _parse_ignoring_invalid,
        _UTF16_DECL
        + b'<rss version="2.0"><channel><title>t</title>'
        + b" " * 40_000
        + b"</channel></rss>",
        id="whitespace-run-bytes",
    ),
    pytest.param(
        _parse_ignoring_invalid,
        '<rss version="2.0"><rss:channel><title>t</title>'
        + " " * 40_000
        + "</rss:channel></rss>",
        id="whitespace-run-str",
    ),
    pytest.param(
        _parse_ignoring_invalid,
        _UTF16_DECL + b'<feed><link href="x">' + b"\n" * 40_000 + b"x</feed>",
        id="newline-run-after-unclosed-link",
    ),
    pytest.param(
        _parse_ignoring_invalid,
        _UTF16_DECL + b"<rss>" + b"<link" * 20_000,
        id="link-starts-without-close",
    ),
    pytest.param(
        _parse_ignoring_invalid,
        "<?xml" * 20_000,
        id="xml-declaration-starts-str",
    ),
    pytest.param(
        lambda page: _extract_meta_refresh_url(page, "https://example.com/"),
        _meta_refresh_page("0; url=" + " " * 40_000 + "'"),
        id="meta-refresh-whitespace-run",
    ),
]


@pytest.mark.parametrize("run, hostile_input", _HOSTILE_INPUTS)
def test_hostile_input_is_handled_in_linear_time(run, hostile_input):
    start = time.perf_counter()
    run(hostile_input)
    assert time.perf_counter() - start < _BUDGET_SECONDS


@pytest.mark.parametrize(
    "broken, repaired",
    [
        (b"<rss rss:version=2.0 a=b>", b'<rss rss:version="2.0" a="b">'),
        (b"<a  b=c d='e' f=g/>", b'<a  b="c" d=\'e\' f="g/">'),
        (b'<link href="x">\n<title>', b'<link href="x"/>\n<title>'),
        (b'<link href="x">  \n \n  <id>', b'<link href="x"/>\n  <id>'),
    ],
)
def test_malformed_xml_is_still_repaired(broken, repaired):
    assert _fix_malformed_xml_bytes(broken) == repaired


@pytest.mark.parametrize(
    "content",
    [
        b'<link href="x">\n</link>',
        b'<link href="x">\n  </link >',
        b'<link href="x"/>\n<id>',
        b'<link href="x">text\n<id>',
        b'<link href="x"> <id>',
    ],
)
def test_well_formed_links_are_left_alone(content):
    assert _fix_malformed_xml_bytes(content) == content


def test_unclosed_links_are_repaired_end_to_end():
    feed = (
        _UTF16_DECL + b'<feed xmlns="http://www.w3.org/2005/Atom"><title>t</title>\n'
        b'<link href="http://e.com/">\n'
        b"<entry><id>1</id><title>e</title>\n"
        b'<link href="http://e.com/1">\n'
        b"</entry></feed>"
    )
    parsed = parse(feed)
    assert parsed.feed.link == "http://e.com/"
    assert [entry.link for entry in parsed.entries] == ["http://e.com/1"]


def test_str_declaration_after_long_leading_whitespace_is_rewritten():
    xml = (
        " " * 5000 + '<?xml version="1.0" encoding="iso-8859-1"?>'
        '<rss version="2.0"><channel><title>café</title></channel></rss>'
    )
    assert parse(xml).feed.title == "café"


@pytest.mark.parametrize(
    "content",
    [
        "0; URL = 'https://example.com/feed.xml'",
        "0; url=' https://example.com/feed.xml'",
        "0;url=https://example.com/feed.xml",
    ],
)
def test_meta_refresh_url_forms_are_still_extracted(content):
    assert (
        _extract_meta_refresh_url(_meta_refresh_page(content), "https://example.com/")
        == "https://example.com/feed.xml"
    )


def test_meta_refresh_with_doubled_quote_is_rejected():
    page = _meta_refresh_page("0; url=''https://example.com/feed.xml")
    assert _extract_meta_refresh_url(page, "https://example.com/") is None
