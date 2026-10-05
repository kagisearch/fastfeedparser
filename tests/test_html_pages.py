"""A URL that answers with an HTML page must not cost a whole-page parse per thread.

The page is only read for a meta-refresh or an error message, both of which
sit at its top, and the tree of the failed parse must be gone before the
redirect is fetched.
"""

import gc
import weakref

import pytest

from fastfeedparser import main, parse
from lxml import etree

_FEED = (
    b'<rss version="2.0"><channel><title>t</title>'
    b"<item><title>e</title></item></channel></rss>"
)
_REFRESH = b'<meta http-equiv="refresh" content="0; url=/feed.xml">'
_BULK = b"<p>filler paragraph</p>" * 100_000


@pytest.fixture
def html_inputs(monkeypatch):
    """Record the bytes handed to lxml together with an HTML parser."""
    inputs = []
    real_fromstring = main.etree.fromstring

    def recording_fromstring(text, parser=None):
        if isinstance(parser, etree.HTMLParser):
            inputs.append(text)
        return real_fromstring(text, parser=parser)

    monkeypatch.setattr(main.etree, "fromstring", recording_fromstring)
    return inputs


@pytest.mark.parametrize("as_text", [False, True])
def test_meta_refresh_is_found_without_parsing_the_whole_page(as_text, html_inputs):
    page = b"<html><head>" + _REFRESH + b"</head><body>" + _BULK + b"</body></html>"
    assert len(page) > 8 * main._HTML_HEAD_BYTES
    content = page.decode() if as_text else page
    url = main._extract_meta_refresh_url(content, "https://example.com/feed/")
    assert url == "https://example.com/feed.xml"
    assert len(html_inputs) == 1
    assert len(html_inputs[0]) <= main._HTML_HEAD_BYTES


def test_meta_refresh_at_the_end_of_the_head_window_is_found(html_inputs):
    padding = b"<!--" + b"x" * (main._HTML_HEAD_BYTES - len(_REFRESH) - 30) + b"-->"
    page = b"<html><head>" + padding + _REFRESH + b"</head></html>"
    assert page.index(_REFRESH) + len(_REFRESH) <= main._HTML_HEAD_BYTES
    assert (
        main._extract_meta_refresh_url(page, "https://example.com/feed/")
        == "https://example.com/feed.xml"
    )


def test_meta_refresh_past_the_head_window_is_not_followed(html_inputs):
    page = b"<html><head></head><body>" + _BULK + _REFRESH + b"</body></html>"
    assert page.index(_REFRESH) > main._HTML_HEAD_BYTES
    assert main._extract_meta_refresh_url(page, "https://example.com/feed/") is None


def test_error_message_fallback_does_not_parse_the_whole_page(html_inputs):
    empty_root = etree.fromstring(b"<html/>")
    page = b"<html><body><p>Service unavailable</p>" + _BULK + b"</body></html>"
    message = main._extract_error_message(empty_root, page)
    assert message.startswith("Service unavailable")
    assert len(html_inputs) == 1
    assert len(html_inputs[0]) <= main._HTML_HEAD_BYTES


def test_tree_of_the_failed_parse_is_gone_before_the_redirect_is_fetched(monkeypatch):
    # This page does not start like HTML, so it is parsed as XML first and
    # only then recognized as a page. That tree must not live through the
    # fetch of the redirect target, where a thread can wait for a long time.
    page = (
        b'<!-- moved --><html><head><meta http-equiv="refresh" content="0; url=/feed.xml">'
        b'</head><body class="x"><p>moved</p></body></html>'
    )
    class Marker:
        """Lives in a frame of the failed parse, next to the one holding the tree."""

    markers = []
    alive_during_fetch = []
    real_raise = main._raise_for_non_feed_root

    def remembering_raise(root, *args):
        marker = Marker()
        markers.append(weakref.ref(marker))
        return real_raise(root, *args)

    def fetch(url):
        if url == "https://example.com/feed/":
            return page
        gc.collect()
        alive_during_fetch.extend(ref() is not None for ref in markers)
        return _FEED

    monkeypatch.setattr(main, "_raise_for_non_feed_root", remembering_raise)
    monkeypatch.setattr(main, "_fetch_url_content", fetch)
    parsed = parse("https://example.com/feed/")
    assert [entry.title for entry in parsed.entries] == ["e"]
    assert alive_during_fetch == [False]


def test_error_for_a_page_without_a_redirect_keeps_its_traceback(monkeypatch):
    page = b'<!-- gone --><html><body class="x"><p>Not here any more</p></body></html>'
    monkeypatch.setattr(main, "_fetch_url_content", lambda url: page)
    with pytest.raises(ValueError, match="Received HTML page instead of feed") as raised:
        parse("https://example.com/feed/")
    functions = [entry.name for entry in raised.traceback]
    assert "_raise_for_non_feed_root" in functions


@pytest.fixture
def xml_inputs(monkeypatch):
    """Record the bytes handed to lxml together with an XML parser."""
    inputs = []
    real_fromstring = main.etree.fromstring

    def recording_fromstring(text, parser=None):
        if isinstance(parser, etree.XMLParser):
            inputs.append(text)
        return real_fromstring(text, parser=parser)

    monkeypatch.setattr(main.etree, "fromstring", recording_fromstring)
    return inputs


_XHTML_PROLOG = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" '
    b'"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n'
)
_LARGE_NON_FEEDS = [
    pytest.param(
        b"<!-- cached --><html><head><title>Page not found</title></head>"
        b'<body class="x">' + _BULK + b"</body></html>",
        "Received HTML page instead of feed: Page not found",
        id="html-after-a-comment",
    ),
    pytest.param(
        _XHTML_PROLOG
        + b'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Page not found</title>'
        b'</head><body class="x">' + _BULK + b"</body></html>",
        "Received HTML page instead of feed: Page not found",
        id="xhtml",
    ),
    pytest.param(
        b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + b"<url><loc>https://example.com/a-page</loc></url>" * 60_000
        + b"</urlset>",
        "Received XML sitemap instead of feed (sitemap is for search engines, not a feed)",
        id="sitemap",
    ),
]


@pytest.mark.parametrize("page, message", _LARGE_NON_FEEDS)
def test_large_document_that_is_not_a_feed_is_only_read_at_its_head(
    page, message, xml_inputs
):
    assert len(page) > 4 * main._HTML_HEAD_BYTES
    with pytest.raises(ValueError) as raised:
        parse(page)
    assert str(raised.value).startswith(message)
    assert xml_inputs
    assert max(len(text) for text in xml_inputs) <= main._HTML_HEAD_BYTES


@pytest.mark.parametrize("page, message", _LARGE_NON_FEEDS)
def test_head_only_error_matches_the_error_from_the_whole_document(
    page, message, monkeypatch
):
    with pytest.raises(ValueError) as from_head:
        parse(page)
    monkeypatch.setattr(main, "_HTML_HEAD_BYTES", 1 << 40)
    with pytest.raises(ValueError) as from_whole:
        parse(page)
    assert str(from_head.value) == str(from_whole.value)


def test_large_feed_that_mentions_html_is_parsed_whole(xml_inputs):
    items = b"<item><title>e</title><description>&lt;html&gt; x</description></item>" * 6000
    feed = (
        b'<?xml version="1.0"?><!-- <html> is not the root here -->'
        b'<rss version="2.0"><channel><title>t</title>' + items + b"</channel></rss>"
    )
    assert len(feed) > main._HTML_HEAD_BYTES
    assert len(parse(feed).entries) == 6000
    assert [len(text) for text in xml_inputs] == [len(feed)]


def test_large_document_that_only_looks_like_a_page_is_parsed_as_the_feed_it_is(
    xml_inputs,
):
    # The first element name found by the cheap look is "html", from inside an
    # entity value. Parsing the head shows the root is <rss>, so the document
    # is then parsed whole.
    items = b"<item><title>e</title></item>" * 12000
    feed = (
        b'<?xml version="1.0"?><!DOCTYPE rss [<!ENTITY e "> <html> ">]>'
        b'<rss version="2.0"><channel><title>t</title>' + items + b"</channel></rss>"
    )
    assert len(feed) > main._HTML_HEAD_BYTES
    assert main._first_element_name(feed) == "html"
    assert len(parse(feed).entries) == 12000
    assert [len(text) for text in xml_inputs] == [main._HTML_HEAD_BYTES, len(feed)]


@pytest.mark.parametrize(
    "head, name",
    [
        (b"<html>", "html"),
        (b'  \n<?xml version="1.0"?>\n<!-- c --><!DOCTYPE html><HTML lang="en">', "html"),
        (b'<?xml version="1.0"?><x:urlset xmlns:x="u">', "urlset"),
        (b"<rss><channel><html/>", "rss"),
        (b"<!-- <html> --><feed>", "feed"),
        (b"<!-- never closed <html>", ""),
        (b"no tags at all", ""),
        (b"<" + b"\xff" * 10 + b">", ""),
        (b" " * 5000 + b"<html>", ""),
    ],
)
def test_first_element_name(head, name):
    assert main._first_element_name(head) == name
