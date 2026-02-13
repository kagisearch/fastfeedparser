from fastfeedparser import parse
from fastfeedparser.main import _extract_meta_refresh_url


def test_parse_str_with_non_utf8_xml_declaration():
    xml = (
        '<?xml version="1.0" encoding="iso-8859-1"?>'
        '<rss version="2.0">'
        "<channel>"
        "<title>café</title>"
        "<item><title>café</title></item>"
        "</channel>"
        "</rss>"
    )
    feed = parse(xml)
    assert feed.feed.title == "café"
    assert feed.entries[0].title == "café"


def test_parse_bytes_with_non_utf8_encoding():
    xml_bytes = (
        b'<?xml version="1.0" encoding="iso-8859-1"?>'
        b'<rss version="2.0">'
        b"<channel>"
        b"<title>caf\xe9</title>"
        b"<item><title>caf\xe9</title></item>"
        b"</channel>"
        b"</rss>"
    )
    feed = parse(xml_bytes)
    assert feed.feed.title == "café"
    assert feed.entries[0].title == "café"


def test_meta_refresh_extraction():
    html = '<!doctype html><html><head><meta http-equiv=refresh content="0; url=https://example.com/feed.xml"></head></html>'
    assert _extract_meta_refresh_url(html, "https://example.com/feed/") == "https://example.com/feed.xml"


def test_meta_refresh_relative_url():
    html = b'<html><head><meta http-equiv="refresh" content="0;url=/index.xml"></head></html>'
    assert _extract_meta_refresh_url(html, "https://example.com/feed/") == "https://example.com/index.xml"


def test_meta_refresh_none_when_missing():
    html = "<html><head><title>Hello</title></head><body></body></html>"
    assert _extract_meta_refresh_url(html, "https://example.com/") is None


def test_meta_refresh_none_when_same_url():
    html = '<html><head><meta http-equiv="refresh" content="0; url=https://example.com/"></head></html>'
    assert _extract_meta_refresh_url(html, "https://example.com/") is None

