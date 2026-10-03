"""Tests for the steps between raw bytes and the parsed tree."""

import pytest

from fastfeedparser import parse

# Either header makes the document look malformed and eligible for repair.
_UTF16_DECL = b'<?xml version="1.0" encoding="utf-16"?>'
_DOUBLE_CLOSE_DECL = b'<?xml version="1.0" encoding="UTF-8"??>'


def _rss(item_children: bytes, decl: bytes = b"") -> bytes:
    return (
        decl + b'<rss version="2.0"><channel><title>t</title>'
        b"<item><title>e</title>" + item_children + b"</item></channel></rss>"
    )


@pytest.mark.parametrize("decl", [_UTF16_DECL, _DOUBLE_CLOSE_DECL])
@pytest.mark.parametrize(
    "text",
    [
        "run with command=/usr/bin/dotnet autostart=true",
        "a b=c",
    ],
)
def test_text_is_not_rewritten_when_document_is_well_formed(decl, text):
    feed = _rss(f"<description>{text}</description>".encode(), decl)
    assert parse(feed).entries[0].description == text


def test_html_in_cdata_is_not_rewritten_when_document_is_well_formed():
    html = '<head><link rel="stylesheet" href="a.css">\n<script src="a.js"></script></head>'
    feed = _rss(f"<description><![CDATA[{html}]]></description>".encode(), _UTF16_DECL)
    assert parse(feed).entries[0].description == html


def test_unquoted_attributes_are_still_repaired():
    feed = _rss(
        b"<description>plain text</description>"
        b"<enclosure url=http://e.com/a.mp3 type=audio/mpeg />",
        _UTF16_DECL,
    )
    entry = parse(feed).entries[0]
    assert entry.description == "plain text"
    assert entry.enclosures == [{"url": "http://e.com/a.mp3", "type": "audio/mpeg"}]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(_UTF16_DECL + b"not a feed at all <<<", id="bytes"),
        pytest.param(_UTF16_DECL, id="declaration-only"),
        pytest.param(
            '<?xml version="1.0" encoding="UTF-8"??>not a feed at all <<<', id="str"
        ),
    ],
)
def test_document_unparseable_even_after_repair_raises(content):
    with pytest.raises(ValueError, match="couldn't be parsed as XML"):
        parse(content)


def test_unclosed_links_are_still_repaired():
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
