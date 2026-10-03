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


@pytest.mark.parametrize("separator", ["\u2028", "\u2029"])
def test_unicode_line_separators_become_newlines(separator):
    feed = (
        f'<rss version="2.0"><channel><title>a{separator}b</title>'
        f"<item><title>x{separator}y</title></item></channel></rss>"
    ).encode()
    parsed = parse(feed)
    assert parsed.feed.title == "a\nb"
    assert parsed.entries[0].title == "x\ny"


def test_line_separator_at_end_of_probe_window_is_found():
    head = b'<rss version="2.0"><channel><title>t</title><item><title>'
    # The separator's three bytes are the last three of the first 64 KB.
    padding = b"x" * (65536 - len(head) - 3)
    feed = head + padding + "\u2028".encode() + b"end</title></item></channel></rss>"
    assert parse(feed).entries[0].title.endswith("x\nend")


_MEDIA_NS = b'xmlns:media="http://search.yahoo.com/mrss/"'


def test_media_in_a_later_item_only_is_parsed():
    feed = (
        b'<rss version="2.0" ' + _MEDIA_NS + b"><channel><title>t</title>"
        b"<item><title>a</title></item>"
        b'<item><title>b</title><media:content url="http://e.com/i.jpg"/></item>'
        b"</channel></rss>"
    )
    media = [entry.get("media_content") for entry in parse(feed).entries]
    assert media == [None, [{"url": "http://e.com/i.jpg"}]]


def test_media_namespace_declared_on_the_element_is_parsed():
    feed = _rss(b"<media:thumbnail " + _MEDIA_NS + b' url="http://e.com/t.jpg"/>')
    assert parse(feed).entries[0].media_content == [
        {"url": "http://e.com/t.jpg", "type": "image/jpeg"}
    ]


@pytest.mark.parametrize("codec", ["utf-16", "utf-16-le", "utf-16-be"])
def test_media_is_parsed_in_utf16_feeds(codec):
    feed = (
        '<?xml version="1.0" encoding="utf-16"?>'
        '<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">'
        "<channel><title>t</title><item><title>a</title>"
        '<media:content url="http://e.com/i.jpg" medium="image"/>'
        "</item></channel></rss>"
    ).encode(codec)
    assert parse(feed).entries[0].media_content == [
        {"url": "http://e.com/i.jpg", "medium": "image"}
    ]


def test_media_namespace_without_media_elements_adds_nothing():
    feed = (
        b'<rss version="2.0" ' + _MEDIA_NS + b"><channel><title>t</title>"
        b"<item><title>a</title><media:rating>nonadult</media:rating></item>"
        b"</channel></rss>"
    )
    assert "media_content" not in parse(feed).entries[0]


def test_media_is_omitted_when_not_requested():
    feed = (
        b'<rss version="2.0" ' + _MEDIA_NS + b"><channel><title>t</title>"
        b'<item><title>a</title><media:content url="http://e.com/i.jpg"/></item>'
        b"</channel></rss>"
    )
    assert "media_content" not in parse(feed, include_media=False).entries[0]


def test_truncated_feed_is_recovered():
    feed = (
        b'<rss version="2.0"><channel><title>t</title>'
        b"<item><title>a</title></item><item><title>b</title>"
    )
    assert [entry.title for entry in parse(feed).entries] == ["a", "b"]


@pytest.mark.parametrize(
    "content, message",
    [
        (b"not xml at all", "couldn't be parsed as XML"),
        (b"<rss", "missing channel element"),
        (b"", "Empty content"),
    ],
)
def test_unparseable_content_raises_value_error(content, message):
    with pytest.raises(ValueError, match=message):
        parse(content)
