"""Tests for the steps between raw bytes and the parsed tree."""

import pytest

from fastfeedparser import main, parse
from fastfeedparser.main import _html_reparse_may_find_more_items

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


def _long_item(title: bytes) -> bytes:
    return (
        b"<item><title>" + title + b"</title>"
        b"<description>" + b"x" * 8000 + b"</description></item>"
    )


def _rss_items(items: list[bytes]) -> bytes:
    return (
        b'<rss version="2.0"><channel><title>t</title>'
        + b"".join(items)
        + b"</channel></rss>"
    )


def test_items_lost_to_an_unterminated_cdata_are_rescued():
    # The XML parser keeps two of these items; the HTML re-parse finds all.
    titles = [b"t%d" % i for i in range(8)]
    titles[1] = b"<![CDATA[t1"
    parsed = parse(_rss_items([_long_item(title) for title in titles]))
    assert len(parsed.entries) == 8


@pytest.fixture
def html_parser_uses(monkeypatch):
    """Count how often parse() builds an HTML parser for the re-parse."""
    uses = []
    real_html_parser = main.etree.HTMLParser

    def counting_html_parser(*args, **kwargs):
        uses.append(1)
        return real_html_parser(*args, **kwargs)

    monkeypatch.setattr(main.etree, "HTMLParser", counting_html_parser)
    return uses


def test_healthy_few_item_feed_is_not_reparsed(html_parser_uses):
    feed = _rss_items([_long_item(b"t%d" % i) for i in range(3)])
    assert len(feed) > 20000
    assert [entry.title for entry in parse(feed).entries] == ["t0", "t1", "t2"]
    assert not html_parser_uses


_ENCODING_MARKERS = [
    b'<meta charset="utf-8">',
    b"http-equiv content charset=utf-8",
    b"HTTP-EQUIV CONTENT CHARSET=utf-8",
]
# A bare "&" is a syntax error the recover parser reads past.
_NOT_WELL_FORMED = b"<x>a & b</x>"


@pytest.mark.parametrize("marker", _ENCODING_MARKERS)
def test_damaged_few_item_feed_that_may_switch_encoding_is_reparsed(
    html_parser_uses, marker
):
    items = [_long_item(b"t%d" % i) for i in range(3)]
    feed = _rss_items([b"<!-- " + marker + b" -->", _NOT_WELL_FORMED] + items)
    assert [entry.title for entry in parse(feed).entries] == ["t0", "t1", "t2"]
    assert html_parser_uses


@pytest.mark.parametrize("marker", _ENCODING_MARKERS)
def test_well_formed_few_item_feed_is_never_reparsed(html_parser_uses, marker):
    items = [_long_item(b"t%d" % i) for i in range(3)]
    feed = _rss_items([b"<!-- " + marker + b" -->"] + items)
    assert [entry.title for entry in parse(feed).entries] == ["t0", "t1", "t2"]
    assert not html_parser_uses


def _item_with_item_markup_in_cdata() -> bytes:
    article = b"<p>How to write a feed:</p>" + b"<item><title>fake</title></item>" * 25
    return (
        b"<item><title>real</title><description><![CDATA["
        + article
        + b"x" * 20000
        + b"]]></description></item>"
    )


def test_item_markup_inside_cdata_is_text_not_entries(html_parser_uses):
    # The HTML parser does not know CDATA, so it reads the article's own
    # <item> examples as elements. A well-formed feed is never given to it.
    feed = _rss_items([_item_with_item_markup_in_cdata()])
    parsed = parse(feed)
    assert [entry.title for entry in parsed.entries] == ["real"]
    assert parsed.entries[0].description.count("<item><title>fake</title></item>") == 25
    assert not html_parser_uses


def test_damaged_feed_with_more_item_tags_is_reparsed(html_parser_uses):
    titles = [b"t%d" % i for i in range(8)]
    titles[1] = b"<![CDATA[t1"
    parse(_rss_items([_long_item(title) for title in titles]))
    assert html_parser_uses


@pytest.mark.parametrize(
    "content, found",
    [
        pytest.param(_rss_items([_long_item(b"t")] * 8), 2, id="more-item-tags"),
        pytest.param(b"<rss>" + b"<ITEM>a</ITEM><Item>b</Item>" * 4, 3, id="any-case"),
        pytest.param(b"<rss><item>a</item></rss>", 0, id="none-found-yet"),
        pytest.param(
            b'<rss><meta charset="utf-7"><item>a</item></rss>', 3, id="meta-tag"
        ),
        pytest.param(
            b"<rss>caf\xc3\xa9 Http-Equiv content charset=utf-16le<item>a</item></rss>",
            3,
            id="raw-text-charset-sniff",
        ),
        pytest.param(
            "<rss><item>a</item></rss>".encode("utf-16"), 3, id="utf-16-with-bom"
        ),
        pytest.param(
            "<rss><item>a</item></rss>".encode("utf-16-le"), 3, id="utf-16-no-bom"
        ),
        pytest.param(b"junk <rss><item>a</item></rss>", 3, id="leading-junk"),
    ],
)
def test_reparse_is_kept_when_it_could_find_more_items(content, found):
    assert _html_reparse_may_find_more_items(content, found)


def _linked_item(i: int, tag: bytes = b"item") -> bytes:
    return (
        b"<" + tag + b"><title>t%d</title><link>http://e.com/%d</link>" % (i, i)
        + b"<description>" + b"x" * 3000 + b"</description></" + tag + b">"
    )


@pytest.mark.parametrize(
    "items",
    [
        pytest.param(
            b"".join(_linked_item(i, b"Item") for i in range(10)), id="mixed-case-tags"
        ),
        pytest.param(
            b"".join(_linked_item(i, b"ITEM") for i in range(10)), id="upper-case-tags"
        ),
        pytest.param(
            _linked_item(0)
            + b"<section>"
            + b"".join(_linked_item(i) for i in range(1, 10))
            + b"</section>",
            id="inside-a-wrapper",
        ),
        pytest.param(
            b"<item><title>t0</title><link>http://e.com/0</link>"
            + b"".join(_linked_item(i) for i in range(1, 10))
            + b"</item>",
            id="nested-in-the-first-item",
        ),
    ],
)
def test_well_formed_feed_gets_its_deeper_items_from_the_xml_tree(
    items, html_parser_uses
):
    parsed = parse(_rss_items([items]))
    assert [entry.title for entry in parsed.entries] == ["t%d" % i for i in range(10)]
    # The HTML parser reads <link> as an empty element and loses its text.
    assert [entry.link for entry in parsed.entries] == [
        "http://e.com/%d" % i for i in range(10)
    ]
    assert not html_parser_uses


def test_items_inside_a_comment_are_not_entries(html_parser_uses):
    commented = b"<!--" + b"".join(_linked_item(i) for i in range(2, 10)) + b"-->"
    parsed = parse(_rss_items([_linked_item(0), _linked_item(1), commented]))
    assert [entry.title for entry in parsed.entries] == ["t0", "t1"]
    assert not html_parser_uses


@pytest.mark.parametrize(
    "feed",
    [
        pytest.param(
            b'<rss version="2.0"><title>f</title><description>'
            + b"x" * 24000
            + b"</description><item><title>T1</title><gallery>"
            + b"<item><title>img</title></item>" * 3
            + b"</gallery></item></rss>",
            id="no-channel",
        ),
        pytest.param(
            b'<rss version="2.0"><channel/><title>f</title><description>'
            + b"x" * 24000
            + b"</description><item><title>T1</title><gallery>"
            + b"<item><title>img</title></item>" * 3
            + b"</gallery></item></rss>",
            id="empty-channel",
        ),
    ],
)
def test_feed_without_a_channel_keeps_only_its_direct_items(feed, html_parser_uses):
    assert [entry.title for entry in parse(feed).entries] == ["T1"]
    assert not html_parser_uses


_QUOTED_ITEMS = b"<p>How to write a feed:</p>" + b"<item><title>fake</title></item>" * 25


def test_item_markup_quoted_in_cdata_of_a_damaged_feed_is_not_entries(html_parser_uses):
    feed = _rss_items([_NOT_WELL_FORMED, _item_with_item_markup_in_cdata()])
    parsed = parse(feed)
    assert html_parser_uses
    assert [entry.title for entry in parsed.entries] == ["real"]
    assert parsed.entries[0].description.count("<item><title>fake</title></item>") == 25


def test_item_markup_quoted_with_cdata_inside_cdata_is_not_entries(html_parser_uses):
    # A section cannot hold "]]>", so an article quoting a feed's own CDATA
    # ends each quoted section with "]]]]><![CDATA[>".
    quoted = (
        b"<item><title>fake</title>"
        b"<description><![CDATA[q]]]]><![CDATA[></description></item>"
    )
    real = (
        b"<item><title>real</title><description><![CDATA["
        + quoted * 25
        + b"x" * 20000
        + b"]]></description></item>"
    )
    parsed = parse(_rss_items([_NOT_WELL_FORMED, real]))
    assert html_parser_uses
    assert [entry.title for entry in parsed.entries] == ["real"]
    assert parsed.entries[0].description.count("<![CDATA[q]]>") == 25


def test_rescued_items_keep_quoted_item_markup_as_text(html_parser_uses):
    # The unterminated section in the second title hides items from the XML
    # parser up to the "]]>" in the seventh, so the HTML re-parse is needed.
    # It must find the real items and read the quoted ones as text.
    titles = [b"t%d" % i for i in range(8)]
    titles[1] = b"<![CDATA[t1"
    items = [_long_item(title) for title in titles]
    items[6] = (
        b"<item><title>t6</title><description><![CDATA["
        + _QUOTED_ITEMS
        + b"]]></description></item>"
    )
    parsed = parse(_rss_items(items))
    assert html_parser_uses
    found = [entry.title for entry in parsed.entries]
    assert len(found) == 8
    assert "fake" not in found
    assert found[6:] == ["t6", "t7"]
    assert parsed.entries[6].description.count("<item><title>fake</title></item>") == 25


@pytest.mark.parametrize(
    "content, expected",
    [
        pytest.param(b"a<![CDATA[<b>&c]]>d", b"a&lt;b>&amp;cd", id="closed-section"),
        pytest.param(b"a<![CDATA[x<y>z", b"a<![CDATA[x<y>z", id="no-end"),
        pytest.param(
            b"<![CDATA[open <i> <![CDATA[<b>]]> tail",
            b"<![CDATA[open <i> &lt;b> tail",
            id="opener-without-its-own-end-is-left",
        ),
        pytest.param(b"a]]>b<![CDATA[<c>]]>", b"a]]>b&lt;c>", id="stray-end-first"),
        pytest.param(b"no sections <here>", b"no sections <here>", id="none"),
        pytest.param(
            b"a<![CDATA[x & y]]>b", b"a<![CDATA[x & y]]>b", id="section-without-markup"
        ),
        pytest.param(
            b"<![CDATA[if (a[b[0]]]]><![CDATA[> c) <x>]]>",
            b"if (a[b[0]]> c) &lt;x>",
            id="split-that-spells-the-end-marker",
        ),
        pytest.param(
            b"<![CDATA[<item><d><![CDATA[q]]]]><![CDATA[></d></item>]]>",
            b"&lt;item>&lt;d>&lt;![CDATA[q]]>&lt;/d>&lt;/item>",
            id="quoted-section-is-part-of-the-outer-one",
        ),
        pytest.param(
            b"<![CDATA[open <i> <![CDATA[<a><![CDATA[q]]]]><![CDATA[></a>]]> tail",
            b"<![CDATA[open <i> &lt;a>&lt;![CDATA[q]]>&lt;/a> tail",
            id="opener-without-an-end-before-a-section-that-quotes-one",
        ),
    ],
)
def test_closed_cdata_sections_become_escaped_text(content, expected):
    assert main._escape_closed_cdata(content) == expected


def test_escaping_cdata_stays_fast_on_hostile_markup():
    import time

    for hostile in (
        b"<![CDATA[" * 200_000,
        b"]]>" * 400_000,
        b"<![CDATA[]]>" * 150_000,
        b"]]]]><![CDATA[>" * 100_000,
        b"<![CDATA[" * 100_000 + b"]]]]><![CDATA[>" * 100_000 + b"]]>",
    ):
        start = time.perf_counter()
        main._escape_closed_cdata(hostile)
        assert time.perf_counter() - start < 1.0
