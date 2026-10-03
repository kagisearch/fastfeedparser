"""Large CDATA sections are cut out of the bytes lxml parses.

A lifted section must come back as the same text libxml2 would have produced,
and a section that cannot be shown to be an element's whole text must be left
for libxml2.
"""

import threading
import time

import pytest

from fastfeedparser import main, parse

_BUDGET_SECONDS = 1.0
_BODY = "<p>caf\u00e9 &amp; cr\u00e8me</p>\n" * 100
_ATOM_NS = 'xmlns="http://www.w3.org/2005/Atom"'
_MIXED_LINE_ENDINGS = "a\r\nb\rc\n" * 300


def _cdata(text: str = _BODY) -> str:
    return f"<![CDATA[{text}]]>"


def _rss(item_children: str, channel_children: str = "") -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        f"<channel><title>t</title>{channel_children}"
        f"<item><title>one</title>{item_children}</item>"
        "<item><title>two</title><description>second</description></item>"
        "</channel></rss>"
    ).encode()


def _atom(entry_children: str) -> bytes:
    return (
        f"<feed {_ATOM_NS}><title>t</title>"
        f"<entry><id>1</id><title>one</title>{entry_children}</entry>"
        "<entry><id>2</id><title>two</title><summary>second</summary></entry>"
        "</feed>"
    ).encode()


def _rdf(item_children: str) -> bytes:
    return (
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns="http://purl.org/rss/1.0/">'
        "<channel><title>t</title></channel>"
        f'<item rdf:about="http://e.com/1"><title>one</title>{item_children}</item>'
        "</rdf:RDF>"
    ).encode()


def _lifted(feed: bytes) -> list:
    return list(main._lift_large_cdata(feed)[1].values())


def _parse_without_lifting(feed, monkeypatch, **options):
    with monkeypatch.context() as patch:
        patch.setattr(main, "_CDATA_LIFT_MIN_BYTES", 1 << 60)
        assert not _lifted(feed)
        try:
            return parse(feed, **options)
        except ValueError as e:
            return str(e)


def _parse(feed, **options):
    try:
        return parse(feed, **options)
    except ValueError as e:
        return str(e)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


_LIFTABLE = [
    pytest.param(_rss(f"<description>{_cdata()}</description>"), id="rss-description"),
    pytest.param(
        _rss(f"<content:encoded>{_cdata()}</content:encoded>"), id="rss-encoded"
    ),
    pytest.param(
        _rss(
            f"<description>{_cdata('short')}</description>"
            f"<content:encoded>{_cdata()}</content:encoded>"
        ),
        id="rss-small-then-large",
    ),
    pytest.param(
        _rss(
            f"<description>{_cdata(_BODY + 'a')}</description>"
            f"<content:encoded>{_cdata(_BODY + 'b')}</content:encoded>"
        ),
        id="rss-two-large",
    ),
    pytest.param(_rss(f"<content>{_cdata()}</content>"), id="rss-content"),
    pytest.param(_rss(f"<summary>{_cdata()}</summary>"), id="rss-summary"),
    pytest.param(
        _rss(f"<description>\n   {_cdata()}\n  </description>"), id="padded"
    ),
    pytest.param(
        _rss(f"<description >{_cdata()}</description >"), id="space-in-tags"
    ),
    pytest.param(
        _rss(f"<description>{_cdata(_MIXED_LINE_ENDINGS)}</description>"),
        id="line-endings",
    ),
    pytest.param(
        _rss(f"<description>{_cdata(_BODY + ']] > ] ]>')}</description>"),
        id="brackets-in-text",
    ),
    pytest.param(
        _rss(f"<description>{_cdata('<description><![CDATA[' + _BODY)}</description>"),
        id="opener-text-in-section",
    ),
    pytest.param(
        _rss("", f"<description>{_cdata()}</description>"), id="channel-description"
    ),
    pytest.param(
        _rss(
            f"<!-- {_cdata()} --><description>{_cdata()}</description>"
            f"<?pi {_cdata()} ?>"
        ),
        id="after-comment-and-before-pi",
    ),
    pytest.param(
        _rss(f"<source><description>{_cdata()}</description></source>"),
        id="nested-in-unread-element",
    ),
    pytest.param(
        _rss(f"<description/><description>{_cdata()}</description>"),
        id="second-description",
    ),
    pytest.param(
        _atom(f'<content type="html">{_cdata()}</content>'), id="atom-content"
    ),
    pytest.param(
        _atom(f'<summary type="html">{_cdata()}</summary>'), id="atom-summary"
    ),
    pytest.param(
        _atom(
            f'<summary type="html">{_cdata(_BODY + "s")}</summary>'
            f'<content type="html">{_cdata(_BODY + "c")}</content>'
        ),
        id="atom-summary-and-content",
    ),
    pytest.param(_rdf(f"<description>{_cdata()}</description>"), id="rdf-description"),
    pytest.param(
        _atom(
            '<content type="xhtml"><div xmlns="http://www.w3.org/1999/xhtml">'
            f"<summary>{_cdata()}</summary><summary> {_cdata(_BODY + '2')} </summary>"
            "</div></content>"
        ),
        id="atom-nested-in-xhtml-content",
    ),
    pytest.param(
        _rss(f'<content type="xhtml"><description>{_cdata()}</description></content>'),
        id="rss-nested-in-xhtml-content",
    ),
    pytest.param(
        _atom(f'<content type="xhtml">{_cdata()}</content>'), id="atom-xhtml-content"
    ),
]


@pytest.mark.parametrize("feed", _LIFTABLE)
@pytest.mark.parametrize("include_content", [True, False])
def test_lifted_sections_parse_like_unlifted_ones(feed, include_content, monkeypatch):
    assert _lifted(feed)
    parsed = _parse(feed, include_content=include_content)
    assert parsed == _parse_without_lifting(
        feed, monkeypatch, include_content=include_content
    )
    assert not [text for text in _strings(parsed) if "\ue000" in text]


def test_lifted_text_is_the_section_text():
    entry = parse(_rss(f"<content:encoded>{_cdata()}</content:encoded>")).entries[0]
    assert entry.content[0]["value"] == _BODY


_NOT_LIFTABLE = [
    pytest.param(_rss(f"<description>{_cdata('x' * 1023)}</description>"), id="small"),
    pytest.param(
        _rss(f"<description>intro {_cdata()}</description>"), id="text-before"
    ),
    pytest.param(
        _rss(f"<description>{_cdata()} outro</description>"), id="text-after"
    ),
    pytest.param(
        _rss(f"<description>{_cdata()}{_cdata()}</description>"), id="two-sections"
    ),
    pytest.param(
        _rss(f"<description>{_cdata()}<b>x</b></description>"), id="element-after"
    ),
    pytest.param(
        _rss(f"<description><b>x</b>{_cdata()}</description>"), id="element-before"
    ),
    pytest.param(
        _rss(f"<description/>{_cdata()}<description></description>"),
        id="after-empty-element",
    ),
    pytest.param(_rss(f"<title>{_cdata()}</title>"), id="element-not-listed"),
    pytest.param(
        _rss(f"<media:description>{_cdata()}</media:description>"),
        id="prefixed-element",
    ),
    pytest.param(_rss(f"<!-- <description>{_cdata()}</description>"), id="open-comment"),
    pytest.param(
        _rss(f"<description>{_cdata()}</descriptions>"), id="other-end-tag"
    ),
    pytest.param(_rss(f"<description><![CDATA[{_BODY}"), id="unterminated"),
    pytest.param(
        _rss(f"<description>{_cdata()}</description>").replace(
            b"UTF-8", b"ISO-8859-1"
        ),
        id="declared-latin-1",
    ),
    pytest.param(
        _rss("<description><![CDATA[" + "x" * 2000 + "]]></description>").replace(
            b"xxxx]]>", b"x\xe9xx]]>"
        ),
        id="invalid-utf-8",
    ),
    pytest.param(
        "<rss><channel><item><description>".encode("utf-16-le")
        + b"<description><![CDATA[" + b"a" * 1025 + b"]]></description>"
        + "</description></item></channel></rss>".encode("utf-16-le"),
        id="utf-16-without-bom-around-an-ascii-section",
    ),
    pytest.param(
        _rss("<description><![CDATA[\x08" + "x" * 1024 + "]]></description>"),
        id="backspace",
    ),
    pytest.param(
        _rss("<description><![CDATA[\x00" + "x" * 1024 + "]]></description>"),
        id="nul",
    ),
    pytest.param(
        _rss("<description><![CDATA[\x1b" + "x" * 1024 + "]]></description>"),
        id="escape",
    ),
    pytest.param(
        _rss("<description><![CDATA[\ufffe" + "x" * 1024 + "]]></description>"),
        id="u-fffe",
    ),
    pytest.param(
        _rss("<description><![CDATA[\uffff" + "x" * 1024 + "]]></description>"),
        id="u-ffff",
    ),
    pytest.param(
        b'<!DOCTYPE rss [<!ENTITY e "x">]>'
        + _rss(f"<description>{_cdata()}</description>"),
        id="doctype",
    ),
]


@pytest.mark.parametrize("feed", _NOT_LIFTABLE)
def test_other_sections_are_left_for_the_xml_parser(feed):
    parse_bytes, lifted = main._lift_large_cdata(feed)
    assert lifted == {}
    assert parse_bytes is feed


def test_scan_gives_up_when_nothing_liftable_starts_early():
    liftable = f"<description>{_cdata()}</description>"
    early_small_section = "<x><![CDATA[c]]></x>"
    padding = "<y>pad</y>" * (main._CDATA_FIRST_LIFT_BYTES // 10 + 1)
    late = _rss(liftable, early_small_section + padding)
    assert late.find(b"<![CDATA[") < main._CDATA_PROBE_BYTES
    assert _lifted(late) == []
    # After a first lift the rest of the document is scanned.
    early_and_late = _rss(liftable, liftable + padding)
    assert _lifted(early_and_late) == [_BODY, _BODY]


def test_only_the_section_that_is_an_elements_whole_text_is_lifted(monkeypatch):
    feed = _rss(
        f"<description>intro {_cdata(_BODY + 'd')}</description>"
        f"<content:encoded>{_cdata(_BODY + 'c')}</content:encoded>"
    )
    assert _lifted(feed) == [_BODY + "c"]
    assert _parse(feed) == _parse_without_lifting(feed, monkeypatch)


def test_section_beyond_the_probe_window_is_not_looked_for():
    filler = "<item><title>f</title></item>" * 1000
    feed = _rss(f"<description>{_cdata()}</description>", filler)
    assert feed.find(b"<![CDATA[") > main._CDATA_PROBE_BYTES
    assert _lifted(feed) == []


def test_error_message_for_a_non_feed_shows_the_section_text(monkeypatch):
    page = f"<error><description>{_cdata('quota exceeded ' * 100)}</description></error>"
    feed = page.encode()
    assert _lifted(feed)
    message = _parse(feed)
    assert message == _parse_without_lifting(feed, monkeypatch)
    assert "quota exceeded" in message


# libxml2 picks the encoding; the declaration is only a hint to the scan.
_READ_AS_LATIN_1 = [
    pytest.param(b' encoding = "iso-8859-1"', id="spaces-around-equals"),
    pytest.param(b" encoding" + b" " * 2000 + b'="iso-8859-1"', id="past-the-declaration-scan"),
    pytest.param(b" encoding='ISO-8859-15'", id="single-quotes"),
]


@pytest.mark.parametrize("declared", _READ_AS_LATIN_1)
def test_document_libxml2_reads_in_another_encoding_is_parsed_again_whole(
    declared, parsed_inputs, monkeypatch
):
    text = "caf" + chr(0xE9) + " " + "x" * 1024
    feed = (
        b'<?xml version="1.0"' + declared + b"?>"
        b'<rss version="2.0"><channel><title>t</title><item><title>one</title>'
        b"<description><![CDATA[" + text.encode("utf-8") + b"]]></description>"
        b"</item></channel></rss>"
    )
    parsed = parse(feed)
    if _lifted(feed):
        assert len(parsed_inputs) == 2
        assert parsed_inputs[1] is feed
    # The UTF-8 bytes of the accented letter read as two Latin-1 characters.
    assert parsed.entries[0].description == text.encode("utf-8").decode("iso-8859-1")
    assert parsed == _parse_without_lifting(feed, monkeypatch)


def test_scan_stops_after_too_many_tokens_without_a_lift():
    liftable = f"<description>{_cdata()}</description>"
    within = _rss(liftable, "<!--x-->" * (main._CDATA_SCAN_SKIP_BASE - 8) + "<x><![CDATA[c]]></x>")
    beyond = _rss(liftable, "<!--x-->" * (main._CDATA_SCAN_SKIP_BASE + 8) + "<x><![CDATA[c]]></x>")
    assert within.find(b"<![CDATA[") < main._CDATA_PROBE_BYTES
    assert _lifted(within) == [_BODY]
    assert _lifted(beyond) == []


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("\ue0000\ue001", id="index-only"),
        pytest.param("\ue000\ue001", id="empty"),
        pytest.param("\ue000 unterminated", id="start-marker-only"),
        pytest.param("\ue0010\ue000", id="reversed"),
    ],
)
def test_text_that_only_looks_like_a_placeholder_is_kept(text):
    feed = _rss(
        f"<description>{text}</description>"
        f"<content:encoded>{_cdata()}</content:encoded>"
    )
    assert _lifted(feed) == [_BODY]
    entry = parse(feed).entries[0]
    assert entry.description == text
    assert entry.content[0]["value"] == _BODY


def test_placeholders_differ_between_parses():
    feed = _rss(f"<content:encoded>{_cdata()}</content:encoded>")
    first = main._lift_large_cdata(feed)[1]
    second = main._lift_large_cdata(feed)[1]
    assert len(first) == len(second) == 1
    assert not set(first) & set(second)


def test_placeholder_characters_inside_a_lifted_section_are_kept():
    text = "\ue0000\ue001" + _BODY
    feed = _rss(f"<content:encoded>{_cdata(text)}</content:encoded>")
    assert _lifted(feed) == [text]
    assert parse(feed).entries[0].content[0]["value"] == text


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(b"<![CDATA[]]>" * 100_000, id="tiny-sections"),
        pytest.param(b"<!---->" * 100_000, id="tiny-comments"),
        pytest.param(b"<??>" * 100_000, id="tiny-instructions"),
        pytest.param(b"<![CDATA[" * 100_000, id="openers-without-end"),
        pytest.param(b"<" * 1_000_000, id="angle-brackets"),
    ],
)
def test_scan_stays_fast_on_hostile_markup(run):
    feed = _rss(f"<description>{_cdata()}</description>") + run

    start = time.perf_counter()
    main._lift_large_cdata(feed)
    elapsed = time.perf_counter() - start

    assert elapsed < _BUDGET_SECONDS


@pytest.fixture
def parsed_inputs(monkeypatch):
    """Record the bytes handed to lxml by each parse."""
    inputs = []
    real_fromstring = main.etree.fromstring

    def recording_fromstring(text, parser=None):
        inputs.append(text)
        return real_fromstring(text, parser=parser)

    monkeypatch.setattr(main.etree, "fromstring", recording_fromstring)
    return inputs


def test_well_formed_document_is_parsed_once_without_the_lifted_text(parsed_inputs):
    feed = _rss(f"<content:encoded>{_cdata()}</content:encoded>")
    assert parse(feed).entries[0].content[0]["value"] == _BODY
    assert len(parsed_inputs) == 1
    assert b"<![CDATA[" not in parsed_inputs[0]
    assert len(parsed_inputs[0]) < len(feed) - len(_BODY)


# A "<?" with no target is not an instruction to libxml2, so the section it
# sees starts at the first opener and runs to the first "]]>". The scan takes
# "<? ... ?>" for an instruction and would lift the inner section.
_MISREAD_BY_THE_SCAN = (
    f"<feed {_ATOM_NS}><title>t</title>"
    "<entry><id>1</id><title>one</title>"
    f"<? bad<summary> <![CDATA[?><summary><![CDATA[{_BODY}]]> </summary></entry>"
    "<entry><id>2</id><title>two</title><summary><![CDATA[x]]></summary></entry>"
    "</feed>"
).encode()


def test_document_libxml2_had_to_recover_is_parsed_again_whole(
    parsed_inputs, monkeypatch
):
    feed = _MISREAD_BY_THE_SCAN
    assert _lifted(feed) == [_BODY]
    parsed = parse(feed)
    assert [entry.title for entry in parsed.entries] == ["one", "two"]
    assert len(parsed_inputs) == 2
    assert parsed_inputs[1] is feed
    assert parsed == _parse_without_lifting(feed, monkeypatch)


def test_the_misread_document_would_lose_an_entry_if_parsed_lifted():
    parse_bytes, _ = main._lift_large_cdata(_MISREAD_BY_THE_SCAN)
    titles = [entry.title for entry in parse(parse_bytes).entries]
    assert titles != ["one", "two"]


_LIFTABLE_FEED = _rss(f"<content:encoded>{_cdata()}</content:encoded>")


def _is_parsed_lifted(parsed_inputs) -> bool:
    del parsed_inputs[:]
    assert parse(_LIFTABLE_FEED).entries[0].content[0]["value"] == _BODY
    assert len(parsed_inputs) == 1
    return parsed_inputs[0] is not _LIFTABLE_FEED


@pytest.mark.parametrize(
    "other_feed, other_fails",
    [
        pytest.param(_LIFTABLE_FEED, False, id="other-parse-succeeds"),
        pytest.param(b"<rss", True, id="other-parse-fails"),
    ],
)
def test_nothing_is_lifted_while_another_thread_is_parsing(
    other_feed, other_fails, monkeypatch
):
    # Lifting holds the GIL where libxml2 releases it, so it only pays when
    # this is the only parse in flight.
    inputs = []
    inside_parse = threading.Event()
    let_go = threading.Event()
    outcome = []
    real_fromstring = main.etree.fromstring

    def fromstring(text, parser=None):
        if threading.current_thread().name == "other":
            inside_parse.set()
            assert let_go.wait(timeout=10)
        else:
            inputs.append(text)
        return real_fromstring(text, parser=parser)

    def other_parse():
        try:
            parse(other_feed)
            outcome.append("parsed")
        except ValueError:
            outcome.append("failed")

    monkeypatch.setattr(main.etree, "fromstring", fromstring)
    other = threading.Thread(target=other_parse, name="other")
    other.start()
    try:
        assert inside_parse.wait(timeout=10)
        assert not _is_parsed_lifted(inputs)
    finally:
        let_go.set()
        other.join(timeout=10)
    assert outcome == ["failed" if other_fails else "parsed"]
    # Once the other parse is over, whether it returned or raised, lifting is back.
    assert _is_parsed_lifted(inputs)


def test_a_parse_that_raises_does_not_stay_in_flight(parsed_inputs):
    with pytest.raises(ValueError):
        parse(b"<rss")
    with pytest.raises(ValueError):
        parse(b"")
    assert _is_parsed_lifted(parsed_inputs)
