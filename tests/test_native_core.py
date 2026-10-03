"""The native core must return what the lxml path returns, or step aside.

Skipped when the optional fastfeedparser_core extension is not installed.
"""

import glob
import json
import os

import pytest

core = pytest.importorskip("fastfeedparser_core")

from feedgen import DATES, FeedGenerator

from fastfeedparser import main, parse

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "integration")
_FEED_FILES = sorted(
    path
    for path in glob.glob(os.path.join(_FIXTURE_DIR, "*"))
    if not path.endswith(".json") or path.endswith("json_feed_sample.json")
)
_ALL_OFF = {
    "include_content": False,
    "include_tags": False,
    "include_media": False,
    "include_enclosures": False,
}
_OPTION_SETS = [
    {},
    _ALL_OFF,
    {"include_media": False},
    {"include_content": False},
    {"include_tags": False},
    {"include_enclosures": False},
]


def _parse(source, use_core, monkeypatch, **options):
    """parse() output as JSON, or the error it raised, on the chosen path."""
    monkeypatch.setattr(main, "_core", core if use_core else None)
    try:
        return json.dumps(parse(source, **options), sort_keys=True)
    except ValueError as error:
        return f"ValueError: {error}"


def _extract(document):
    return core.parse_entries(document, main.FastFeedParserDict, main._parse_date)


def _rss(item_children, root_attrs=""):
    return (
        f'<rss version="2.0"{root_attrs}><channel><title>t</title>'
        f"<item>{item_children}</item></channel></rss>"
    ).encode()


@pytest.mark.parametrize("path", _FEED_FILES, ids=os.path.basename)
@pytest.mark.parametrize("as_text", [False, True], ids=["bytes", "str"])
def test_fixtures_parse_the_same_on_both_paths(path, as_text, monkeypatch):
    with open(path, "rb") as feed_file:
        source = feed_file.read()
    if as_text:
        source = source.decode("utf-8", errors="replace")
    for options in _OPTION_SETS:
        with_lxml = _parse(source, False, monkeypatch, **options)
        with_core = _parse(source, True, monkeypatch, **options)
        assert with_core == with_lxml, options


def test_well_formed_feeds_are_handled_natively():
    rss = _rss("<title>a</title><link>http://e.com/a</link>")
    atom = (
        b'<feed xmlns="http://www.w3.org/2005/Atom"><title>t</title>'
        b"<entry><id>1</id><title>a</title></entry></feed>"
    )
    for document, kind in ((rss, "rss"), (atom, "atom")):
        result = _extract(document)
        assert result[0] == kind
        assert [entry["title"] for entry in result[3]] == ["a"]
        assert b"<item>" not in result[2] and b"<entry>" not in result[2]


@pytest.mark.parametrize("seed", range(4))
def test_generated_feeds_parse_the_same_on_both_paths(seed, monkeypatch):
    generator = FeedGenerator(seed)
    handled = 0
    for index in range(400):
        document = generator.document()
        if index % 3 == 0:
            document = generator.mutate(document)
        options = _OPTION_SETS[index % len(_OPTION_SETS)]
        with_lxml = _parse(document, False, monkeypatch, **options)
        with_core = _parse(document, True, monkeypatch, **options)
        assert with_core == with_lxml, document
        handled += not isinstance(_extract(document), str)
    # Guard against the comparison becoming vacuous.
    assert handled > 100


_NOT_HANDLED = {
    "undefined entity": _rss("<title>a&nbsp;b</title>"),
    "bare ampersand": _rss("<title>AT&T</title>"),
    "undeclared prefix": _rss("<dc:creator>x</dc:creator>"),
    "mismatched end tag": _rss("<title>a</titel>"),
    "truncated": _rss("<title>a</title>")[:-20],
    "cdata end in text": _rss("<title>a ]]> b</title>"),
    "attributes not separated": _rss('<enclosure url="http://e.com/a"type="a/b"/>'),
    "duplicate attribute": _rss(
        '<enclosure url="http://e.com/a" url="http://e.com/b"/>'
    ),
    "control character": _rss("<title>a\x0bb</title>"),
    "invalid character reference": _rss("<title>a&#0;b</title>"),
    "double hyphen in comment": _rss("<!-- a -- b --><title>a</title>"),
    "malformed declaration": b'<?xml version="1.0"encoding="utf-8"?>' + _rss(""),
    "other encoding": b'<?xml version="1.0" encoding="iso-8859-1"?>' + _rss(""),
    "invalid utf-8": _rss("<title>X</title>").replace(b"X", b"caf\xe9"),
    "internal dtd subset": b"<!DOCTYPE rss [<!ENTITY e 'v'>]>"
    + _rss("<title>&e;</title>"),
    "malformed-looking header": b'<?xml version="1.0" encoding="utf-16"?>'
    + _rss("<title>a</title>"),
    "rdf root": (
        b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        b' xmlns="http://purl.org/rss/1.0/"><item><title>a</title></item></rdf:RDF>'
    ),
    "channel without items": b'<rss version="2.0"><channel><title>t</title></channel></rss>',
    "xhtml content": (
        b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>a</title>'
        b'<content type="xhtml"><div xmlns="http://www.w3.org/1999/xhtml">x</div>'
        b"</content></entry></feed>"
    ),
}


@pytest.mark.parametrize("document", _NOT_HANDLED.values(), ids=_NOT_HANDLED.keys())
def test_unsupported_documents_fall_back_to_lxml(document, monkeypatch):
    results = []
    parse_with_core = main._parse_with_core

    def recording(*args, **kwargs):
        results.append(parse_with_core(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(main, "_parse_with_core", recording)
    with_lxml = _parse(document, False, monkeypatch)
    assert _parse(document, True, monkeypatch) == with_lxml
    # Either the core was never asked, or it handed the document back.
    assert all(result is None for result in results)


@pytest.mark.parametrize(
    "trailing", [b"junk", b"\n<script>x</script>", b"<rss/>", b"<a><b></a>&"]
)
def test_content_after_the_root_is_ignored_on_both_paths(trailing, monkeypatch):
    document = _rss("<title>a</title><link>http://e.com/a</link>") + trailing
    result = _extract(document)
    assert not isinstance(result, str)
    with_lxml = _parse(document, False, monkeypatch)
    assert _parse(document, True, monkeypatch) == with_lxml
    assert '"title": "a"' in with_lxml


def test_date_callback_errors_propagate():
    def failing_parse_date(raw):
        raise RuntimeError("boom " + raw)

    document = _rss("<title>a</title><pubDate>next tuesday</pubDate>")
    with pytest.raises(RuntimeError, match="boom next tuesday"):
        core.parse_entries(document, dict, failing_parse_date)


def _date_candidates():
    zones = [
        "GMT",
        "UTC",
        "+0000",
        "-0000",
        "+0530",
        "-0800",
        "EST",
        "PDT",
        "CEST",
        "Z",
        "XYZ",
        "+2400",
    ]
    yield from DATES
    for zone in zones:
        yield f"Mon, 02 Jan 2006 15:04:05 {zone}"
        yield f"29 Feb 2024 23:59:59 {zone}"
        yield f"31 Dec 9999 23:59:59 {zone}"
        yield f"01 Jan 0001 00:00:00 {zone}"
    for offset in [
        "Z",
        "z",
        "+00:00",
        "-00:00",
        "+05:30",
        "-08:00",
        "+23:59",
        "+24:00",
        "+0100",
    ]:
        for fraction in ["", ".5", ".123", ".123456", ".1234567"]:
            for separator in ["T", " ", "t"]:
                yield f"2024-02-29{separator}23:59:59{fraction}{offset}"
        yield f"0001-01-01T00:00:00{offset}"
        yield f"9999-12-31T23:59:59{offset}"
        yield f"2023-02-29T10:00:00{offset}"
        yield f"2024-13-01T10:00:00{offset}"
        yield f"2024-01-15T24:00:00{offset}"


@pytest.mark.parametrize("raw", sorted(set(_date_candidates())))
def test_fast_date_agrees_with_parse_date(raw):
    state, value = core.fast_date(raw)
    if state == 2:  # left to Python
        return
    try:
        expected = main._parse_date(raw)
    except ValueError:
        pytest.fail(f"fast path decided {value!r} where _parse_date raises")
    assert value == expected
    assert (state == 1) == (expected is None)


def test_fast_date_decides_the_common_shapes():
    for raw in (
        "Mon, 02 Jan 2006 15:04:05 GMT",
        "2024-01-15T10:30:00Z",
        "2024-01-15T10:30:00+02:00",
    ):
        assert core.fast_date(raw)[0] == 0
