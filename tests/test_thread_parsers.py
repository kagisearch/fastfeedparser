"""Tests that parses running in different threads do not share an lxml parser.

lxml holds a parser's lock for the whole of a parse, so threads that share
one parser object parse one at a time.
"""

import threading

import pytest

from fastfeedparser import main, parse

# These tests are about the lxml path; the native core would bypass it.
pytestmark = pytest.mark.usefixtures("lxml_path_only")

_ITEM = b"<item><title>e</title></item>"
_WELL_FORMED = b'<rss version="2.0"><channel><title>t</title>' + _ITEM + b"</channel></rss>"
# This header looks malformed, which sends the document to the strict parser.
_UTF16_DECL = b'<?xml version="1.0" encoding="utf-16"?>'
# The unquoted attribute fails the strict parse, so the recover parser runs too.
_NEEDS_REPAIR = (
    _UTF16_DECL + b'<rss version="2.0"><channel><title>t</title>'
    b"<item><title>e</title><enclosure url=http://e.com/a.mp3 /></item>"
    b"</channel></rss>"
)


def _parsers_used_per_thread(monkeypatch, feed, threads=2, parses=2):
    """Parse `feed` in several live threads; return each thread's parser objects."""
    used = {}
    real_fromstring = main.etree.fromstring

    def recording_fromstring(text, parser=None):
        used.setdefault(threading.current_thread().name, []).append(parser)
        return real_fromstring(text, parser=parser)

    monkeypatch.setattr(main.etree, "fromstring", recording_fromstring)
    start = threading.Barrier(threads)
    errors = []

    def work():
        try:
            start.wait(timeout=10)
            for _ in range(parses):
                assert [entry.title for entry in parse(feed).entries] == ["e"]
        except Exception as e:
            errors.append(e)

    workers = [threading.Thread(target=work, name=f"w{i}") for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert not errors
    assert sorted(used) == [worker.name for worker in workers]
    return used


@pytest.mark.parametrize(
    "feed",
    [
        pytest.param(_WELL_FORMED, id="recover-parser"),
        pytest.param(_UTF16_DECL + _WELL_FORMED, id="strict-parser"),
        pytest.param(_NEEDS_REPAIR, id="strict-then-recover"),
    ],
)
def test_threads_do_not_share_a_parser(monkeypatch, feed):
    used = _parsers_used_per_thread(monkeypatch, feed)
    first, second = ({id(parser) for parser in parsers} for parsers in used.values())
    assert None not in used["w0"] + used["w1"]
    assert not first & second


def test_a_thread_reuses_its_parser_across_parses(monkeypatch):
    used = _parsers_used_per_thread(monkeypatch, _WELL_FORMED, parses=3)
    for parsers in used.values():
        assert len(parsers) == 3
        assert len({id(parser) for parser in parsers}) == 1


def test_a_parse_that_fails_in_one_thread_leaves_other_threads_working():
    start = threading.Barrier(2)
    results = {}

    def bad():
        start.wait(timeout=10)
        for _ in range(20):
            try:
                parse(_UTF16_DECL + b"not a feed at all <<<")
            except ValueError as e:
                results["bad"] = str(e)

    def good():
        start.wait(timeout=10)
        results["good"] = [
            [entry.title for entry in parse(_WELL_FORMED).entries] for _ in range(20)
        ]

    workers = [threading.Thread(target=bad), threading.Thread(target=good)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert "couldn't be parsed as XML" in results["bad"]
    assert results["good"] == [["e"]] * 20
