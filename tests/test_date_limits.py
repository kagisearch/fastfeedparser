"""Overlong date strings must not reach the slow date parsers.

dateutil tokenizes in quadratic time, so 200 KB of "1." in a date element
held parse() for several seconds.
"""

import time

from fastfeedparser import parse
from fastfeedparser.main import _MAX_DATE_CHARS, _parse_date

_BUDGET_SECONDS = 1.0
_RFC822 = "Mon, 02 Jan 2006 15:04:05 GMT"
_PARSED = "2006-01-02T15:04:05+00:00"


def test_overlong_date_in_feed_is_dropped_quickly():
    feed = (
        '<rss version="2.0"><channel><title>t</title><item><title>e</title>'
        f"<pubDate>{'1.' * 100_000}</pubDate></item></channel></rss>"
    )

    start = time.perf_counter()
    parsed = parse(feed)
    elapsed = time.perf_counter() - start

    assert elapsed < _BUDGET_SECONDS
    assert parsed.entries[0].title == "e"
    assert "published" not in parsed.entries[0]


def test_date_just_over_the_limit_is_rejected():
    assert _parse_date("x" * (_MAX_DATE_CHARS + 1)) is None


def test_surrounding_whitespace_does_not_count_toward_the_limit():
    assert _parse_date(" " * 1000 + _RFC822 + "\n" * 1000) == _PARSED


def test_verbose_real_world_date_is_within_the_limit():
    verbose = "Monday, 02 January 2006 15:04:05 +0000 (Coordinated Universal Time)"
    assert _parse_date(verbose) == _PARSED
