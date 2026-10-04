"""A date that cannot exist must not take the feed down or come out as text.

Every value below either raised out of parse(), losing the whole feed, or was
returned as a timestamp that names no real moment.
"""

import datetime

import pytest

from fastfeedparser import parse
from fastfeedparser.main import _parse_date

_IMPOSSIBLE = [
    pytest.param("Wed, 31 Feb 2026 06:34:46 +0530", id="rfc822-offset-day"),
    pytest.param("31 Feb 2026 06:34:46 PST", id="rfc822-zone-name-day"),
    pytest.param("Wed, 31 Feb 2026 24:34:46 +0530", id="rfc822-offset-hour-24-day"),
    pytest.param("Fri, 31 Dec 9999 23:59:59 -0500", id="rfc822-offset-past-year-9999"),
    pytest.param("Fri, 31 Dec 9999 24:59:59 -0500", id="rfc822-hour-24-past-year-9999"),
    pytest.param("2023-02-30T24:00:00Z", id="iso-hour-24-day"),
    pytest.param("9999-12-31T24:00:00", id="iso-hour-24-past-year-9999"),
    pytest.param("Wed, 31 Feb 2026 06:34:46 GMT", id="rfc822-utc-day"),
    pytest.param("Mon, 82 Jan 2006 15:04:05 GMT", id="rfc822-utc-day-82"),
    pytest.param("Mon, 00 Jan 2006 15:04:05 GMT", id="rfc822-utc-day-0"),
    pytest.param("Wed, 29 Feb 2023 15:04:05 GMT", id="rfc822-utc-feb-29-not-leap"),
    pytest.param("Mon, 02 Jan 2006 25:04:05 GMT", id="rfc822-utc-hour-25"),
    pytest.param("Mon, 02 Jan 2006 15:61:05 GMT", id="rfc822-utc-minute-61"),
    pytest.param("Mon, 02 Jan 2006 15:04:61 +0000", id="rfc822-utc-second-61"),
    pytest.param("Mon, 2 Jan 2006 15:61:05 GMT", id="rfc822-utc-1-digit-day-minute-61"),
    pytest.param("Mon, 02 Jan 2006 24:61:00 GMT", id="rfc822-utc-hour-24-minute-61"),
    pytest.param("Wed, 31 Feb 2026 24:34846140530", id="number-too-large-for-a-date"),
]


# The general parsers read a year below 100 as a two-digit year, so these come
# out as dates in 2000 and 2001 instead of being dropped.
_YEAR_READ_AS_TWO_DIGITS = [
    pytest.param("Mon, 01 Jan 0001 00:00:00 +0530", id="rfc822-offset-before-year-1"),
    pytest.param("Mon, 02 Jan 0000 15:04:05 GMT", id="rfc822-utc-year-0"),
]


def _is_real_utc_timestamp(value: str) -> bool:
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.utcoffset() == datetime.timedelta(0) and value.isascii()


@pytest.mark.parametrize("value", _IMPOSSIBLE)
def test_impossible_date_gives_no_date(value):
    assert _parse_date.__wrapped__(value) is None


@pytest.mark.parametrize("value", _YEAR_READ_AS_TWO_DIGITS)
def test_year_out_of_range_gives_a_real_date_or_none(value):
    parsed = _parse_date.__wrapped__(value)
    assert parsed is None or _is_real_utc_timestamp(parsed)


@pytest.mark.parametrize("value", _IMPOSSIBLE + _YEAR_READ_AS_TWO_DIGITS)
def test_impossible_date_does_not_cost_the_feed(value):
    feed = (
        '<rss version="2.0"><channel><title>t</title>'
        f"<item><title>bad</title><pubDate>{value}</pubDate></item>"
        "<item><title>good</title><pubDate>Mon, 02 Jan 2006 15:04:05 GMT</pubDate></item>"
        "</channel></rss>"
    )
    bad, good = parse(feed).entries
    assert (bad.title, good.title) == ("bad", "good")
    assert "published" not in bad or _is_real_utc_timestamp(bad.published)
    assert good.published == "2006-01-02T15:04:05+00:00"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Thu, 29 Feb 2024 15:04:05 GMT", "2024-02-29T15:04:05+00:00"),
        ("Sun, 31 Dec 2006 23:59:59 +0000", "2006-12-31T23:59:59+00:00"),
        ("Mon, 02 Jan 2006 24:00:00 GMT", "2006-01-03T00:00:00+00:00"),
        ("Sun, 31 Dec 2006 24:04:05 GMT", "2007-01-01T00:04:05+00:00"),
        ("Mon, 02 Jan 2006 24:04:05 +0530", "2006-01-02T18:34:05+00:00"),
        ("Mon, 02 Jan 2006 15:04:05 +0530", "2006-01-02T09:34:05+00:00"),
        ("Mon, 2 Jan 2006 15:04:05 EST", "2006-01-02T20:04:05+00:00"),
        ("Sat, 01 Jan 0001 00:00:00 GMT", "0001-01-01T00:00:00+00:00"),
        ("Fri, 31 Dec 9999 23:59:59 GMT", "9999-12-31T23:59:59+00:00"),
        ("2006-01-02T24:00:00Z", "2006-01-03T00:00:00+00:00"),
    ],
)
def test_dates_at_the_edges_still_parse(value, expected):
    parsed = _parse_date.__wrapped__(value)
    assert parsed == expected
    assert _is_real_utc_timestamp(parsed)
