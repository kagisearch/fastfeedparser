"""Dates in the two most common layouts are converted by position.

Each near miss below differs from a fast-path layout in one way and must
come out exactly as the general parsers produce it.
"""

import importlib.util

import pytest

from fastfeedparser.main import _parse_date

# The optional dateparser package answers a date for strings the other
# parsers reject, so "no date" is only the result without it.
_no_dateparser = pytest.mark.skipif(
    importlib.util.find_spec("dateparser") is not None,
    reason="dateparser guesses a date for this value",
)


def _parse(value):
    # Bypass the LRU cache so every case runs the parsing code.
    return _parse_date.__wrapped__(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Mon, 02 Jan 2006 15:04:05 GMT", "2006-01-02T15:04:05+00:00"),
        ("Mon, 02 Jan 2006 15:04:05 UTC", "2006-01-02T15:04:05+00:00"),
        ("Mon, 02 Jan 2006 15:04:05 WET", "2006-01-02T15:04:05+00:00"),
        ("Mon, 02 Jan 2006 15:04:05 +0000", "2006-01-02T15:04:05+00:00"),
        ("Mon, 02 Jan 2006 15:04:05 -0000", "2006-01-02T15:04:05+00:00"),
        ("Sun, 31 Dec 2023 23:59:59 GMT", "2023-12-31T23:59:59+00:00"),
        ("  Mon, 02 Jan 2006 15:04:05 GMT\n", "2006-01-02T15:04:05+00:00"),
    ],
)
def test_rfc822_utc_layout(value, expected):
    assert _parse(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(
            "Mon, 02 Jan 2006 24:04:05 GMT", "2006-01-03T00:04:05+00:00", id="hour-24"
        ),
        pytest.param(
            "Mon, 02 Jan 2006 15:04:05 EST", "2006-01-02T20:04:05+00:00", id="zone-name"
        ),
        pytest.param(
            "Mon, 02 Jan 2006 15:04:05 +0530", "2006-01-02T09:34:05+00:00", id="offset"
        ),
        pytest.param(
            "Mon, 02 jan 2006 15:04:05 GMT", "2006-01-02T15:04:05+00:00", id="lower-month"
        ),
        pytest.param(
            "Mon, 02 JAN 2006 15:04:05 GMT", "2006-01-02T15:04:05+00:00", id="upper-month"
        ),
        pytest.param("Mon, 02 Xyz 2006 15:04:05 GMT", None, id="unknown-month"),
        pytest.param(
            "Mon, 2 Jan 2006 15:04:05 +0000", "2006-01-02T15:04:05+00:00", id="1-digit-day"
        ),
        pytest.param(
            "Mon,  2 Jan 2006 15:04:05 GMT", "2006-01-02T15:04:05+00:00", id="padded-day"
        ),
        pytest.param(
            "Mon, 02 Jan 2006 15:04:05  GMT", "2006-01-02T15:04:05+00:00", id="two-spaces"
        ),
        pytest.param(
            "Mon, 02 Jan 2006 15:04:05\tGMT", "2006-01-02T15:04:05+00:00", id="tab"
        ),
        pytest.param(
            "Mon, 02 Jan 2006 15:04:05 GMT+1", "2006-01-02T15:04:05+00:00", id="trailing"
        ),
        pytest.param(
            "Mon, 02 Jan 2006 15.04.05 GMT", "2006-01-02T15:04:05+00:00", id="dot-clock"
        ),
        pytest.param(
            "Mon, 02 Jan 2006 1x:04:05 GMT", None, id="non-digit-hour"
        ),
        pytest.param(
            "Mon, \u0660\u0662 Jan 2006 15:04:05 GMT", "2006-01-02T15:04:05+00:00", id="non-ascii-day"
        ),
    ],
)
def test_rfc822_near_misses_use_the_general_parsers(value, expected):
    assert _parse(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2006-01-02T15:04:05Z", "2006-01-02T15:04:05+00:00"),
        ("2006-01-02T15:04:05z", "2006-01-02T15:04:05+00:00"),
        ("2006-01-02T15:04:05+00:00", "2006-01-02T15:04:05+00:00"),
        ("2024-02-29T00:00:00Z", "2024-02-29T00:00:00+00:00"),
    ],
)
def test_iso_utc_layout(value, expected):
    assert _parse(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param("2006-01-02T24:00:00Z", "2006-01-03T00:00:00+00:00", id="hour-24"),
        pytest.param("2006-01-02t15:04:05Z", "2006-01-02T15:04:05+00:00", id="lower-t"),
        pytest.param("2006-01-02 15:04:05Z", "2006-01-02T15:04:05+00:00", id="space"),
        pytest.param(
            "2006-01-02T15:04:05.5Z", "2006-01-02T15:04:05.500000+00:00", id="fraction"
        ),
        pytest.param(
            "2006-01-02T15:04:05-00:00", "2006-01-02T15:04:05+00:00", id="minus-zero"
        ),
        pytest.param(
            "2006-01-02T15:04:05+05:30", "2006-01-02T09:34:05+00:00", id="offset"
        ),
        pytest.param("2006-02-30T15:04:05Z", None, id="day-out-of-range", marks=_no_dateparser),
        pytest.param("2006-13-02T15:04:05Z", None, id="month-out-of-range", marks=_no_dateparser),
        pytest.param("2006-01-02T15:04:60Z", None, id="second-out-of-range", marks=_no_dateparser),
        pytest.param("2023-02-29T10:00:00Z", "2023-02-28T10:00:00+00:00", id="feb-29"),
        pytest.param("0000-01-02T15:04:05Z", None, id="year-zero", marks=_no_dateparser),
    ],
)
def test_iso_near_misses_use_the_general_parsers(value, expected):
    assert _parse(value) == expected
