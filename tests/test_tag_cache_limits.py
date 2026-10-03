"""Regression tests for the RSS item tag classification cache.

lxml tags are "{namespace-uri}local", so a feed that declares one long
namespace URI and uses many distinct child names in it would otherwise pin a
copy of the URI per cached tag for the life of the process.
"""

import pytest

import fastfeedparser.main as main
from fastfeedparser import parse

_ATOM_NS = "http://www.w3.org/2005/Atom"
_LONG_URI = "urn:" + "a" * 10_000


@pytest.fixture(autouse=True)
def lxml_path_only(monkeypatch):
    """The tag cache belongs to the lxml path; the native core never fills it."""
    monkeypatch.setattr(main, "_core", None)


def _rss_with_long_namespace(item_children: str) -> str:
    return (
        f'<rss version="2.0" xmlns:x="{_LONG_URI}"><channel><title>feed</title>'
        f"<item>{item_children}</item></channel></rss>"
    )


def test_long_tags_are_not_retained_in_tag_cache():
    main._rss_tag_info_cache.cache_clear()
    children = "<title>entry</title>" + "".join(f"<x:t{i}/>" for i in range(50))

    parsed = parse(_rss_with_long_namespace(children))

    assert parsed.entries[0].title == "entry"
    cache = main._rss_tag_info_cache(_ATOM_NS)
    # Short tags are still memoized, so the length check below is not vacuous.
    assert "title" in cache
    cached_tags = [tag for tag in cache if isinstance(tag, str)]
    assert max(len(tag) for tag in cached_tags) <= main._TAG_CACHE_MAX_TAG_LEN


def test_uncached_long_tags_are_still_parsed():
    main._rss_tag_info_cache.cache_clear()
    children = (
        "<x:title>entry</x:title>"
        "<x:category>news</x:category>"
        "<x:category>tech</x:category>"
    )

    # Parse twice: the second pass must classify the long tags again rather
    # than rely on a cache hit.
    for _ in range(2):
        entry = parse(_rss_with_long_namespace(children)).entries[0]
        assert entry.title == "entry"
        assert [tag["term"] for tag in entry.tags] == ["news", "tech"]

    cache = main._rss_tag_info_cache(_ATOM_NS)
    assert not any(isinstance(tag, str) and _LONG_URI in tag for tag in cache)
