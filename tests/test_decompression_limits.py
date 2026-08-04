"""Bounded-decompression tests for _fetch_url_content.

A compressed response body must not be allowed to expand without limit:
~1.7KB of brotli inflates to 1GB, which is a cheap remote memory-exhaustion.
"""

import gzip
import tracemalloc
import zlib

import pytest

import fastfeedparser.main as main
from fastfeedparser.main import HAS_BROTLI, _fetch_url_content, _inflate_bounded

FEED = (
    b'<?xml version="1.0"?><rss version="2.0"><channel><title>ok</title>'
    b"<item><title>one</title></item></channel></rss>"
)


class _FakeHeaders(dict):
    def get_content_charset(self):
        return self.get("charset")


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict):
        self._body = body
        self.headers = _FakeHeaders(headers)

    def read(self, amt=None):
        return self._body if amt is None else self._body[:amt]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    def __init__(self, response):
        self._response = response

    def open(self, request, timeout=None):
        return self._response


def _serve(monkeypatch, body: bytes, headers: dict):
    """Point _fetch_url_content at a canned response instead of the network."""
    monkeypatch.setattr(
        main, "build_opener", lambda *a, **kw: _FakeOpener(_FakeResponse(body, headers))
    )


def _raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def test_gzip_bomb_is_refused(monkeypatch):
    monkeypatch.setattr(main, "_MAX_CONTENT_BYTES", 1024 * 1024)
    _serve(monkeypatch, gzip.compress(b"\0" * (64 * 1024 * 1024)), {"Content-Encoding": "gzip"})

    with pytest.raises(ValueError, match="decompressed response exceeds"):
        _fetch_url_content("https://attacker.example/feed")


def test_deflate_bomb_is_refused(monkeypatch):
    monkeypatch.setattr(main, "_MAX_CONTENT_BYTES", 1024 * 1024)
    _serve(monkeypatch, _raw_deflate(b"\0" * (64 * 1024 * 1024)), {"Content-Encoding": "deflate"})

    with pytest.raises(ValueError, match="decompressed response exceeds"):
        _fetch_url_content("https://attacker.example/feed")


@pytest.mark.skipif(not HAS_BROTLI, reason="brotli not installed")
def test_brotli_bomb_is_refused(monkeypatch):
    import brotli

    monkeypatch.setattr(main, "_MAX_CONTENT_BYTES", 1024 * 1024)
    _serve(
        monkeypatch,
        brotli.compress(b"\0" * (128 * 1024 * 1024), quality=5),
        {"Content-Encoding": "br"},
    )

    with pytest.raises(ValueError, match="decompressed response exceeds"):
        _fetch_url_content("https://attacker.example/feed")


@pytest.mark.skipif(not HAS_BROTLI, reason="brotli not installed")
def test_brotli_bomb_does_not_materialize_in_memory(monkeypatch):
    """The point of the cap: refusing must not require inflating the bomb first."""
    import brotli

    cap = 1024 * 1024
    monkeypatch.setattr(main, "_MAX_CONTENT_BYTES", cap)
    bomb = brotli.compress(b"\0" * (512 * 1024 * 1024), quality=5)
    _serve(monkeypatch, bomb, {"Content-Encoding": "br"})

    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="decompressed response exceeds"):
            _fetch_url_content("https://attacker.example/feed")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # 512MB bomb, 1MB cap: peak must stay near the cap, not near the payload.
    assert peak < 16 * 1024 * 1024, f"peak allocation was {peak} bytes"


def test_uncompressed_body_over_limit_is_refused(monkeypatch):
    monkeypatch.setattr(main, "_MAX_CONTENT_BYTES", 4096)
    _serve(monkeypatch, b"x" * 8192, {})

    with pytest.raises(ValueError, match="response body exceeds"):
        _fetch_url_content("https://attacker.example/feed")


def test_body_exactly_at_limit_is_allowed(monkeypatch):
    monkeypatch.setattr(main, "_MAX_CONTENT_BYTES", 4096)
    _serve(monkeypatch, b"x" * 4096, {})

    assert _fetch_url_content("https://example.com/feed") == b"x" * 4096


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", None])
def test_normal_feed_still_round_trips(monkeypatch, encoding):
    if encoding == "gzip":
        body = gzip.compress(FEED)
    elif encoding == "deflate":
        body = _raw_deflate(FEED)
    elif encoding == "br":
        if not HAS_BROTLI:
            pytest.skip("brotli not installed")
        import brotli

        body = brotli.compress(FEED)
    else:
        body = FEED

    _serve(monkeypatch, body, {"Content-Encoding": encoding} if encoding else {})
    assert _fetch_url_content("https://example.com/feed") == FEED


def test_multi_member_gzip_is_not_truncated(monkeypatch):
    """gzip.decompress joined concatenated members; the bounded path must too."""
    _serve(
        monkeypatch,
        gzip.compress(b"<rss>one</rss>") + gzip.compress(b"<rss>two</rss>"),
        {"Content-Encoding": "gzip"},
    )
    assert _fetch_url_content("https://example.com/feed") == b"<rss>one</rss><rss>two</rss>"


def test_truncated_gzip_is_not_accepted_as_partial_body(monkeypatch):
    """A cut-off stream must fail, not return a truncated feed as if complete."""
    _serve(monkeypatch, gzip.compress(FEED)[:-8], {"Content-Encoding": "gzip"})

    # gzip.decompress raised EOFError here; the bounded path raises zlib.error.
    with pytest.raises((zlib.error, EOFError)):
        _fetch_url_content("https://example.com/feed")


def test_inflate_bounded_returns_one_byte_over_limit():
    """The overflow signal the caller checks: limit+1 bytes, never a silent cut."""
    data = gzip.compress(b"\0" * 5000)
    assert len(_inflate_bounded(data, 16 + zlib.MAX_WBITS, 1000)) == 1001
    assert len(_inflate_bounded(data, 16 + zlib.MAX_WBITS, 5000)) == 5000
