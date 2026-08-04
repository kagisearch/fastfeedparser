"""Regression tests for GHSA-xwxr-vfq4-mq54.

A meta-refresh tag in an attacker-controlled HTML response must not be able
to steer the follow-up fetch at a non-http(s) scheme (file://, ftp://, data:).
"""

import pytest

import fastfeedparser.main as main
from fastfeedparser import parse
from fastfeedparser.main import (
    _extract_meta_refresh_url,
    _fetch_url_content,
    _SchemeRestrictedRedirectHandler,
)


def _meta_refresh_html(target: str) -> str:
    return (
        "<!doctype html><html><head>"
        f'<meta http-equiv="refresh" content="0; url={target}">'
        "</head><body>not a feed</body></html>"
    )


@pytest.mark.parametrize(
    "target",
    [
        "file:///etc/hosts",
        "FILE:///etc/hosts",
        "ftp://attacker.example/payload",
        "data:text/html,<html></html>",
        "jar:file:///etc/hosts!/",
    ],
)
def test_meta_refresh_rejects_non_http_scheme(target):
    html = _meta_refresh_html(target)
    assert _extract_meta_refresh_url(html, "http://attacker.example/x") is None


def test_meta_refresh_still_follows_http_targets():
    html = _meta_refresh_html("https://example.com/feed.xml")
    assert (
        _extract_meta_refresh_url(html, "http://attacker.example/x")
        == "https://example.com/feed.xml"
    )


def test_meta_refresh_still_follows_relative_targets():
    html = _meta_refresh_html("/index.xml")
    assert (
        _extract_meta_refresh_url(html, "https://example.com/feed/")
        == "https://example.com/index.xml"
    )


def test_fetch_url_content_refuses_file_scheme(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("SENTINEL-do-not-leak")

    with pytest.raises(ValueError, match="non-http"):
        _fetch_url_content(secret.as_uri())


@pytest.mark.parametrize(
    "url",
    ["ftp://attacker.example/payload", "data:text/plain,hello", "/etc/hosts"],
)
def test_fetch_url_content_refuses_other_non_http_schemes(url):
    with pytest.raises(ValueError, match="non-http"):
        _fetch_url_content(url)


def test_redirect_handler_refuses_non_http_location():
    handler = _SchemeRestrictedRedirectHandler()
    with pytest.raises(ValueError, match="non-http"):
        handler.redirect_request(
            None, None, 302, "Found", {}, "ftp://attacker.example/payload"
        )


def test_parse_does_not_leak_local_file_via_meta_refresh(tmp_path, monkeypatch):
    """End-to-end PoC from the advisory: the second fetch must not read the file."""
    secret = tmp_path / "secret.txt"
    secret.write_text("SENTINEL-do-not-leak")

    real_fetch = main._fetch_url_content
    calls = []

    def fake_fetch(url: str):
        calls.append(url)
        if len(calls) == 1:
            # Stands in for the attacker's web server, not for the library.
            return _meta_refresh_html(secret.as_uri())
        return real_fetch(url)

    monkeypatch.setattr(main, "_fetch_url_content", fake_fetch)

    with pytest.raises(ValueError) as excinfo:
        parse("http://attacker.example/feed")

    assert calls == ["http://attacker.example/feed"]
    chain = []
    exc = excinfo.value
    while exc is not None:
        chain.append(str(exc))
        exc = exc.__cause__ or exc.__context__
    assert "SENTINEL-do-not-leak" not in "\n".join(chain)
