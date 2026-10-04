from __future__ import annotations

import datetime
from email.utils import parsedate_to_datetime
import html as _html_mod
import json
import os
import re
import threading
import zlib
from functools import lru_cache, partial
from xml.sax.saxutils import escape as _xml_escape

try:
    import brotli

    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

try:
    import orjson

    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, Literal
from urllib.parse import urljoin, urlsplit
from urllib.request import (
    HTTPErrorProcessor,
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from dateutil import parser as dateutil_parser
from lxml import etree

if TYPE_CHECKING:
    from typing import Protocol

    from lxml.etree import _Element

    class _ElementValueGetter(Protocol):
        def __call__(self, path: str, attribute: Optional[str] = None) -> Optional[str]: ...

_FeedType = Literal["rss", "atom", "rdf"]


_UTC = datetime.timezone.utc

# Pre-compiled regex patterns for performance
_RE_XML_DECL_ENCODING = re.compile(
    r'(<\?xml[^>]*encoding=["\'])([^"\']+)(["\'][^>]*\?>)', re.IGNORECASE
)
_RE_XML_DECL_ENCODING_BYTES = re.compile(
    rb'(<\?xml[^>]*encoding=["\'])([^"\']+)(["\'][^>]*\?>)', re.IGNORECASE
)
_RE_DOUBLE_XML_DECL_BYTES = re.compile(rb"<\?xml\?xml\s+", re.IGNORECASE)
_RE_DOUBLE_CLOSE_BYTES = re.compile(rb"\?\?>\s*")
# The patterns below run over whole, untrusted documents and must stay linear
# (GHSA-3r75-qcwc-78f2). The lookbehind limits attempts to the start of each
# whitespace run; a match can only begin there.
_RE_UNQUOTED_ATTR_BYTES = re.compile(rb'(?<!\s)(\s+[\w:]+)=([^\s>"\']+)')
_RE_UTF16_ENCODING_BYTES = re.compile(
    rb'(<\?xml[^>]*encoding=["\'])utf-16(-le|-be)?(["\'][^>]*\?>)', re.IGNORECASE
)
# The tag body excludes "<" so a scan from one "<link" stops at the next tag
# instead of running to a distant ">". The "next tag is not </link>" check
# runs once per tag, then the match ends before the last newline of the
# whitespace run that follows.
_RE_UNCLOSED_LINK_BYTES = re.compile(
    rb"<link([^<>]*[^/<>])>(?=\s*<(?!/link\s*>))\s*(?=\n)", re.MULTILINE
)
_RE_UNICODE_LINE_SEP_BYTES = re.compile(rb"\xe2\x80[\xa8\xa9]")
_RE_FEB29 = re.compile(r"(\d{4})-02-29")
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_ISO_TZ_NO_COLON = re.compile(r"([+-]\d{2})(\d{2})$")
_RE_ISO_TZ_HOUR_ONLY = re.compile(r"([+-]\d{2})$")
_RE_ISO_FRACTION = re.compile(r"\.(\d{7,})(?=(?:[+-]\d{2}:?\d{2}|Z|$))", re.IGNORECASE)
_RE_RFC822 = re.compile(
    r"(?:\w{3},\s+)?(\d{1,2})\s+(\w{3})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\s+([+-]\d{4}|[A-Z]{2,5})"
)
_RE_HOUR24 = re.compile(r"(\d{4}-\d{2}-\d{2})[T ]24:(\d{2}):(\d{2})")
# A UTC timestamp already in the form datetime.isoformat() gives it. Hour 24
# is excluded because fromisoformat() reads it as 00 on the next day.
_RE_ISO_UTC_SECONDS = re.compile(
    r"\d{4}-\d\d-\d\dT(?!24)\d\d:\d\d:\d\d\+00:00", re.ASCII
)
_MONTHS_RFC822: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_XML_NS = "{http://www.w3.org/XML/1998/namespace}"
_XML_LANG_ATTR = _XML_NS + "lang"
_XML_BASE_ATTR = _XML_NS + "base"
_RDF_ABOUT_ATTR = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
_RSS_CONTENT_ENCODED_TAG = "{http://purl.org/rss/1.0/modules/content/}encoded"
_DC_SUBJECT_TAG = "{http://purl.org/dc/elements/1.1/}subject"
_MEDIA_CONTENT_TAG = "{http://search.yahoo.com/mrss/}content"
_MEDIA_THUMBNAIL_TAG = "{http://search.yahoo.com/mrss/}thumbnail"
_MEDIA_TITLE_TAG = "{http://search.yahoo.com/mrss/}title"
_MEDIA_TEXT_TAG = "{http://search.yahoo.com/mrss/}text"
_MEDIA_DESCRIPTION_TAG = "{http://search.yahoo.com/mrss/}description"
_MEDIA_CREDIT_TAG = "{http://search.yahoo.com/mrss/}credit"


@lru_cache(maxsize=4)
def _atom_ns_tags(atom_ns: str) -> dict[str, str]:
    """Pre-compute namespace-prefixed tag strings once per unique namespace.

    Avoids thousands of redundant f-string / concatenation operations when
    parsing feeds with many entries.
    """
    ns = f"{{{atom_ns}}}"
    is_atom_03 = atom_ns == "http://purl.org/atom/ns#"
    return {
        "ns": ns,
        "id": ns + "id",
        "title": ns + "title",
        "summary": ns + "summary",
        "link": ns + "link",
        "content": ns + "content",
        "author": ns + "author",
        "author_name": ns + "author/" + ns + "name",
        "category": ns + "category",
        "published": ns + ("issued" if is_atom_03 else "published"),
        "updated": ns + ("modified" if is_atom_03 else "updated"),
        "pub_fallback": ns + ("published" if is_atom_03 else "issued"),
        "upd_fallback": ns + ("updated" if is_atom_03 else "modified"),
    }


class FastFeedParserDict(dict):
    """A dictionary that allows access to its keys as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'FastFeedParserDict' object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _detect_xml_encoding(content: bytes) -> str:
    """Detect encoding from XML declaration or BOM.

    Returns the detected encoding or 'utf-8' as default.
    """
    # Check for BOM (Byte Order Mark)
    if content.startswith(b"\xff\xfe"):
        return "utf-16"
    elif content.startswith(b"\xfe\xff"):
        return "utf-16"
    elif content.startswith(b"\xef\xbb\xbf"):
        return "utf-8"

    encoding_match = _RE_XML_DECL_ENCODING_BYTES.search(content[:2000])
    if encoding_match:
        try:
            return encoding_match.group(2).decode("ascii", errors="replace").lower()
        except Exception:
            return "utf-8"

    return "utf-8"


_XML_DECL_SCAN_CHARS = 2048


def _ensure_utf8_xml_declaration(content: str) -> str:
    """Ensure the XML declaration's encoding matches the UTF-8 bytes we emit."""
    stripped = content.lstrip()
    if not stripped.startswith("<?xml"):
        return content
    # Only rewrite within the head of the document: the declaration is the
    # first thing in it, and the pattern is quadratic on a whole document.
    start = len(content) - len(stripped)
    end = start + _XML_DECL_SCAN_CHARS
    head = _RE_XML_DECL_ENCODING.sub(r"\1utf-8\3", content[start:end], count=1)
    return content[:start] + head + content[end:]


def _clean_feed_bytes(content: bytes) -> bytes:
    """Clean feed bytes by extracting the XML document (if it's embedded in junk)."""
    stripped_content = content.lstrip()
    preview = stripped_content[:2000]
    preview_lower = preview.lower()

    # Skip UTF-8 BOM when doing ASCII prefix checks
    if preview_lower.startswith(b"\xef\xbb\xbf"):
        preview_lower = preview_lower[3:]
        stripped_content = stripped_content[3:]

    if preview_lower.startswith((b"<?xml", b"<rss", b"<feed", b"<rdf")):
        return stripped_content

    if preview_lower.startswith(b"<!doctype html") or preview_lower.startswith(
        b"<html"
    ):
        raise ValueError("Content appears to be HTML, not a valid RSS/Atom feed")

    xml_start_patterns = (
        b"<?xml",
        b"<rss",
        b"<feed",
        b"<rdf:rdf",
        b"<?xml-stylesheet",
    )

    # Search for XML start patterns without splitting entire content into lines.
    # For large feeds (multi-MB), splitlines() creates thousands of byte string
    # objects; find() scans in-place with zero allocations.
    search_limit = min(len(content), 8192)
    search_chunk = content[:search_limit].lower()
    earliest = -1
    for pattern in xml_start_patterns:
        idx = search_chunk.find(pattern)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
    if earliest != -1:
        return content[earliest:]

    if b"<script>" in preview_lower or b"<body>" in preview_lower:
        raise ValueError("Content appears to be HTML, not a valid RSS/Atom feed")

    return content


def _fix_xml_header_bytes(content: bytes, actual_encoding: str = "utf-8") -> bytes:
    # XML declarations and encoding definitions live at the top of the file.
    # Run declaration-fixing regexes only on the first 2 KB to avoid scanning
    # multi-megabyte payloads with patterns that can only match the header.
    header = content[:2048]
    tail = content[2048:]

    # Fix double XML declarations like "<?xml?xml version="1.0"?>"
    header = _RE_DOUBLE_XML_DECL_BYTES.sub(b"<?xml ", header)

    # Fix double closing ?> in XML declaration like "??>>"
    header = _RE_DOUBLE_CLOSE_BYTES.sub(b"?>", header)

    # Update encoding in XML declaration to match actual encoding when a feed was transcoded.
    if actual_encoding.lower() != "utf-16":
        replacement = (
            rb"\1" + actual_encoding.encode("ascii", errors="replace") + rb"\3"
        )
        header = _RE_UTF16_ENCODING_BYTES.sub(replacement, header)

    return header + tail


def _repair_xml_body_bytes(content: bytes) -> bytes:
    """Rewrite body-wide syntax errors. These patterns also match article text."""
    # Fix malformed attribute syntax like rss:version=2.0 (missing quotes)
    content = _RE_UNQUOTED_ATTR_BYTES.sub(rb'\1="\2"', content)

    # Fix unclosed link tags - common in Atom feeds
    return _RE_UNCLOSED_LINK_BYTES.sub(rb"<link\1/>", content)


def _prepare_xml_bytes(xml_content: str | bytes) -> tuple[bytes, bool]:
    """Return the cleaned document and whether its header looked malformed."""
    if isinstance(xml_content, bytes):
        cleaned = _clean_feed_bytes(xml_content)
        if not cleaned:
            raise ValueError("Empty content")

        # Replace Unicode LINE SEPARATOR (U+2028) and PARAGRAPH SEPARATOR (U+2029)
        # with regular newlines — these are invalid in XML 1.0 and cause lxml to fail.
        # These are extremely rare; probe a small prefix to avoid full O(n) scan on
        # multi-MB feeds.  If neither appears in the first 64 KB, skip the scan.
        # One regex pass is ~7x faster here than two bytes `in` searches.
        if _RE_UNICODE_LINE_SEP_BYTES.search(cleaned, 0, 65536):
            cleaned = cleaned.replace(b"\xe2\x80\xa8", b"\n").replace(
                b"\xe2\x80\xa9", b"\n"
            )

        detected_encoding = _detect_xml_encoding(cleaned)
        actual_encoding = detected_encoding
        if detected_encoding.startswith("utf-16") and b"\x00" not in cleaned[:200]:
            actual_encoding = "utf-8"

        looks_malformed = (
            b"?xml?xml" in cleaned[:200].lower()
            or b"??>" in cleaned[:200]
            or (
                b"rss:" in cleaned[:500].lower()
                and b"xmlns:rss" not in cleaned[:1000].lower()
            )
            or (b"utf-16" in cleaned[:200].lower() and actual_encoding != "utf-16")
        )
        if looks_malformed:
            cleaned = _fix_xml_header_bytes(cleaned, actual_encoding=actual_encoding)
        return cleaned, looks_malformed

    # Str input: fix encoding declaration, encode to bytes, then use bytes path.
    xml_content = _ensure_utf8_xml_declaration(xml_content)
    return _prepare_xml_bytes(xml_content.encode("utf-8", errors="replace"))


# libxml2 reads a CDATA section one character at a time, and CDATA is where
# many feeds keep their HTML. A section of at least this many bytes is cut out
# of the bytes lxml parses and decoded here instead, which is much faster.
_CDATA_LIFT_MIN_BYTES = 1024
# A document with no CDATA this early is not scanned at all.
_CDATA_PROBE_BYTES = 16384
# The scan visits every comment, instruction and section in Python. It stops
# once it has passed this many without lifting, so a feed whose sections are
# all small pays for a few dozen visits at most. Every feed in the benchmark
# corpus that has a large section reaches its first one within 22 visits and
# the next one within 42.
_CDATA_SCAN_SKIP_BASE = 32
_CDATA_SCAN_SKIP_PER_LIFT = 64
# The scan also gives up when nothing liftable starts this early, so a large
# feed with a few scattered small sections is not walked to its end. The first
# lift in every corpus feed starts within 8 KB.
_CDATA_FIRST_LIFT_BYTES = 65536
# How far before a section its parent's start tag may begin.
_CDATA_PARENT_WINDOW = 256
# A lifted section leaves a placeholder between these two private-use
# characters.
_CDATA_MARK_START = "\ue000"
_CDATA_MARK_END = "\ue001"
_RE_SCAN_TOKEN_START = re.compile(rb"<(?=!\[CDATA\[|!--|!DOCTYPE|\?)")
_RE_CDATA_END = re.compile(rb"\]\]>")
# Elements whose text every reader passes through _restore_lifted_cdata.
# Inside xhtml content, which is serialized from the tree,
# _restore_lifted_cdata_in_xml puts their text back.
_RE_LIFT_PARENT_START = re.compile(
    rb"<(description|content:encoded|content|summary)(?:\s[^<>]*)?(?<!/)>\s*\Z"
)
_RE_LIFT_PARENT_END = re.compile(
    rb"\s*</(description|content:encoded|content|summary)\s*>"
)
_RE_CDATA_PLACEHOLDER = re.compile(
    _CDATA_MARK_START + "[0-9a-f]+" + _CDATA_MARK_END
)
# Control characters XML does not allow. libxml2 ends a CDATA section at one.
_XML_INVALID_CONTROL_BYTES = bytes(c for c in range(32) if c not in (9, 10, 13))


def _liftable_section_text(
    content: bytes, region_start: int, start: int, end: int
) -> Optional[str]:
    """Text of the CDATA section content[start:end + 3], or None to leave it.

    The section is lifted only when it is its parent's whole text apart from
    whitespace: the parent's start tag ends just before it, inside
    content[region_start:start], and the parent's end tag follows it. A
    section libxml2 would not read to its end, because it holds invalid UTF-8
    or a character XML does not allow, is left for libxml2.
    """
    window_start = max(region_start, start - _CDATA_PARENT_WINDOW)
    parent = _RE_LIFT_PARENT_START.search(content, window_start, start)
    if parent is None:
        return None
    closing = _RE_LIFT_PARENT_END.match(content, end + 3)
    if closing is None or closing.group(1) != parent.group(1):
        return None
    section = content[start + 9 : end]
    if len(section.translate(None, _XML_INVALID_CONTROL_BYTES)) != len(section):
        return None
    # An XML parser reads every line ending as "\n".
    if b"\r" in section:
        section = section.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    try:
        text = section.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\ufffe" in text or "\uffff" in text:
        return None
    return text


def _lift_large_cdata(content: bytes) -> tuple[bytes, dict[str, str]]:
    """Replace large CDATA sections with placeholders.

    Returns the bytes to parse and each lifted section's text keyed by its
    placeholder; the same `content` object and an empty dict when nothing is
    lifted. Placeholders carry a random value, so document text cannot name
    one. Comments, processing instructions and every CDATA section are skipped
    whole, so a section opener inside one of them is never taken for a real
    one. Documents that carry a DOCTYPE, start as UTF-16 or UTF-32, or declare
    another encoding are left alone. The declaration is only read loosely
    here; _parse_xml_root_lifting_cdata checks the encoding libxml2 used.
    """
    lifted: dict[str, str] = {}
    if content.find(b"<![CDATA[", 0, _CDATA_PROBE_BYTES) == -1:
        return content, lifted
    if b"\x00" in content[:4]:
        return content, lifted
    if _detect_xml_encoding(content) not in ("utf-8", "utf8"):
        return content, lifted

    mark = _CDATA_MARK_START + os.urandom(8).hex()
    pieces: list[bytes] = []
    copied = 0  # content[:copied] is accounted for in pieces
    pos = 0  # end of the last comment, instruction or section
    skipped = 0
    while skipped <= _CDATA_SCAN_SKIP_BASE + _CDATA_SCAN_SKIP_PER_LIFT * len(lifted):
        limit = len(content) if lifted else _CDATA_FIRST_LIFT_BYTES
        token = _RE_SCAN_TOKEN_START.search(content, pos, limit)
        if token is None:
            break
        start = token.start()
        kind = content[start + 1 : start + 3]
        if kind == b"![":
            end_match = _RE_CDATA_END.search(content, start + 9)
            if end_match is None:
                break
            end = end_match.start()
            if end - start - 9 >= _CDATA_LIFT_MIN_BYTES:
                text = _liftable_section_text(content, pos, start, end)
                if text is not None:
                    placeholder = f"{mark}{len(lifted)}{_CDATA_MARK_END}"
                    pieces.append(content[copied:start])
                    pieces.append(placeholder.encode())
                    lifted[placeholder] = text
                    copied = pos = end + 3
                    continue
            pos = end + 3
        elif kind == b"!-":
            end = content.find(b"-->", start + 4)
            if end == -1:
                break
            pos = end + 3
        elif kind == b"!D":
            return content, {}
        else:
            end = content.find(b"?>", start + 2)
            if end == -1:
                break
            pos = end + 2
        skipped += 1

    if not lifted:
        return content, lifted
    pieces.append(content[copied:])
    return b"".join(pieces), lifted


def _restore_lifted_cdata(text: str, lifted: dict[str, str]) -> str:
    """Put a lifted section back into the text of the element it came from.

    Text that holds the marker characters without being a placeholder is
    returned unchanged.
    """
    start = text.find(_CDATA_MARK_START)
    if start == -1:
        return text
    end = text.find(_CDATA_MARK_END, start)
    section = lifted.get(text[start : end + 1])
    if section is None:
        return text
    if start == 0 and end == len(text) - 1:
        return section
    return text[:start] + section + text[end + 1 :]


def _restore_lifted_cdata_in_xml(serialized: str, lifted: dict[str, str]) -> str:
    """Put lifted sections back into serialized XML, escaped as text."""

    def section(match: re.Match[str]) -> str:
        text = lifted.get(match.group(0))
        return match.group(0) if text is None else _xml_escape(text)

    return _RE_CDATA_PLACEHOLDER.sub(section, serialized)


def _parse_json_feed(
    json_data: dict,
    *,
    include_content: bool = True,
    include_tags: bool = True,
    include_enclosures: bool = True,
) -> FastFeedParserDict:
    """Parse a JSON Feed and convert to FastFeedParserDict format.

    JSON Feed spec: https://jsonfeed.org/
    """
    feed = FastFeedParserDict()

    # Parse feed-level metadata
    feed_info = FastFeedParserDict()
    feed_info["title"] = json_data.get("title", "")
    feed_info["link"] = json_data.get("home_page_url", "")
    feed_info["subtitle"] = json_data.get("description", "")
    feed_info["id"] = json_data.get("feed_url", "")
    feed_info["language"] = json_data.get("language")

    # Add feed icon
    icon = json_data.get("icon")
    if icon:
        feed_info["icon"] = icon
    favicon = json_data.get("favicon")
    if favicon:
        feed_info["favicon"] = favicon

    # Add feed authors
    authors = json_data.get("authors")
    if authors and len(authors) > 0:
        feed_info["author"] = authors[0].get("name", "")

    # Add links
    feed_info["links"] = []
    home_page_url = json_data.get("home_page_url")
    if home_page_url:
        feed_info["links"].append(
            {"rel": "alternate", "type": "text/html", "href": home_page_url}
        )
    feed_url = json_data.get("feed_url")
    if feed_url:
        feed_info["links"].append(
            {"rel": "self", "type": "application/json", "href": feed_url}
        )

    feed["feed"] = feed_info

    # Parse items
    entries = []
    for item in json_data.get("items", []):
        entry = FastFeedParserDict()

        entry["id"] = item.get("id", item.get("url", ""))
        entry["title"] = item.get("title", "")
        entry["link"] = item.get("url", "")

        # Handle content - prefer content_html, fall back to content_text
        content_html = item.get("content_html")
        content_text = item.get("content_text")
        summary = item.get("summary", "")

        if content_html:
            if include_content:
                entry["content"] = [{"type": "text/html", "value": content_html}]
            entry["description"] = summary
        elif content_text:
            if include_content:
                entry["content"] = [{"type": "text/plain", "value": content_text}]
            entry["description"] = summary or content_text[:512]
        else:
            entry["description"] = summary

        # Parse dates
        date_published = item.get("date_published")
        if date_published:
            entry["published"] = _parse_date(date_published)
        date_modified = item.get("date_modified")
        if date_modified:
            entry["updated"] = _parse_date(date_modified)

        # Add images
        image = item.get("image")
        if image:
            entry["image"] = image
        banner_image = item.get("banner_image")
        if banner_image:
            entry["banner_image"] = banner_image

        # Add author
        authors = item.get("authors")
        if authors and len(authors) > 0:
            entry["author"] = authors[0].get("name", "")
        else:
            author = item.get("author")
            if author:
                # JSON Feed 1.0 uses singular 'author'
                entry["author"] = author.get("name", "")

        # Add tags
        tags = item.get("tags")
        if include_tags and tags:
            entry["tags"] = [
                {"term": tag, "scheme": None, "label": None} for tag in tags
            ]

        # Add attachments as enclosures
        attachments = item.get("attachments")
        if include_enclosures and attachments:
            enclosures = []
            for attachment in attachments:
                url = attachment.get("url", "")
                if url:  # Only add if has URL
                    enc = {
                        "url": url,
                        "type": attachment.get("mime_type", ""),
                    }
                    size = attachment.get("size_in_bytes")
                    if size:
                        enc["length"] = size
                    enclosures.append(enc)
            if enclosures:
                entry["enclosures"] = enclosures

        # Derive feedparser-compatible author fields
        _author = entry.get("author")
        if _author:
            _detail = {"name": _author}
            entry["author_detail"] = _detail
            entry["authors"] = [_detail]

        # Add links
        entry["links"] = []
        item_url = item.get("url")
        if item_url:
            entry["links"].append(
                {"rel": "alternate", "type": "text/html", "href": item_url}
            )
        external_url = item.get("external_url")
        if external_url:
            entry["links"].append(
                {"rel": "related", "type": "text/html", "href": external_url}
            )

        entries.append(entry)

    feed["entries"] = entries
    return feed


_ALLOWED_URL_SCHEMES = frozenset(("http", "https"))


def _is_http_url(url: str) -> bool:
    """True only for http(s) URLs.

    urlsplit lowercases the scheme and rejects one that is not
    ``[a-zA-Z][a-zA-Z0-9+.-]*``, so "FILE://" and "+file://" both fail here.
    urllib.request derives Request.type the same way (lowercased text before
    the first colon), so a URL that passes this check cannot dispatch to
    FileHandler, FTPHandler or DataHandler.
    """
    return urlsplit(url).scheme in _ALLOWED_URL_SCHEMES


class _SchemeRestrictedRedirectHandler(HTTPRedirectHandler):
    """Refuse 30x redirects that leave http(s).

    urllib's default handler permits redirecting to ftp:// as well, which
    would let a server reach a non-http scheme through the same fetch.
    """

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Any:
        if not _is_http_url(newurl):
            raise ValueError(f"refusing redirect to non-http(s) URL: {newurl[:100]}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Cap on both the bytes read off the wire and the bytes a compressed response
# is allowed to expand into. A ~1.7KB brotli body can otherwise inflate to 1GB.
_MAX_CONTENT_BYTES = 32 * 1024 * 1024
_BROTLI_INPUT_CHUNK = 64 * 1024


def _inflate_bounded(data: bytes, wbits: int, limit: int) -> bytes:
    """Inflate `data`, stopping once more than `limit` bytes are produced.

    Returns up to limit + 1 bytes so the caller can detect the overflow.
    `wbits` selects the container: 16 + MAX_WBITS for gzip, -MAX_WBITS for raw
    deflate. gzip bodies may concatenate members, which gzip.decompress joined,
    so a finished stream with trailing bytes is resumed rather than truncated.
    """
    parts: list[bytes] = []
    produced = 0
    while True:
        # max_length of 0 means "unlimited" to zlib; the produced > limit break
        # below keeps limit + 1 - produced at 1 or more.
        decompressor = zlib.decompressobj(wbits)
        parts.append(decompressor.decompress(data, limit + 1 - produced))
        produced += len(parts[-1])
        if produced > limit:
            break
        if not decompressor.eof:
            # Under the limit with the stream unfinished means it ended early.
            # The one-shot decompressors raised on truncated input (gzip with
            # EOFError, zlib with zlib.error); keep failing rather than
            # returning a partial body as if it were the whole feed.
            raise zlib.error("incomplete or truncated stream")
        data = decompressor.unused_data
        if not data:
            break
    return b"".join(parts)


def _brotli_decompress_bounded(data: bytes, limit: int) -> bytes:
    """Brotli-decompress `data`, producing at most limit + 1 bytes."""
    try:
        return brotli.Decompressor().process(data, output_buffer_limit=limit + 1)
    except TypeError:
        # brotli < 1.2.0 has no output cap, so feed the input in slices and
        # check the running total instead.
        pass
    decompressor = brotli.Decompressor()
    parts: list[bytes] = []
    produced = 0
    for start in range(0, len(data), _BROTLI_INPUT_CHUNK):
        parts.append(decompressor.process(data[start : start + _BROTLI_INPUT_CHUNK]))
        produced += len(parts[-1])
        if produced > limit:
            break
    return b"".join(parts)


def _fetch_url_content(url: str) -> str | bytes:
    if not _is_http_url(url):
        raise ValueError(f"refusing to fetch non-http(s) URL: {url[:100]}")
    accept_encoding = "gzip, deflate, br" if HAS_BROTLI else "gzip, deflate"
    request = Request(
        url,
        method="GET",
        headers={
            "Accept-Encoding": accept_encoding,
            "User-Agent": "fastfeedparser (+https://github.com/kagisearch/fastfeedparser)",
        },
    )
    opener = build_opener(_SchemeRestrictedRedirectHandler(), HTTPErrorProcessor())
    with opener.open(request, timeout=30) as response:
        limit = _MAX_CONTENT_BYTES
        content: bytes = response.read(limit + 1)
        if len(content) > limit:
            raise ValueError(f"response body exceeds {limit} bytes")
        content_encoding = response.headers.get("Content-Encoding")
        if content_encoding == "gzip":
            content = _inflate_bounded(content, 16 + zlib.MAX_WBITS, limit)
        elif content_encoding == "deflate":
            content = _inflate_bounded(content, -zlib.MAX_WBITS, limit)
        elif content_encoding == "br":
            if not HAS_BROTLI:
                raise ValueError(
                    "Received brotli-compressed response but 'brotli' is not installed"
                )
            content = _brotli_decompress_bounded(content, limit)
        if len(content) > limit:
            raise ValueError(f"decompressed response exceeds {limit} bytes")
        content_charset = response.headers.get_content_charset()
        if content_charset:
            try:
                return content.decode(content_charset)
            except (UnicodeDecodeError, LookupError):
                # Server lied about charset; return raw bytes and let
                # lxml detect encoding from the XML declaration/BOM.
                return content
        return content


def _maybe_parse_json_feed(
    content: str | bytes,
    *,
    include_content: bool = True,
    include_tags: bool = True,
    include_enclosures: bool = True,
) -> FastFeedParserDict | None:
    if isinstance(content, bytes):
        if not content.lstrip().startswith(b"{"):
            return None
    else:
        if not content.lstrip().startswith("{"):
            return None

    try:
        json_data = _json_loads(content)
    except Exception:
        return None

    if not isinstance(json_data, dict):
        return None

    version = json_data.get("version")
    if isinstance(version, str) and "jsonfeed.org" in version:
        return _parse_json_feed(
            json_data,
            include_content=include_content,
            include_tags=include_tags,
            include_enclosures=include_enclosures,
        )

    if isinstance(json_data.get("items"), list):
        return _parse_json_feed(
            json_data,
            include_content=include_content,
            include_tags=include_tags,
            include_enclosures=include_enclosures,
        )

    return None


class _XMLParsers:
    """A strict and a recover parser."""

    def __init__(self) -> None:
        self.strict = etree.XMLParser(
            ns_clean=True,
            recover=False,
            collect_ids=False,
            resolve_entities=False,
            huge_tree=True,
        )
        self.recover = etree.XMLParser(
            ns_clean=True,
            recover=True,
            collect_ids=False,
            resolve_entities=False,
            huge_tree=True,
        )


class _ThreadXMLParsers(threading.local, _XMLParsers):
    """A pair of parsers for each thread.

    lxml holds a parser's lock for a whole parse, so threads sharing one
    parser object parse one at a time.
    """


_THREAD_XML_PARSERS = _ThreadXMLParsers()
# One pair for all threads. Its lock makes the documents given to it parse one
# at a time, which is what keeps memory flat when many large ones arrive.
_SHARED_XML_PARSERS = _XMLParsers()

# With a parser each, every thread's tree exists at once. Threads keep their
# own parsers only while the trees in flight are estimated to stay under this.
# Ordinary feeds come to about 1.5 MB each, so up to 16 threads rarely reach it.
_MAX_TREE_BYTES_IN_FLIGHT = 32 * 1024 * 1024
# thread id -> estimated tree bytes of the document that thread is parsing with
# its own parsers; 0 if it has not picked parsers yet or uses the shared pair.
_IN_FLIGHT: dict[int, int] = {}
_TREE_SAMPLE_BYTES = 4096


def _reset_after_fork() -> None:
    # The child has only the forking thread, and the shared parsers' lock may
    # have been held by a thread that does not exist there.
    global _SHARED_XML_PARSERS
    _IN_FLIGHT.clear()
    _SHARED_XML_PARSERS = _XMLParsers()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _estimated_tree_bytes(xml_content: bytes) -> int:
    """Roughly what a parse of this document holds in memory while it runs.

    Four times the document plus 200 bytes for each tag: measured parses held
    3 to 7 times a text-heavy feed and about 110 bytes per tag of a feed made
    of small elements. Tags are counted in three samples, not the whole
    document.
    """
    size = len(xml_content)
    sample = _TREE_SAMPLE_BYTES
    if size <= 3 * sample:
        tags = xml_content.count(b"<")
    else:
        middle = size // 2
        sampled = (
            xml_content.count(b"<", 0, sample)
            + xml_content.count(b"<", middle, middle + sample)
            + xml_content.count(b"<", size - sample)
        )
        tags = sampled * size // (3 * sample)
    return 4 * size + 200 * tags


def _xml_parsers(xml_content: bytes) -> _XMLParsers:
    """The parsers to parse this document with.

    This thread's own, unless other threads are parsing and the trees in
    flight plus this one would pass _MAX_TREE_BYTES_IN_FLIGHT; then the shared
    pair. Nothing here waits or takes a lock, so a wrong or stale entry in
    _IN_FLIGHT can send documents to the shared pair but cannot block one.
    """
    if len(_IN_FLIGHT) <= 1:
        return _THREAD_XML_PARSERS
    thread_id = threading.get_ident()
    _IN_FLIGHT[thread_id] = _estimated_tree_bytes(xml_content)
    if sum(list(_IN_FLIGHT.values())) > _MAX_TREE_BYTES_IN_FLIGHT:
        _IN_FLIGHT[thread_id] = 0
        return _SHARED_XML_PARSERS
    return _THREAD_XML_PARSERS


def _parse_xml_root(xml_content: bytes) -> _Element:
    # The recover parser builds the same tree as the strict one for
    # well-formed input, so trying strict first would only add a wasted parse
    # for malformed documents.
    try:
        root = etree.fromstring(xml_content, parser=_xml_parsers(xml_content).recover)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Failed to parse XML content: {str(e)}")

    if root is None:
        preview = xml_content[:500].decode("utf-8", errors="replace").strip()
        if preview:
            raise ValueError(
                "Failed to parse XML: received content that couldn't be parsed as XML "
                f"(first 200 chars: {preview[:200]})"
            )
        raise ValueError("Failed to parse XML: received empty content")

    return root


def _parse_xml_root_lifting_cdata(
    xml_content: bytes,
) -> tuple[_Element, dict[str, str]]:
    """Parse a document with its large CDATA sections lifted out.

    Nothing is lifted while another thread is parsing: the scan and decode
    hold the GIL where libxml2 releases it, which costs more throughput than
    it saves once parses run side by side.

    Returns the root and the lifted sections. The scan in _lift_large_cdata
    follows the XML grammar, so it only matches what libxml2 does while the
    document is well-formed, and the placeholders are UTF-8. If libxml2
    reports any error on the bytes left after lifting, or read them in another
    encoding, the document is parsed again whole and nothing is lifted.
    """
    if len(_IN_FLIGHT) > 1:
        return _parse_xml_root(xml_content), {}
    parse_bytes, lifted = _lift_large_cdata(xml_content)
    if lifted:
        # This thread's own parser, so the error log read below is this parse's.
        parser = _THREAD_XML_PARSERS.recover
        try:
            root = etree.fromstring(parse_bytes, parser=parser)
        except etree.XMLSyntaxError:
            root = None
        if root is not None and not parser.error_log:
            encoding = root.getroottree().docinfo.encoding or "utf-8"
            if encoding.lower() in ("utf-8", "utf8"):
                return root, lifted
    return _parse_xml_root(xml_content), {}


def _parse_repairable_xml_root(xml_content: bytes) -> tuple[_Element, bytes]:
    """Parse a document whose header looked malformed.

    Returns the root and the bytes it was parsed from. The body is only
    repaired when the document does not parse as it is, because the repair
    patterns also rewrite matching article text.
    """
    try:
        parser = _xml_parsers(xml_content).strict
        return etree.fromstring(xml_content, parser=parser), xml_content
    except etree.XMLSyntaxError:
        xml_content = _repair_xml_body_bytes(xml_content)
        return _parse_xml_root(xml_content), xml_content


def _root_tag_local(root: _Element) -> str:
    return root.tag.split("}")[-1].lower() if "}" in root.tag else root.tag.lower()


def _extract_error_message(root: _Element, raw_bytes: Optional[bytes] = None) -> str:
    error_msg = root.text or ""

    if not error_msg:
        for tag in ["message", "title", "h1", "h2", "h3", "h4", "p", "code"]:
            try:
                elem = root.find(f".//{tag}")
                if elem is None:
                    elem = root.find(tag)
                if elem is not None and elem.text:
                    return elem.text
                elems = root.xpath(f".//*[local-name()='{tag}']")
                if elems and elems[0].text:
                    return elems[0].text
            except Exception:
                continue

    if not error_msg or len(error_msg.strip()) < 5:
        try:
            all_text = " ".join(
                text.strip() for text in root.itertext() if text and text.strip()
            )
            all_text = " ".join(all_text.split())
            if all_text:
                return all_text[:300]
        except Exception:
            pass

        # XML parser may strip children from malformed HTML (e.g. unquoted
        # attributes); re-parse with the lenient HTML parser as a fallback.
        if raw_bytes:
            try:
                html_root = etree.fromstring(raw_bytes, parser=etree.HTMLParser())
                all_text = " ".join(
                    t.strip() for t in html_root.itertext() if t and t.strip()
                )
                all_text = " ".join(all_text.split())
                if all_text:
                    return all_text[:300]
            except Exception:
                pass

        return "No error message"

    return error_msg


_NON_FEED_MESSAGES: dict[str, str] = {
    "html": "Received HTML page instead of feed",
    "div": "Received HTML fragment instead of feed",
    "body": "Received HTML fragment instead of feed",
    "br": "Received HTML fragment instead of feed",
    "status": "Feed server returned status message",
    "error": "Feed server returned error",
    "opml": "Received OPML document instead of feed (OPML is an outline format, not a feed)",
    "urlset": "Received XML sitemap instead of feed (sitemap is for search engines, not a feed)",
    "sitemapindex": "Received XML sitemap instead of feed (sitemap is for search engines, not a feed)",
}


def _raise_for_non_feed_root(
    root: _Element, root_tag_local: str, raw_bytes: Optional[bytes] = None
) -> None:
    base_msg = _NON_FEED_MESSAGES.get(root_tag_local)
    if base_msg is None:
        return

    error_msg = (
        _extract_error_message(root, raw_bytes).strip()[:300] or "No error message"
    )

    if error_msg != "No error message" and len(error_msg) > 10:
        raise ValueError(f"{base_msg}: {error_msg[:150]}")
    raise ValueError(base_msg)


# The optional quote owns the whitespace after it, so a whitespace run is only
# ever consumed one way (GHSA-3r75-qcwc-78f2).
_RE_META_REFRESH_URL = re.compile(r'url\s*=\s*(?:["\']\s*)?([^"\'>\s]+)', re.IGNORECASE)
_MAX_META_REDIRECTS = 3


def _extract_meta_refresh_url(content: str | bytes, base_url: str) -> str | None:
    """Extract redirect URL from an HTML meta-refresh tag."""
    html_bytes = content.encode("utf-8") if isinstance(content, str) else content
    try:
        doc = etree.fromstring(html_bytes, parser=etree.HTMLParser())
    except Exception:
        return None
    if doc is None:
        return None

    for meta in doc.iter("meta"):
        if (meta.get("http-equiv") or "").lower() == "refresh":
            match = _RE_META_REFRESH_URL.search(meta.get("content", ""))
            if match:
                url = urljoin(base_url, match.group(1))
                # urljoin keeps an absolute reference's own scheme, so an
                # attacker page can name file:// or ftp:// here. Apply the
                # same allow-list parse() applies to its source.
                if url != base_url and _is_http_url(url):
                    return url
    return None


_RE_ITEM_OR_META_START_BYTES = re.compile(rb"<(item|meta)", re.IGNORECASE)
# "http-equiv" in any letter case; the literal "-" keeps the scan fast.
_RE_HTTP_EQUIV_BYTES = re.compile(rb"-[eE][qQ][uU][iI][vV]")


def _html_reparse_may_find_more_items(xml_content: bytes, found: int) -> bool:
    """Whether an HTML re-parse could yield more than twice ``found`` items.

    The HTML parser builds an item element from a "<item" in the bytes, in
    any letter case, so their count bounds what it can find. The bound needs
    the parser to read the bytes as ASCII-compatible throughout. libxml2 can
    switch encoding on a <meta> tag, or on an "http-equiv ... charset=" run
    in raw text at the first non-ASCII byte, so a document with either, or
    one that does not start as ASCII-compatible, is always re-parsed.
    """
    if not xml_content.startswith(b"<") or b"\x00" in xml_content[:4]:
        return True
    limit = found * 2
    matches = _RE_ITEM_OR_META_START_BYTES.finditer(xml_content)
    for count, match in enumerate(matches, 1):
        if match.group(1)[:1] in b"mM" or count > limit:
            return True
    return _RE_HTTP_EQUIV_BYTES.search(xml_content) is not None


def _is_well_formed_xml(xml_content: bytes) -> bool:
    try:
        etree.fromstring(xml_content, parser=_xml_parsers(xml_content).strict)
    except etree.XMLSyntaxError:
        return False
    return True


def _detect_feed_structure(
    root: _Element, xml_content: bytes, root_tag_local: str
) -> tuple[_FeedType, _Element, list[_Element], Optional[str]]:
    feed_type: _FeedType
    atom_namespace: Optional[str] = None

    if root_tag_local == "rss":
        feed_type = "rss"
        channel = root.find("channel")
        if channel is None:
            for child in root:
                if not isinstance(child.tag, str):
                    continue
                tag_lower = child.tag.lower()
                if (
                    child.tag.endswith("}channel")
                    or child.tag == "channel"
                    or tag_lower == "rss:channel"
                    or tag_lower.endswith(":channel")
                ):
                    channel = child
                    break

        if channel is None:
            has_atom_elements = any(
                isinstance(child.tag, str)
                and child.tag
                in {"entry", "title", "subtitle", "updated", "id", "author", "link"}
                for child in root
            )
            if has_atom_elements:
                channel = root
            else:
                raise ValueError("Invalid RSS feed: missing channel element")
        elif len(channel) == 0 and any(
            isinstance(child.tag, str) and child.tag == "item" for child in root
        ):
            channel = root

        items = channel.findall("item")
        if not items:
            for child in channel:
                if not isinstance(child.tag, str):
                    continue
                tag_lower = child.tag.lower()
                if (
                    child.tag.endswith("}item")
                    or child.tag == "item"
                    or tag_lower == "rss:item"
                    or tag_lower.endswith(":item")
                ):
                    if not items:
                        items = []
                    items.append(child)
            if not items:
                items = channel.xpath(".//item") or channel.xpath(
                    ".//*[local-name()='item']"
                )

            if not items:
                items = channel.findall("entry")
                if not items:
                    for child in channel:
                        if not isinstance(child.tag, str):
                            continue
                        if child.tag.endswith("}entry") or child.tag == "entry":
                            if not items:
                                items = []
                            items.append(child)

        # The HTML re-parse rescues items a damaged document hides from the
        # XML parser. It does not know CDATA, so it would also read item
        # markup quoted inside an article as entries; a well-formed document
        # has nothing to rescue and is never given to it.
        if (
            len(items) < 5
            and len(xml_content) > 20000
            and _html_reparse_may_find_more_items(xml_content, len(items))
            and not _is_well_formed_xml(xml_content)
        ):
            try:
                html_parser = etree.HTMLParser(recover=True, collect_ids=False)
                html_root = etree.fromstring(xml_content, parser=html_parser)
                html_channel = html_root.find(".//channel")
                if html_channel is not None:
                    html_items = html_channel.findall(".//item")
                    if len(html_items) > len(items) * 2:
                        channel = html_channel
                        items = html_items
            except Exception:
                pass

        return feed_type, channel, items, atom_namespace

    if root_tag_local == "feed":
        if "}" not in root.tag:
            raise ValueError(f"Unknown Atom namespace in feed type: {root.tag}")
        atom_namespace = root.tag[1:].split("}", 1)[0]
        if atom_namespace not in {
            "http://www.w3.org/2005/Atom",
            "https://www.w3.org/2005/Atom",
            "http://purl.org/atom/ns#",
        }:
            raise ValueError(f"Unknown Atom namespace in feed type: {root.tag}")

        feed_type = "atom"
        channel = root
        items = channel.findall(f".//{{{atom_namespace}}}entry")
        return feed_type, channel, items, atom_namespace

    if root.tag == "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF":
        feed_type = "rdf"
        channel = root
        items = channel.findall(".//{http://purl.org/rss/1.0/}item") or channel.findall(
            "item"
        )
        return feed_type, channel, items, atom_namespace

    raise ValueError(f"Unknown feed type: {root.tag}")


def _parse_content(
    xml_content: str | bytes,
    *,
    include_content: bool = True,
    include_tags: bool = True,
    include_media: bool = True,
    include_enclosures: bool = True,
) -> FastFeedParserDict:
    """Parse feed content (XML or JSON) that has already been fetched."""
    json_feed = _maybe_parse_json_feed(
        xml_content,
        include_content=include_content,
        include_tags=include_tags,
        include_enclosures=include_enclosures,
    )
    if json_feed is not None:
        return json_feed

    xml_content, looks_malformed = _prepare_xml_bytes(xml_content)
    lifted_cdata: dict[str, str] = {}
    if looks_malformed:
        root, xml_content = _parse_repairable_xml_root(xml_content)
    else:
        root, lifted_cdata = _parse_xml_root_lifting_cdata(xml_content)
    root_tag_local = _root_tag_local(root)
    if lifted_cdata and root_tag_local in _NON_FEED_MESSAGES:
        # The error message is built from the document's text.
        root = _parse_xml_root(xml_content)
    _raise_for_non_feed_root(root, root_tag_local, xml_content)

    feed_type, channel, items, atom_namespace = _detect_feed_structure(
        root, xml_content, root_tag_local
    )

    feed = _parse_feed_info(
        channel,
        feed_type,
        atom_namespace,
        include_tags=include_tags,
        lifted_cdata=lifted_cdata,
    )

    # Detect once whether the tree holding the items has any element that
    # _parse_media_content reads.
    has_media_elements = include_media and (
        next(
            channel.getroottree().iter(_MEDIA_CONTENT_TAG, _MEDIA_THUMBNAIL_TAG),
            None,
        )
        is not None
    )

    # Parse entries — resolve parser once per feed instead of per entry
    entry_options = dict(
        include_content=include_content,
        include_tags=include_tags,
        include_media=include_media,
        include_enclosures=include_enclosures,
        lifted_cdata=lifted_cdata,
    )
    atom_ns = atom_namespace or "http://www.w3.org/2005/Atom"
    parse_entry: Callable[[_Element], FastFeedParserDict]
    if feed_type == "rss":
        parse_entry = partial(
            _parse_rss_feed_entry_fast,
            atom_ns=atom_ns,
            has_media_elements=has_media_elements,
            **entry_options,
        )
    elif feed_type == "atom":
        parse_entry = partial(
            _parse_atom_feed_entry_fast,
            atom_ns=atom_ns,
            has_media_elements=has_media_elements,
            **entry_options,
        )
    else:
        parse_entry = partial(
            _parse_feed_entry,
            feed_type=feed_type,
            atom_namespace=atom_namespace,
            has_media_elements=has_media_elements,
            **entry_options,
        )

    entries: list[FastFeedParserDict] = []
    feed["entries"] = entries
    for item in items:
        entry = parse_entry(item)
        # Ensure that titles and descriptions are always present
        entry["title"] = entry.get("title", "").strip()
        entry["description"] = entry.get("description", "").strip()
        # Derive feedparser-compatible author fields
        _author = entry.get("author")
        if _author:
            _detail = {"name": _author}
            entry["author_detail"] = _detail
            entry["authors"] = [_detail]
        entries.append(entry)

    return feed


def _parse_content_in_flight(
    content: str | bytes, **options: bool
) -> FastFeedParserDict:
    """Run _parse_content with this thread listed in _IN_FLIGHT."""
    thread_id = threading.get_ident()
    _IN_FLIGHT[thread_id] = 0
    try:
        return _parse_content(content, **options)
    finally:
        _IN_FLIGHT.pop(thread_id, None)


def parse(
    source: str | bytes,
    *,
    include_content: bool = True,
    include_tags: bool = True,
    include_media: bool = True,
    include_enclosures: bool = True,
) -> FastFeedParserDict:
    """Parse a feed from a URL or XML content.

    Args:
        source: URL string or XML content string/bytes
        include_content: Include per-entry content blobs and synthesized descriptions
        include_tags: Include feed and entry tags/categories
        include_media: Include media namespace content (media:content/media:thumbnail)
        include_enclosures: Include RSS enclosures and JSON-feed attachments

    Returns:
        FastFeedParserDict containing parsed feed data

    Raises:
        ValueError: If content is empty or invalid
        HTTPError: If URL fetch fails
    """
    is_url = isinstance(source, str) and source.startswith(("http://", "https://"))
    if is_url:
        assert isinstance(source, str)
        content = _fetch_url_content(source)
    else:
        content = source

    parse_kwargs = dict(
        include_content=include_content,
        include_tags=include_tags,
        include_media=include_media,
        include_enclosures=include_enclosures,
    )

    redirects_left = _MAX_META_REDIRECTS
    while True:
        try:
            return _parse_content_in_flight(content, **parse_kwargs)
        except ValueError as e:
            if not is_url:
                raise
            assert isinstance(source, str)
            err_msg = str(e)
            if "HTML" not in err_msg and "not a valid RSS/Atom feed" not in err_msg:
                raise
            if redirects_left <= 0:
                raise ValueError("too many meta-refresh redirects") from e
            redirect_url = _extract_meta_refresh_url(content, source)
            if redirect_url is None:
                raise
            content = _fetch_url_content(redirect_url)
            source = redirect_url
            redirects_left -= 1


def _parse_feed_info(
    channel: _Element,
    feed_type: _FeedType,
    atom_namespace: Optional[str] = None,
    *,
    include_tags: bool = True,
    lifted_cdata: Optional[dict[str, str]] = None,
) -> FastFeedParserDict:
    # Use dynamic atom namespace or fallback to default
    atom_ns = atom_namespace or "http://www.w3.org/2005/Atom"

    # Check if this is Atom 0.3 to use different date field names
    is_atom_03 = atom_ns == "http://purl.org/atom/ns#"

    # Atom 0.3 uses 'modified', Atom 1.0 uses 'updated'
    updated_field = f"{{{atom_ns}}}modified" if is_atom_03 else f"{{{atom_ns}}}updated"

    fields: tuple[tuple[str, str, str, str, bool], ...] = (
        (
            "title",
            "title",
            f"{{{atom_ns}}}title",
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/rss/1.0/}title",
            False,
        ),
        (
            "link",
            "link",
            f"{{{atom_ns}}}link",
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/rss/1.0/}link",
            True,
        ),
        (
            "subtitle",
            "description",
            f"{{{atom_ns}}}subtitle",
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/rss/1.0/}description",
            False,
        ),
        (
            "generator",
            "generator",
            f"{{{atom_ns}}}generator",
            "{http://purl.org/rss/1.0/}channel/{http://webns.net/mvcb/}generatorAgent",
            False,
        ),
        (
            "publisher",
            "publisher",
            f"{{{atom_ns}}}publisher",
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/dc/elements/1.1/}publisher",
            False,
        ),
        (
            "author",
            "author",
            f"{{{atom_ns}}}author/{{{atom_ns}}}name",
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/dc/elements/1.1/}creator",
            False,
        ),
        (
            "updated",
            "lastBuildDate",
            updated_field,
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/dc/elements/1.1/}date",
            False,
        ),
    )

    feed = FastFeedParserDict()
    element_get = _cached_element_value_factory(channel, lifted_cdata)
    get_field_value = _field_value_getter(channel, feed_type, cached_get=element_get)
    for field in fields:
        value = get_field_value(*field[1:])
        if value:
            feed[field[0]] = value

    feed_lang = channel.get(_XML_LANG_ATTR)
    feed_base = channel.get(_XML_BASE_ATTR)
    feed["language"] = feed_lang

    # Add title_detail and subtitle_detail
    if "title" in feed:
        feed["title_detail"] = {
            "type": "text/plain",
            "language": feed_lang,
            "base": feed_base,
            "value": feed["title"],
        }
    if "subtitle" in feed:
        feed["subtitle_detail"] = {
            "type": "text/plain",
            "language": feed_lang,
            "base": feed_base,
            "value": feed["subtitle"],
        }

    # Add links
    feed_links: list[dict[str, Optional[str]]] = []
    feed["links"] = feed_links
    feed_link: Optional[str] = None
    for link in channel.findall(f"{{{atom_ns}}}link"):
        rel = link.get("rel")
        href = link.get("href") or link.get("link")
        if rel == "alternate" and href and not feed_link:
            feed_link = href
            feed_links.append(
                {
                    "rel": rel,
                    "type": link.get("type"),
                    "href": href,
                    "title": link.get("title"),
                }
            )
        elif rel is None and href:
            if not feed_link:
                feed_link = href
        elif rel not in {"hub", "self", "replies", "edit"}:
            feed_links.append(
                {
                    "rel": rel,
                    "type": link.get("type"),
                    "href": href,
                    "title": link.get("title"),
                }
            )
    if feed_link:
        feed["link"] = feed_link
        feed_links.insert(
            0, {"rel": "alternate", "type": "text/html", "href": feed_link}
        )

    # Add id
    feed["id"] = element_get(f"{{{atom_ns}}}id")

    # Add generator_detail
    generator = channel.find(f"{{{atom_ns}}}generator")
    if generator is not None:
        feed["generator_detail"] = {
            "name": generator.text,
            "version": generator.get("version"),
            "href": generator.get("uri"),
        }

    if feed_type == "rss":
        comments = element_get("comments")
        if comments:
            feed["comments"] = comments

    # Additional checks for publisher and author
    if "publisher" not in feed:
        webmaster = element_get("webMaster")
        if webmaster:
            feed["publisher"] = webmaster
    if "author" not in feed:
        managing_editor = element_get("managingEditor")
        if managing_editor:
            feed["author"] = managing_editor

    # Parse feed-level image/icon/logo
    if feed_type == "atom":
        icon_el = channel.find(f"{{{atom_ns}}}icon")
        if icon_el is not None and icon_el.text:
            feed["icon"] = icon_el.text.strip()
        logo_el = channel.find(f"{{{atom_ns}}}logo")
        if logo_el is not None and logo_el.text:
            feed["logo"] = logo_el.text.strip()
    elif feed_type == "rss":
        image_el = channel.find("image")
        if image_el is not None:
            image: dict[str, Optional[str]] = {}
            for sub_tag in ("url", "title", "link"):
                sub_el = image_el.find(sub_tag)
                if sub_el is not None and sub_el.text:
                    image[sub_tag] = sub_el.text.strip()
            if image.get("url"):
                feed["image"] = image
    elif feed_type == "rdf":
        rdf_image_el = channel.find("{http://purl.org/rss/1.0/}image")
        if rdf_image_el is not None:
            image = {}
            for sub_tag in ("title", "link", "url"):
                sub_el = rdf_image_el.find(f"{{http://purl.org/rss/1.0/}}{sub_tag}")
                if sub_el is not None and sub_el.text:
                    image[sub_tag] = sub_el.text.strip()
            if image.get("url"):
                feed["image"] = image

    # Parse feed-level tags/categories
    if include_tags:
        tags = _parse_tags(channel, feed_type, atom_ns)
        if tags:
            feed["tags"] = tags

    return FastFeedParserDict(feed=feed)


def _parse_tags(
    element: _Element, feed_type: _FeedType, atom_namespace: Optional[str] = None
) -> list[dict[str, str | None]] | None:
    """Parse tags/categories from an element based on feed type."""
    tags_list: list[dict[str, str | None]] = []
    if feed_type == "rss":
        # RSS uses <category> elements
        for cat in element.findall("category"):
            term = cat.text.strip() if cat.text else None
            if term:
                tags_list.append(
                    {"term": term, "scheme": cat.get("domain"), "label": None}
                )
        # RSS might also use <dc:subject>
        for subject in element.findall(_DC_SUBJECT_TAG):
            term = subject.text.strip() if subject.text else None
            if term:
                tags_list.append({"term": term, "scheme": None, "label": None})
    elif feed_type == "atom":
        # Atom uses <category> elements with attributes
        atom_ns = atom_namespace or "http://www.w3.org/2005/Atom"
        for cat in element.findall(_atom_ns_tags(atom_ns)["category"]):
            term = cat.get("term")
            if term:
                tags_list.append(
                    {
                        "term": term,
                        "scheme": cat.get("scheme"),
                        "label": cat.get("label"),
                    }
                )
    elif feed_type == "rdf":
        # RDF uses <dc:subject> or <taxo:topic>
        for subject in element.findall(_DC_SUBJECT_TAG):
            term = subject.text.strip() if subject.text else None
            if term:
                tags_list.append({"term": term, "scheme": None, "label": None})
        # Example for taxo:topic (might need refinement based on actual usage)
        for topic in element.findall(
            "{http://purl.org/rss/1.0/modules/taxonomy/}topic"
        ):
            # rdf:resource often contains the tag URL which could be scheme+term
            resource = topic.get(
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
            )
            term = (
                topic.text.strip() if topic.text else resource
            )  # Use text or resource as term
            if term:
                tags_list.append({"term": term, "scheme": resource, "label": None})

    return tags_list if tags_list else None


def _drop_none_values(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


def _coerce_int_fields(mapping: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        value = mapping.get(field)
        if value is None:
            continue
        try:
            mapping[field] = int(value)
        except (ValueError, TypeError):
            mapping.pop(field, None)


def _populate_entry_links_from_elements(
    entry: FastFeedParserDict,
    atom_links: list[_Element],
    *,
    guid_text: Optional[str] = None,
    guid_is_permalink: bool = False,
) -> None:
    entry_links: list[dict[str, Optional[str]]] = []
    alternate_link: Optional[dict[str, Optional[str]]] = None
    for link in atom_links:
        rel = link.get("rel")
        href = link.get("href") or link.get("link")
        if not href:
            continue
        link_dict = {
            "rel": rel,
            "type": link.get("type"),
            "href": href,
            "title": link.get("title"),
        }
        if rel == "alternate":
            if alternate_link is None:
                alternate_link = link_dict
            else:
                entry_links.append(link_dict)
        elif rel not in {"edit", "self"}:
            entry_links.append(link_dict)

    is_guid_url = guid_text is not None and guid_text.startswith(
        ("http://", "https://")
    )

    if is_guid_url and "link" not in entry:
        entry["link"] = guid_text
        if alternate_link:
            entry_links.insert(
                0, {"rel": "alternate", "type": "text/html", "href": guid_text}
            )
    elif alternate_link:
        entry["link"] = alternate_link["href"]
        entry_links.insert(0, alternate_link)
    elif ("link" not in entry) and guid_is_permalink:
        entry["link"] = guid_text

    entry["links"] = entry_links


def _populate_entry_links(
    entry: FastFeedParserDict, item: _Element, atom_ns: str
) -> None:
    tags = _atom_ns_tags(atom_ns)
    guid = item.find("guid")
    guid_text = guid.text.strip() if guid is not None and guid.text else None
    _populate_entry_links_from_elements(
        entry,
        item.findall(tags["link"]),
        guid_text=guid_text,
        guid_is_permalink=guid is not None and guid.get("isPermaLink") == "true",
    )


_SYNTH_DESCRIPTION_LEN = 512
# Normalized text is first computed on this many leading chars; only a
# whitespace-heavy prefix that yields < _SYNTH_DESCRIPTION_LEN chars falls
# back to normalizing the whole string.
_SYNTH_DESCRIPTION_SCAN = 640
# HTML content is only stripped of tags up to 2048 chars. The fast path first
# tries a prefix ending at the first ">" at or after this offset.
_SYNTH_HTML_LIMIT = 2048
_SYNTH_HTML_CUT = 800


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _needs_whitespace_collapse(value: str) -> bool:
    return "  " in value or "\n" in value or "\t" in value or "\r" in value


def _normalize_description_prefix(value: str, normalize: Callable[[str], str]) -> str:
    """Return normalize(value)[:_SYNTH_DESCRIPTION_LEN] without touching the
    whole string when a prefix suffices.

    Both normalizers (strip, collapse) map a prefix of `value` to a prefix of
    normalize(value), so a long-enough result from the prefix is exact.
    """
    if len(value) > _SYNTH_DESCRIPTION_SCAN:
        head = normalize(value[:_SYNTH_DESCRIPTION_SCAN])
        if len(head) >= _SYNTH_DESCRIPTION_LEN:
            return head[:_SYNTH_DESCRIPTION_LEN]
    return normalize(value)[:_SYNTH_DESCRIPTION_LEN]


def _html_description_from_prefix(html: str) -> Optional[str]:
    """Synthesize the description from a prefix of `html`, or None if the
    prefix cannot be shown to give the same result as the full pipeline.

    The prefix ends just after a ">", so no _RE_HTML_TAGS match spans the cut
    (a match ends at the first ">" after its "<"). If that ">" closed a tag,
    the stripped prefix ends with the " " it was replaced by, and no entity
    reference spans a space, so unescaping the prefix equals the prefix of the
    unescaped text. A whitespace run found in the prefix exists in the full
    text too, which selects the collapse branch, and collapsing is
    prefix-preserving.
    """
    cut = html.find(">", _SYNTH_HTML_CUT, _SYNTH_HTML_LIMIT - 1)
    if cut == -1:
        return None
    head = _RE_HTML_TAGS.sub(" ", html[: cut + 1])
    if not head.endswith(" "):
        return None
    if "&" in head:
        head = _html_mod.unescape(head)
    if not _needs_whitespace_collapse(head):
        return None
    collapsed = _collapse_whitespace(head)
    if len(collapsed) < _SYNTH_DESCRIPTION_LEN:
        return None
    return collapsed[:_SYNTH_DESCRIPTION_LEN]


def _synthesize_entry_description(entry: FastFeedParserDict) -> None:
    if "description" in entry or "content" not in entry:
        return

    content_value = entry["content"][0]["value"]
    if content_value:
        if "<" in content_value and ">" in content_value:
            description = _html_description_from_prefix(content_value)
            if description is not None:
                entry["description"] = description
                return
            content_value = _RE_HTML_TAGS.sub(" ", content_value[:_SYNTH_HTML_LIMIT])
            if "&" in content_value:
                content_value = _html_mod.unescape(content_value)
        if _needs_whitespace_collapse(content_value):
            # " ".join(split()) collapses \s+ runs identically to the regex
            # but ~4x faster (split/join are C-level); only on the runs the
            # guard already confirmed need collapsing.
            normalize = _collapse_whitespace
        else:
            normalize = str.strip
        content_value = _normalize_description_prefix(content_value, normalize)
    entry["description"] = content_value[:_SYNTH_DESCRIPTION_LEN]


def _populate_entry_content_preparsed(
    entry: FastFeedParserDict,
    item: _Element,
    *,
    content_el: Optional[_Element],
    rss_description_text: Optional[str],
    content_text: Optional[str] = None,
    lifted_cdata: Optional[dict[str, str]] = None,
) -> None:
    """Fill entry["content"] from a pre-located content element.

    ``content_text`` is ``content_el.text`` when the caller already read it;
    passing it avoids decoding a large content blob a second time.
    """
    if content_el is not None:
        content_type = content_el.get("type", "text/html")
        if content_type in {"xhtml", "application/xhtml+xml"}:
            content_value = etree.tostring(content_el, encoding="unicode", method="xml")
            if lifted_cdata and _CDATA_MARK_START in content_value:
                content_value = _restore_lifted_cdata_in_xml(
                    content_value, lifted_cdata
                )
        elif content_text is not None:
            content_value = content_text
        else:
            content_value = content_el.text or ""
        entry["content"] = [
            {
                "type": content_type,
                "language": content_el.get(_XML_LANG_ATTR),
                "base": content_el.get(_XML_BASE_ATTR),
                "value": content_value,
            }
        ]
    elif rss_description_text:
        entry["content"] = [
            {
                "type": "text/html",
                "language": item.get(_XML_LANG_ATTR),
                "base": item.get(_XML_BASE_ATTR),
                "value": rss_description_text,
            }
        ]

    _synthesize_entry_description(entry)


def _populate_entry_content(
    entry: FastFeedParserDict,
    item: _Element,
    feed_type: _FeedType,
    atom_ns: str,
    lifted_cdata: Optional[dict[str, str]] = None,
) -> None:
    content_el: Optional[_Element] = None
    rss_description_text: Optional[str] = None
    if feed_type == "rss":
        content_el = item.find(_RSS_CONTENT_ENCODED_TAG)
        if content_el is None:
            content_el = item.find("content")
        description = item.find("description")
        if description is not None:
            rss_description_text = description.text
            if lifted_cdata and rss_description_text:
                rss_description_text = _restore_lifted_cdata(
                    rss_description_text, lifted_cdata
                )
    elif feed_type == "atom":
        content_el = item.find(_atom_ns_tags(atom_ns)["content"])

    content_text = content_el.text if content_el is not None else None
    if lifted_cdata and content_text:
        content_text = _restore_lifted_cdata(content_text, lifted_cdata)

    _populate_entry_content_preparsed(
        entry,
        item,
        content_el=content_el,
        rss_description_text=rss_description_text,
        content_text=content_text,
        lifted_cdata=lifted_cdata,
    )


def _first_media_desc_credit(
    parent: _Element,
) -> tuple[Optional[_Element], Optional[_Element]]:
    """First media:description and media:credit children of `parent`."""
    desc: Optional[_Element] = None
    credit: Optional[_Element] = None
    for child in parent.iterchildren(_MEDIA_DESCRIPTION_TAG, _MEDIA_CREDIT_TAG):
        if child.tag == _MEDIA_DESCRIPTION_TAG:
            if desc is None:
                desc = child
        elif credit is None:
            credit = child
        if desc is not None and credit is not None:
            break
    return desc, credit


def _parse_media_content(
    item: _Element, lifted_cdata: Optional[dict[str, str]] = None
) -> list[dict[str, Any]] | None:
    media_contents: list[dict[str, Any]] = []
    # Siblings under one item or media:group share a parent; look its
    # description/credit up once. lxml keeps one proxy per node while it is
    # referenced, so identity comparison is reliable here.
    last_parent: Optional[_Element] = None
    parent_desc: Optional[_Element] = None
    parent_credit: Optional[_Element] = None

    # iter() walks descendants in C; it also yields `item` itself, but an item
    # is never a media:content element.
    for media in item.iter(_MEDIA_CONTENT_TAG):
        # Same keys, order and int coercion as building the full dict and
        # then dropping None values.
        media_item: dict[str, Any] = {}
        for attr in ("url", "type", "medium"):
            value = media.get(attr)
            if value is not None:
                media_item[attr] = value
        for attr in ("width", "height"):
            value = media.get(attr)
            if value is not None:
                try:
                    media_item[attr] = int(value)
                except ValueError:
                    pass

        title = text = desc = credit = thumbnail = None
        for child in media:
            tag = child.tag
            if tag == _MEDIA_TITLE_TAG:
                if title is None:
                    title = child
            elif tag == _MEDIA_TEXT_TAG:
                if text is None:
                    text = child
            elif tag == _MEDIA_DESCRIPTION_TAG:
                if desc is None:
                    desc = child
            elif tag == _MEDIA_CREDIT_TAG:
                if credit is None:
                    credit = child
            elif tag == _MEDIA_THUMBNAIL_TAG:
                if thumbnail is None:
                    thumbnail = child

        if desc is None or credit is None:
            parent = media.getparent()
            if parent is not None:
                if parent is not last_parent:
                    last_parent = parent
                    parent_desc, parent_credit = _first_media_desc_credit(parent)
                if desc is None:
                    desc = parent_desc
                if credit is None:
                    credit = parent_credit

        if title is not None and title.text:
            media_item["title"] = title.text.strip()
        if text is not None and text.text:
            media_item["text"] = text.text.strip()
        if desc is not None and desc.text:
            # A description in a default media namespace is written as a
            # plain <description>, which _lift_large_cdata lifts.
            desc_text = desc.text
            if lifted_cdata:
                desc_text = _restore_lifted_cdata(desc_text, lifted_cdata)
            media_item["description"] = desc_text.strip()
        if credit is not None and credit.text:
            media_item["credit"] = credit.text.strip()
            credit_scheme = credit.get("scheme")
            if credit_scheme is not None:
                media_item["credit_scheme"] = credit_scheme
        if thumbnail is not None:
            thumbnail_url = thumbnail.get("url")
            if thumbnail_url is not None:
                media_item["thumbnail_url"] = thumbnail_url

        if media_item:
            media_contents.append(media_item)

    if not media_contents:
        for thumbnail in item.iter(_MEDIA_THUMBNAIL_TAG):
            parent = thumbnail.getparent()
            if parent is None or parent.tag == _MEDIA_CONTENT_TAG:
                continue
            thumb_item: dict[str, str | int | None] = {
                "url": thumbnail.get("url"),
                "type": "image/jpeg",
                "width": thumbnail.get("width"),
                "height": thumbnail.get("height"),
            }
            _coerce_int_fields(thumb_item, ("width", "height"))
            cleaned = _drop_none_values(thumb_item)
            if cleaned:
                media_contents.append(cleaned)

    return media_contents or None


def _parse_enclosures(item: _Element) -> list[dict[str, Any]] | None:
    enclosures: list[dict[str, Any]] = []
    for enclosure in item.findall("enclosure"):
        cleaned = _parse_enclosure_element(enclosure)
        if cleaned.get("url"):
            enclosures.append(cleaned)

    return enclosures or None


def _parse_enclosure_element(enclosure: _Element) -> dict[str, Any]:
    enc_item: dict[str, str | int | None] = {
        "url": enclosure.get("url"),
        "type": enclosure.get("type"),
        "length": enclosure.get("length"),
    }
    length = enc_item.get("length")
    if length:
        try:
            enc_item["length"] = int(length)
        except (ValueError, TypeError):
            enc_item.pop("length", None)
    return _drop_none_values(enc_item)


_RSS_KIND_OTHER = 0
_RSS_KIND_ATOM_LINK = 1
_RSS_KIND_ATOM_ID = 2
_RSS_KIND_GUID = 3
_RSS_KIND_ENCODED = 4
_RSS_KIND_CONTENT = 5
_RSS_KIND_DESCRIPTION = 6
_RSS_KIND_ENCLOSURE = 7
_RSS_KIND_CATEGORY = 8
_RSS_KIND_SUBJECT = 9
_RSS_KIND_ATOM_AUTHOR = 10

# Bound on per-namespace tag classification caches. Feeds are untrusted and
# can carry arbitrary tag names, so stop memoizing past this many.
_TAG_CACHE_MAX = 4096
# Tags longer than this are classified on every use instead of memoized. A tag
# is "{namespace-uri}local", so one long URI would be copied into every key.
_TAG_CACHE_MAX_TAG_LEN = 128


@lru_cache(maxsize=4)
def _rss_tag_info_cache(atom_ns: str) -> dict[Any, tuple[Optional[str], int]]:
    # Comments, PIs and entities carry these factories as their .tag.
    return {
        etree.Comment: (None, _RSS_KIND_OTHER),
        etree.ProcessingInstruction: (None, _RSS_KIND_OTHER),
        etree.Entity: (None, _RSS_KIND_OTHER),
    }


# Local names whose first-occurrence text _parse_rss_feed_entry_fast reads.
# Other children's text is never fetched unless their kind needs it.
_RSS_TEXT_LOCALS = frozenset(
    (
        "guid",
        "title",
        "description",
        "summary",
        "link",
        "pubdate",
        "published",
        "issued",
        "date",
        "lastbuilddate",
        "updated",
        "modified",
        "author",
        "creator",
        "comments",
    )
)


def _classify_rss_tag(tag: str, atom_tags: dict[str, str]) -> tuple[Optional[str], int]:
    """Return (tracked local name or None, _RSS_KIND_*) for an RSS item child.

    The local name is lowercased and only returned when it is in
    _RSS_TEXT_LOCALS.
    """
    if "{" in tag:
        local = tag.rsplit("}", 1)[1].lower()
    elif ":" in tag:
        local = tag.split(":", 1)[1].lower()
    else:
        local = tag.lower()

    if tag == atom_tags["link"]:
        kind = _RSS_KIND_ATOM_LINK
    elif tag == atom_tags["id"]:
        kind = _RSS_KIND_ATOM_ID
    elif tag == "guid":
        kind = _RSS_KIND_GUID
    elif tag == _RSS_CONTENT_ENCODED_TAG:
        kind = _RSS_KIND_ENCODED
    elif tag == "content":
        kind = _RSS_KIND_CONTENT
    elif tag == "description":
        kind = _RSS_KIND_DESCRIPTION
    elif tag == "enclosure":
        kind = _RSS_KIND_ENCLOSURE
    elif local == "category":
        kind = _RSS_KIND_CATEGORY
    elif tag == _DC_SUBJECT_TAG:
        kind = _RSS_KIND_SUBJECT
    elif tag == atom_tags["author"]:
        kind = _RSS_KIND_ATOM_AUTHOR
    else:
        kind = _RSS_KIND_OTHER
    return (local if local in _RSS_TEXT_LOCALS else None), kind


def _parse_rss_feed_entry_fast(
    item: _Element,
    atom_ns: str,
    has_media_elements: bool = True,
    *,
    include_content: bool = True,
    include_tags: bool = True,
    include_media: bool = True,
    include_enclosures: bool = True,
    lifted_cdata: Optional[dict[str, str]] = None,
) -> FastFeedParserDict:
    atom_tags = _atom_ns_tags(atom_ns)
    tag_info = _rss_tag_info_cache(atom_ns)
    text_by_local: dict[str, Optional[str]] = {}
    atom_id_text: Optional[str] = None
    atom_links: list[_Element] = []
    guid_element: Optional[_Element] = None
    encoded_content_el: Optional[_Element] = None
    encoded_content_text: Optional[str] = None
    raw_content_el: Optional[_Element] = None
    raw_content_text: Optional[str] = None
    rss_description_text: Optional[str] = None
    tag_categories: list[dict[str, str | None]] = []
    tag_subjects: list[dict[str, str | None]] = []
    enclosures: list[dict[str, Any]] = []
    has_atom_author = False

    for child in item:
        tag = child.tag
        info = tag_info.get(tag)
        if info is None:
            if not isinstance(tag, str):
                continue
            info = _classify_rss_tag(tag, atom_tags)
            if (
                len(tag_info) < _TAG_CACHE_MAX
                and len(tag) <= _TAG_CACHE_MAX_TAG_LEN
            ):
                tag_info[tag] = info
        local, kind = info

        if local is not None and local not in text_by_local:
            text_value = child.text or None
            if lifted_cdata and text_value:
                text_value = _restore_lifted_cdata(text_value, lifted_cdata)
            text_by_local[local] = text_value
        elif kind == _RSS_KIND_OTHER:
            continue
        else:
            text_value = child.text or None
            if lifted_cdata and text_value:
                text_value = _restore_lifted_cdata(text_value, lifted_cdata)

        if kind == _RSS_KIND_OTHER:
            continue
        if kind == _RSS_KIND_CATEGORY:
            if include_tags:
                term = text_value.strip() if text_value else None
                if term:
                    tag_categories.append(
                        {"term": term, "scheme": child.get("domain"), "label": None}
                    )
        elif kind == _RSS_KIND_ATOM_LINK:
            atom_links.append(child)
        elif kind == _RSS_KIND_ATOM_ID:
            if atom_id_text is None:
                atom_id_text = text_value
        elif kind == _RSS_KIND_GUID:
            if guid_element is None:
                guid_element = child
        elif kind == _RSS_KIND_ENCODED:
            if encoded_content_el is None:
                encoded_content_el = child
                encoded_content_text = text_value
        elif kind == _RSS_KIND_CONTENT:
            if raw_content_el is None:
                raw_content_el = child
                raw_content_text = text_value
        elif kind == _RSS_KIND_DESCRIPTION:
            if rss_description_text is None:
                rss_description_text = text_value
        elif kind == _RSS_KIND_ENCLOSURE:
            if include_enclosures:
                cleaned = _parse_enclosure_element(child)
                if cleaned.get("url"):
                    enclosures.append(cleaned)
        elif kind == _RSS_KIND_ATOM_AUTHOR:
            has_atom_author = True
        elif kind == _RSS_KIND_SUBJECT:
            if include_tags:
                term = text_value.strip() if text_value else None
                if term:
                    tag_subjects.append({"term": term, "scheme": None, "label": None})

    entry = FastFeedParserDict()
    atom_id = atom_id_text
    rss_guid = text_by_local.get("guid")
    rdf_about = item.get(_RDF_ABOUT_ATTR)
    entry_id: Optional[str] = atom_id or rss_guid or rdf_about
    if entry_id:
        entry["id"] = entry_id.strip()

    title = text_by_local.get("title")
    if title:
        entry["title"] = title.strip()

    description = text_by_local.get("description") or text_by_local.get("summary")
    if description:
        entry["description"] = description.strip()

    link = text_by_local.get("link")
    if link:
        entry["link"] = link.strip()

    published_source = (
        text_by_local.get("pubdate")
        or text_by_local.get("published")
        or text_by_local.get("issued")
        or text_by_local.get("date")
    )
    if published_source:
        published = _parse_date(published_source)
        if published:
            entry["published"] = published

    updated_source = (
        text_by_local.get("lastbuilddate")
        or text_by_local.get("updated")
        or text_by_local.get("modified")
    )
    if updated_source:
        updated = _parse_date(updated_source)
        if updated:
            entry["updated"] = updated

    if (
        "published" not in entry
        and rss_guid
        and not rss_guid.startswith(("http://", "https://"))
    ):
        guid_date = _parse_date(rss_guid)
        if guid_date:
            entry["published"] = guid_date

    if "updated" in entry and "published" not in entry:
        entry["published"] = entry["updated"]

    if atom_links:
        guid_text = (
            guid_element.text.strip()
            if guid_element is not None and guid_element.text
            else None
        )
        _populate_entry_links_from_elements(
            entry,
            atom_links,
            guid_text=guid_text,
            guid_is_permalink=guid_element is not None
            and guid_element.get("isPermaLink") == "true",
        )
    else:
        entry["links"] = []
        if (
            "link" not in entry
            and rss_guid
            and rss_guid.startswith(("http://", "https://"))
        ):
            entry["link"] = rss_guid

    if "id" not in entry and "link" in entry:
        entry["id"] = entry["link"]

    if include_content:
        if encoded_content_el is not None:
            content_el, content_text = encoded_content_el, encoded_content_text
        else:
            content_el, content_text = raw_content_el, raw_content_text
        _populate_entry_content_preparsed(
            entry,
            item,
            content_el=content_el,
            rss_description_text=rss_description_text,
            content_text=content_text,
            lifted_cdata=lifted_cdata,
        )

    if include_media and has_media_elements:
        media_contents = _parse_media_content(item, lifted_cdata)
        if media_contents:
            entry["media_content"] = media_contents

    if include_enclosures and enclosures:
        entry["enclosures"] = enclosures

    author = text_by_local.get("author") or text_by_local.get("creator")
    if not author and has_atom_author:
        atom_author = item.find(atom_tags["author_name"])
        author = (
            atom_author.text.strip()
            if atom_author is not None and atom_author.text
            else None
        )
    if author:
        entry["author"] = author.strip()

    comments = text_by_local.get("comments")
    if comments:
        entry["comments"] = comments.strip()

    if include_tags and (tag_categories or tag_subjects):
        entry["tags"] = tag_categories + tag_subjects

    return entry


(
    _ATOM_KIND_ID,
    _ATOM_KIND_TITLE,
    _ATOM_KIND_SUMMARY,
    _ATOM_KIND_PUBLISHED,
    _ATOM_KIND_UPDATED,
    _ATOM_KIND_PUB_FALLBACK,
    _ATOM_KIND_UPD_FALLBACK,
    _ATOM_KIND_LINK,
    _ATOM_KIND_CONTENT,
    _ATOM_KIND_AUTHOR,
    _ATOM_KIND_CATEGORY,
    _ATOM_KIND_ENCLOSURE,
) = range(12)


@lru_cache(maxsize=4)
def _atom_entry_tag_kinds(atom_ns: str) -> dict[str, int]:
    """Map the Atom entry child tags we extract to an _ATOM_KIND_* code."""
    t = _atom_ns_tags(atom_ns)
    return {
        t["id"]: _ATOM_KIND_ID,
        t["title"]: _ATOM_KIND_TITLE,
        t["summary"]: _ATOM_KIND_SUMMARY,
        t["published"]: _ATOM_KIND_PUBLISHED,
        t["updated"]: _ATOM_KIND_UPDATED,
        t["pub_fallback"]: _ATOM_KIND_PUB_FALLBACK,
        t["upd_fallback"]: _ATOM_KIND_UPD_FALLBACK,
        t["link"]: _ATOM_KIND_LINK,
        t["content"]: _ATOM_KIND_CONTENT,
        t["ns"] + "author": _ATOM_KIND_AUTHOR,
        t["category"]: _ATOM_KIND_CATEGORY,
        "enclosure": _ATOM_KIND_ENCLOSURE,
    }


def _parse_atom_feed_entry_fast(
    item: _Element,
    atom_ns: str,
    has_media_elements: bool = True,
    *,
    include_content: bool = True,
    include_tags: bool = True,
    include_media: bool = True,
    include_enclosures: bool = True,
    lifted_cdata: Optional[dict[str, str]] = None,
) -> FastFeedParserDict:
    atom_name_tag = _atom_ns_tags(atom_ns)["ns"] + "name"
    atom_links: list[_Element] = []
    atom_categories: list[dict[str, str | None]] = []
    enclosures: list[dict[str, Any]] = []
    content_el: Optional[_Element] = None
    author_name: Optional[str] = None
    first_link_href: Optional[str] = None
    published_source: Optional[str] = None
    updated_source: Optional[str] = None
    published_fallback_source: Optional[str] = None
    updated_fallback_source: Optional[str] = None

    kinds = _atom_entry_tag_kinds(atom_ns)
    content_text: Optional[str] = None

    entry = FastFeedParserDict()
    for child in item:
        # Comments/PIs have a non-str tag and never match a kind.
        kind = kinds.get(child.tag)
        if kind is None:
            continue

        if kind == _ATOM_KIND_LINK:
            atom_links.append(child)
            href = child.get("href")
            if href and first_link_href is None:
                first_link_href = href.strip()
        elif kind == _ATOM_KIND_CATEGORY:
            if include_tags:
                term = child.get("term")
                if term:
                    atom_categories.append(
                        {
                            "term": term,
                            "scheme": child.get("scheme"),
                            "label": child.get("label"),
                        }
                    )
        elif kind == _ATOM_KIND_AUTHOR:
            if author_name is None:
                author_name_el = child.find(atom_name_tag)
                if author_name_el is not None and author_name_el.text:
                    author_name = author_name_el.text.strip()
        elif kind == _ATOM_KIND_ENCLOSURE:
            if include_enclosures:
                cleaned = _parse_enclosure_element(child)
                if cleaned.get("url"):
                    enclosures.append(cleaned)
        elif kind == _ATOM_KIND_CONTENT:
            if include_content and content_el is None:
                content_el = child
                content_text = child.text
                if lifted_cdata and content_text:
                    content_text = _restore_lifted_cdata(content_text, lifted_cdata)
        else:
            text_value = child.text
            if not text_value:
                continue
            if lifted_cdata:
                text_value = _restore_lifted_cdata(text_value, lifted_cdata)
            if kind == _ATOM_KIND_ID:
                if "id" not in entry:
                    entry["id"] = text_value.strip()
            elif kind == _ATOM_KIND_TITLE:
                if "title" not in entry:
                    entry["title"] = text_value.strip()
            elif kind == _ATOM_KIND_SUMMARY:
                if "description" not in entry:
                    entry["description"] = text_value.strip()
            elif kind == _ATOM_KIND_PUBLISHED:
                if published_source is None:
                    published_source = text_value
            elif kind == _ATOM_KIND_UPDATED:
                if updated_source is None:
                    updated_source = text_value
            elif kind == _ATOM_KIND_PUB_FALLBACK:
                if published_fallback_source is None:
                    published_fallback_source = text_value
            elif kind == _ATOM_KIND_UPD_FALLBACK:
                if updated_fallback_source is None:
                    updated_fallback_source = text_value

    if first_link_href:
        entry["link"] = first_link_href

    if published_source:
        published = _parse_date(published_source)
        if published:
            entry["published"] = published

    if updated_source:
        updated = _parse_date(updated_source)
        if updated:
            entry["updated"] = updated

    if "published" not in entry and published_fallback_source:
        published = _parse_date(published_fallback_source)
        if published:
            entry["published"] = published

    if "updated" not in entry and updated_fallback_source:
        updated = _parse_date(updated_fallback_source)
        if updated:
            entry["updated"] = updated

    if "updated" in entry and "published" not in entry:
        entry["published"] = entry["updated"]

    _populate_entry_links_from_elements(entry, atom_links)

    if "id" not in entry and "link" in entry:
        entry["id"] = entry["link"]

    if include_content:
        _populate_entry_content_preparsed(
            entry,
            item,
            content_el=content_el,
            rss_description_text=None,
            content_text=content_text,
            lifted_cdata=lifted_cdata,
        )

    if include_media and has_media_elements:
        media_contents = _parse_media_content(item, lifted_cdata)
        if media_contents:
            entry["media_content"] = media_contents

    if include_enclosures and enclosures:
        entry["enclosures"] = enclosures

    if author_name:
        entry["author"] = author_name

    if include_tags and atom_categories:
        entry["tags"] = atom_categories

    return entry


def _parse_feed_entry(
    item: _Element,
    feed_type: _FeedType,
    atom_namespace: Optional[str] = None,
    has_media_elements: bool = True,
    *,
    include_content: bool = True,
    include_tags: bool = True,
    include_media: bool = True,
    include_enclosures: bool = True,
    lifted_cdata: Optional[dict[str, str]] = None,
) -> FastFeedParserDict:
    # Use dynamic atom namespace or fallback to default
    atom_ns = atom_namespace or "http://www.w3.org/2005/Atom"

    if feed_type == "rss":
        return _parse_rss_feed_entry_fast(
            item,
            atom_ns,
            has_media_elements,
            include_content=include_content,
            include_tags=include_tags,
            include_media=include_media,
            include_enclosures=include_enclosures,
            lifted_cdata=lifted_cdata,
        )

    if feed_type == "atom":
        return _parse_atom_feed_entry_fast(
            item,
            atom_ns,
            has_media_elements,
            include_content=include_content,
            include_tags=include_tags,
            include_media=include_media,
            include_enclosures=include_enclosures,
            lifted_cdata=lifted_cdata,
        )

    # RDF path uses the generic field machinery
    # Check if this is Atom 0.3 to use different date field names
    is_atom_03 = atom_ns == "http://purl.org/atom/ns#"

    # Atom 0.3 uses 'issued' and 'modified', Atom 1.0 uses 'published' and 'updated'
    # However, some feeds mix namespaces, so we'll check both formats
    published_field = (
        f"{{{atom_ns}}}issued" if is_atom_03 else f"{{{atom_ns}}}published"
    )
    updated_field = f"{{{atom_ns}}}modified" if is_atom_03 else f"{{{atom_ns}}}updated"

    # Also define fallback fields for mixed namespace scenarios
    published_fallback = (
        f"{{{atom_ns}}}published" if is_atom_03 else f"{{{atom_ns}}}issued"
    )
    updated_fallback = (
        f"{{{atom_ns}}}updated" if is_atom_03 else f"{{{atom_ns}}}modified"
    )

    fields: tuple[tuple[str, str, str, str, bool], ...] = (
        (
            "title",
            "title",
            f"{{{atom_ns}}}title",
            "{http://purl.org/rss/1.0/}title",
            False,
        ),
        (
            "link",
            "link",
            f"{{{atom_ns}}}link",
            "{http://purl.org/rss/1.0/}link",
            True,
        ),
        (
            "description",
            "description",
            f"{{{atom_ns}}}summary",
            "{http://purl.org/rss/1.0/}description",
            False,
        ),
        (
            "published",
            "pubDate",
            published_field,
            "{http://purl.org/dc/elements/1.1/}date",
            False,
        ),
        (
            "updated",
            "lastBuildDate",
            updated_field,
            "{http://purl.org/dc/terms/}modified",
            False,
        ),
    )

    element_get = _cached_element_value_factory(item, lifted_cdata)
    entry = FastFeedParserDict()
    # ------------------------------------------------------------------
    # 1) Collect a stable identifier for this entry.
    #    Atom   → <id>
    #    RSS    → <guid>
    #    RDF    → rdf:about attribute on the <item>
    # ------------------------------------------------------------------
    atom_id = element_get(f"{{{atom_ns}}}id")
    rss_guid = element_get("guid")
    rdf_about = item.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
    entry_id: Optional[str] = atom_id or rss_guid or rdf_about
    if entry_id:
        entry["id"] = entry_id.strip()
    get_field_value = _field_value_getter(item, feed_type, cached_get=element_get)
    for field in fields:
        value = get_field_value(*field[1:])
        if value:
            name = field[0]
            if name in {"published", "updated"}:
                value = _parse_date(value)
            entry[name] = value

    # Check for fallback date fields if primary fields are missing
    if "published" not in entry:
        fallback_published = element_get(published_fallback)
        if fallback_published:
            entry["published"] = _parse_date(fallback_published)

    if "updated" not in entry:
        fallback_updated = element_get(updated_fallback)
        if fallback_updated:
            entry["updated"] = _parse_date(fallback_updated)

    # Try to extract date from GUID as final fallback
    if (
        "published" not in entry
        and rss_guid
        and not rss_guid.startswith(("http://", "https://"))
    ):
        guid_date = _parse_date(rss_guid)
        if guid_date:
            entry["published"] = guid_date

    # If published is missing but updated exists, use updated as published
    if "updated" in entry and "published" not in entry:
        entry["published"] = entry["updated"]

    _populate_entry_links(entry, item, atom_ns)

    # ------------------------------------------------------------------
    # 2) Guarantee that every entry has an id.  If none of the dedicated
    #    id sources were present, fall back to the chosen link.
    # ------------------------------------------------------------------
    if "id" not in entry and "link" in entry:
        entry["id"] = entry["link"]

    if include_content:
        _populate_entry_content(entry, item, feed_type, atom_ns, lifted_cdata)

    if include_media and has_media_elements:
        media_contents = _parse_media_content(item, lifted_cdata)
        if media_contents:
            entry["media_content"] = media_contents

    if include_enclosures:
        enclosures = _parse_enclosures(item)
        if enclosures:
            entry["enclosures"] = enclosures

    author = get_field_value(
        "author",
        f"{{{atom_ns}}}author/{{{atom_ns}}}name",
        "{http://purl.org/dc/elements/1.1/}creator",
        False,
    )
    if not author:
        author = element_get(
            "{http://purl.org/dc/elements/1.1/}creator"
        ) or element_get("author")
    if author:
        entry["author"] = author

    # Parse entry-level tags/categories
    if include_tags:
        tags = _parse_tags(item, feed_type, atom_ns)
        if tags:
            entry["tags"] = tags

    return entry


def _field_value_getter(
    root: _Element,
    feed_type: _FeedType,
    cached_get: Optional[_ElementValueGetter] = None,
) -> Callable[[str, str, str, bool], str | None]:
    get_value: _ElementValueGetter = cached_get or _cached_element_value_factory(root)

    if feed_type == "rss":

        def wrapper(
            rss_css: str, atom_css: str, rdf_css: str, is_attr: bool
        ) -> str | None:
            # First try standard RSS field (most common case)
            result = get_value(rss_css)
            if result:
                return result

            # Try case-insensitive for mixed-case fields (pubdate vs pubDate)
            # Only try if field has uppercase letters
            if rss_css != rss_css.lower():
                result = get_value(rss_css.lower())
                if result:
                    return result

            # For attributes, try with href/link attributes
            if is_attr:
                result = get_value(atom_css, attribute="href")
                if result:
                    return result
                result = get_value(atom_css, attribute="link")
                if result:
                    return result
            else:
                # Try Atom and RDF fields for non-attribute lookups
                result = get_value(atom_css)
                if result:
                    return result
                result = get_value(rdf_css)
                if result:
                    return result

            # Last resort: Try unnamespaced Atom field for malformed RSS
            # Only if atom_css has namespace
            if "{" in atom_css:
                unnamespaced_atom = atom_css.split("}", 1)[1]
                result = get_value(unnamespaced_atom)
                if result:
                    return result

            return None

    elif feed_type == "atom":

        def wrapper(
            rss_css: str, atom_css: str, rdf_css: str, is_attr: bool
        ) -> str | None:
            if is_attr:
                return get_value(atom_css, attribute="href") or get_value(
                    atom_css, attribute="link"
                )
            return get_value(atom_css)

    elif feed_type == "rdf":

        def wrapper(
            rss_css: str, atom_css: str, rdf_css: str, is_attr: bool
        ) -> str | None:
            return get_value(rdf_css)

    return wrapper


_RE_PATH_TAG_STEP = re.compile(r"(?:\{[^}]*\})?[^/{}]+")
_RE_PATH_SPECIAL = re.compile(r"[.*\[\]@():]")


@lru_cache(maxsize=256)
def _path_tag_steps(path: str) -> Optional[tuple[str, ...]]:
    """Split an ElementPath of plain "{ns}tag" steps joined by "/".

    Returns None for anything else (wildcards, predicates, prefixes, "." or
    "//"), which callers hand to find(). The "/" inside a "{uri}" is not a
    step separator.
    """
    steps = tuple(_RE_PATH_TAG_STEP.findall(path))
    if not steps or "/".join(steps) != path:
        return None
    if any(_RE_PATH_SPECIAL.search(step.rpartition("}")[2]) for step in steps):
        return None
    return steps


def _cached_element_value_factory(
    root: _Element,
    lifted_cdata: Optional[dict[str, str]] = None,
) -> _ElementValueGetter:
    """Create a getter for the text or attribute of a child of `root`.

    Simple paths resolve through a one-pass child index instead of
    root.find(), which rescans every child (all items, for an RSS channel) on
    each miss. A miss on a plain name also tries the rss:/atom:/dc: prefixed
    names that malformed feeds leave unresolved.
    """
    # First child per exact tag, matching root.find(tag).
    first_by_tag: dict[str, _Element] = {}
    # Last child per lowercased tag, for the case-insensitive prefixed
    # fallback. Only an unnamespaced tag containing ":" can equal "rss:x"
    # and the like ("{uri}x" contains ":" too, but starts with "{").
    prefixed_index: dict[str, _Element] = {}
    for child in root:
        tag = child.tag
        if not isinstance(tag, str):
            continue
        if tag not in first_by_tag:
            first_by_tag[tag] = child
        if ":" in tag and tag[0] != "{":
            prefixed_index[tag.lower()] = child

    def getter(path: str, attribute: Optional[str] = None) -> Optional[str]:
        steps = _path_tag_steps(path)
        if steps is None:
            el = root.find(path)
        elif len(steps) > 1:
            # "a/b" can only match below a child tagged "a".
            el = root.find(path) if steps[0] in first_by_tag else None
        else:
            el = first_by_tag.get(path)
            if el is None and prefixed_index and "{" not in path:
                path_lower = path.lower()
                for prefix in ("rss:", "atom:", "dc:"):
                    el = prefixed_index.get(prefix + path_lower)
                    if el is not None:
                        break

        if el is None:
            return None

        if attribute is not None:
            attr_value = el.get(attribute)
            return attr_value.strip() if attr_value else None
        text_value = el.text
        if lifted_cdata and text_value:
            text_value = _restore_lifted_cdata(text_value, lifted_cdata)
        return text_value.strip() if text_value else None

    return getter


def _normalize_iso_datetime_string(value: str) -> str:
    """Coerce flexible ISO-8601 inputs into a form datetime.fromisoformat can parse."""
    cleaned = value.strip()
    if not cleaned:
        return cleaned

    # Fast path: 'Z' suffix (most common in Atom feeds)
    if cleaned[-1] in ("Z", "z"):
        return cleaned[:-1] + "+00:00"

    # Fast path: already has proper +HH:MM or -HH:MM timezone
    if len(cleaned) > 6 and cleaned[-6] in ("+", "-") and cleaned[-3] == ":":
        return cleaned

    upper_cleaned = cleaned.upper()
    for suffix in (" UTC", " GMT", " Z"):
        if upper_cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip() + "+00:00"
            upper_cleaned = cleaned.upper()
            break

    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1] + "+00:00"

    if (
        " " in cleaned
        and "T" not in cleaned[:11]
        and len(cleaned) >= 10
        and cleaned[4] == "-"
        and cleaned[0:4].isdigit()
    ):
        date_part, rest = cleaned.split(" ", 1)
        if rest and rest[0].isdigit():
            cleaned = f"{date_part}T{rest}"

    match = _RE_ISO_TZ_NO_COLON.search(cleaned)
    if match:
        cleaned = cleaned[:-5] + f"{match.group(1)}:{match.group(2)}"
    else:
        match = _RE_ISO_TZ_HOUR_ONLY.search(cleaned)
        if match:
            cleaned = cleaned[:-3] + f"{match.group(1)}:00"

    cleaned = _RE_ISO_FRACTION.sub(lambda m: "." + m.group(1)[:6], cleaned, count=1)
    return cleaned


def _ensure_utc(dt: datetime.datetime) -> Optional[datetime.datetime]:
    """Return a timezone-aware datetime normalized to UTC."""
    try:
        return dt.replace(tzinfo=_UTC) if dt.tzinfo is None else dt.astimezone(_UTC)
    except (ValueError, OverflowError):
        return None


# What _fast_rfc822_to_iso returns for a date in RFC-822 form that names no
# real moment. Falsy, and not None: _parse_date stops there instead of letting
# the looser parsers guess at it.
_IMPOSSIBLE_DATE = ""


def _fast_rfc822_to_iso(value: str) -> Optional[str]:
    """RFC-822 date to a UTC ISO string.

    Returns None when the value is not in RFC-822 form, and _IMPOSSIBLE_DATE
    when it is but the date or time cannot exist.
    """
    m = _RE_RFC822.match(value)
    if not m:
        return None
    day, mon_str, year, hour, minute, second, tz = m.groups()
    month = _MONTHS_RFC822.get(mon_str.lower())
    if month is None:
        return None
    if tz[0] in "+-":
        tz_offset_seconds = (int(tz[1:3]) * 3600 + int(tz[3:5]) * 60) * (
            1 if tz[0] == "+" else -1
        )
    else:
        tz_offset_seconds = _custom_tzinfos.get(tz)
        if tz_offset_seconds is None:
            return None  # Unknown tz name, fall through to full parser
    # Python requires offset strictly between -24h and +24h
    if not (-86400 < tz_offset_seconds < 86400):
        return None
    h = int(hour)
    try:
        # Hour 24 is not an hour of the day; read it as 00 on the next day.
        local = datetime.datetime(
            int(year), month, int(day), 0 if h == 24 else h, int(minute), int(second)
        )
        if h == 24:
            local += datetime.timedelta(days=1)
        utc = local - datetime.timedelta(seconds=tz_offset_seconds)
    except (ValueError, OverflowError):
        # No such date, or it falls outside the years datetime can hold.
        return _IMPOSSIBLE_DATE
    return utc.isoformat() + "+00:00"


def _parsedate_to_utc(value: str) -> Optional[datetime.datetime]:
    """RFC-822 / RFC-2822 parsing via email.utils (fallback)."""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed is None:
        return None
    return _ensure_utc(parsed)


_custom_tzinfos: dict[str, int] = {
    "UTC": 0,
    "UT": 0,
    "GMT": 0,
    "WET": 0,
    "WEST": 3600,
    "BST": 3600,
    "CET": 3600,
    "CEST": 7200,
    "EET": 7200,
    "EEST": 10800,
    "MSK": 10800,
    "IST": 19800,
    "PST": -28800,
    "PDT": -25200,
    "MST": -25200,
    "MDT": -21600,
    "CST": -21600,
    "CDT": -18000,
    "EST": -18000,
    "EDT": -14400,
    "AKST": -32400,
    "AKDT": -28800,
    "HST": -36000,
    "HAST": -36000,
    "HADT": -32400,
    "AEST": 36000,
    "AEDT": 39600,
    "ACST": 34200,
    "ACDT": 37800,
    "AWST": 28800,
    "NZST": 43200,
    "NZDT": 46800,
    "JST": 32400,
    "KST": 32400,
    "SGT": 28800,
    "SST": 28800,  # Legacy alias for Singapore Standard Time
    "China Standard Time": 28800,
    "Australian Eastern Standard Time": 36000,
    "Australian Eastern Daylight Time": 39600,
}

_DATEPARSER_SETTINGS = {
    "TIMEZONE": "UTC",
    "RETURN_AS_TIMEZONE_AWARE": True,
}


@lru_cache(maxsize=512)
def _slow_dateutil_parse(value: str) -> Optional[datetime.datetime]:
    try:
        return dateutil_parser.parse(value, tzinfos=_custom_tzinfos, ignoretz=False)
    except (ValueError, TypeError, ArithmeticError):
        # ArithmeticError covers OverflowError and the decimal.InvalidOperation
        # dateutil raises for a number too long to be a time.
        return None


@lru_cache(maxsize=256)
def _slow_dateparser(value: str) -> Optional[datetime.datetime]:
    try:
        import dateparser as _dateparser  # optional dependency
    except ImportError:
        return None
    try:
        return _dateparser.parse(
            value, languages=["en"], settings=_DATEPARSER_SETTINGS
        )
    except (ValueError, TypeError, ArithmeticError):
        return None


# Zone spellings that _fast_rfc822_to_iso reads as a zero offset.
_RFC822_UTC_ZONES = frozenset(
    [name for name, offset in _custom_tzinfos.items() if offset == 0]
    + ["+0000", "-0000"]
)
_MONTH_DIGITS_RFC822 = {
    name.capitalize(): f"{number:02d}" for name, number in _MONTHS_RFC822.items()
}


def _fixed_rfc822_utc_to_iso(value: str) -> Optional[str]:
    """Convert "Mon, 02 Jan 2006 15:04:05 GMT" to ISO by character position.

    The caller has checked that the zone, value[26:], is one of
    _RFC822_UTC_ZONES. Returns None for any other layout, for hour 24, and
    for a date or time that cannot exist; _fast_rfc822_to_iso handles those.
    For the strings it accepts, it returns what _fast_rfc822_to_iso returns.
    """
    month = _MONTH_DIGITS_RFC822.get(value[8:11])
    if month is None:
        return None
    day = value[5:7]
    year = value[12:16]
    hour = value[17:19]
    minute = value[20:22]
    second = value[23:25]
    if not (
        value.isascii()
        and value[:3].isalpha()
        and value[4] == value[7] == value[11] == value[16] == value[25] == " "
        and value[19] == value[22] == ":"
        and day.isdigit()
        and year.isdigit()
        and hour.isdigit()
        and minute.isdigit()
        and second.isdigit()
        # Two ASCII digits compare like the numbers they spell.
        and hour < "24"
        and minute < "60"
        and second < "60"
    ):
        return None
    if not ("01" <= day <= "28" and year != "0000"):
        try:
            datetime.date(int(year), int(month), int(day))
        except ValueError:
            return None
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}+00:00"


# Longer strings are not dates, and dateutil tokenizes in quadratic time.
_MAX_DATE_CHARS = 256


@lru_cache(maxsize=8192)
def _parse_date(date_str: str) -> Optional[str]:
    """Parse date string and return as an ISO 8601 formatted UTC string.

    Args:
        date_str: Date string in any common format

    Returns:
        ISO‑8601 formatted UTC date string, or None when parsing fails
    """
    if not date_str:
        return None

    candidate = date_str.strip()
    if not candidate:
        return None

    clen = len(candidate)
    if clen > _MAX_DATE_CHARS:
        return None

    # Fast path: the layout most RSS feeds use, with a 3-char or 5-char zone
    if (
        (clen == 29 or clen == 31)
        and candidate[3] == ","
        and candidate[26:] in _RFC822_UTC_ZONES
    ):
        rfc822_utc = _fixed_rfc822_utc_to_iso(candidate)
        if rfc822_utc is not None:
            return rfc822_utc

    # Fast path: clean ISO-8601 (covers >90% of Atom/modern RSS dates)
    if clen >= 20 and candidate[4] == "-" and candidate[0:4].isdigit():
        last = candidate[-1]
        # Most common: ends with 'Z' (e.g., 2024-01-15T10:30:00Z)
        if last in ("Z", "z"):
            iso = candidate[:-1] + "+00:00"
            try:
                dt = datetime.datetime.fromisoformat(iso)
                if clen == 20 and _RE_ISO_UTC_SECONDS.fullmatch(iso):
                    return iso
                return dt.isoformat()
            except ValueError:
                pass  # Fall through to full parsing
        # Second most common: ends with +HH:MM (e.g., 2024-01-15T10:30:00+00:00)
        elif clen > 6 and candidate[-6] in ("+", "-") and candidate[-3] == ":":
            try:
                dt = datetime.datetime.fromisoformat(candidate)
                if clen == 25 and _RE_ISO_UTC_SECONDS.fullmatch(candidate):
                    return candidate
                if dt.tzinfo is _UTC:
                    return dt.isoformat()
                utc_dt = dt.astimezone(_UTC)
                return utc_dt.isoformat()
            except (ValueError, OverflowError):
                pass  # Fall through to full parsing

    if "\n" in candidate or "\r" in candidate or "\t" in candidate or "  " in candidate:
        candidate = _RE_WHITESPACE.sub(" ", candidate)

    # Fix invalid leap year dates (Feb 29 in non-leap years)
    # This handles feeds with incorrect dates like "2023-02-29"
    if "-02-29" in candidate:
        year_match = _RE_FEB29.match(candidate)
        if year_match:
            year = int(year_match.group(1))
            if not ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
                candidate = candidate.replace(f"{year}-02-29", f"{year}-02-28")

    if "T24:" in candidate or " 24:" in candidate:
        m24 = _RE_HOUR24.search(candidate)
        if m24:
            try:
                base = datetime.date.fromisoformat(m24.group(1))
                next_day = base + datetime.timedelta(days=1)
            except (ValueError, OverflowError):
                # No such day, or no day after it; the parsers below reject it.
                next_day = None
            if next_day is not None:
                mins, secs = int(m24.group(2)), int(m24.group(3))
                candidate = (
                    candidate[: m24.start()]
                    + f"{next_day}T00:{mins:02d}:{secs:02d}"
                    + candidate[m24.end() :]
                )

    dt: Optional[datetime.datetime] = None

    is_iso_like = (
        len(candidate) >= 10 and candidate[4] == "-" and candidate[0:4].isdigit()
    )
    if is_iso_like:
        iso_candidate = _normalize_iso_datetime_string(candidate)
        try:
            dt = datetime.datetime.fromisoformat(iso_candidate)
        except ValueError:
            dt = None
        if dt is not None:
            utc_dt = _ensure_utc(dt)
            if utc_dt is not None:
                return utc_dt.isoformat()

    rfc822_result = _fast_rfc822_to_iso(candidate)
    if rfc822_result is not None:
        return rfc822_result or None

    dt = _parsedate_to_utc(candidate)
    if dt is not None:
        return dt.isoformat()

    slow_dt = _slow_dateutil_parse(candidate)
    if slow_dt is not None:
        utc_dt = _ensure_utc(slow_dt)
        if utc_dt is not None:
            return utc_dt.isoformat()

    parsed = _slow_dateparser(candidate)
    if parsed is not None:
        utc_dt = _ensure_utc(parsed)
        if utc_dt is not None:
            return utc_dt.isoformat()

    # If all parsing attempts fail, return None
    return None
