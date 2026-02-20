from __future__ import annotations

import datetime
from email.utils import parsedate_to_datetime
import gzip
import html as _html_mod
import json
import re
import zlib
from functools import lru_cache

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
from urllib.parse import urljoin
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
_RE_UNQUOTED_ATTR_BYTES = re.compile(rb'(\s+[\w:]+)=([^\s>"\']+)')
_RE_UTF16_ENCODING_BYTES = re.compile(
    rb'(<\?xml[^>]*encoding=["\'])utf-16(-le|-be)?(["\'][^>]*\?>)', re.IGNORECASE
)
_RE_UNCLOSED_LINK_BYTES = re.compile(
    rb"<link([^>]*[^/])>\s*(?=\n\s*<(?!/link\s*>))", re.MULTILINE
)
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


def _ensure_utf8_xml_declaration(content: str) -> str:
    """Ensure the XML declaration's encoding matches the UTF-8 bytes we emit."""
    if not content.lstrip().startswith("<?xml"):
        return content
    return _RE_XML_DECL_ENCODING.sub(r"\1utf-8\3", content, count=1)


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


def _fix_malformed_xml_bytes(content: bytes, actual_encoding: str = "utf-8") -> bytes:
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

    # Reassemble before running body-wide fixes
    content = header + tail

    # Fix malformed attribute syntax like rss:version=2.0 (missing quotes)
    content = _RE_UNQUOTED_ATTR_BYTES.sub(rb'\1="\2"', content)

    # Fix unclosed link tags - common in Atom feeds
    content = _RE_UNCLOSED_LINK_BYTES.sub(rb"<link\1/>", content)

    return content


def _prepare_xml_bytes(xml_content: str | bytes) -> bytes:
    if isinstance(xml_content, bytes):
        cleaned = _clean_feed_bytes(xml_content)
        if not cleaned.strip():
            raise ValueError("Empty content")

        # Replace Unicode LINE SEPARATOR (U+2028) and PARAGRAPH SEPARATOR (U+2029)
        # with regular newlines — these are invalid in XML 1.0 and cause lxml to fail.
        if b"\xe2\x80\xa8" in cleaned or b"\xe2\x80\xa9" in cleaned:
            cleaned = cleaned.replace(b"\xe2\x80\xa8", b"\n").replace(
                b"\xe2\x80\xa9", b"\n"
            )

        detected_encoding = _detect_xml_encoding(cleaned)
        actual_encoding = detected_encoding
        if detected_encoding.startswith("utf-16") and b"\x00" not in cleaned[:200]:
            actual_encoding = "utf-8"

        needs_fixing = (
            b"?xml?xml" in cleaned[:200].lower()
            or b"??>" in cleaned[:200]
            or (
                b"rss:" in cleaned[:500].lower()
                and b"xmlns:rss" not in cleaned[:1000].lower()
            )
            or (b"utf-16" in cleaned[:200].lower() and actual_encoding != "utf-16")
        )
        if needs_fixing:
            cleaned = _fix_malformed_xml_bytes(cleaned, actual_encoding=actual_encoding)
        return cleaned

    # Str input: fix encoding declaration, encode to bytes, then use bytes path.
    xml_content = _ensure_utf8_xml_declaration(xml_content)
    return _prepare_xml_bytes(xml_content.encode("utf-8", errors="replace"))


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


def _fetch_url_content(url: str) -> str | bytes:
    accept_encoding = "gzip, deflate, br" if HAS_BROTLI else "gzip, deflate"
    request = Request(
        url,
        method="GET",
        headers={
            "Accept-Encoding": accept_encoding,
            "User-Agent": "fastfeedparser (+https://github.com/kagisearch/fastfeedparser)",
        },
    )
    opener = build_opener(HTTPRedirectHandler(), HTTPErrorProcessor())
    with opener.open(request, timeout=30) as response:
        content: bytes = response.read()
        content_encoding = response.headers.get("Content-Encoding")
        if content_encoding == "gzip":
            content = gzip.decompress(content)
        elif content_encoding == "deflate":
            content = zlib.decompress(content, -zlib.MAX_WBITS)
        elif content_encoding == "br":
            if not HAS_BROTLI:
                raise ValueError(
                    "Received brotli-compressed response but 'brotli' is not installed"
                )
            content = brotli.decompress(content)
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


_STRICT_XML_PARSER = etree.XMLParser(
    ns_clean=True,
    recover=False,
    collect_ids=False,
    resolve_entities=False,
)
_RECOVER_XML_PARSER = etree.XMLParser(
    ns_clean=True,
    recover=True,
    collect_ids=False,
    resolve_entities=False,
)


def _parse_xml_root(xml_content: bytes) -> _Element:
    try:
        root = etree.fromstring(xml_content, parser=_STRICT_XML_PARSER)
    except etree.XMLSyntaxError:
        try:
            root = etree.fromstring(xml_content, parser=_RECOVER_XML_PARSER)
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


_RE_META_REFRESH_URL = re.compile(r'url\s*=\s*["\']?\s*([^"\'>\s]+)', re.IGNORECASE)


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
                if url != base_url:
                    return url
    return None


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

        if len(items) < 5 and len(xml_content) > 20000:
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

    xml_content = _prepare_xml_bytes(xml_content)
    root = _parse_xml_root(xml_content)
    root_tag_local = _root_tag_local(root)
    _raise_for_non_feed_root(root, root_tag_local, xml_content)

    feed_type, channel, items, atom_namespace = _detect_feed_structure(
        root, xml_content, root_tag_local
    )

    feed = _parse_feed_info(
        channel, feed_type, atom_namespace, include_tags=include_tags
    )

    # Detect once whether media namespace is used anywhere in the document
    has_media_ns = (
        b"search.yahoo.com/mrss" in xml_content
        if isinstance(xml_content, bytes)
        else "search.yahoo.com/mrss" in xml_content
    )

    # Parse entries — resolve parser once per feed instead of per entry
    entries: list[FastFeedParserDict] = []
    feed["entries"] = entries
    for item in items:
        entry = _parse_feed_entry(
            item,
            feed_type,
            atom_namespace,
            has_media_ns,
            include_content=include_content,
            include_tags=include_tags,
            include_media=include_media,
            include_enclosures=include_enclosures,
        )
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

    try:
        return _parse_content(
            content,
            include_content=include_content,
            include_tags=include_tags,
            include_media=include_media,
            include_enclosures=include_enclosures,
        )
    except ValueError as e:
        if not is_url:
            raise
        assert isinstance(source, str)
        err_msg = str(e)
        if "HTML" not in err_msg and "not a valid RSS/Atom feed" not in err_msg:
            raise
        redirect_url = _extract_meta_refresh_url(content, source)
        if redirect_url is None:
            raise
        return parse(
            redirect_url,
            include_content=include_content,
            include_tags=include_tags,
            include_media=include_media,
            include_enclosures=include_enclosures,
        )


def _parse_feed_info(
    channel: _Element,
    feed_type: _FeedType,
    atom_namespace: Optional[str] = None,
    *,
    include_tags: bool = True,
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
    element_get = _cached_element_value_factory(channel)
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
        if rel is None and href:
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
            alternate_link = link_dict
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


def _synthesize_entry_description(entry: FastFeedParserDict) -> None:
    if "description" in entry or "content" not in entry:
        return

    content_value = entry["content"][0]["value"]
    if content_value:
        if "<" in content_value and ">" in content_value:
            content_value = _RE_HTML_TAGS.sub(" ", content_value[:2048])
            if "&" in content_value:
                content_value = _html_mod.unescape(content_value)
        if (
            "  " in content_value
            or "\n" in content_value
            or "\t" in content_value
            or "\r" in content_value
        ):
            content_value = _RE_WHITESPACE.sub(" ", content_value).strip()
        else:
            content_value = content_value.strip()
    entry["description"] = content_value[:512]


def _populate_entry_content_preparsed(
    entry: FastFeedParserDict,
    item: _Element,
    *,
    content_el: Optional[_Element],
    rss_description_text: Optional[str],
) -> None:
    if content_el is not None:
        content_type = content_el.get("type", "text/html")
        if content_type in {"xhtml", "application/xhtml+xml"}:
            content_value = etree.tostring(content_el, encoding="unicode", method="xml")
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
    entry: FastFeedParserDict, item: _Element, feed_type: _FeedType, atom_ns: str
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
    elif feed_type == "atom":
        content_el = item.find(_atom_ns_tags(atom_ns)["content"])

    _populate_entry_content_preparsed(
        entry,
        item,
        content_el=content_el,
        rss_description_text=rss_description_text,
    )


def _parse_media_content(item: _Element) -> list[dict[str, Any]] | None:
    media_contents: list[dict[str, Any]] = []

    for media in item.findall(f".//{_MEDIA_CONTENT_TAG}"):
        media_item: dict[str, str | int | None] = {
            "url": media.get("url"),
            "type": media.get("type"),
            "medium": media.get("medium"),
            "width": media.get("width"),
            "height": media.get("height"),
        }
        _coerce_int_fields(media_item, ("width", "height"))

        title = media.find(_MEDIA_TITLE_TAG)
        if title is not None and title.text:
            media_item["title"] = title.text.strip()

        text = media.find(_MEDIA_TEXT_TAG)
        if text is not None and text.text:
            media_item["text"] = text.text.strip()

        desc = media.find(_MEDIA_DESCRIPTION_TAG)
        if desc is None:
            parent = media.getparent()
            if parent is not None:
                desc = parent.find(_MEDIA_DESCRIPTION_TAG)
        if desc is not None and desc.text:
            media_item["description"] = desc.text.strip()

        credit = media.find(_MEDIA_CREDIT_TAG)
        if credit is None:
            parent = media.getparent()
            if parent is not None:
                credit = parent.find(_MEDIA_CREDIT_TAG)
        if credit is not None and credit.text:
            media_item["credit"] = credit.text.strip()
            media_item["credit_scheme"] = credit.get("scheme")

        thumbnail = media.find(_MEDIA_THUMBNAIL_TAG)
        if thumbnail is not None:
            media_item["thumbnail_url"] = thumbnail.get("url")

        cleaned = _drop_none_values(media_item)
        if cleaned:
            media_contents.append(cleaned)

    if not media_contents:
        for thumbnail in item.findall(f".//{_MEDIA_THUMBNAIL_TAG}"):
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


def _first_non_empty(
    mapping: dict[str, Optional[str]], keys: tuple[str, ...]
) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


def _parse_rss_feed_entry_fast(
    item: _Element,
    atom_ns: str,
    has_media_ns: bool = True,
    *,
    include_content: bool = True,
    include_tags: bool = True,
    include_media: bool = True,
    include_enclosures: bool = True,
) -> FastFeedParserDict:
    atom_tags = _atom_ns_tags(atom_ns)
    text_by_local: dict[str, Optional[str]] = {}
    text_by_full: dict[str, Optional[str]] = {}
    atom_links: list[_Element] = []
    guid_element: Optional[_Element] = None
    encoded_content_el: Optional[_Element] = None
    raw_content_el: Optional[_Element] = None
    rss_description_text: Optional[str] = None
    tag_categories: list[dict[str, str | None]] = []
    tag_subjects: list[dict[str, str | None]] = []
    enclosures: list[dict[str, Any]] = []

    for child in item:
        tag = child.tag
        if not isinstance(tag, str):
            continue

        text_value = child.text or None
        if tag not in text_by_full:
            text_by_full[tag] = text_value

        if "{" in tag:
            local = tag.rsplit("}", 1)[1].lower()
        elif ":" in tag:
            local = tag.split(":", 1)[1].lower()
        else:
            local = tag.lower()
        if local not in text_by_local:
            text_by_local[local] = text_value

        if tag == atom_tags["link"]:
            atom_links.append(child)
        elif tag == "guid":
            if guid_element is None:
                guid_element = child
        elif tag == _RSS_CONTENT_ENCODED_TAG:
            if encoded_content_el is None:
                encoded_content_el = child
        elif tag == "content":
            if raw_content_el is None:
                raw_content_el = child
        elif tag == "description":
            if rss_description_text is None:
                rss_description_text = text_value

        if include_enclosures and tag == "enclosure":
            cleaned = _parse_enclosure_element(child)
            if cleaned.get("url"):
                enclosures.append(cleaned)

        if include_tags:
            if local == "category":
                term = text_value.strip() if text_value else None
                if term:
                    tag_categories.append(
                        {"term": term, "scheme": child.get("domain"), "label": None}
                    )
            elif tag == _DC_SUBJECT_TAG:
                term = text_value.strip() if text_value else None
                if term:
                    tag_subjects.append({"term": term, "scheme": None, "label": None})

    entry = FastFeedParserDict()
    atom_id = text_by_full.get(atom_tags["id"])
    rss_guid = text_by_local.get("guid")
    rdf_about = item.get(_RDF_ABOUT_ATTR)
    entry_id: Optional[str] = atom_id or rss_guid or rdf_about
    if entry_id:
        entry["id"] = entry_id.strip()

    title = text_by_local.get("title")
    if title:
        entry["title"] = title.strip()

    description = _first_non_empty(text_by_local, ("description", "summary"))
    if description:
        entry["description"] = description.strip()

    link = text_by_local.get("link")
    if link:
        entry["link"] = link.strip()

    published_source = _first_non_empty(
        text_by_local, ("pubdate", "published", "issued", "date")
    )
    if published_source:
        published = _parse_date(published_source)
        if published:
            entry["published"] = published

    updated_source = _first_non_empty(
        text_by_local, ("lastbuilddate", "updated", "modified")
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
        _populate_entry_content_preparsed(
            entry,
            item,
            content_el=(
                encoded_content_el if encoded_content_el is not None else raw_content_el
            ),
            rss_description_text=rss_description_text,
        )

    if include_media and has_media_ns:
        media_contents = _parse_media_content(item)
        if media_contents:
            entry["media_content"] = media_contents

    if include_enclosures and enclosures:
        entry["enclosures"] = enclosures

    author = _first_non_empty(text_by_local, ("author", "creator"))
    if not author:
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


def _parse_atom_feed_entry_fast(
    item: _Element,
    atom_ns: str,
    has_media_ns: bool = True,
    *,
    include_content: bool = True,
    include_tags: bool = True,
    include_media: bool = True,
    include_enclosures: bool = True,
) -> FastFeedParserDict:
    t = _atom_ns_tags(atom_ns)
    atom_link_tag = t["link"]
    atom_author_tag = t["ns"] + "author"
    atom_name_tag = t["ns"] + "name"
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

    entry = FastFeedParserDict()
    for child in item:
        tag = child.tag
        if not isinstance(tag, str):
            continue

        text_value = child.text
        if tag == t["id"] and "id" not in entry and text_value:
            entry["id"] = text_value.strip()
        elif tag == t["title"] and "title" not in entry and text_value:
            entry["title"] = text_value.strip()
        elif tag == t["summary"] and "description" not in entry and text_value:
            entry["description"] = text_value.strip()
        elif tag == t["published"] and published_source is None and text_value:
            published_source = text_value
        elif tag == t["updated"] and updated_source is None and text_value:
            updated_source = text_value
        elif (
            tag == t["pub_fallback"]
            and published_fallback_source is None
            and text_value
        ):
            published_fallback_source = text_value
        elif (
            tag == t["upd_fallback"] and updated_fallback_source is None and text_value
        ):
            updated_fallback_source = text_value
        elif tag == atom_link_tag:
            atom_links.append(child)
            href = child.get("href")
            if href and first_link_href is None:
                first_link_href = href.strip()
        elif include_content and tag == t["content"] and content_el is None:
            content_el = child
        elif tag == atom_author_tag and author_name is None:
            author_name_el = child.find(atom_name_tag)
            if author_name_el is not None and author_name_el.text:
                author_name = author_name_el.text.strip()

        if include_tags and tag == t["category"]:
            term = child.get("term")
            if term:
                atom_categories.append(
                    {
                        "term": term,
                        "scheme": child.get("scheme"),
                        "label": child.get("label"),
                    }
                )

        if include_enclosures and tag == "enclosure":
            cleaned = _parse_enclosure_element(child)
            if cleaned.get("url"):
                enclosures.append(cleaned)

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
        )

    if include_media and has_media_ns:
        media_contents = _parse_media_content(item)
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
    has_media_ns: bool = True,
    *,
    include_content: bool = True,
    include_tags: bool = True,
    include_media: bool = True,
    include_enclosures: bool = True,
) -> FastFeedParserDict:
    # Use dynamic atom namespace or fallback to default
    atom_ns = atom_namespace or "http://www.w3.org/2005/Atom"

    if feed_type == "rss":
        return _parse_rss_feed_entry_fast(
            item,
            atom_ns,
            has_media_ns,
            include_content=include_content,
            include_tags=include_tags,
            include_media=include_media,
            include_enclosures=include_enclosures,
        )

    if feed_type == "atom":
        return _parse_atom_feed_entry_fast(
            item,
            atom_ns,
            has_media_ns,
            include_content=include_content,
            include_tags=include_tags,
            include_media=include_media,
            include_enclosures=include_enclosures,
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

    element_get = _cached_element_value_factory(item)
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
        _populate_entry_content(entry, item, feed_type, atom_ns)

    if include_media and has_media_ns:
        media_contents = _parse_media_content(item)
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


def _get_element_value(
    root: _Element,
    path: str,
    attribute: Optional[str] = None,
    child_index: Optional[dict[str, _Element]] = None,
) -> Optional[str]:
    """Get text content or attribute value of an element.

    Also tries common namespace prefixes (rss:, atom:) for malformed feeds.
    """
    el = root.find(path)

    # If not found and path is a simple element name, try with common prefixes
    if el is None and "/" not in path and "{" not in path:
        path_lower = path.lower()
        if child_index is not None:
            for prefix in ("rss:", "atom:", "dc:"):
                found = child_index.get(f"{prefix}{path_lower}")
                if found is not None:
                    el = found
                    break
        else:
            prefixed_paths = [
                f"rss:{path_lower}",
                f"atom:{path_lower}",
                f"dc:{path_lower}",
            ]
            for child in root:
                if not isinstance(child.tag, str):
                    continue
                if child.tag.lower() in prefixed_paths:
                    el = child
                    break

    if el is None:
        return None

    if attribute is not None:
        attr_value = el.get(attribute)
        return attr_value.strip() if attr_value else None
    text_value = el.text
    return text_value.strip() if text_value else None


def _cached_element_value_factory(
    root: _Element,
) -> _ElementValueGetter:
    """Create a closure with a child tag index for fast namespace-prefix lookups."""
    # Build child tag index once: O(children) instead of O(children × misses)
    child_index: dict[str, _Element] = {}
    for child in root:
        if isinstance(child.tag, str):
            child_index[child.tag.lower()] = child

    def getter(path: str, attribute: Optional[str] = None) -> Optional[str]:
        return _get_element_value(
            root, path, attribute=attribute, child_index=child_index
        )

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


def _fast_rfc822_to_iso(value: str) -> Optional[str]:
    """Fast RFC-822 date to ISO string, bypassing datetime objects for UTC dates."""
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
    d = int(day)
    h = int(hour)
    mi = int(minute)
    s = int(second)
    # Hour 24 is invalid (even ISO only allows 24:00:00); roll to next day at 00:mm:ss
    if h == 24:
        base = datetime.date(int(year), month, d) + datetime.timedelta(days=1)
        h = 0
        if tz_offset_seconds == 0:
            return f"{base.year:04d}-{base.month:02d}-{base.day:02d}T{h:02d}:{mi:02d}:{s:02d}+00:00"
        dt = datetime.datetime(
            base.year,
            base.month,
            base.day,
            h,
            mi,
            s,
            tzinfo=datetime.timezone(datetime.timedelta(seconds=tz_offset_seconds)),
        )
        utc = dt.astimezone(_UTC)
        return f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}T{utc.hour:02d}:{utc.minute:02d}:{utc.second:02d}+00:00"
    if tz_offset_seconds == 0:
        return f"{year}-{month:02d}-{d:02d}T{hour}:{minute}:{second}+00:00"
    dt = datetime.datetime(
        int(year),
        month,
        d,
        h,
        mi,
        s,
        tzinfo=datetime.timezone(datetime.timedelta(seconds=tz_offset_seconds)),
    )
    utc = dt.astimezone(_UTC)
    return f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}T{utc.hour:02d}:{utc.minute:02d}:{utc.second:02d}+00:00"


def _parsedate_to_utc(value: str) -> Optional[datetime.datetime]:
    """RFC-822 / RFC-2822 parsing via email.utils (fallback)."""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
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
    except (ValueError, TypeError, OverflowError):
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
    except (ValueError, TypeError):
        return None


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

    # Fast path: clean ISO-8601 (covers >90% of Atom/modern RSS dates)
    clen = len(candidate)
    if clen >= 20 and candidate[4] == "-" and candidate[0:4].isdigit():
        last = candidate[-1]
        # Most common: ends with 'Z' (e.g., 2024-01-15T10:30:00Z)
        if last in ("Z", "z"):
            iso = candidate[:-1] + "+00:00"
            try:
                dt = datetime.datetime.fromisoformat(iso)
                return dt.isoformat()
            except ValueError:
                pass  # Fall through to full parsing
        # Second most common: ends with +HH:MM (e.g., 2024-01-15T10:30:00+00:00)
        elif clen > 6 and candidate[-6] in ("+", "-") and candidate[-3] == ":":
            try:
                dt = datetime.datetime.fromisoformat(candidate)
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
            base = datetime.date.fromisoformat(m24.group(1))
            mins, secs = int(m24.group(2)), int(m24.group(3))
            next_day = base + datetime.timedelta(days=1)
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
        return rfc822_result

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
