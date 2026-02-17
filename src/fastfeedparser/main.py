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
from typing import Any, Callable, Optional, TYPE_CHECKING, Literal
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
    from lxml.etree import _Element

_FeedType = Literal["rss", "atom", "rdf"]

_UTC = datetime.timezone.utc

# Pre-compiled regex patterns for performance
_RE_XML_DECL_ENCODING = re.compile(
    r'(<\?xml[^>]*encoding=["\'])([^"\']+)(["\'][^>]*\?>)', re.IGNORECASE
)
_RE_XML_DECL_ENCODING_BYTES = re.compile(
    br'(<\?xml[^>]*encoding=["\'])([^"\']+)(["\'][^>]*\?>)', re.IGNORECASE
)
_RE_DOUBLE_XML_DECL = re.compile(r"<\?xml\?xml\s+", re.IGNORECASE)
_RE_DOUBLE_XML_DECL_BYTES = re.compile(br"<\?xml\?xml\s+", re.IGNORECASE)
_RE_DOUBLE_CLOSE = re.compile(r"\?\?>\s*")
_RE_DOUBLE_CLOSE_BYTES = re.compile(br"\?\?>\s*")
_RE_UNQUOTED_ATTR = re.compile(r'(\s+[\w:]+)=([^\s>"\']+)')
_RE_UNQUOTED_ATTR_BYTES = re.compile(br'(\s+[\w:]+)=([^\s>"\']+)')
_RE_UTF16_ENCODING = re.compile(
    r'(<\?xml[^>]*encoding=["\'])utf-16(-le|-be)?(["\'][^>]*\?>)', re.IGNORECASE
)
_RE_UTF16_ENCODING_BYTES = re.compile(
    br'(<\?xml[^>]*encoding=["\'])utf-16(-le|-be)?(["\'][^>]*\?>)', re.IGNORECASE
)
_RE_UNCLOSED_LINK = re.compile(
    r"<link([^>]*[^/])>\s*(?=\n\s*<(?!/link\s*>))", re.MULTILINE
)
_RE_UNCLOSED_LINK_BYTES = re.compile(
    br"<link([^>]*[^/])>\s*(?=\n\s*<(?!/link\s*>))", re.MULTILINE
)
_RE_FEB29 = re.compile(r"(\d{4})-02-29")
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_ISO_TZ_NO_COLON = re.compile(r"([+-]\d{2})(\d{2})$")
_RE_ISO_TZ_HOUR_ONLY = re.compile(r"([+-]\d{2})$")
_RE_ISO_FRACTION = re.compile(r"\.(\d{7,})(?=(?:[+-]\d{2}:?\d{2}|Z|$))", re.IGNORECASE)


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


def _clean_feed_text(content: str) -> str:
    """Clean feed text by extracting the XML document (if it's embedded in junk)."""
    stripped_content = content.lstrip()
    stripped_lower = stripped_content[:2000].lower()
    if stripped_lower.startswith(("<?xml", "<rss", "<feed", "<rdf")):
        return stripped_content

    if stripped_lower.startswith("<!doctype html") or stripped_lower.startswith("<html"):
        raise ValueError("Content appears to be HTML, not a valid RSS/Atom feed")

    xml_start_patterns = (
        "<?xml",
        "<rss",
        "<feed",
        "<rdf:rdf",
        "<?xml-stylesheet",
    )

    content_lines = content.splitlines()
    for i, line in enumerate(content_lines):
        line_stripped = line.strip().lower()
        if any(line_stripped.startswith(pattern) for pattern in xml_start_patterns):
            return "\n".join(content_lines[i:])

    if "<script>" in stripped_lower or "<body>" in stripped_lower:
        raise ValueError("Content appears to be HTML, not a valid RSS/Atom feed")

    return content


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

    if preview_lower.startswith(b"<!doctype html") or preview_lower.startswith(b"<html"):
        raise ValueError("Content appears to be HTML, not a valid RSS/Atom feed")

    xml_start_patterns = (
        b"<?xml",
        b"<rss",
        b"<feed",
        b"<rdf:rdf",
        b"<?xml-stylesheet",
    )

    lines = content.splitlines()
    for i, line in enumerate(lines):
        line_stripped = line.strip().lower()
        if any(line_stripped.startswith(pattern) for pattern in xml_start_patterns):
            return b"\n".join(lines[i:])

    if b"<script>" in preview_lower or b"<body>" in preview_lower:
        raise ValueError("Content appears to be HTML, not a valid RSS/Atom feed")

    return content


def _fix_malformed_xml_bytes(content: bytes, actual_encoding: str = "utf-8") -> bytes:
    # Fix double XML declarations like "<?xml?xml version="1.0"?>"
    content = _RE_DOUBLE_XML_DECL_BYTES.sub(b"<?xml ", content)

    # Fix double closing ?> in XML declaration like "??>>"
    content = _RE_DOUBLE_CLOSE_BYTES.sub(b"?>", content)

    # Fix malformed attribute syntax like rss:version=2.0 (missing quotes)
    content = _RE_UNQUOTED_ATTR_BYTES.sub(br'\1="\2"', content)

    # Update encoding in XML declaration to match actual encoding when a feed was transcoded.
    if actual_encoding.lower() != "utf-16":
        replacement = br"\1" + actual_encoding.encode("ascii", errors="replace") + br"\3"
        content = _RE_UTF16_ENCODING_BYTES.sub(replacement, content)

    # Fix unclosed link tags - common in Atom feeds
    content = _RE_UNCLOSED_LINK_BYTES.sub(br"<link\1/>", content)

    return content


def _prepare_xml_bytes(xml_content: str | bytes) -> bytes:
    if isinstance(xml_content, bytes):
        cleaned = _clean_feed_bytes(xml_content)
        if not cleaned.strip():
            raise ValueError("Empty content")

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

    cleaned_text = _clean_feed_text(xml_content)
    if not cleaned_text.strip():
        raise ValueError("Empty content")

    needs_fixing = (
        "?xml?xml" in cleaned_text[:200]
        or "??>" in cleaned_text[:200]
        or (
            "rss:" in cleaned_text[:500] and "xmlns:rss" not in cleaned_text[:1000]
        )
        or ("utf-16" in cleaned_text[:200].lower())
    )
    if needs_fixing:
        cleaned_text = _fix_malformed_xml(cleaned_text, actual_encoding="utf-8")

    cleaned_text = _ensure_utf8_xml_declaration(cleaned_text)
    return cleaned_text.encode("utf-8", errors="replace")


def _fix_malformed_xml(content: str, actual_encoding: str = "utf-8") -> str:
    """Fix common malformed XML issues in feeds.

    Some feeds have malformed XML like unclosed link tags or other issues
    that can be automatically corrected.

    Args:
        content: The XML content as a string
        actual_encoding: The actual encoding used (default: utf-8)
    """
    # Fix double XML declarations like "<?xml?xml version="1.0"?>"
    # This is found in dylanharris.org feed
    content = _RE_DOUBLE_XML_DECL.sub(r"<?xml ", content)

    # Fix double closing ?> in XML declaration like "??>>"
    content = _RE_DOUBLE_CLOSE.sub(r"?>", content)

    # Fix malformed attribute syntax like rss:version=2.0 (missing quotes)
    # This is found in dylanharris.org feed
    content = _RE_UNQUOTED_ATTR.sub(r'\1="\2"', content)

    # Update encoding in XML declaration to match actual encoding
    # This handles cases where content was transcoded from UTF-16 to UTF-8
    if actual_encoding.lower() != "utf-16":
        content = _RE_UTF16_ENCODING.sub(rf"\1{actual_encoding}\3", content)

    # Fix unclosed link tags - common in Atom feeds
    # Pattern: <link ...> followed by whitespace and another tag (not </link>)
    # should be <link .../>
    # Only fix link tags that are clearly malformed:
    # - End with > instead of />
    # - Are followed by whitespace and another tag (not a closing </link>)
    content = _RE_UNCLOSED_LINK.sub(r"<link\1/>", content)

    return content


def _parse_json_feed(json_data: dict) -> FastFeedParserDict:
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
            entry["content"] = [{"type": "text/html", "value": content_html}]
            entry["description"] = summary
        elif content_text:
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
        if tags:
            entry["tags"] = [
                {"term": tag, "scheme": None, "label": None} for tag in tags
            ]

        # Add attachments as enclosures
        attachments = item.get("attachments")
        if attachments:
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
        return content.decode(content_charset) if content_charset else content


def _maybe_parse_json_feed(content: str | bytes) -> FastFeedParserDict | None:
    if isinstance(content, bytes):
        if not content.lstrip().startswith(b"{"):
            return None
        json_str = content.decode("utf-8", errors="replace")
    else:
        if not content.lstrip().startswith("{"):
            return None
        json_str = content

    try:
        json_data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(json_data, dict):
        return None

    version = json_data.get("version")
    if isinstance(version, str) and "jsonfeed.org" in version:
        return _parse_json_feed(json_data)

    if isinstance(json_data.get("items"), list):
        return _parse_json_feed(json_data)

    return None


def _parse_xml_root(xml_content: bytes) -> _Element:
    try:
        strict_parser = etree.XMLParser(
            ns_clean=True,
            recover=False,
            collect_ids=False,
            resolve_entities=False,
        )
        root = etree.fromstring(xml_content, parser=strict_parser)
    except etree.XMLSyntaxError:
        recover_parser = etree.XMLParser(
            ns_clean=True,
            recover=True,
            collect_ids=False,
            resolve_entities=False,
        )
        try:
            root = etree.fromstring(xml_content, parser=recover_parser)
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


def _raise_for_non_feed_root(
    root: _Element, root_tag_local: str, raw_bytes: Optional[bytes] = None
) -> None:
    non_feed_tags = {
        "status", "error", "html", "opml", "br", "div", "body",
        "urlset", "sitemapindex",
    }
    if root_tag_local not in non_feed_tags:
        return

    error_msg = _extract_error_message(root, raw_bytes).strip()[:300] or "No error message"

    if root_tag_local == "html":
        if error_msg != "No error message" and len(error_msg) > 10:
            raise ValueError(f"Received HTML page instead of feed: {error_msg[:150]}")
        raise ValueError(
            "Received HTML page instead of feed (possible redirect, 404, or server error)"
        )
    if root_tag_local in {"div", "body"}:
        if error_msg != "No error message" and len(error_msg) > 10:
            raise ValueError(f"Received HTML fragment instead of feed: {error_msg[:150]}")
        raise ValueError("Received HTML fragment instead of feed")
    if root_tag_local == "br":
        if error_msg != "No error message" and len(error_msg) > 10:
            raise ValueError(f"Received HTML error instead of feed: {error_msg[:150]}")
        raise ValueError("Received HTML fragment instead of feed")
    if root_tag_local == "status":
        raise ValueError(f"Feed server returned status message: {error_msg}")
    if root_tag_local == "error":
        if error_msg != "No error message":
            raise ValueError(f"Feed server returned error: {error_msg}")
        raise ValueError("Feed server returned error (no details provided)")
    if root_tag_local == "opml":
        raise ValueError(
            "Received OPML document instead of feed (OPML is an outline format, not a feed)"
        )
    if root_tag_local in {"urlset", "sitemapindex"}:
        raise ValueError(
            "Received XML sitemap instead of feed (sitemap is for search engines, not a feed)"
        )


_RE_META_REFRESH_URL = re.compile(
    r'url\s*=\s*["\']?\s*([^"\'>\s]+)', re.IGNORECASE
)


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
                and child.tag in {"entry", "title", "subtitle", "updated", "id", "author", "link"}
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
                items = channel.xpath(".//item") or channel.xpath(".//*[local-name()='item']")

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
        items = channel.findall(".//{http://purl.org/rss/1.0/}item") or channel.findall("item")
        return feed_type, channel, items, atom_namespace

    raise ValueError(f"Unknown feed type: {root.tag}")


def _should_parse_media_content(root: _Element, xml_content: bytes) -> bool:
    """Check if feed likely contains Media RSS fields."""
    ns_values = root.nsmap.values() if root.nsmap else ()
    for ns_value in ns_values:
        if not ns_value:
            continue
        if "search.yahoo.com/mrss" in ns_value:
            return True

    # Fallback for feeds with undeclared/late namespace usage.
    return b"search.yahoo.com/mrss" in xml_content or b"<media:" in xml_content


def _should_parse_enclosures(feed_type: _FeedType, xml_content: bytes) -> bool:
    """Check if feed likely contains RSS enclosure elements."""
    return feed_type == "rss" and b"<enclosure" in xml_content


def _parse_content(xml_content: str | bytes) -> FastFeedParserDict:
    """Parse feed content (XML or JSON) that has already been fetched."""
    json_feed = _maybe_parse_json_feed(xml_content)
    if json_feed is not None:
        return json_feed

    xml_content = _prepare_xml_bytes(xml_content)
    root = _parse_xml_root(xml_content)
    root_tag_local = _root_tag_local(root)
    _raise_for_non_feed_root(root, root_tag_local, xml_content)

    feed_type, channel, items, atom_namespace = _detect_feed_structure(
        root, xml_content, root_tag_local
    )
    parse_media_content = _should_parse_media_content(root, xml_content)
    parse_enclosures = _should_parse_enclosures(feed_type, xml_content)

    feed = _parse_feed_info(channel, feed_type, atom_namespace)

    # Parse entries
    entries: list[FastFeedParserDict] = []
    feed["entries"] = entries
    for item in items:
        entry = _parse_feed_entry(
            item,
            feed_type,
            atom_namespace,
            parse_media_content=parse_media_content,
            parse_enclosures=parse_enclosures,
        )
        # Ensure that titles and descriptions are always present
        entry["title"] = entry.get("title", "").strip()
        entry["description"] = entry.get("description", "").strip()
        entries.append(entry)

    return feed


def parse(source: str | bytes) -> FastFeedParserDict:
    """Parse a feed from a URL or XML content.

    Args:
        source: URL string or XML content string/bytes

    Returns:
        FastFeedParserDict containing parsed feed data

    Raises:
        ValueError: If content is empty or invalid
        HTTPError: If URL fetch fails
    """
    is_url = isinstance(source, str) and source.startswith(("http://", "https://"))
    if is_url:
        content = _fetch_url_content(source)
    else:
        content = source

    try:
        return _parse_content(content)
    except ValueError as e:
        if not is_url:
            raise
        err_msg = str(e)
        if "HTML" not in err_msg and "not a valid RSS/Atom feed" not in err_msg:
            raise
        redirect_url = _extract_meta_refresh_url(content, source)
        if redirect_url is None:
            raise
        return parse(redirect_url)


def _parse_feed_info(
    channel: _Element, feed_type: _FeedType, atom_namespace: Optional[str] = None
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

    feed_lang = channel.get("{http://www.w3.org/XML/1998/namespace}lang")
    feed_base = channel.get("{http://www.w3.org/XML/1998/namespace}base")
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
        for subject in element.findall("{http://purl.org/dc/elements/1.1/}subject"):
            term = subject.text.strip() if subject.text else None
            if term:
                tags_list.append({"term": term, "scheme": None, "label": None})
    elif feed_type == "atom":
        # Atom uses <category> elements with attributes
        atom_ns = atom_namespace or "http://www.w3.org/2005/Atom"
        for cat in element.findall(f"{{{atom_ns}}}category"):
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
        for subject in element.findall("{http://purl.org/dc/elements/1.1/}subject"):
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


def _populate_entry_links(entry: FastFeedParserDict, item: _Element, atom_ns: str) -> None:
    entry_links: list[dict[str, Optional[str]]] = []
    alternate_link: Optional[dict[str, Optional[str]]] = None
    for link in item.findall(f"{{{atom_ns}}}link"):
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

    guid = item.find("guid")
    guid_text = guid.text.strip() if guid is not None and guid.text else None
    is_guid_url = guid_text is not None and guid_text.startswith(("http://", "https://"))

    if is_guid_url and "link" not in entry:
        entry["link"] = guid_text
        if alternate_link:
            entry_links.insert(
                0, {"rel": "alternate", "type": "text/html", "href": guid_text}
            )
    elif alternate_link:
        entry["link"] = alternate_link["href"]
        entry_links.insert(0, alternate_link)
    elif (
        ("link" not in entry)
        and (guid is not None)
        and guid.get("isPermaLink") == "true"
    ):
        entry["link"] = guid_text

    entry["links"] = entry_links


def _populate_entry_content(
    entry: FastFeedParserDict, item: _Element, feed_type: _FeedType, atom_ns: str
) -> None:
    content_el = None
    if feed_type == "rss":
        content_el = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
        if content_el is None:
            content_el = item.find("content")
    elif feed_type == "atom":
        content_el = item.find(f"{{{atom_ns}}}content")

    if content_el is not None:
        content_type = content_el.get("type", "text/html")
        if content_type in {"xhtml", "application/xhtml+xml"}:
            content_value = etree.tostring(content_el, encoding="unicode", method="xml")
        else:
            content_value = content_el.text or ""
        entry["content"] = [
            {
                "type": content_type,
                "language": content_el.get("{http://www.w3.org/XML/1998/namespace}lang"),
                "base": content_el.get("{http://www.w3.org/XML/1998/namespace}base"),
                "value": content_value,
            }
        ]

    if "content" not in entry:
        description = item.find("description")
        if description is not None and description.text:
            entry["content"] = [
                {
                    "type": "text/html",
                    "language": item.get("{http://www.w3.org/XML/1998/namespace}lang"),
                    "base": item.get("{http://www.w3.org/XML/1998/namespace}base"),
                    "value": description.text,
                }
            ]

    if "description" not in entry and "content" in entry:
        content_value = entry["content"][0]["value"]
        if content_value:
            if "<" in content_value:
                content_value = _RE_HTML_TAGS.sub(" ", content_value[:2048])
                content_value = _html_mod.unescape(content_value)
            content_value = _RE_WHITESPACE.sub(" ", content_value).strip()
        entry["description"] = content_value[:512]


def _parse_media_content(item: _Element) -> list[dict[str, Any]] | None:
    media_contents: list[dict[str, Any]] = []

    for media in item.findall(".//{http://search.yahoo.com/mrss/}content"):
        media_item: dict[str, str | int | None] = {
            "url": media.get("url"),
            "type": media.get("type"),
            "medium": media.get("medium"),
            "width": media.get("width"),
            "height": media.get("height"),
        }
        _coerce_int_fields(media_item, ("width", "height"))

        title = media.find("{http://search.yahoo.com/mrss/}title")
        if title is not None and title.text:
            media_item["title"] = title.text.strip()

        text = media.find("{http://search.yahoo.com/mrss/}text")
        if text is not None and text.text:
            media_item["text"] = text.text.strip()

        desc = media.find("{http://search.yahoo.com/mrss/}description")
        if desc is None:
            parent = media.getparent()
            if parent is not None:
                desc = parent.find("{http://search.yahoo.com/mrss/}description")
        if desc is not None and desc.text:
            media_item["description"] = desc.text.strip()

        credit = media.find("{http://search.yahoo.com/mrss/}credit")
        if credit is None:
            parent = media.getparent()
            if parent is not None:
                credit = parent.find("{http://search.yahoo.com/mrss/}credit")
        if credit is not None and credit.text:
            media_item["credit"] = credit.text.strip()
            media_item["credit_scheme"] = credit.get("scheme")

        thumbnail = media.find("{http://search.yahoo.com/mrss/}thumbnail")
        if thumbnail is not None:
            media_item["thumbnail_url"] = thumbnail.get("url")

        cleaned = _drop_none_values(media_item)
        if cleaned:
            media_contents.append(cleaned)

    if not media_contents:
        for thumbnail in item.findall(".//{http://search.yahoo.com/mrss/}thumbnail"):
            parent = thumbnail.getparent()
            if parent is None or parent.tag == "{http://search.yahoo.com/mrss/}content":
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

        cleaned = _drop_none_values(enc_item)
        if cleaned.get("url"):
            enclosures.append(cleaned)

    return enclosures or None


def _build_rss_item_text_maps(item: _Element) -> tuple[dict[str, Optional[str]], dict[str, Optional[str]]]:
    by_local: dict[str, Optional[str]] = {}
    by_full: dict[str, Optional[str]] = {}
    for child in item:
        tag = child.tag
        if not isinstance(tag, str):
            continue
        text_value = child.text.strip() if child.text else None
        if tag not in by_full:
            by_full[tag] = text_value
        local = tag.rsplit("}", 1)[-1].lower()
        if ":" in local:
            local = local.split(":", 1)[1]
        if local not in by_local:
            by_local[local] = text_value
    return by_local, by_full


def _first_non_empty(mapping: dict[str, Optional[str]], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


def _parse_rss_feed_entry_fast(
    item: _Element,
    atom_ns: str,
    parse_media_content: bool = True,
    parse_enclosures: bool = True,
) -> FastFeedParserDict:
    text_by_local, text_by_full = _build_rss_item_text_maps(item)

    entry = FastFeedParserDict()
    atom_id = text_by_full.get(f"{{{atom_ns}}}id")
    rss_guid = text_by_local.get("guid")
    rdf_about = item.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
    entry_id: Optional[str] = atom_id or rss_guid or rdf_about
    if entry_id:
        entry["id"] = entry_id.strip()

    title = text_by_local.get("title")
    if title:
        entry["title"] = title

    description = _first_non_empty(text_by_local, ("description", "summary"))
    if description:
        entry["description"] = description

    link = text_by_local.get("link")
    if link:
        entry["link"] = link

    published_source = _first_non_empty(text_by_local, ("pubdate", "published", "issued", "date"))
    if published_source:
        published = _parse_date(published_source)
        if published:
            entry["published"] = published

    updated_source = _first_non_empty(text_by_local, ("lastbuilddate", "updated", "modified"))
    if updated_source:
        updated = _parse_date(updated_source)
        if updated:
            entry["updated"] = updated

    if "published" not in entry and rss_guid:
        guid_date = _parse_date(rss_guid)
        if guid_date:
            entry["published"] = guid_date

    if "updated" in entry and "published" not in entry:
        entry["published"] = entry["updated"]

    _populate_entry_links(entry, item, atom_ns)
    if "id" not in entry and "link" in entry:
        entry["id"] = entry["link"]

    _populate_entry_content(entry, item, "rss", atom_ns)

    if parse_media_content:
        media_contents = _parse_media_content(item)
        if media_contents:
            entry["media_content"] = media_contents

    if parse_enclosures:
        enclosures = _parse_enclosures(item)
        if enclosures:
            entry["enclosures"] = enclosures

    author = _first_non_empty(text_by_local, ("author", "creator"))
    if not author:
        atom_author = item.find(f"{{{atom_ns}}}author/{{{atom_ns}}}name")
        author = atom_author.text.strip() if atom_author is not None and atom_author.text else None
    if author:
        entry["author"] = author

    comments = text_by_local.get("comments")
    if comments:
        entry["comments"] = comments

    tags = _parse_tags(item, "rss", atom_ns)
    if tags:
        entry["tags"] = tags

    return entry


def _parse_feed_entry(
    item: _Element,
    feed_type: _FeedType,
    atom_namespace: Optional[str] = None,
    *,
    parse_media_content: bool = True,
    parse_enclosures: bool = True,
) -> FastFeedParserDict:
    # Use dynamic atom namespace or fallback to default
    atom_ns = atom_namespace or "http://www.w3.org/2005/Atom"

    if feed_type == "rss":
        return _parse_rss_feed_entry_fast(
            item,
            atom_ns,
            parse_media_content=parse_media_content,
            parse_enclosures=parse_enclosures,
        )

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
    if "published" not in entry and rss_guid:
        # Check if GUID contains date information
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

    _populate_entry_content(entry, item, feed_type, atom_ns)

    if parse_media_content:
        media_contents = _parse_media_content(item)
        if media_contents:
            entry["media_content"] = media_contents

    if parse_enclosures:
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
        author = element_get("{http://purl.org/dc/elements/1.1/}creator") or element_get("author")
    if author:
        entry["author"] = author

    # Parse entry-level tags/categories
    tags = _parse_tags(item, feed_type, atom_ns)
    if tags:
        entry["tags"] = tags

    return entry


def _field_value_getter(
    root: _Element,
    feed_type: _FeedType,
    cached_get: Optional[Callable[[str, Optional[str]], Optional[str]]] = None,
) -> Callable[[str, str, str, bool], str | None]:
    get_value = cached_get or _cached_element_value_factory(root)

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
            prefixed_paths = [f"rss:{path_lower}", f"atom:{path_lower}", f"dc:{path_lower}"]
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
) -> Callable[[str, Optional[str]], Optional[str]]:
    """Create a closure with a child tag index for fast namespace-prefix lookups."""
    # Build child tag index once: O(children) instead of O(children × misses)
    child_index: dict[str, _Element] = {}
    for child in root:
        if isinstance(child.tag, str):
            child_index[child.tag.lower()] = child

    def getter(path: str, attribute: Optional[str] = None) -> Optional[str]:
        return _get_element_value(root, path, attribute=attribute, child_index=child_index)

    return getter


def _normalize_iso_datetime_string(value: str) -> str:
    """Coerce flexible ISO-8601 inputs into a form datetime.fromisoformat can parse."""
    cleaned = value.strip()
    if not cleaned:
        return cleaned

    upper_cleaned = cleaned.upper()
    for suffix in (" UTC", " GMT", " Z"):
        if upper_cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip() + "+00:00"
            upper_cleaned = cleaned.upper()
            break

    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1] + "+00:00"

    if " " in cleaned and "T" not in cleaned[:11] and len(cleaned) >= 10 and cleaned[4] == "-" and cleaned[0:4].isdigit():
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


def _parsedate_to_utc(value: str) -> Optional[datetime.datetime]:
    """Fast RFC-822 / RFC-2822 parsing via email.utils."""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed is None:
        return None
    return _ensure_utc(parsed)


custom_tzinfos: dict[str, int] = {
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
        return dateutil_parser.parse(value, tzinfos=custom_tzinfos, ignoretz=False)
    except (ValueError, TypeError, OverflowError):
        return None


@lru_cache(maxsize=256)
def _slow_dateparser(value: str) -> Optional[datetime.datetime]:
    try:
        import dateparser as _dateparser  # optional dependency
    except ImportError:
        return None
    try:
        return _dateparser.parse(value, languages=["en"], settings=_DATEPARSER_SETTINGS)
    except (ValueError, TypeError):
        return None


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

    if "24:00" in candidate:
        candidate = candidate.replace("24:00:00", "00:00:00").replace(
            " 24:00", " 00:00"
        )

    dt: Optional[datetime.datetime] = None

    is_iso_like = len(candidate) >= 10 and candidate[4] == "-" and candidate[0:4].isdigit()
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
