from __future__ import annotations

import datetime
from email.utils import parsedate_to_datetime
import gzip
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
from urllib.request import (
    HTTPErrorProcessor,
    HTTPRedirectHandler,
    Request,
    build_opener,
)

import dateparser
from dateutil import parser as dateutil_parser
from lxml import etree

if TYPE_CHECKING:
    from lxml.etree import _Element

_FeedType = Literal["rss", "atom", "rdf"]

_UTC = datetime.timezone.utc

# Pre-compiled regex patterns for performance
_RE_XML_ENCODING = re.compile(r'<\?xml[^>]*encoding=["\']([^"\']+)["\']')
_RE_DOUBLE_XML_DECL = re.compile(r"<\?xml\?xml\s+", re.IGNORECASE)
_RE_DOUBLE_CLOSE = re.compile(r"\?\?>\s*")
_RE_UNQUOTED_ATTR = re.compile(r'(\s+[\w:]+)=([^\s>"\']+)')
_RE_UTF16_ENCODING = re.compile(
    r'(<\?xml[^>]*encoding=["\'])utf-16(-le|-be)?(["\'][^>]*\?>)', re.IGNORECASE
)
_RE_UNCLOSED_LINK = re.compile(
    r"<link([^>]*[^/])>\s*(?=\n\s*<(?!/link\s*>))", re.MULTILINE
)
_RE_FEB29 = re.compile(r"(\d{4})-02-29")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_ISO_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_RE_ISO_TZ_NO_COLON = re.compile(r"([+-]\d{2})(\d{2})$")
_RE_ISO_TZ_HOUR_ONLY = re.compile(r"([+-]\d{2})$")
_RE_ISO_FRACTION = re.compile(r"\.(\d{7,})(?=(?:[+-]\d{2}:?\d{2}|Z|$))", re.IGNORECASE)
_RE_COMMA_WEEKDAY = re.compile(r"^[A-Za-z]{3}, ")


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
        return "utf-16-le"
    elif content.startswith(b"\xfe\xff"):
        return "utf-16-be"
    elif content.startswith(b"\xef\xbb\xbf"):
        return "utf-8"

    # Try to read the first 1000 bytes to find XML declaration
    # Try UTF-8 first
    try:
        preview = content[:1000].decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        # If UTF-8 fails, try UTF-16
        try:
            preview = content[:1000].decode("utf-16", errors="strict")
        except UnicodeDecodeError:
            # Fall back to latin-1 which never fails
            preview = content[:1000].decode("latin-1", errors="replace")

    # Look for encoding in XML declaration
    encoding_match = _RE_XML_ENCODING.search(preview)
    if encoding_match:
        return encoding_match.group(1).lower()

    return "utf-8"


def _clean_feed_content(content: str | bytes) -> tuple[str, str]:
    """Clean feed content by finding and extracting the actual XML.

    Handles cases where PHP warnings, HTML, or other content appears before the XML.

    Returns:
        Tuple of (cleaned_content, encoding_used)
    """
    if isinstance(content, bytes):
        # Detect encoding from XML declaration
        declared_encoding = _detect_xml_encoding(content)

        # Try to decode with the declared encoding
        try:
            content = content.decode(declared_encoding, errors="strict")
            encoding_used = declared_encoding
        except (UnicodeDecodeError, LookupError):
            # If declared encoding fails, try UTF-8
            try:
                content = content.decode("utf-8", errors="strict")
                encoding_used = "utf-8"
            except UnicodeDecodeError:
                # Last resort: use declared encoding with error replacement
                content = content.decode(declared_encoding, errors="replace")
                encoding_used = declared_encoding
    else:
        # Already a string, assume UTF-8
        encoding_used = "utf-8"

    # Look for XML declaration or root elements
    xml_start_patterns = [
        "<?xml",  # XML declaration
        "<rss",  # RSS feed
        "<feed",  # Atom feed
        "<rdf:RDF",  # RDF feed
        "<?xml-stylesheet",  # Sometimes comes before <?xml
    ]

    stripped_content = content.lstrip()
    stripped_lower = stripped_content[:2000].lower()
    if stripped_lower.startswith(("<?xml", "<rss", "<feed", "<rdf")):
        return stripped_content, encoding_used

    if stripped_lower.startswith("<!doctype html") or stripped_lower.startswith(
        "<html"
    ):
        raise ValueError("Content appears to be HTML, not a valid RSS/Atom feed")

    content_lines = content.splitlines()
    xml_start_line = -1

    # Find the first line that looks like XML
    for i, line in enumerate(content_lines):
        line_stripped = line.strip()
        if any(line_stripped.startswith(pattern) for pattern in xml_start_patterns):
            xml_start_line = i
            break

    if xml_start_line >= 0:
        # Return content starting from the XML line
        return "\n".join(content_lines[xml_start_line:]), encoding_used

    # Check if content looks like HTML (only check first 2000 chars for performance)
    content_preview = content[:2000].lower()
    if (
        content_preview.strip().startswith("<!doctype html")
        or content_preview.strip().startswith("<html")
        or "<script>" in content_preview
        or "<body>" in content_preview
    ):
        raise ValueError("Content appears to be HTML, not a valid RSS/Atom feed")

    # If no XML patterns found, return original content (let XML parser handle the error)
    return content, encoding_used


def _fix_malformed_xml(content: str, actual_encoding: str = "utf-8") -> str:
    """Fix common malformed XML issues in feeds.

    Some feeds have malformed XML like unclosed link tags or other issues
    that can be automatically corrected.

    Args:
        content: The XML content as a string
        actual_encoding: The actual encoding used (default: utf-8)
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

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
    # Handle URL input
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        request = Request(
            source,
            method="GET",
            headers={
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": "fastfeedparser (+https://github.com/kagisearch/fastfeedparser)",
            },
        )
        opener = build_opener(HTTPRedirectHandler(), HTTPErrorProcessor())
        with opener.open(request, timeout=30) as response:
            response.begin()
            content: bytes = response.read()
            content_encoding = response.headers.get("Content-Encoding")
            if content_encoding == "gzip":
                content = gzip.decompress(content)
            elif content_encoding == "deflate":
                content = zlib.decompress(content, -zlib.MAX_WBITS)
            elif content_encoding == "br" and HAS_BROTLI:
                content = brotli.decompress(content)
            content_charset = response.headers.get_content_charset()
            xml_content = (
                content.decode(content_charset) if content_charset else content
            )
    else:
        xml_content = source

    # Fast-path: Skip JSON detection for content that clearly starts with XML
    # This avoids unnecessary JSON decode attempts for 99% of feeds
    if isinstance(xml_content, bytes):
        content_start = xml_content[:10].lstrip()
    else:
        content_start = xml_content[:10].lstrip().encode("utf-8", errors="replace")

    # Only try JSON parsing if content starts with '{' (potential JSON)
    if content_start.startswith(b"{"):
        try:
            if isinstance(xml_content, bytes):
                json_str = xml_content.decode("utf-8", errors="replace")
            else:
                json_str = xml_content

            # Quick check if it looks like JSON
            json_str_stripped = json_str.strip()
            if json_str_stripped.startswith("{"):
                try:
                    json_data = json.loads(json_str_stripped)
                    # Check if it's a JSON Feed (has version field pointing to jsonfeed.org)
                    if isinstance(json_data, dict) and "version" in json_data:
                        version = json_data["version"]
                        if isinstance(version, str) and "jsonfeed.org" in version:
                            return _parse_json_feed(json_data)
                        # Also check for 'items' which is required in JSON Feed
                        elif "items" in json_data and isinstance(
                            json_data["items"], list
                        ):
                            # Might be a JSON Feed without explicit version
                            return _parse_json_feed(json_data)
                except (json.JSONDecodeError, ValueError):
                    # Not JSON or invalid JSON, continue to XML parsing
                    pass
        except Exception:
            # If JSON detection fails, continue to XML parsing
            pass

    # Clean content to handle PHP warnings/HTML before XML
    xml_content, detected_encoding = _clean_feed_content(xml_content)

    # Fast-path: Skip malformed XML fixes for well-formed feeds
    # Check if content has any of the patterns we fix
    needs_fixing = (
        "?xml?xml" in xml_content[:200]  # Double XML declaration
        or "??>" in xml_content[:200]  # Double closing
        or (
            "rss:" in xml_content[:500] and "xmlns:rss" not in xml_content[:1000]
        )  # Undeclared prefix
        or (
            "utf-16" in xml_content[:200].lower() and detected_encoding != "utf-16"
        )  # Encoding mismatch
    )

    # Only fix if needed (skip for ~90% of well-formed feeds)
    if needs_fixing:
        xml_content = _fix_malformed_xml(xml_content, actual_encoding=detected_encoding)

    # Ensure we have bytes for lxml
    if isinstance(xml_content, str):
        xml_content = xml_content.encode("utf-8", errors="replace")

    # Handle empty content
    if not xml_content.strip():
        raise ValueError("Empty content")

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
        # Try to provide helpful context about what we received
        try:
            preview = (
                xml_content[:500].decode("utf-8", errors="replace")
                if isinstance(xml_content, bytes)
                else str(xml_content)[:500]
            )
            preview = preview.strip()
            if preview:
                raise ValueError(
                    f"Failed to parse XML: received content that couldn't be parsed as XML (first 200 chars: {preview[:200]})"
                )
            else:
                raise ValueError("Failed to parse XML: received empty content")
        except Exception:
            raise ValueError(
                "Failed to parse XML: root element is None (invalid or empty content)"
            )

    # Check if this is an error/status XML document, not a feed
    root_tag_local = (
        root.tag.split("}")[-1].lower() if "}" in root.tag else root.tag.lower()
    )
    non_feed_tags = {"status", "error", "html", "opml", "br", "div", "body"}
    if root_tag_local in non_feed_tags:
        # Try to extract error message from various sources
        error_msg = root.text or ""

        # Try common error message paths
        if not error_msg:
            # Try <message>, <title>, <h1>, etc. - use XPath to handle namespaces
            for tag in ["message", "title", "h1", "h2", "p", "code"]:
                try:
                    # Try with and without namespace
                    elem = root.find(f".//{tag}") or root.find(tag)
                    if elem is not None and elem.text:
                        error_msg = elem.text
                        break
                    # Try XPath for case-insensitive
                    elems = root.xpath(f".//*[local-name()='{tag}']")
                    if elems and elems[0].text:
                        error_msg = elems[0].text
                        break
                except Exception:
                    pass

        # Get all text content as fallback
        if not error_msg or len(error_msg.strip()) < 5:
            try:
                all_text = " ".join(
                    text.strip() for text in root.itertext() if text and text.strip()
                )
                # Clean up whitespace
                all_text = " ".join(all_text.split())
                error_msg = all_text[:300] if all_text else "No error message"
            except Exception:
                error_msg = "No error message"

        error_msg = error_msg.strip()[:300] if error_msg else "No error message"

        # Provide helpful error messages
        if root_tag_local == "html":
            if error_msg and error_msg != "No error message" and len(error_msg) > 10:
                raise ValueError(
                    f"Received HTML page instead of feed: {error_msg[:150]}"
                )
            else:
                raise ValueError(
                    "Received HTML page instead of feed (possible redirect, 404, or server error)"
                )
        elif root_tag_local in ["div", "body"]:
            if error_msg and error_msg != "No error message" and len(error_msg) > 10:
                raise ValueError(
                    f"Received HTML fragment instead of feed: {error_msg[:150]}"
                )
            else:
                raise ValueError("Received HTML fragment instead of feed")
        elif root_tag_local == "status":
            raise ValueError(f"Feed server returned status message: {error_msg}")
        elif root_tag_local == "error":
            if error_msg and error_msg != "No error message":
                raise ValueError(f"Feed server returned error: {error_msg}")
            else:
                raise ValueError("Feed server returned error (no details provided)")
        elif root_tag_local == "opml":
            raise ValueError(
                "Received OPML document instead of feed (OPML is an outline format, not a feed)"
            )
        else:
            raise ValueError(
                f"Not a valid feed: {root_tag_local} element found - {error_msg[:100]}"
            )

    # Determine a feed type based on the content structure
    feed_type: _FeedType
    atom_namespace: Optional[str] = None

    if (
        root.tag == "rss"
        or root.tag.endswith("}rss")
        or (root.tag.lower().split("}")[-1] == "rss")
    ):
        feed_type = "rss"
        # Handle both namespaced and non-namespaced RSS
        channel = root.find("channel")
        if channel is None:
            # Try to find channel with any namespace or prefix
            for child in root:
                # Skip non-Element children (comments, processing instructions)
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
            # Fallback: Check if this is a malformed RSS with Atom-style elements
            # This handles feeds like seancdavis.com that declare RSS but use Atom structure
            has_atom_elements = any(
                isinstance(child.tag, str)
                and child.tag
                in ["entry", "title", "subtitle", "updated", "id", "author", "link"]
                for child in root
            )
            if has_atom_elements:
                # Treat the RSS root as the channel for malformed feeds
                channel = root
            else:
                raise ValueError("Invalid RSS feed: missing channel element")
        elif (
            len(list(channel)) == 0
            and len(
                [
                    child
                    for child in root
                    if isinstance(child.tag, str) and child.tag == "item"
                ]
            )
            > 0
        ):
            # Handle malformed RSS where channel is empty but items are at root level
            # This handles feeds like canolcer.eu that have <channel></channel> but items outside
            channel = root
        # Find items with or without namespace
        items = channel.findall("item")
        if not items:
            # Try to find items with any namespace or prefix
            for child in channel:
                # Skip non-Element children (comments, processing instructions)
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
            # If still no items found using findall with any namespace
            if not items:
                items = [
                    child
                    for child in channel
                    if isinstance(child.tag, str)
                    and (
                        child.tag.endswith("}item")
                        or child.tag == "item"
                        or child.tag.lower() == "rss:item"
                        or child.tag.lower().endswith(":item")
                    )
                ]
            # Try recursive search for deeply nested items (minified feeds)
            if not items:
                items = channel.xpath(".//item") or channel.xpath(
                    ".//*[local-name()='item']"
                )

            # Fallback for malformed RSS: look for Atom-style <entry> elements
            if not items:
                items = channel.findall("entry")
                if not items:
                    # Try to find entries with any namespace
                    for child in channel:
                        # Skip non-Element children
                        if not isinstance(child.tag, str):
                            continue
                        if child.tag.endswith("}entry") or child.tag == "entry":
                            if not items:
                                items = []
                            items.append(child)
                    # If still no entries found using findall with any namespace
                    if not items:
                        items = [
                            child
                            for child in channel
                            if isinstance(child.tag, str)
                            and (child.tag.endswith("}entry") or child.tag == "entry")
                        ]

        # Last resort fallback: If we found very few items but feed is large, try HTMLParser
        # This handles feeds with malformed CDATA or other XML errors that XMLParser can't recover from
        # Only try if feed is >20KB (suggests it should have more content)
        if len(items) < 5 and len(xml_content) > 20000:
            try:
                # Try HTMLParser which is more forgiving with malformed content
                html_parser = etree.HTMLParser(recover=True, collect_ids=False)
                html_root = etree.fromstring(xml_content, parser=html_parser)
                html_channel = html_root.find(".//channel")
                if html_channel is not None:
                    html_items = html_channel.findall(".//item")
                    # Only use HTMLParser results if we get significantly more items
                    if len(html_items) > len(items) * 2:  # At least 2x more items
                        root = html_root
                        channel = html_channel
                        items = html_items
            except Exception:
                # If HTMLParser fails, continue with XMLParser results
                pass
    elif root.tag.endswith("}feed"):
        # Detect Atom namespace dynamically
        if "{http://www.w3.org/2005/Atom}" in root.tag:
            atom_namespace = "http://www.w3.org/2005/Atom"
        elif "{https://www.w3.org/2005/Atom}" in root.tag:
            atom_namespace = "https://www.w3.org/2005/Atom"
        elif "{http://purl.org/atom/ns#}" in root.tag:
            atom_namespace = "http://purl.org/atom/ns#"
        else:
            raise ValueError(f"Unknown Atom namespace in feed type: {root.tag}")

        feed_type = "atom"
        channel = root
        items = channel.findall(f".//{{{atom_namespace}}}entry")
    elif root.tag == "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF":
        feed_type = "rdf"
        channel = root
        items = channel.findall(".//{http://purl.org/rss/1.0/}item")
        if not items:
            items = channel.findall("item")
    else:
        raise ValueError(f"Unknown feed type: {root.tag}")

    feed = _parse_feed_info(channel, feed_type, atom_namespace)

    # Parse entries
    entries: list[FastFeedParserDict] = []
    feed["entries"] = entries
    for item in items:
        entry = _parse_feed_entry(item, feed_type, atom_namespace)
        # Ensure that titles and descriptions are always present
        entry["title"] = entry.get("title", "").strip()
        entry["description"] = entry.get("description", "").strip()
        entries.append(entry)

    return feed


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


def _parse_feed_entry(
    item: _Element, feed_type: _FeedType, atom_namespace: Optional[str] = None
) -> FastFeedParserDict:
    # Use dynamic atom namespace or fallback to default
    atom_ns = atom_namespace or "http://www.w3.org/2005/Atom"

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

    # Handle links
    entry_links: list[dict[str, Optional[str]]] = []
    entry["links"] = entry_links
    alternate_link: Optional[dict[str, Optional[str]]] = None
    for link in item.findall(f"{{{atom_ns}}}link"):
        rel = link.get("rel")
        href = link.get("href") or link.get("link")
        if not href:
            continue
        if rel == "alternate":
            alternate_link = {
                "rel": rel,
                "type": link.get("type"),
                "href": href,
                "title": link.get("title"),
            }
        elif rel not in {"edit", "self"}:
            entry_links.append(
                {
                    "rel": rel,
                    "type": link.get("type"),
                    "href": href,
                    "title": link.get("title"),
                }
            )

    # Check for guid that looks like a URL
    guid = item.find("guid")
    guid_text = guid.text.strip() if guid is not None and guid.text else None
    is_guid_url = guid_text and guid_text.startswith(("http://", "https://"))

    if is_guid_url and "link" not in entry:  # Only use guid if link doesn't exist
        # Prefer guid as link when it looks like a URL
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

    # ------------------------------------------------------------------
    # 2) Guarantee that every entry has an id.  If none of the dedicated
    #    id sources were present, fall back to the chosen link.
    # ------------------------------------------------------------------
    if "id" not in entry and "link" in entry:
        entry["id"] = entry["link"]

    content = None
    if feed_type == "rss":
        content = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
        if content is None:
            content = item.find("content")
    elif feed_type == "atom":
        content = item.find(f"{{{atom_ns}}}content")

    if content is not None:
        content_type = content.get("type", "text/html")  # Default to text/html
        if content_type in {"xhtml", "application/xhtml+xml"}:
            # For XHTML content, serialize the entire content
            content_value = etree.tostring(content, encoding="unicode", method="xml")
        else:
            content_value = content.text or ""
        entry["content"] = [
            {
                "type": content_type,
                "language": content.get("{http://www.w3.org/XML/1998/namespace}lang"),
                "base": content.get("{http://www.w3.org/XML/1998/namespace}base"),
                "value": content_value,
            },
        ]

    # If content is still empty, try to use description
    if "content" not in entry:
        description = item.find("description")
        if description is not None and description.text:
            entry["content"] = [
                {
                    "type": "text/html",
                    "language": item.get("{http://www.w3.org/XML/1998/namespace}lang"),
                    "base": item.get("{http://www.w3.org/XML/1998/namespace}base"),
                    "value": description.text,
                },
            ]

    # If description is empty, derive it from content (removing non-text content)
    if "description" not in entry and "content" in entry:
        content = entry["content"][0]["value"]
        if content:
            try:
                html_content = etree.HTML(content)
                if html_content is not None:
                    content_text = html_content.xpath("string()")
                    if isinstance(content_text, str):
                        content = _RE_WHITESPACE.sub(" ", content_text)
            except etree.ParserError:
                pass
        entry["description"] = content[:512]

    # Handle media content
    media_contents: list[dict[str, int | str | None]] = []

    # Process media:content elements
    for media in item.findall(".//{http://search.yahoo.com/mrss/}content"):
        media_item: dict[str, str | int | None] = {
            "url": media.get("url"),
            "type": media.get("type"),
            "medium": media.get("medium"),
            "width": media.get("width"),
            "height": media.get("height"),
        }

        # Convert width/height to integers if present
        for dim in ("width", "height"):
            value = media_item[dim]
            if value:
                try:
                    media_item[dim] = int(value)
                except (ValueError, TypeError):
                    del media_item[dim]

        # Handle sibling elements
        # Handle title
        title = media.find("{http://search.yahoo.com/mrss/}title")
        if title is not None and title.text:
            media_item["title"] = title.text.strip()

        # Handle credit
        credit = media.find("{http://search.yahoo.com/mrss/}credit")
        if credit is not None and credit.text:
            media_item["credit"] = credit.text.strip()
            media_item["credit_scheme"] = credit.get("scheme")

        # Handle text
        text = media.find("{http://search.yahoo.com/mrss/}text")
        if text is not None and text.text:
            media_item["text"] = text.text.strip()

        # Handle description - check both direct child and sibling elements
        desc = media.find("{http://search.yahoo.com/mrss/}description")
        if desc is None:
            parent = media.getparent()
            if parent is not None:
                desc = parent.find("{http://search.yahoo.com/mrss/}description")
        if desc is not None and desc.text:
            media_item["description"] = desc.text.strip()

        # Handle credit - check both direct child and sibling elements
        credit = media.find("{http://search.yahoo.com/mrss/}credit")
        if credit is None:
            parent = media.getparent()
            if parent is not None:
                credit = parent.find("{http://search.yahoo.com/mrss/}credit")
        if credit is not None and credit.text:
            media_item["credit"] = credit.text.strip()

        # Handle thumbnail as a separate URL field
        thumbnail = media.find("{http://search.yahoo.com/mrss/}thumbnail")
        if thumbnail is not None:
            media_item["thumbnail_url"] = thumbnail.get("url")

        # Remove None values
        media_item = {k: v for k, v in media_item.items() if v is not None}
        if media_item:  # Only append if we have some content
            media_contents.append(media_item)

    # If no media:content but there are standalone thumbnails, add them
    if not media_contents:
        for thumbnail in item.findall(".//{http://search.yahoo.com/mrss/}thumbnail"):
            parent = thumbnail.getparent()
            if parent is None or parent.tag == "{http://search.yahoo.com/mrss/}content":
                continue
            thumb_item = {
                "url": thumbnail.get("url"),
                "type": "image/jpeg",  # Default type for thumbnails
                "width": thumbnail.get("width"),
                "height": thumbnail.get("height"),
            }
            # Convert dimensions to integers if present
            for dim in ("width", "height"):
                value = thumb_item[dim]
                if value:
                    try:
                        thumb_item[dim] = int(value)
                    except (ValueError, TypeError):
                        del thumb_item[dim]

            # Remove None values
            thumb_item = {k: v for k, v in thumb_item.items() if v is not None}
            if thumb_item:
                media_contents.append(thumb_item)

    if media_contents:
        entry["media_content"] = media_contents

    # Handle enclosures
    enclosures: list[dict[str, int | str | None]] = []
    for enclosure in item.findall("enclosure"):
        enc_item: dict[str, str | int | None] = {
            "url": enclosure.get("url"),
            "type": enclosure.get("type"),
            "length": enclosure.get("length"),
        }
        # Convert length to integer if present and valid
        length = enc_item["length"]
        if length:
            try:
                enc_item["length"] = int(length)
            except (ValueError, TypeError):
                del enc_item["length"]

        # Remove None values
        enc_item = {k: v for k, v in enc_item.items() if v is not None}
        if enc_item.get("url"):  # Only append if we have a URL
            enclosures.append(enc_item)

    if enclosures:
        entry["enclosures"] = enclosures

    author = (
        get_field_value(
            "author",
            f"{{{atom_ns}}}author/{{{atom_ns}}}name",
            "{http://purl.org/dc/elements/1.1/}creator",
            False,
        )
        or get_field_value(
            "{http://purl.org/dc/elements/1.1/}creator",
            "{http://purl.org/dc/elements/1.1/}creator",
            "{http://purl.org/dc/elements/1.1/}creator",
            False,
        )
        or element_get("{http://purl.org/dc/elements/1.1/}creator")
        or element_get("author")
    )
    if author:
        entry["author"] = author

    if feed_type == "rss":
        comments = element_get("comments")
        if comments:
            entry["comments"] = comments

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
    root: _Element, path: str, attribute: Optional[str] = None
) -> Optional[str]:
    """Get text content or attribute value of an element.

    Also tries common namespace prefixes (rss:, atom:) for malformed feeds.
    """
    el = root.find(path)

    # If not found and path is a simple element name, try with common prefixes
    # We iterate manually because lxml doesn't support prefixes without a namespace map
    if el is None and "/" not in path and "{" not in path:
        # Pre-build the prefixed paths to avoid repeated string concatenation
        path_lower = path.lower()
        prefixed_paths = [f"rss:{path_lower}", f"atom:{path_lower}", f"dc:{path_lower}"]

        # Single pass through children, checking all prefixes
        for child in root:
            # Skip non-Element children (comments, processing instructions)
            if not isinstance(child.tag, str):
                continue
            child_tag_lower = child.tag.lower()
            if child_tag_lower in prefixed_paths:
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
    """Create a closure that memoises element lookups for a given root node."""
    cache: dict[tuple[str, Optional[str]], Optional[str]] = {}

    def getter(path: str, attribute: Optional[str] = None) -> Optional[str]:
        key = (path, attribute)
        if key in cache:
            return cache[key]
        value = _get_element_value(root, path, attribute=attribute)
        cache[key] = value
        return value

    return getter


def _normalize_iso_datetime_string(value: str) -> str:
    """Coerce flexible ISO-8601 inputs into a form datetime.fromisoformat can parse."""
    cleaned = value.strip()
    if not cleaned:
        return cleaned

    cleaned = _RE_WHITESPACE.sub(" ", cleaned)
    upper_cleaned = cleaned.upper()
    for suffix in (" UTC", " GMT", " Z"):
        if upper_cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip() + "+00:00"
            upper_cleaned = cleaned.upper()
            break

    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1] + "+00:00"

    if " " in cleaned and "T" not in cleaned[:11] and _RE_ISO_LIKE.match(cleaned):
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


def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    """Return a timezone-aware datetime normalized to UTC."""
    return dt.replace(tzinfo=_UTC) if dt.tzinfo is None else dt.astimezone(_UTC)


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
        return dateparser.parse(value, languages=["en"], settings=_DATEPARSER_SETTINGS)
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

    candidate = _RE_WHITESPACE.sub(" ", date_str.strip())
    if not candidate:
        return None

    # Fix invalid leap year dates (Feb 29 in non-leap years)
    # This handles feeds with incorrect dates like "2023-02-29"
    year_match = _RE_FEB29.match(candidate)
    if year_match:
        year = int(year_match.group(1))
        if not ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
            # Not a leap year, change Feb 29 to Feb 28
            candidate = candidate.replace(f"{year}-02-29", f"{year}-02-28")

    if "24:00" in candidate:
        candidate = candidate.replace("24:00:00", "00:00:00").replace(
            " 24:00", " 00:00"
        )

    dt: Optional[datetime.datetime] = None

    if _RE_ISO_LIKE.match(candidate):
        iso_candidate = _normalize_iso_datetime_string(candidate)
        try:
            dt = datetime.datetime.fromisoformat(iso_candidate)
        except ValueError:
            # Retry after trimming overly precise fractional seconds
            trimmed = _RE_ISO_FRACTION.sub(
                lambda m: "." + m.group(1)[:6], iso_candidate, count=1
            )
            if trimmed != iso_candidate:
                try:
                    dt = datetime.datetime.fromisoformat(trimmed)
                except ValueError:
                    dt = None
        if dt is not None:
            return _ensure_utc(dt).isoformat()

    dt = _parsedate_to_utc(candidate)
    if dt is not None:
        return dt.isoformat()

    slow_dt = _slow_dateutil_parse(candidate)
    if slow_dt is not None:
        return _ensure_utc(slow_dt).isoformat()

    parsed = _slow_dateparser(candidate)
    if parsed is not None:
        return _ensure_utc(parsed).isoformat()

    # If all parsing attempts fail, return None
    return None
