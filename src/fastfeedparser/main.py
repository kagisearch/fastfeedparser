from __future__ import annotations

import datetime
import gzip
import re
import zlib
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
from dateutil.tz import gettz
from lxml import etree

if TYPE_CHECKING:
    from lxml.etree import _Element

_FeedType = Literal["rss", "atom", "rdf"]

_UTC = datetime.timezone.utc


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


def _clean_feed_content(content: str | bytes) -> str:
    """Clean feed content by finding and extracting the actual XML.
    
    Handles cases where PHP warnings, HTML, or other content appears before the XML.
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    
    # Look for XML declaration or root elements
    xml_start_patterns = [
        '<?xml',  # XML declaration
        '<rss',   # RSS feed
        '<feed',  # Atom feed  
        '<rdf:RDF',  # RDF feed
        '<?xml-stylesheet'  # Sometimes comes before <?xml
    ]
    
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
        return '\n'.join(content_lines[xml_start_line:])
    
    # Check if content looks like HTML
    content_lower = content.lower()
    if (content_lower.strip().startswith('<!doctype html') or 
        content_lower.strip().startswith('<html') or
        '<script>' in content_lower or
        '<body>' in content_lower):
        raise ValueError("Content appears to be HTML, not a valid RSS/Atom feed")
    
    # If no XML patterns found, return original content (let XML parser handle the error)
    return content


def _fix_malformed_xml(content: str) -> str:
    """Fix common malformed XML issues in feeds.
    
    Some feeds have malformed XML like unclosed link tags or other issues
    that can be automatically corrected.
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    
    # Fix unclosed link tags - common in Atom feeds
    # Pattern: <link ...> followed by whitespace and another tag (not </link>)
    # should be <link .../> 
    import re
    
    # Only fix link tags that are clearly malformed:
    # - End with > instead of />  
    # - Are followed by whitespace and another tag (not a closing </link>)
    content = re.sub(
        r'<link([^>]*[^/])>\s*(?=\n\s*<(?!/link\s*>))',
        r'<link\1/>',
        content,
        flags=re.MULTILINE
    )
    
    return content


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

    # Clean content to handle PHP warnings/HTML before XML
    xml_content = _clean_feed_content(xml_content)
    
    # Fix common malformed XML issues
    xml_content = _fix_malformed_xml(xml_content)
    
    # Ensure we have bytes for lxml
    if isinstance(xml_content, str):
        xml_content = xml_content.encode("utf-8", errors="replace")

    # Handle empty content
    if not xml_content.strip():
        raise ValueError("Empty content")

    parser = etree.XMLParser(
        ns_clean=True,
        recover=True,
        collect_ids=False,
        resolve_entities=False,
    )
    try:
        root = etree.fromstring(xml_content, parser=parser)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Failed to parse XML content: {str(e)}")
    if root is None:
        raise ValueError("Failed to parse XML content: root element is None")

    # Determine a feed type based on the content structure
    feed_type: _FeedType
    atom_namespace: Optional[str] = None
    
    if root.tag == "rss" or root.tag.endswith("}rss") or (root.tag.lower().split("}")[-1] == "rss"):
        feed_type = "rss"
        # Handle both namespaced and non-namespaced RSS
        channel = root.find("channel")
        if channel is None:
            # Try to find channel with any namespace
            for child in root:
                if child.tag.endswith("}channel") or child.tag == "channel":
                    channel = child
                    break
        if channel is None:
            # Fallback: Check if this is a malformed RSS with Atom-style elements
            # This handles feeds like seancdavis.com that declare RSS but use Atom structure
            has_atom_elements = any(
                child.tag in ['entry', 'title', 'subtitle', 'updated', 'id', 'author', 'link']
                for child in root
            )
            if has_atom_elements:
                # Treat the RSS root as the channel for malformed feeds
                channel = root
            else:
                raise ValueError("Invalid RSS feed: missing channel element")
        elif len(list(channel)) == 0 and len([child for child in root if child.tag == "item"]) > 0:
            # Handle malformed RSS where channel is empty but items are at root level
            # This handles feeds like canolcer.eu that have <channel></channel> but items outside
            channel = root
        # Find items with or without namespace
        items = channel.findall("item")
        if not items:
            # Try to find items with any namespace
            for child in channel:
                if child.tag.endswith("}item") or child.tag == "item":
                    if not items:
                        items = []
                    items.append(child)
            # If still no items found using findall with any namespace
            if not items:
                items = [child for child in channel if child.tag.endswith("}item") or child.tag == "item"]
            # Try recursive search for deeply nested items (minified feeds)
            if not items:
                items = channel.xpath(".//item") or channel.xpath(".//*[local-name()='item']")
            
            # Fallback for malformed RSS: look for Atom-style <entry> elements
            if not items:
                items = channel.findall("entry")
                if not items:
                    # Try to find entries with any namespace
                    for child in channel:
                        if child.tag.endswith("}entry") or child.tag == "entry":
                            if not items:
                                items = []
                            items.append(child)
                    # If still no entries found using findall with any namespace
                    if not items:
                        items = [child for child in channel if child.tag.endswith("}entry") or child.tag == "entry"]
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


def _parse_feed_info(channel: _Element, feed_type: _FeedType, atom_namespace: Optional[str] = None) -> FastFeedParserDict:
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
    get_field_value = _field_value_getter(channel, feed_type)
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
    feed["id"] = _get_element_value(channel, f"{{{atom_ns}}}id")

    # Add generator_detail
    generator = channel.find(f"{{{atom_ns}}}generator")
    if generator is not None:
        feed["generator_detail"] = {
            "name": generator.text,
            "version": generator.get("version"),
            "href": generator.get("uri"),
        }

    if feed_type == "rss":
        comments = _get_element_value(channel, "comments")
        if comments:
            feed["comments"] = comments

    # Additional checks for publisher and author
    if "publisher" not in feed:
        webmaster = _get_element_value(channel, "webMaster")
        if webmaster:
            feed["publisher"] = webmaster
    if "author" not in feed:
        managing_editor = _get_element_value(channel, "managingEditor")
        if managing_editor:
            feed["author"] = managing_editor

    # Parse feed-level tags/categories
    tags = _parse_tags(channel, feed_type, atom_ns)
    if tags:
        feed["tags"] = tags

    return FastFeedParserDict(feed=feed)


def _parse_tags(element: _Element, feed_type: _FeedType, atom_namespace: Optional[str] = None) -> list[dict[str, str | None]] | None:
    """Parse tags/categories from an element based on feed type."""
    tags_list: list[dict[str, str | None]] = []
    if feed_type == "rss":
        # RSS uses <category> elements
        for cat in element.findall("category"):
            term = cat.text.strip() if cat.text else None
            if term:
                tags_list.append({"term": term, "scheme": cat.get("domain"), "label": None})
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
                tags_list.append({"term": term, "scheme": cat.get("scheme"), "label": cat.get("label")})
    elif feed_type == "rdf":
        # RDF uses <dc:subject> or <taxo:topic>
        for subject in element.findall("{http://purl.org/dc/elements/1.1/}subject"):
            term = subject.text.strip() if subject.text else None
            if term:
                tags_list.append({"term": term, "scheme": None, "label": None})
        # Example for taxo:topic (might need refinement based on actual usage)
        for topic in element.findall("{http://purl.org/rss/1.0/modules/taxonomy/}topic"):
             # rdf:resource often contains the tag URL which could be scheme+term
             resource = topic.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
             term = topic.text.strip() if topic.text else resource # Use text or resource as term
             if term:
                 tags_list.append({"term": term, "scheme": resource, "label": None})


    return tags_list if tags_list else None


def _parse_feed_entry(item: _Element, feed_type: _FeedType, atom_namespace: Optional[str] = None) -> FastFeedParserDict:
    # Use dynamic atom namespace or fallback to default
    atom_ns = atom_namespace or "http://www.w3.org/2005/Atom"
    
    # Check if this is Atom 0.3 to use different date field names
    is_atom_03 = atom_ns == "http://purl.org/atom/ns#"
    
    # Atom 0.3 uses 'issued' and 'modified', Atom 1.0 uses 'published' and 'updated'
    # However, some feeds mix namespaces, so we'll check both formats
    published_field = f"{{{atom_ns}}}issued" if is_atom_03 else f"{{{atom_ns}}}published"
    updated_field = f"{{{atom_ns}}}modified" if is_atom_03 else f"{{{atom_ns}}}updated"
    
    # Also define fallback fields for mixed namespace scenarios
    published_fallback = f"{{{atom_ns}}}published" if is_atom_03 else f"{{{atom_ns}}}issued"  
    updated_fallback = f"{{{atom_ns}}}updated" if is_atom_03 else f"{{{atom_ns}}}modified"
    
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

    entry = FastFeedParserDict()
    # ------------------------------------------------------------------
    # 1) Collect a stable identifier for this entry.
    #    Atom   → <id>
    #    RSS    → <guid>
    #    RDF    → rdf:about attribute on the <item>
    # ------------------------------------------------------------------
    atom_id = _get_element_value(item, f"{{{atom_ns}}}id")
    rss_guid = _get_element_value(item, "guid")
    rdf_about = item.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
    entry_id: Optional[str] = (atom_id or rss_guid or rdf_about)
    if entry_id:
        entry["id"] = entry_id.strip()
    get_field_value = _field_value_getter(item, feed_type)
    for field in fields:
        value = get_field_value(*field[1:])
        if value:
            name = field[0]
            if name in {"published", "updated"}:
                value = _parse_date(value)
            entry[name] = value

    # Check for fallback date fields if primary fields are missing
    if "published" not in entry:
        fallback_published = _get_element_value(item, published_fallback)
        if fallback_published:
            entry["published"] = _parse_date(fallback_published)
    
    if "updated" not in entry:
        fallback_updated = _get_element_value(item, updated_fallback)
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
                        content = re.sub(r"\s+", " ", content_text)
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
        or _get_element_value(item, "{http://purl.org/dc/elements/1.1/}creator")
        or _get_element_value(item, "author")
    )
    if author:
        entry["author"] = author

    if feed_type == "rss":
        comments = _get_element_value(item, "comments")
        if comments:
            entry["comments"] = comments

    # Parse entry-level tags/categories
    tags = _parse_tags(item, feed_type, atom_ns)
    if tags:
        entry["tags"] = tags

    return entry


def _field_value_getter(
    root: _Element, feed_type: _FeedType
) -> Callable[[str, str, str, bool], str | None]:
    if feed_type == "rss":

        def wrapper(
            rss_css: str, atom_css: str, rdf_css: str, is_attr: bool
        ) -> str | None:
            # First try standard RSS fields
            result = _get_element_value(root, rss_css)
            if result:
                return result
            
            # Try RSS fields with Atom namespace (malformed feeds like ajxs.me)
            atom_ns_rss = f"{{http://www.w3.org/2005/Atom}}{rss_css}"
            result = _get_element_value(root, atom_ns_rss)
            if result:
                return result
            
            # Try case-insensitive RSS fields (some feeds use pubdate instead of pubDate)
            if rss_css.lower() != rss_css:
                result = _get_element_value(root, rss_css.lower())
                if result:
                    return result
            
            # Try alternative RSS field names for dates
            if rss_css == "pubDate":
                # Some feeds use <published> instead of <pubDate>
                result = _get_element_value(root, "published")
                if result:
                    return result
                
            # Try standard Atom fields
            if is_attr:
                result = (_get_element_value(root, atom_css, attribute="href")
                         or _get_element_value(root, atom_css, attribute="link"))
            else:
                result = _get_element_value(root, atom_css) or _get_element_value(root, rdf_css)
            
            if result:
                return result
            
            # Try unnamespaced Atom fields for malformed RSS feeds like seancdavis.com
            # Extract the local name from the namespaced atom_css
            if atom_css.startswith("{") and "}" in atom_css:
                unnamespaced_atom = atom_css.split("}", 1)[1]
                result = _get_element_value(root, unnamespaced_atom)
                if result:
                    return result
            
            return None

    elif feed_type == "atom":

        def wrapper(
            rss_css: str, atom_css: str, rdf_css: str, is_attr: bool
        ) -> str | None:
            return _get_element_value(root, atom_css) or (
                (
                    _get_element_value(root, atom_css, attribute="href")
                    or _get_element_value(root, atom_css, attribute="link")
                )
                if is_attr
                else None
            )

    elif feed_type == "rdf":

        def wrapper(
            rss_css: str, atom_css: str, rdf_css: str, is_attr: bool
        ) -> str | None:
            return _get_element_value(root, rdf_css)

    return wrapper


def _get_element_value(
    root: _Element, path: str, attribute: Optional[str] = None
) -> Optional[str]:
    """Get text content or attribute value of an element."""
    el = root.find(path)
    if el is None:
        return None
    if attribute is not None:
        attr_value = el.get(attribute)
        return attr_value.strip() if attr_value else None
    text_value = el.text
    return text_value.strip() if text_value else None

custom_tzinfos: dict[str, int] = {
    'EST': -5 * 3600,  # Eastern Standard Time
    'CST': -6 * 3600,  # Central Standard Time
    'PST': -8 * 3600,  # Pacific Standard Time
    'MST': -7 * 3600,  # Mountain Standard Time
    'EDT': -4 * 3600,  # Eastern Daylight Time
    'CDT': -5 * 3600,  # Central Daylight Time
    'PDT': -7 * 3600,  # Pacific Daylight Time
    'MDT': -6 * 3600,  # Mountain Daylight Time
    'GMT': 0,          # Greenwich Mean Time
    'BST': 1 * 3600,   # British Summer Time
    'CET': 1 * 3600,   # Central European Time
    'CEST': 2 * 3600,  # Central European Summer Time
    'EET': 2 * 3600,   # Eastern European Time 
    'EEST': 3 * 3600,  # Eastern European Summer Time
    'MSK': 3 * 3600,   # Moscow Time
    'IST': 5.5 * 3600, # Indian Standard Time
    'SST': 8 * 3600,   # Singapore Standard Time
    'CST': 8 * 3600,   # China Standard Time
    'JST': 9 * 3600,   # Japan Standard Time
    'KST': 9 * 3600,   # Korea Standard Time
    'AEST': 10 * 3600, # Australian Eastern Standard Time
    'AEDT': 11 * 3600, # Australian Eastern Daylight Time
    'ACST': 9.5 * 3600,# Australian Central Standard Time
    'ACDT': 10.5 * 3600,# Australian Central Daylight Time
    'AWST': 8 * 3600,  # Australian Western Standard Time
    'NZST': 12 * 3600, # New Zealand Standard Time
    'NZDT': 13 * 3600, # New Zealand Daylight Time
    'HAST': -10 * 3600,# Hawaii-Aleutian Standard Time
    'HADT': -9 * 3600, # Hawaii-Aleutian Daylight Time
    'AKST': -9 * 3600, # Alaska Standard Time
    'AKDT': -8 * 3600, # Alaska Daylight Time
    'WET': 0,          # Western European Time
    'WEST': 1 * 3600,  # Western European Summer Time
    # Add more timezones as needed
}

def _parse_date(date_str: str) -> Optional[str]:
    """Parse date string and return as an ISO 8601 formatted UTC string.

    Args:
        date_str: Date string in any common format

    Returns:
        ISO‑8601 formatted UTC date string, or None when parsing fails
    """
    if not date_str:
        return None

    # Fix invalid leap year dates (Feb 29 in non-leap years)
    # This handles feeds with incorrect dates like "2023-02-29"
    import re
    if re.match(r'(\d{4})-02-29', date_str):
        year_match = re.match(r'(\d{4})-02-29', date_str)
        if year_match:
            year = int(year_match.group(1))
            if not ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
                # Not a leap year, change Feb 29 to Feb 28
                date_str = date_str.replace(f'{year}-02-29', f'{year}-02-28')
    
    # Try dateutil.parser first
    try:
        dt = dateutil_parser.parse(date_str, tzinfos=custom_tzinfos, ignoretz=False)
        return dt.astimezone(_UTC).isoformat()
    except ValueError as e:
        # Try parsing just the date portion if full datetime parse fails
        try:
           # Handle ISO8601 format with explicit offset
           if 'T' in date_str and ('+' in date_str or '-' in date_str.split('T')[1]):
               dt = dateutil_parser.parse(date_str)
           elif '24:00:00' in date_str:
                   date_str = date_str.replace('24:00:00', '00:00:00')
                   dt = dateutil_parser.parse(date_str)
           else:
                   dt = dateutil_parser.parse(date_str.split()[0], ignoretz=True)
           # Since no time info, set to start of day UTC
           return dt.astimezone(_UTC).isoformat()
        except ValueError as e:
            pass
    except Exception as e:
        pass

    # Fall back to parsedatetime
    try:
         dt = dateparser.parse(date_str)
         if dt:
            return dt.astimezone(_UTC).isoformat()
    except ValueError:
         pass

    
    # If all parsing attempts fail, return None
    return None

