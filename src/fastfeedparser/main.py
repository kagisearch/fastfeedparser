from __future__ import annotations

import datetime
import gzip
import re
import zlib
from typing import Literal, TYPE_CHECKING
from urllib.request import (
    HTTPErrorProcessor,
    HTTPRedirectHandler,
    Request,
    build_opener,
)

import parsedatetime
from dateutil import parser as dateutil_parser
from lxml import etree

if TYPE_CHECKING:
    from collections.abc import Callable
    from lxml.etree import _Element

_FeedType = Literal["rss", "atom", "rdf"]

_CAL = parsedatetime.Calendar()
_UTC = datetime.timezone.utc


class FastFeedParserDict(dict):
    """A dictionary that allows access to its keys as attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'FastFeedParserDict' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value):
        self[name] = value


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
            content_charset = response.headers.get_content_charset()
            xml_content = (
                content.decode(content_charset) if content_charset else content
            )
    else:
        xml_content = source

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
    root_tag = root.tag.split('}')[-1].lower()  # Get local part of tag name
    
    if root_tag == "rss":
        feed_type = "rss"
        channel = root.find("channel")
        if channel is None:
            raise ValueError("Invalid RSS feed: missing channel element")
        items = channel.findall("item")
    elif root_tag == "feed":
        feed_type = "atom"
        channel = root
        
        # Try multiple namespace variations since feeds can be inconsistent
        namespaces = [
            "{http://www.w3.org/2005/Atom}",  # Standard Atom namespace
            "{http://purl.org/atom/ns#}",     # Old Atom namespace
            "{http://www.w3.org/2005/Atom1}",  # Common typo
            "",  # No namespace
        ]
        
        # Find entries trying different namespaces
        items = []
        for ns in namespaces:
            items = root.findall(f".//{ns}entry")
            if items:
                break
                
        if not items:
            # Try finding entries without namespace if none found
            items = root.findall(".//entry") 
            
        if not items:
            nsmap = getattr(root, 'nsmap', {})
            default_ns = nsmap.get(None, '')
            xmlns = root.get('xmlns', '')
            
            raise ValueError(
                f"Feed tag 'feed' found but no entries found. "
                f"Found namespaces - default: {default_ns}, xmlns: {xmlns}"
            )
    elif root_tag == "rdf":
        feed_type = "rdf"
        channel = root
        items = channel.findall(".//{http://purl.org/rss/1.0/}item")
        if not items:
            items = channel.findall("item")
    else:
        raise ValueError(
            f"Unknown feed type: {root.tag}\n"
            f"Expected one of: RSS (<rss>), Atom (<feed>) or RDF (<rdf:RDF>)"
        )

    feed = _parse_feed_info(channel, feed_type)

    # Parse entries
    entries: list[FastFeedParserDict] = []
    feed["entries"] = entries
    for item in items:
        entry = _parse_feed_entry(item, feed_type)
        # Ensure that titles and descriptions are always present
        entry["title"] = entry.get("title", "").strip()
        entry["description"] = entry.get("description", "").strip()
        entries.append(entry)

    return feed


def _parse_feed_info(channel: _Element, feed_type: _FeedType) -> FastFeedParserDict:
    fields: tuple[tuple[str, str, str, str, bool], ...] = (
        (
            "title",
            "title", 
            "title",  # Try without namespace first
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/rss/1.0/}title",
            False,
        ),
        (
            "link",
            "link",
            "{http://www.w3.org/2005/Atom}link",
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/rss/1.0/}link",
            True,
        ),
        (
            "subtitle",
            "description",
            "{http://www.w3.org/2005/Atom}subtitle",
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/rss/1.0/}description",
            False,
        ),
        (
            "generator",
            "generator",
            "{http://www.w3.org/2005/Atom}generator",
            "{http://purl.org/rss/1.0/}channel/{http://webns.net/mvcb/}generatorAgent",
            False,
        ),
        (
            "publisher",
            "publisher",
            "{http://www.w3.org/2005/Atom}publisher",
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/dc/elements/1.1/}publisher",
            False,
        ),
        (
            "author",
            "author",
            "{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name",
            "{http://purl.org/rss/1.0/}channel/{http://purl.org/dc/elements/1.1/}creator",
            False,
        ),
        (
            "updated",
            "lastBuildDate",
            "{http://www.w3.org/2005/Atom}updated",
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
    feed_links: list[dict] = []
    feed["links"] = feed_links
    feed_link: str | None = None
    for link in channel.findall("{http://www.w3.org/2005/Atom}link"):
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
    feed["id"] = _get_element_value(channel, "{http://www.w3.org/2005/Atom}id")

    # Add generator_detail
    generator = channel.find("{http://www.w3.org/2005/Atom}generator")
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

    return FastFeedParserDict(feed=feed)


def _parse_feed_entry(item: _Element, feed_type: _FeedType) -> FastFeedParserDict:
    fields: tuple[tuple[str, str, str, str, bool], ...] = (
        (
            "title",
            "title",
            "title",  # Try without namespace first
            "{http://purl.org/rss/1.0/}title",
            False,
        ),
        (
            "link",
            "link",
            "{http://www.w3.org/2005/Atom}link",
            "{http://purl.org/rss/1.0/}link",
            True,
        ),
        (
            "description",
            "description",
            "{http://www.w3.org/2005/Atom}summary",
            "{http://purl.org/rss/1.0/}description",
            False,
        ),
        (
            "published",
            "pubDate",
            "{http://www.w3.org/2005/Atom}published",
            "{http://purl.org/dc/elements/1.1/}date",
            False,
        ),
        (
            "updated",
            "lastBuildDate",
            "{http://www.w3.org/2005/Atom}updated",
            "{http://purl.org/dc/terms/}modified",
            False,
        ),
    )

    entry = FastFeedParserDict()
    get_field_value = _field_value_getter(item, feed_type)
    for field in fields:
        value = get_field_value(*field[1:])
        if value:
            name = field[0]
            if name in {"published", "updated"}:
                value = _parse_date(value)
            entry[name] = value

    # If published is missing but updated exists, use updated as published
    if "updated" in entry and "published" not in entry:
        entry["published"] = entry["updated"]

    # Handle links
    entry_links: list[dict] = []
    entry["links"] = entry_links
    alternate_link: dict | None = None
    # Try both namespaced and non-namespaced link elements
    for link in item.findall(".//link") + item.findall(".//{http://www.w3.org/2005/Atom}link"):
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

    if is_guid_url:
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

    content = None
    if feed_type == "rss":
        content = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
        if content is None:
            content = item.find("content")
    elif feed_type == "atom":
        content = item.find("{http://www.w3.org/2005/Atom}content")

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
    media_contents: list[dict] = []

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
    enclosures: list[dict] = []
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
            "{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name",
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

    return entry


def _field_value_getter(
    root: _Element, feed_type: _FeedType
) -> Callable[[str, str, str, bool], str | None]:
    if feed_type == "rss":

        def wrapper(
            rss_css: str, atom_css: str, rdf_css: str, is_attr: bool
        ) -> str | None:
            return _get_element_value(root, rss_css) or (
                (
                    _get_element_value(root, atom_css, attribute="href")
                    or _get_element_value(root, atom_css, attribute="link")
                )
                if is_attr
                else _get_element_value(root, atom_css)
            )

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
    root: _Element, path: str, attribute: str | None = None
) -> str | None:
    """Get text content or attribute value of an element."""
    el = root.find(path)
    if el is None:
        return None
    if attribute is not None:
        return el.get(attribute)
    return el.text


def _parse_date(date_str: str) -> str | None:
    """Parse date string and return as an ISO 8601 formatted UTC string.

    Args:
        date_str: Date string in any common format

    Returns:
        ISO 8601 formatted UTC date string or None if parsing fails
    """
    if not date_str:
        return None

    # Try dateutil.parser first
    try:
        dt = dateutil_parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UTC)
        else:
            dt = dt.astimezone(_UTC)
        return dt.isoformat()
    except (ValueError, OverflowError):
        pass

    # Fall back to parsedatetime
    try:
        # Parse with timezone awareness
        time_struct, parse_status = _CAL.parseDT(date_str, tzinfo=_UTC)
        if parse_status:
            # time_struct is already a datetime with proper timezone
            return time_struct.astimezone(_UTC).isoformat()
    except ValueError:
        pass

    # If all parsing attempts fail, return None instead of original string
    return None
