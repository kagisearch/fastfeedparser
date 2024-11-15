import datetime
import httpx
from lxml import etree
import parsedatetime
from dateutil import parser as dateutil_parser

MEDIA_NS = "http://search.yahoo.com/mrss/"


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


def parse(xml_content):
    """Parse the XML content of a feed."""
    if not xml_content.strip():
        raise ValueError("Empty content")

    # Handle decoding if content is bytes
    if isinstance(xml_content, bytes):
        encodings = ["utf-8", "iso-8859-1", "windows-1252"]
        decoded = None
        for encoding in encodings:
            try:
                decoded = xml_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if decoded is None:
            raise ValueError("Could not decode content with any supported encoding")
        xml_content = decoded

    # Ensure we have bytes for lxml
    xml_content = xml_content.encode("utf-8", errors="replace")

    parser = etree.XMLParser(recover=True)
    try:
        root = etree.fromstring(xml_content, parser=parser)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Failed to parse XML content: {str(e)}")

    # Check if root is None
    if root is None:
        raise ValueError("Failed to parse XML content: root element is None")

    namespaces = root.nsmap

    # Clean up the XML tree
    for element in root.iter():
        if element.text:
            element.text = element.text.replace("\x00", "")  # Remove null characters
        if element.tail:
            element.tail = element.tail.replace("\x00", "")  # Remove null characters

    feed = FastFeedParserDict()
    entries = []

    # Determine feed type based on content structure
    if root.tag == "rss" or root.tag == f"{{{namespaces.get(None, '')}}}rss":
        feed_type = "rss"
        channel = root.find("channel")
        if channel is None:
            raise ValueError("Invalid RSS feed: missing channel element")
        items = channel.findall("item")
    elif (
        root.tag.endswith("feed")
        or root.tag
        == f"{{{namespaces.get('atom', 'http://www.w3.org/2005/Atom')}}}feed"
    ):
        feed_type = "atom"
        channel = root
        items = root.findall(
            f".//{{{namespaces.get('atom', 'http://www.w3.org/2005/Atom')}}}entry"
        ) or root.findall("entry")
    elif (
        root.tag.endswith("RDF")
        or root.tag
        == f"{{{namespaces.get('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')}}}RDF"
    ):
        feed_type = "rdf"
        channel = root
        items = root.findall(
            f".//{{{namespaces.get('rss', 'http://purl.org/rss/1.0/')}}}item"
        ) or root.findall("item")
    else:
        raise ValueError(f"Unknown feed type: {root.tag}")

    if not items:
        raise ValueError("No entries found in the feed")

    def parse_feed_info(channel, feed_type, namespaces):
        feed = FastFeedParserDict()

        def get_feed_value(rss_field, atom_field, rdf_field=None, is_attr=False):
            if feed_type == "rss":
                value = get_element_value(channel, rss_field, namespaces) or (
                    (
                        get_element_value(
                            channel, atom_field, namespaces, attribute="href"
                        )
                        or get_element_value(
                            channel, atom_field, namespaces, attribute="link"
                        )
                    )
                    if is_attr
                    else get_element_value(channel, atom_field, namespaces)
                )
            elif feed_type == "atom":
                value = get_element_value(channel, atom_field, namespaces) or (
                    (
                        get_element_value(
                            channel, atom_field, namespaces, attribute="href"
                        )
                        or get_element_value(
                            channel, atom_field, namespaces, attribute="link"
                        )
                    )
                    if is_attr
                    else ""
                )
            else:  # RDF
                value = (
                    get_element_value(channel, rdf_field, namespaces)
                    if rdf_field
                    else ""
                )
            return value if value else None

        fields = [
            (
                "title",
                "title",
                "{http://www.w3.org/2005/Atom}title",
                "{http://purl.org/rss/1.0/}channel/{http://purl.org/rss/1.0/}title",
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
            ),
            (
                "generator",
                "generator",
                "{http://www.w3.org/2005/Atom}generator",
                "{http://purl.org/rss/1.0/}channel/{http://webns.net/mvcb/}generatorAgent",
            ),
            (
                "publisher",
                "publisher",
                "{http://www.w3.org/2005/Atom}publisher",
                "{http://purl.org/rss/1.0/}channel/{http://purl.org/dc/elements/1.1/}publisher",
            ),
            (
                "author",
                "author",
                "{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name",
                "{http://purl.org/rss/1.0/}channel/{http://purl.org/dc/elements/1.1/}creator",
            ),
            (
                "updated",
                "lastBuildDate",
                "{http://www.w3.org/2005/Atom}updated",
                "{http://purl.org/rss/1.0/}channel/{http://purl.org/dc/elements/1.1/}date",
            ),
        ]

        for field in fields:
            value = get_feed_value(*field[1:])
            if value:
                feed[field[0]] = value

        # Add title_detail and subtitle_detail
        if "title" in feed:
            feed["title_detail"] = {
                "type": "text/plain",
                "language": channel.get("{http://www.w3.org/XML/1998/namespace}lang"),
                "base": channel.get("{http://www.w3.org/XML/1998/namespace}base"),
                "value": feed["title"],
            }
        if "subtitle" in feed:
            feed["subtitle_detail"] = {
                "type": "text/plain",
                "language": channel.get("{http://www.w3.org/XML/1998/namespace}lang"),
                "base": channel.get("{http://www.w3.org/XML/1998/namespace}base"),
                "value": feed["subtitle"],
            }

        # Add links
        feed["links"] = []
        feed_link = None
        for link in channel.findall("{http://www.w3.org/2005/Atom}link"):
            rel = link.get("rel")
            href = link.get("href")
            if rel is None and href:
                feed_link = href
            elif rel not in ["hub", "self", "replies", "edit"]:
                feed["links"].append(
                    {
                        "rel": rel,
                        "type": link.get("type"),
                        "href": href,
                        "title": link.get("title"),
                    }
                )

        if feed_link:
            feed["link"] = feed_link
            feed["links"].insert(
                0, {"rel": "alternate", "type": "text/html", "href": feed_link}
            )

        # Add id
        feed["id"] = get_element_value(
            channel, "{http://www.w3.org/2005/Atom}id", namespaces
        )

        # Add generator_detail
        generator = channel.find("{http://www.w3.org/2005/Atom}generator")
        if generator is not None:
            feed["generator_detail"] = {
                "name": generator.text,
                "version": generator.get("version"),
                "href": generator.get("uri"),
            }

        feed["language"] = channel.get("{http://www.w3.org/XML/1998/namespace}lang")
        feed["guidislink"] = False

        if feed_type == "rss":
            comments = get_element_value(channel, "comments", namespaces)
            if comments:
                feed["comments"] = comments

        # Additional checks for publisher and author
        if "publisher" not in feed:
            webmaster = get_element_value(channel, "webMaster", namespaces)
            if webmaster:
                feed["publisher"] = webmaster
        if "author" not in feed:
            managing_editor = get_element_value(channel, "managingEditor", namespaces)
            if managing_editor:
                feed["author"] = managing_editor

        return {"feed": feed}

    feed.update(parse_feed_info(channel, feed_type, namespaces))

    # Parse entries
    def parse_feed_entry(item, feed_type, namespaces):
        entry = FastFeedParserDict()

        def get_entry_value(rss_field, atom_field, rdf_field=None, is_attr=False):
            if feed_type == "rss":
                value = get_element_value(item, rss_field, namespaces) or (
                    (
                        get_element_value(
                            item, atom_field, namespaces, attribute="href"
                        )
                        or get_element_value(
                            item, atom_field, namespaces, attribute="link"
                        )
                    )
                    if is_attr
                    else get_element_value(item, atom_field, namespaces)
                )
            elif feed_type == "atom":
                value = get_element_value(item, atom_field, namespaces) or (
                    (
                        get_element_value(
                            item, atom_field, namespaces, attribute="href"
                        )
                        or get_element_value(
                            item, atom_field, namespaces, attribute="link"
                        )
                    )
                    if is_attr
                    else ""
                )
            else:  # RDF
                value = (
                    get_element_value(item, rdf_field, namespaces) if rdf_field else ""
                )
            return value if value else None

        fields = [
            (
                "title",
                "title",
                "{http://www.w3.org/2005/Atom}title",
                "{http://purl.org/rss/1.0/}title",
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
            ),
            (
                "published",
                "pubDate",
                "{http://www.w3.org/2005/Atom}published",
                "{http://purl.org/dc/elements/1.1/}date",
            ),
            (
                "updated",
                "lastBuildDate",
                "{http://www.w3.org/2005/Atom}updated",
                "{http://purl.org/dc/terms/}modified",
            ),
        ]

        for field in fields:
            value = get_entry_value(*field[1:])
            if value:
                if field[0] in ["published", "updated"]:
                    value = parse_date(value)
                entry[field[0]] = value

        # If published is missing but updated exists, use updated as published
        if "updated" in entry and "published" not in entry:
            entry["published"] = entry["updated"]

        # Handle links
        entry["links"] = []
        alternate_link = None
        for link in item.findall("{http://www.w3.org/2005/Atom}link"):
            rel = link.get("rel")
            href = link.get("href") or link.get(
                "link"
            )  # Check both 'href' and 'link' attributes
            if href:
                if rel == "alternate":
                    alternate_link = {
                        "rel": rel,
                        "type": link.get("type"),
                        "href": href,
                        "title": link.get("title"),
                    }
                elif rel not in ["edit", "self"]:
                    entry["links"].append(
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
        is_guid_url = guid_text and (
            guid_text.startswith("http://") or guid_text.startswith("https://")
        )

        if is_guid_url:
            # Prefer guid as link when it looks like a URL
            entry["link"] = guid_text
            if alternate_link:
                entry["links"].insert(
                    0, {"rel": "alternate", "type": "text/html", "href": guid_text}
                )
        elif alternate_link:
            entry["links"].insert(0, alternate_link)
            entry["link"] = alternate_link["href"]
        elif (
            "link" not in entry
            and guid is not None
            and guid.get("isPermaLink") == "true"
        ):
            entry["link"] = guid.text

        content = None
        if feed_type == "rss":
            content = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
            if content is None:
                for ns, uri in namespaces.items():
                    if uri == "http://purl.org/rss/1.0/modules/content/":
                        content = item.find(f"{{{uri}}}encoded")
                        break
            if content is None:
                content = item.find("content")
        elif feed_type == "atom":
            content = item.find("{http://www.w3.org/2005/Atom}content")

        if content is not None:
            content_type = content.get("type", "text/html")  # Default to text/html
            if content_type in ["html", "xhtml"]:
                # For XHTML content, serialize the entire content
                content_value = etree.tostring(
                    content, encoding="unicode", method="xml"
                )
            else:
                content_value = content.text if content.text else ""

            entry["content"] = [
                {
                    "type": content_type,
                    "language": content.get(
                        "{http://www.w3.org/XML/1998/namespace}lang"
                    ),
                    "base": content.get("{http://www.w3.org/XML/1998/namespace}base"),
                    "value": content_value,
                }
            ]

        # If content is still empty, try to use description
        if "content" not in entry or not entry["content"]:
            description = item.find("description")
            if description is not None and description.text:
                entry["content"] = [{"type": "text/html", "value": description.text}]

        if ("description" not in entry) and ("content" in entry or "summary" in entry):
            content = (
                entry.get("content", [{}])[0].get("value", "")
                if entry.get("content")
                else ""
            )
            if content:
                try:
                    html_content = etree.HTML(content)
                    if html_content is not None:
                        content_text = html_content.xpath("string()")
                        entry["description"] = " ".join(content_text.split()[:256])
                    else:
                        entry["description"] = content[:512]
                except etree.ParserError:
                    entry["description"] = content[:512]
            else:
                entry["description"] = entry.get("summary", "")[:512]

        # Handle media content
        media_contents = []

        # Process media:content elements
        for media in item.findall(f".//{{{MEDIA_NS}}}content"):
            media_item = {
                "url": media.get("url"),
                "type": media.get("type"),
                "medium": media.get("medium"),
                "width": media.get("width"),
                "height": media.get("height"),
            }

            # Convert width/height to integers if present
            for dim in ["width", "height"]:
                if media_item.get(dim):
                    try:
                        media_item[dim] = int(media_item[dim])
                    except (ValueError, TypeError):
                        del media_item[dim]

            # Handle sibling elements
            # Handle title
            title = media.find(f"{{{MEDIA_NS}}}title")
            if title is not None and title.text:
                media_item["title"] = title.text.strip()

            # Handle credit
            credit = media.find(f"{{{MEDIA_NS}}}credit")
            if credit is not None:
                media_item["credit"] = credit.text.strip() if credit.text else None
                media_item["credit_scheme"] = credit.get("scheme")

            # Handle text
            text = media.find(f"{{{MEDIA_NS}}}text")
            if text is not None and text.text:
                media_item["text"] = text.text.strip()

            # Handle description - check both direct child and sibling elements
            desc = media.find(f"{{{MEDIA_NS}}}description")
            if desc is None:
                desc = media.getparent().find(f"{{{MEDIA_NS}}}description")
            if desc is not None and desc.text:
                media_item["description"] = desc.text.strip()

            # Handle credit - check both direct child and sibling elements
            credit = media.find(f"{{{MEDIA_NS}}}credit")
            if credit is None:
                credit = media.getparent().find(f"{{{MEDIA_NS}}}credit")
            if credit is not None and credit.text:
                media_item["credit"] = credit.text.strip()

            # Handle thumbnail as a separate URL field
            thumbnail = media.find(f"{{{MEDIA_NS}}}thumbnail")
            if thumbnail is not None:
                media_item["thumbnail_url"] = thumbnail.get("url")

            # Remove None values
            media_item = {k: v for k, v in media_item.items() if v is not None}

            if media_item:  # Only append if we have some content
                media_contents.append(media_item)

        # If no media:content but there are standalone thumbnails, add them
        if not media_contents:
            for thumbnail in item.findall(f".//{{{MEDIA_NS}}}thumbnail"):
                if thumbnail.getparent().tag != f"{{{MEDIA_NS}}}content":
                    thumb_item = {
                        "url": thumbnail.get("url"),
                        "type": "image/jpeg",  # Default type for thumbnails
                        "width": thumbnail.get("width"),
                        "height": thumbnail.get("height"),
                    }
                    # Convert dimensions to integers if present
                    for dim in ["width", "height"]:
                        if thumb_item.get(dim):
                            try:
                                thumb_item[dim] = int(thumb_item[dim])
                            except (ValueError, TypeError):
                                del thumb_item[dim]

                    # Remove None values
                    thumb_item = {k: v for k, v in thumb_item.items() if v is not None}

                    if thumb_item:
                        media_contents.append(thumb_item)

        if media_contents:
            entry["media_content"] = media_contents

        # Handle enclosures
        enclosures = []
        for enclosure in item.findall("enclosure"):
            enc_item = {
                "url": enclosure.get("url"),
                "type": enclosure.get("type"),
                "length": enclosure.get("length"),
            }
            # Convert length to integer if present and valid
            if enc_item["length"]:
                try:
                    enc_item["length"] = int(enc_item["length"])
                except (ValueError, TypeError):
                    del enc_item["length"]

            # Remove None values
            enc_item = {k: v for k, v in enc_item.items() if v is not None}

            if enc_item.get("url"):  # Only append if we have a URL
                enclosures.append(enc_item)

        if enclosures:
            entry["enclosures"] = enclosures

        author = (
            get_entry_value(
                "author",
                "{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name",
                "{http://purl.org/dc/elements/1.1/}creator",
            )
            or get_entry_value(
                "dc:creator",
                "{http://purl.org/dc/elements/1.1/}creator",
                "{http://purl.org/dc/elements/1.1/}creator",
            )
            or get_element_value(
                item, "{http://purl.org/dc/elements/1.1/}creator", namespaces
            )
            or get_element_value(item, "author", namespaces)
        )
        if author:
            entry["author"] = author

        if feed_type == "rss":
            comments = get_element_value(item, "comments", namespaces)
            if comments:
                entry["comments"] = comments

        return entry

    # Usage:
    for item in items:
        entry = parse_feed_entry(item, feed_type, namespaces)
        entries.append(entry)

    # Trim titles and descriptions
    for entry in entries:
        entry["title"] = trim_text(entry.get("title", ""))
        entry["description"] = trim_text(entry.get("description", ""))
        entry["summary"] = trim_text(entry.get("summary", ""))

    feed["entries"] = entries
    return feed


def trim_text(text):
    """Trim leading and trailing whitespace from text."""
    return text.strip() if text else ""


def get_element_value(element, tag, namespaces, attribute=None):
    """Get text content or attribute value of an element."""
    if ":" in tag and not tag.startswith("{"):
        prefix, tag_name = tag.split(":")
        uri = namespaces.get(prefix, "")
        tag = f"{{{uri}}}{tag_name}"
    el = element.find(tag)
    if el is not None:
        if attribute:
            return el.get(attribute)
        else:
            return el.text
    return None


# Initialize parsedatetime Calendar
cal = parsedatetime.Calendar()


def parse_date(date_str):
    """Parse date string and return as a standard string in UTC."""
    if not date_str:
        return None

    # Try dateutil.parser first
    try:
        dt = dateutil_parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            dt = dt.astimezone(datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except (ValueError, OverflowError):
        pass

    # Fall back to parsedatetime
    try:
        time_struct, parse_status = cal.parse(date_str)
        if parse_status:
            dt = datetime.datetime(*time_struct[:6], tzinfo=datetime.timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        pass

    # If all parsing attempts fail, return the original string
    return date_str


def parse_url(url):
    """Parse a URL and return a FastFeedParserDict object."""
    return parse(url)


def fetch_url(url):
    """Fetch content from a URL."""
    with httpx.Client() as client:
        response = client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.text
